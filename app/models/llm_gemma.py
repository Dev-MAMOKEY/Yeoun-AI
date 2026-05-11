"""GemmaLLM — `google/gemma-4-E4B-it` 멀티모달 로더 + 추론.

- 부팅 시 AWQ INT4 가중치 우선 로드. Blackwell(sm_120) 등에서 AWQ 커널이
  미지원이면 BF16 으로 폴백한다.
- 음성 입력은 멀티모달 audio-in 으로 Gemma 가 직접 이해해 한국어 응답을 만든다.
- 토큰은 transformers `TextIteratorStreamer` 를 별도 스레드에 띄워 async 큐로
  yield (단일 워커 + asyncio 단일 루프 전제).
- `GPU_ENABLED=false` 모드는 모든 호출을 더미 텍스트/토큰 반환으로 단락 처리.

이슈 #6 범위는 로더·전사·스트림 API 까지. 시스템 프롬프트 조립과 conversation
파이프라인은 #10 에서 본 클래스를 호출한다.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Literal

logger = logging.getLogger("yeoun")

# model card 권장 샘플링 — 모든 추론 호출 공통.
_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 64}

# 명세서 결정 — KV 캐시 절약 위해 컨텍스트 8K 로 제한 (모델 자체는 256K 까지 지원).
# 시스템 프롬프트 + 히스토리 + 현재 입력 합계. 호출자(#10) 가 히스토리를 자른다.
_MAX_CONTEXT_TOKENS = 8192
# 한 응답이 생성하는 최대 토큰 수 — 한국어 단문 응답 위주라 1024 면 충분.
_MAX_RESPONSE_TOKENS = 1024

# 스트리밍 종료를 알리는 sentinel — `run_in_executor` 로 next 결과를 await 할 때
# StopIteration 이 그대로 yield 되지 못해 별도 표식 객체로 대신한다.
_STOP_SENTINEL: object = object()


def _next_or_stop(iterator):  # type: ignore[no-untyped-def]
    """sync iterator 의 다음 값을 가져오거나 StopIteration → sentinel 로 변환."""
    try:
        return next(iterator)
    except StopIteration:
        return _STOP_SENTINEL

_DUMMY_RESPONSE_TOKENS = (
    "[",
    "더미",
    " 응답",
    "]",
    " GPU_ENABLED=false",
    " 모드",
    " — ",
    "이",
    " 메시지는",
    " 실제",
    " 모델",
    " 출력이",
    " 아닙니다",
    ".",
)

LlmStatus = Literal["not_loaded", "loading", "loaded", "error"]


class GemmaLLM:
    """Gemma 4 E4B-it 멀티모달 LLM 래퍼."""

    def __init__(
        self,
        *,
        awq_path: str | None,
        bf16_path: str | None,
        gpu_enabled: bool,
    ) -> None:
        self._awq_path = awq_path
        self._bf16_path = bf16_path
        self._gpu_enabled = gpu_enabled
        self._status: LlmStatus = "not_loaded"
        self._model = None
        self._processor = None
        self._loaded_variant: Literal["awq", "bf16"] | None = None

    # --- 라이프사이클 -------------------------------------------------------
    async def load(self) -> None:
        """가중치를 GPU 에 적재.

        AWQ INT4 우선 시도 → 실패 시(Blackwell 등에서 AWQ 커널 미지원) BF16 폴백.
        둘 다 실패하면 `error` 상태로 예외 raise. 더미 모드는 즉시 `loaded`.
        """
        if not self._gpu_enabled:
            logger.info("GPU_ENABLED=false → GemmaLLM 더미 모드")
            self._status = "loaded"
            return

        self._status = "loading"

        if self._awq_path:
            try:
                await asyncio.to_thread(self._load_variant, "awq", self._awq_path)
                self._loaded_variant = "awq"
                self._status = "loaded"
                return
            except Exception as exc:  # noqa: BLE001 — 어떤 실패든 BF16 폴백 시도
                logger.warning(
                    "AWQ 로드 실패 (Blackwell 커널 미지원 가능) — BF16 폴백 시도: %s",
                    exc,
                )

        if self._bf16_path:
            try:
                await asyncio.to_thread(self._load_variant, "bf16", self._bf16_path)
                self._loaded_variant = "bf16"
                self._status = "loaded"
                return
            except Exception:
                logger.exception("BF16 로드도 실패")
                self._status = "error"
                raise

        self._status = "error"
        raise RuntimeError(
            "LLM 가중치 경로가 모두 비어 있습니다. LLM_AWQ_PATH 또는 LLM_BF16_PATH 중 "
            "최소 하나를 설정하거나 GPU_ENABLED=false 로 더미 모드를 사용하십시오."
        )

    def _load_variant(self, variant: Literal["awq", "bf16"], path: str) -> None:
        """단일 변형 가중치를 동기 로드. `asyncio.to_thread` 안에서 호출된다."""
        # transformers >= 4.50 의 멀티모달 클래스. 일부 버전엔 이 이름이 없어
        # AutoModelForCausalLM 으로 폴백 — Gemma 4 model card 의 멀티모달 예시는
        # AutoModelForMultimodalLM 을 쓰지만 가용성이 환경마다 다르다.
        try:
            from transformers import AutoModelForMultimodalLM as _AutoModel  # type: ignore[attr-defined]
        except ImportError:
            from transformers import AutoModelForCausalLM as _AutoModel  # type: ignore[assignment]
        from transformers import AutoProcessor

        logger.info("Gemma %s 로드 시작: %s", variant.upper(), path)
        self._processor = AutoProcessor.from_pretrained(path)

        load_kwargs: dict[str, object] = {"device_map": "auto"}
        if variant == "awq":
            # AWQ 양자화 가중치는 dtype 을 그대로 따른다 ("auto").
            load_kwargs["dtype"] = "auto"
        else:
            import torch

            load_kwargs["dtype"] = torch.bfloat16

        self._model = _AutoModel.from_pretrained(path, **load_kwargs)
        logger.info("Gemma %s 로드 완료", variant.upper())

    async def unload(self) -> None:
        """모델 자원을 해제. 더미 모드면 no-op."""
        if not self._gpu_enabled:
            self._status = "not_loaded"
            return
        self._model = None
        self._processor = None
        self._loaded_variant = None
        self._status = "not_loaded"
        # GPU 메모리 회수 — torch 가 import 가능할 때만.
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # --- 상태 --------------------------------------------------------------
    @property
    def status(self) -> LlmStatus:
        return self._status

    @property
    def variant(self) -> Literal["awq", "bf16"] | None:
        return self._loaded_variant

    # --- 추론 ---------------------------------------------------------------
    async def transcribe(self, audio_path: str | Path) -> str:
        """짧은 오디오 → 한국어 전사. 더미 모드는 고정 문자열.

        전사는 deterministic 결과가 필요하므로 `do_sample=False` (greedy) 로 호출.
        멀티모달 입력은 `apply_chat_template` 의 mixed content 메시지에 `audio` 타입
        부분을 넣어 전달한다.
        """
        if not self._gpu_enabled:
            return f"[더미 전사] {Path(audio_path).name}"
        if self._model is None or self._processor is None:
            raise RuntimeError("GemmaLLM 이 로드되지 않았습니다.")
        return await asyncio.to_thread(self._run_transcribe, str(audio_path))

    def _run_transcribe(self, audio_path: str) -> str:
        """동기 전사 — `asyncio.to_thread` 안에서 호출."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "url": audio_path},
                    {
                        "type": "text",
                        "text": (
                            "위 음성을 한국어로 그대로 전사해 주세요. "
                            "부가 설명 없이 발화 텍스트만 반환합니다."
                        ),
                    },
                ],
            }
        ]
        inputs = self._processor.apply_chat_template(  # type: ignore[union-attr]
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self._model.device)  # type: ignore[union-attr]
        input_len = inputs["input_ids"].shape[-1]
        outputs = self._model.generate(  # type: ignore[union-attr]
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )
        decoded = self._processor.decode(  # type: ignore[union-attr]
            outputs[0][input_len:], skip_special_tokens=True
        )
        return decoded.strip()

    async def stream_response(
        self,
        *,
        system_prompt: str,
        history: list[dict],
        user_audio_path: str | Path,
    ) -> AsyncIterator[str]:
        """audio-in 응답을 토큰 단위로 yield.

        구성: 시스템 프롬프트 + 텍스트 히스토리 + 현재 사용자 음성 메시지.
        `TextIteratorStreamer` 를 별도 스레드에 띄우고, sync iterator 결과를
        `loop.run_in_executor` 로 가져와 async 시퀀스로 변환한다.
        더미 모드는 고정 토큰 시퀀스.

        Args:
            system_prompt: 시스템 프롬프트 본문. 페르소나 정체성·인터뷰 답변
                10 개·응답 정책이 합쳐진 문자열 (#10 에서 조립).
            history: 텍스트 메시지 히스토리. 각 항목 `{"role": "user"|"assistant",
                "text": str}` 형식.
            user_audio_path: 현재 사용자 음성 파일 경로.
        """
        if not self._gpu_enabled:
            for token in _DUMMY_RESPONSE_TOKENS:
                # 실모드 흉내내기 위해 한 토큰당 짧은 await.
                await asyncio.sleep(0)
                yield token
            return
        if self._model is None or self._processor is None:
            raise RuntimeError("GemmaLLM 이 로드되지 않았습니다.")

        from threading import Thread

        from transformers import TextIteratorStreamer  # type: ignore[import-not-found]

        # 메시지 조립 — 시스템 + 히스토리 + 현재 audio.
        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["text"]})
        messages.append(
            {
                "role": "user",
                "content": [{"type": "audio", "url": str(user_audio_path)}],
            }
        )

        inputs = self._processor.apply_chat_template(  # type: ignore[union-attr]
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self._model.device)  # type: ignore[union-attr]

        streamer = TextIteratorStreamer(
            self._processor.tokenizer,  # type: ignore[union-attr]
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=_MAX_RESPONSE_TOKENS,
            do_sample=True,
            **_SAMPLING,
        )

        def _generate_safely() -> None:
            # `model.generate` 가 예외로 죽으면 `TextIteratorStreamer` 의 내부 큐에
            # StopIteration 신호가 안 들어가 `next(streamer)` 가 영구 블록된다.
            # finally 에서 `streamer.end()` 를 직접 호출해 호출 측 async iterator 가
            # 정상 종료 시그널을 받도록 한다.
            try:
                self._model.generate(**generation_kwargs)  # type: ignore[union-attr]
            except Exception:
                logger.exception("Gemma generate 예외 — streamer 종료 신호 강제 주입")
                raise
            finally:
                try:
                    streamer.end()  # type: ignore[attr-defined]
                except Exception:
                    logger.exception("streamer.end() 호출 실패")

        thread = Thread(target=_generate_safely, daemon=True)
        thread.start()

        loop = asyncio.get_running_loop()
        try:
            while True:
                token = await loop.run_in_executor(None, _next_or_stop, streamer)
                if token is _STOP_SENTINEL:
                    break
                yield token
        finally:
            # 생성이 끝났거나 호출자가 중단했을 때 스레드 회수.
            await loop.run_in_executor(None, thread.join, 5.0)
            if thread.is_alive():
                # 5 초 타임아웃 후에도 살아있으면 daemon 이라도 GPU 점유 우려.
                # 후속 stream_response 호출에서 같은 GPU 자원이 겹칠 수 있으므로
                # 로그로 분명히 남긴다 — `gpu_semaphore` 직렬화는 호출자(#10) 책임.
                logger.warning(
                    "Gemma generate 스레드가 join 타임아웃 후에도 alive — GPU 점유 가능"
                )
