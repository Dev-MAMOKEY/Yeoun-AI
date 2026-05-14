"""DittoTalkingHead — `antgroup/ditto-talkinghead` 영상 생성 래퍼.

HF README(<https://huggingface.co/digital-avatar/ditto-talkinghead>) 패턴 따름:
- 코드 repo: `git clone github.com/antgroup/ditto-talkinghead` → `DITTO_VENDOR_DIR`
- 가중치: `cd $DITTO_VENDOR_DIR && git clone hf.co/digital-avatar/ditto-talkinghead checkpoints`
- 추론: `python <vendor>/inference.py --data_root ... --cfg_pkl ... --audio_path ... --source_path ... --output_path ...`

본 모듈은 위 CLI 를 `asyncio.create_subprocess_exec` 으로 호출한다.
- 매 호출이 새 프로세스 → GPU 메모리 자동 회수 (명세 "온디맨드 로드/언로드" 자연 충족).
- `GPU_ENABLED=false` 모드는 placeholder bytes 를 output_path 에 쓰고 반환.
- GPU 직렬화는 호출자 가 `ModelRegistry.ditto_semaphore` 로 보호 (Gemma/TTS 와 분리한 락).
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Literal

logger = logging.getLogger("yeoun")

# Ditto inference.py 가 입력으로 받는 wav 의 가정 sample rate — Hubert 기반.
_SILENCE_SAMPLE_RATE = 16_000

# 더미 모드 placeholder mp4 — 최소 유효 ftyp box. ffmpeg 가 없어도 쓸 수 있게 raw bytes.
_DUMMY_MP4_BYTES = (
    b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00isomiso2mp41"
)

DittoStatus = Literal["not_loaded", "loading", "loaded", "error"]


class FaceDetectionError(RuntimeError):
    """얼굴 미검출로 인한 Ditto 합성 실패.

    호출자(`process_persona` · `process_message`) 가 이 도메인 예외를 잡아
    `RuntimeError:` 타입 prefix 없이 `str(exc)` 만 사용자 친화 메시지로 노출한다.
    """


def _is_face_detection_failure(stderr_tail: str) -> bool:
    """vendor 의 stderr 가 얼굴 미검출 패턴인지 판정.

    `source2info._crop()` / `avatar_registrar` 가 얼굴을 못 찾으면 빈 결과를
    반환 → 호출 측이 `img_crop, M_c2o, lmk203 = ...` 처럼 unpack 하다 None
    `TypeError: cannot unpack non-iterable NoneType object` 으로 죽는다.
    vendor 가 명시적 예외 raise 를 안 하므로 본 헬퍼가 stderr 마지막 부분을
    보고 얼굴 미검출 케이스를 분류해 사용자 친화 메시지로 환원한다.

    정밀도 강화: `cannot unpack non-iterable NoneType` 는 Python 의 generic
    TypeError 라 같은 vendor 모듈의 다른 unpack 오류도 매칭될 수 있어
    `_crop`(메서드명) 또는 `img_crop`(unpack 변수명) 추가 매칭으로 false positive
    회피. 테스트가 본 헬퍼·메시지 상수를 직접 import 해 단위 검증한다.
    """
    return (
        "cannot unpack non-iterable NoneType" in stderr_tail
        and ("source2info" in stderr_tail or "avatar_registrar" in stderr_tail)
        and ("_crop" in stderr_tail or "img_crop" in stderr_tail)
    )


_FACE_DETECTION_FAILURE_MESSAGE = (
    "얼굴 검출 실패: 업로드한 사진에서 얼굴을 찾을 수 없습니다. "
    "정면 얼굴이 또렷한 사진을 사용해 주세요."
)


class DittoTalkingHead:
    """Ditto-TalkingHead inference.py CLI 래퍼."""

    def __init__(
        self,
        *,
        vendor_dir: str | None,
        data_root: str | None,
        cfg_pkl: str | None,
        gpu_enabled: bool,
    ) -> None:
        # 인스턴스 생성 시점(워커 부팅, cwd 명시적) 에 한 번 resolve.
        # subprocess cwd 를 vendor 디렉토리로 바꿔도 인자 경로가 cwd 안에서 또
        # append 되어 이중 중첩 → FileNotFoundError 가 나는 문제를 한 곳에서 차단.
        # load() 와 render_speak() 가 같은 절대경로를 공유해 검증·실행 불일치도 제거.
        self._vendor_dir = str(Path(vendor_dir).resolve()) if vendor_dir else None
        self._data_root = str(Path(data_root).resolve()) if data_root else None
        self._cfg_pkl = str(Path(cfg_pkl).resolve()) if cfg_pkl else None
        self._gpu_enabled = gpu_enabled
        self._status: DittoStatus = "not_loaded"

    # --- 라이프사이클 -------------------------------------------------------
    async def load(self) -> None:
        """경로 존재 검증. 더미 모드는 즉시 `loaded`.

        Ditto 는 매 호출 subprocess 라 메모리에 모델을 상주시키지 않는다. `load()`
        는 vendor / data_root / cfg_pkl 경로가 유효한지만 검증해 운영 부팅 시점에
        문제를 빠르게 드러낸다.
        """
        if not self._gpu_enabled:
            logger.info("GPU_ENABLED=false → DittoTalkingHead 더미 모드")
            self._status = "loaded"
            return

        self._status = "loading"
        missing = []
        if not self._vendor_dir or not (Path(self._vendor_dir) / "inference.py").exists():
            missing.append(f"DITTO_VENDOR_DIR/inference.py ({self._vendor_dir})")
        if not self._data_root or not Path(self._data_root).exists():
            missing.append(f"DITTO_DATA_ROOT ({self._data_root})")
        if not self._cfg_pkl or not Path(self._cfg_pkl).exists():
            missing.append(f"DITTO_CFG_PKL ({self._cfg_pkl})")
        if missing:
            self._status = "error"
            raise RuntimeError(
                "Ditto 환경 경로가 유효하지 않습니다: " + ", ".join(missing)
            )
        self._status = "loaded"
        logger.info("DittoTalkingHead 경로 검증 완료")

    async def unload(self) -> None:
        """subprocess 모델은 자원 보유 없음 — 상태만 reset."""
        self._status = "not_loaded"

    # --- 상태 --------------------------------------------------------------
    @property
    def status(self) -> DittoStatus:
        return self._status

    # --- 렌더 ---------------------------------------------------------------
    async def render_speak(
        self,
        *,
        image_path: str | Path,
        audio_path: str | Path,
        output_path: str | Path,
    ) -> Path:
        """발화 영상 렌더. 사진 + 발화 wav → 입 모양 동기화 mp4.

        Ditto repo 의 `inference.py` CLI 를 subprocess 로 호출한다. 매 호출이
        독립 프로세스라 종료 시 GPU 메모리가 자동 회수된다 (명세 "온디맨드
        로드/언로드" 의도와 일치). 호출자 는 `ModelRegistry.ditto_semaphore`
        로 한 번에 1 건 렌더만 허용해야 한다.

        **경로 인자 검증 책임은 호출자**: `image_path` / `audio_path` / `output_path`
        는 그대로 subprocess argv 로 전달된다. 외부 입력(예: Spring 프록시 경유 사용자
        업로드 경로) 을 검증 없이 넘기면 path traversal 위험이 있으므로, 라우터
        레이어 에서 `Path.resolve().is_relative_to(persona_dir)` 등으로
        반드시 allowlist 화한 뒤 전달할 것.
        """
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        if self._status != "loaded":
            raise RuntimeError("DittoTalkingHead 가 로드되지 않았습니다.")

        if not self._gpu_enabled:
            await asyncio.to_thread(_write_dummy_mp4, target)
            return target

        assert self._vendor_dir and self._data_root and self._cfg_pkl

        vendor = Path(self._vendor_dir)
        inference_py = vendor / "inference.py"
        cmd = [
            sys.executable,
            str(inference_py),
            "--data_root", self._data_root,
            "--cfg_pkl", self._cfg_pkl,
            "--audio_path", str(audio_path),
            "--source_path", str(image_path),
            "--output_path", str(target),
        ]

        logger.info("Ditto inference 시작: %s", target)
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(vendor),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                # 운영자가 stderr 마지막 부분을 보고 빠르게 진단할 수 있게 잘라 전달.
                stderr_tail = stderr.decode(errors="replace")[-1000:]
                # 얼굴 미검출은 사용자 입력(사진 자체) 문제라 진단 stderr 대신
                # 한국어 친화 메시지로 환원. stderr_tail 은 운영자용 로그에만 남긴다.
                if _is_face_detection_failure(stderr_tail):
                    logger.error(
                        "Ditto 얼굴 미검출 (사진 자체 문제): stderr_tail=%s",
                        stderr_tail,
                    )
                    raise FaceDetectionError(_FACE_DETECTION_FAILURE_MESSAGE)
                raise RuntimeError(
                    f"Ditto inference 실패 (rc={proc.returncode}): {stderr_tail}"
                )
            if not target.exists():
                raise RuntimeError(f"Ditto inference 가 output 을 생성하지 않음: {target}")
        except BaseException:
            # 실패·취소 시 부분 mp4 가 남아 다음 단계가 성공 산출물로 오인하는 일이
            # 없도록 정리. inference.py 는 `<output>.tmp.mp4` 임시 파일로 렌더 후
            # ffmpeg muxing 으로 최종 target 을 만드는 흐름이라 양쪽 모두 정리한다.
            for stray in (target, Path(str(target) + ".tmp.mp4")):
                try:
                    stray.unlink(missing_ok=True)
                except OSError:
                    logger.warning("실패 산출물 정리 중 OSError: %s", stray)
            raise
        logger.info("Ditto inference 완료: %s", target)
        return target

    async def render_idle(
        self,
        *,
        image_path: str | Path,
        output_path: str | Path,
        duration_seconds: float = 5.0,
    ) -> Path:
        """무음 talking head 영상 (idle 클립).

        지정 길이의 무음 wav 를 임시 생성한 뒤 `render_speak` 에 위임한다.
        입 모양은 거의 정지 — 명세서 "Idle talking head 영상" 의도와 일치.
        """
        import tempfile

        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        fd, silence_name = tempfile.mkstemp(suffix=".wav", prefix="ditto_idle_")
        import os

        os.close(fd)
        silence_path = Path(silence_name)
        try:
            await asyncio.to_thread(_write_silence_wav, silence_path, duration_seconds)
            return await self.render_speak(
                image_path=image_path,
                audio_path=silence_path,
                output_path=target,
            )
        finally:
            try:
                silence_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("임시 무음 wav 삭제 실패: %s", silence_path)


def _write_dummy_mp4(target: Path) -> None:
    """더미 모드용 placeholder mp4 바이트 작성."""
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(_DUMMY_MP4_BYTES)


def _write_silence_wav(target: Path, duration_seconds: float) -> None:
    """`render_idle` 보조 helper — N초 무음 wav 를 16kHz float32 zeros 로 작성."""
    import numpy as np
    import soundfile as sf

    target.parent.mkdir(parents=True, exist_ok=True)
    samples = int(_SILENCE_SAMPLE_RATE * max(0.0, duration_seconds))
    sf.write(str(target), np.zeros(samples, dtype="float32"), _SILENCE_SAMPLE_RATE)
