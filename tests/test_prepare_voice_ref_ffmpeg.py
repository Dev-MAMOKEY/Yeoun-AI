"""_prepare_voice_ref ffmpeg trim 단위 테스트 — 호스트 ffmpeg 가용 시에만 실행.

dual review code-reviewer P1 conf 85 채택: autouse stub 이 적용되면 trim 로직
(8초 / 16kHz / mono / pcm_s16le / 마이그레이션 잔재 정리 / 실패 시 부분 파일 정리)
이 단위로 검증되지 않아 CI 에서 ffmpeg flag 오타·분기 결함이 통과될 수 있음. 본
파일은 conftest 의 autouse stub 을 명시 해제 후 실제 ffmpeg 바이너리로 동작 검증.
ffmpeg 미설치 환경에선 skip (호스트 venv 격리 유지).
"""

import shutil
import wave
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="호스트 ffmpeg 미설치 — 도커/CI 의 ffmpeg 가용 환경에서만 검증",
)


@pytest.fixture
def _disable_ffmpeg_stub(monkeypatch):
    """conftest 의 autouse stub 해제 — 실제 ffmpeg 호출 경로 활성."""
    from app.pipeline import persona_creation
    from app.pipeline.persona_creation import _prepare_voice_ref as _real_prepare

    monkeypatch.setattr(persona_creation, "_prepare_voice_ref", _real_prepare)


def _make_silent_wav(path: Path, seconds: float, sample_rate: int = 44100) -> None:
    """주어진 길이의 무음 WAV 생성 — ffmpeg 입력 시드용."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(b"\x00\x00" * int(sample_rate * seconds) * 2)


def _make_padded_tone_wav(
    path: Path,
    silence_seconds: float,
    tone_seconds: float,
    sample_rate: int = 44100,
    frequency: int = 440,
) -> None:
    """앞 무음 + 톤 발화 WAV 생성 — silenceremove 검증 시드용."""
    import math

    path.parent.mkdir(parents=True, exist_ok=True)
    silence_frames = int(sample_rate * silence_seconds)
    tone_frames = int(sample_rate * tone_seconds)
    silence_bytes = b"\x00\x00" * silence_frames

    amplitude = 16000
    tone_buf = bytearray(tone_frames * 2)
    for i in range(tone_frames):
        val = int(amplitude * math.sin(2 * math.pi * frequency * i / sample_rate))
        tone_buf[i * 2 : i * 2 + 2] = val.to_bytes(2, "little", signed=True)

    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(silence_bytes + bytes(tone_buf))


async def test_prepare_voice_ref_trims_to_8_seconds(_disable_ffmpeg_stub, tmp_path):
    """30초 입력이 정확히 8초로 잘려 16kHz mono PCM WAV 로 저장되는지."""
    from app.pipeline.persona_creation import _prepare_voice_ref

    voice = tmp_path / "voice" / "long.wav"
    _make_silent_wav(voice, seconds=30.0)
    ref = tmp_path / "voice_ref" / "ref_audio.wav"

    await _prepare_voice_ref(voice, ref)

    assert ref.exists()
    with wave.open(str(ref), "rb") as f:
        assert f.getnchannels() == 1, "mono 강제 실패"
        assert f.getframerate() == 16000, "16kHz 강제 실패"
        assert f.getsampwidth() == 2, "pcm_s16le 강제 실패"
        duration = f.getnframes() / f.getframerate()
        # ffmpeg `-t 8` 은 정확히 8초 ± 1 프레임 수준.
        assert 7.9 < duration <= 8.1, f"trim 길이 어긋남: {duration:.3f}s"


async def test_prepare_voice_ref_idempotent_skip(_disable_ffmpeg_stub, tmp_path):
    """ref_audio 가 이미 있으면 ffmpeg 재호출 없이 스킵."""
    from app.pipeline.persona_creation import _prepare_voice_ref

    voice = tmp_path / "voice" / "v.wav"
    _make_silent_wav(voice, seconds=10.0)
    ref = tmp_path / "voice_ref" / "ref_audio.wav"

    await _prepare_voice_ref(voice, ref)
    first_mtime = ref.stat().st_mtime_ns

    await _prepare_voice_ref(voice, ref)
    assert ref.stat().st_mtime_ns == first_mtime, "멱등 스킵 실패"


async def test_prepare_voice_ref_removes_legacy_mp3_stray(_disable_ffmpeg_stub, tmp_path):
    """이전 동작이 남긴 ref_audio.mp3 같은 마이그레이션 잔재가 trim 성공 후 제거되는지."""
    from app.pipeline.persona_creation import _prepare_voice_ref

    voice = tmp_path / "voice" / "v.wav"
    _make_silent_wav(voice, seconds=10.0)
    ref_dir = tmp_path / "voice_ref"
    ref_dir.mkdir(parents=True)
    legacy_mp3 = ref_dir / "ref_audio.mp3"
    legacy_mp3.write_bytes(b"\x00" * 1024)  # 가짜 mp3 잔재

    ref = ref_dir / "ref_audio.wav"
    await _prepare_voice_ref(voice, ref)

    assert ref.exists()
    assert not legacy_mp3.exists(), "이전 ref_audio.mp3 잔재 제거 실패"


async def test_prepare_voice_ref_strips_leading_silence(_disable_ffmpeg_stub, tmp_path):
    """앞 5초 무음 + 5초 톤 입력 시 silenceremove 가 무음 제거 — 결과 첫 100ms 가 비-무음.

    #65 핵심 회귀 방지: 앞 무음이 있으면 fixed 8초 trim 이 발화 부족으로 OmniVoice
    voice clone 컨텍스트 부족. silenceremove 필터로 앞 무음 자동 컷.
    """
    from app.pipeline.persona_creation import _prepare_voice_ref

    voice = tmp_path / "voice" / "padded.wav"
    _make_padded_tone_wav(voice, silence_seconds=5.0, tone_seconds=5.0)
    ref = tmp_path / "voice_ref" / "ref_audio.wav"

    await _prepare_voice_ref(voice, ref)

    assert ref.exists()
    with wave.open(str(ref), "rb") as f:
        sr = f.getframerate()
        head_frames = f.readframes(int(0.1 * sr))
    # 첫 100ms 안에 non-zero 16-bit sample 존재 — 무음 제거됐다는 신호.
    non_zero = sum(
        1
        for i in range(0, len(head_frames), 2)
        if int.from_bytes(head_frames[i : i + 2], "little", signed=True) != 0
    )
    assert non_zero > 0, "silenceremove 후에도 앞 100ms 가 여전히 무음"


async def test_prepare_voice_ref_failure_cleans_partial(_disable_ffmpeg_stub, tmp_path):
    """ffmpeg 가 입력 디코딩 실패 시 부분 산출물 정리 + RuntimeError."""
    from app.pipeline.persona_creation import _prepare_voice_ref

    # 가짜 voice — wav 헤더 없음 → ffmpeg 디코딩 실패 유도.
    voice = tmp_path / "voice" / "broken.wav"
    voice.parent.mkdir(parents=True)
    voice.write_bytes(b"NOT-A-WAV-FILE")

    ref = tmp_path / "voice_ref" / "ref_audio.wav"
    with pytest.raises(RuntimeError, match="ref_audio trim"):
        await _prepare_voice_ref(voice, ref)

    assert not ref.exists(), "실패 시 부분 산출물 정리 실패"
