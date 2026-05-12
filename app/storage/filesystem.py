"""파일시스템 안전 경로 + HTTP Range 처리.

`safe_resolve` 는 외부에서 들어온 부분 경로가 지정 루트(`PERSONA_DIR`) 밖으로
나가지 못하도록 강제하고, `parse_range` / `iter_file_range` 는 HTTP Range
스트리밍을 지원한다.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import AsyncIterator

# 기본 청크 — 너무 크면 메모리 압박·작은 latency, 너무 작으면 syscall 오버헤드.
# NVMe + uvicorn 단일 워커 환경에서 64KiB 가 적절한 절충(syscall ~µs).
_RANGE_CHUNK_SIZE = 64 * 1024


class RangeNotSatisfiable(ValueError):
    """`Range` 헤더가 파일 크기 범위 밖일 때 — 라우터가 416 으로 환원."""


def safe_resolve(root: str | Path, *parts: str | Path) -> Path:
    """`root` 하위에 위치하는 절대 경로를 안전하게 합성.

    `parts` 안에 `..` / 절대 경로 / 심볼릭 탈출이 포함되면 `PermissionError`.
    PERSONA_DIR 같은 신뢰 루트 안의 자원만 외부에 노출하기 위한 가드.
    """
    root_resolved = Path(root).resolve()
    candidate = (root_resolved.joinpath(*[str(p) for p in parts])).resolve()
    if root_resolved != candidate and root_resolved not in candidate.parents:
        raise PermissionError(f"경로가 허용 루트 밖입니다: {candidate} (root={root_resolved})")
    return candidate


def parse_range(header: str | None, file_size: int) -> tuple[int, int] | None:
    """`Range: bytes=start-end` 헤더 파싱. 없으면 None, 잘못이면 RangeNotSatisfiable.

    반환: `(start, end)` — 둘 다 포함 인덱스 (HTTP 표준).
    - `bytes=0-`        → (0, size-1)
    - `bytes=-500`      → (size-500, size-1)
    - `bytes=100-200`   → (100, 200)
    """
    if not header:
        return None
    if not header.startswith("bytes="):
        raise RangeNotSatisfiable(f"지원되지 않는 unit: {header!r}")
    spec = header[len("bytes="):].strip()
    if "," in spec:
        # 다중 Range 는 지원 안 함 — 라우터가 416 환원.
        raise RangeNotSatisfiable("다중 Range 미지원")
    start_str, _, end_str = spec.partition("-")
    try:
        if not start_str:
            # suffix 형태: 끝에서 N 바이트.
            n = int(end_str)
            if n <= 0:
                raise RangeNotSatisfiable("suffix 길이가 0 이하")
            start = max(0, file_size - n)
            end = file_size - 1
        else:
            start = int(start_str)
            end = int(end_str) if end_str else file_size - 1
    except ValueError as exc:
        raise RangeNotSatisfiable(f"Range 숫자 파싱 실패: {spec!r}") from exc
    if start < 0 or end < start or start >= file_size:
        raise RangeNotSatisfiable(f"Range 가 파일 크기 밖: start={start}, end={end}, size={file_size}")
    end = min(end, file_size - 1)
    return start, end


async def safe_rmtree(root: str | Path, *parts: str | Path) -> bool:
    """`root` 하위의 부분 경로를 트리째 삭제. 루트 밖 인자는 `PermissionError`.

    반환: 실제로 삭제했으면 True, 대상이 없으면 False. 삭제 실패(IO error) 는
    `OSError` raise — 호출자(라우터)가 DB 변경 보류 + 500 환원.
    """
    target = safe_resolve(root, *parts)
    if not target.exists():
        return False
    await asyncio.to_thread(shutil.rmtree, target)
    return True


async def iter_file_range(
    path: Path,
    start: int,
    end: int,
    chunk_size: int = _RANGE_CHUNK_SIZE,
) -> AsyncIterator[bytes]:
    """`[start, end]` 구간을 chunk 단위로 비동기 yield.

    동기 IO 호출은 짧은 chunk 단위라 이벤트 루프 차단 영향이 작지만, 큰 파일에서
    안정성을 위해 호출자가 `StreamingResponse` 안에 위치시키는 게 일반적.
    """
    remaining = end - start + 1
    with path.open("rb") as f:
        f.seek(start)
        while remaining > 0:
            data = f.read(min(chunk_size, remaining))
            if not data:
                break
            remaining -= len(data)
            yield data
