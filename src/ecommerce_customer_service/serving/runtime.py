from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
from typing import Iterator


@contextmanager
def exclusive_runtime(directory: Path) -> Iterator[None]:
    """Prevent a second process from falsely recovering a live process's work."""
    directory.mkdir(parents=True, exist_ok=True)
    stream = (directory / "service.lock").open("a+b")
    if stream.seek(0, os.SEEK_END) == 0:
        stream.write(b"0")
        stream.flush()
    stream.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        stream.close()
        raise RuntimeError("data_directory_already_in_use") from exc
    try:
        yield
    finally:
        stream.close()
