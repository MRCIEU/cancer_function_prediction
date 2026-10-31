"""File fingerprints computed during a sequential read."""

from __future__ import annotations

import hashlib
from pathlib import Path


def blake2_file(path: Path, progress=None, digest_size: int = 32) -> str:
    hasher = hashlib.blake2b(digest_size=digest_size)
    with path.open("rb") as fh:
        while True:
            block = fh.read(8 * 1024 * 1024)
            if not block:
                break
            hasher.update(block)
            if progress is not None:
                progress.update(nbytes=len(block))
    if progress is not None:
        progress.update(force=True)
    return hasher.hexdigest()
