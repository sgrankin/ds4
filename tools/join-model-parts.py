#!/usr/bin/env python3
"""Bounded binary assembly: flock, seek/truncate, fsync and a streaming digest.

The Nushell downloader owns model selection and publication policy. Keep this
small I/O boundary in Python instead of buffering hundreds of GB in the shell.
"""
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import sys

out = Path(sys.argv[1])
boundary, total = map(int, sys.argv[2:4])
expected = sys.argv[4]
first, second, pending = (Path(str(out) + suffix)
                          for suffix in (".part1", ".part2", ".assembling"))

def verify(path):
    if path.stat().st_size != total:
        sys.exit("Incorrect file size: " + str(path))
    print("Verifying SHA-256: " + str(path), flush=True)
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for block in iter(lambda: fp.read(16 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        sys.exit("Checksum mismatch: " + str(path) +
                 ". Move this file aside before retrying; it was not accepted.")

# Keep this lock file: unlinking it could allow two different locks for the
# same download. The first part becomes the output without a second full copy.
with Path(str(pending) + ".lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    if out.exists():
        verify(out)
        sys.exit(0)
    if not pending.exists():
        if first.stat().st_size != boundary:
            sys.exit("Incorrect file size: " + str(first))
        first.rename(pending)
    size = pending.stat().st_size
    if not boundary <= size <= total:
        sys.exit("Invalid partial assembly: " + str(pending) +
                 ". Move it aside before retrying.")
    if second.stat().st_size != total - boundary:
        sys.exit("Incorrect file size: " + str(second))
    # An interrupted append restarts only the smaller tail, never the prefix.
    with pending.open("r+b") as dst:
        dst.truncate(boundary)
        dst.seek(boundary)
        if shutil.disk_usage(out.parent).free < total - boundary + (1 << 30):
            sys.exit("Not enough disk space to join Q4; keep the parts and retry.")
        print("Joining Q4 download parts", flush=True)
        with second.open("rb") as src:
            shutil.copyfileobj(src, dst, 16 << 20)
        dst.flush()
        os.fsync(dst.fileno())
    verify(pending)
    pending.rename(out)
    second.unlink()
