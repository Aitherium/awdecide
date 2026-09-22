"""hostpaths -- a Windows drive-letter path must never become a relative one.

Measured 2026-09-19: `arc-world-model-svc/D:/awdecide/wm-contrib/manifest.jsonl`
existed on disk -- a 1 KB manifest holding 154 verdicts from a real validation run.
Every tool here defaults to `D:/awdecide/...`; run from a POSIX shell (Git Bash,
WSL, or inside the container) that string is a RELATIVE path, so the run created a
directory literally named `D:` under the repo, wrote the manifest into it, printed
success, and the real manifest never moved.

`host_path()` makes that failure loud instead of silent: a drive-letter path is
either usable as an absolute path on this platform, or it is remapped through the
matching env var / mount, or the caller is told to pass one. It never returns a
path that would be created relative to the current directory.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Optional

_DRIVE = re.compile(r"^[A-Za-z]:[\\/]")
# Where the same tree is mounted when this runs somewhere D: does not exist.
# WM_HOST_ROOT=/data means D:/awdecide/x is read as /data/awdecide/x.
_HOST_ROOT_ENV = "WM_HOST_ROOT"


class HostPathError(RuntimeError):
    """Raised instead of silently writing into a phantom `D:` directory."""


def looks_like_drive_path(value: str) -> bool:
    return bool(_DRIVE.match(str(value or "")))


def drive_root_usable(value: str) -> bool:
    """True when this platform can really open that drive letter."""
    if not looks_like_drive_path(value):
        return True
    return os.path.isdir(str(value)[:3])


def host_path(value: str, what: str = "path", host_root: Optional[str] = None) -> str:
    """Return an absolute path for *value*, or raise HostPathError.

    - a plain absolute/relative path is returned as-is (relative stays relative:
      that is a deliberate caller choice, not a drive letter turning into one)
    - `D:/x` on a machine that has D: is returned unchanged
    - `D:/x` elsewhere is remapped under $WM_HOST_ROOT when that is set
    - otherwise it raises, naming the env var to set
    """
    value = str(value or "")
    if not looks_like_drive_path(value):
        return value
    if drive_root_usable(value):
        return value
    root = host_root if host_root is not None else os.environ.get(_HOST_ROOT_ENV, "")
    if root:
        rest = value[3:].replace("\\", "/")
        return os.path.join(root, rest)
    raise HostPathError(
        f"{what}={value!r} names drive {value[0]}: which does not exist here, and it would "
        f"be created as a RELATIVE directory called {value[:2]!r} under {os.getcwd()!r} "
        f"(that is how 154 validation verdicts were stranded on 2026-09-19). "
        f"Pass an absolute path, or set {_HOST_ROOT_ENV} to where that tree is mounted."
    )


def self_test() -> int:
    ok = True

    def check(cond: bool, msg: str) -> None:
        nonlocal ok
        if not cond:
            print(f"SELF-TEST FAILED: {msg}")
            ok = False

    check(looks_like_drive_path("D:/a/b"), "D:/a/b is a drive path")
    check(looks_like_drive_path(r"C:\a"), "C:\\a is a drive path")
    check(not looks_like_drive_path("/data/a"), "/data/a is not a drive path")
    check(not looks_like_drive_path("relative/a"), "relative/a is not a drive path")
    check(host_path("/data/x") == "/data/x", "plain absolute passes through")

    missing = "Q:/awdecide/wm-contrib/manifest.jsonl"
    if not drive_root_usable(missing):
        try:
            host_path(missing, "manifest", host_root="")
            check(False, "an unusable drive letter must raise, not be returned")
        except HostPathError:
            pass
        check(
            host_path(missing, "manifest", host_root="/mnt/host")
            == os.path.join("/mnt/host", "awdecide/wm-contrib/manifest.jsonl"),
            "WM_HOST_ROOT remaps the drive path",
        )
    else:
        print("note: drive Q: exists here, skipping the unusable-drive arm")

    print("SELF-TEST PASSED: a drive path never degrades into a relative one" if ok else "")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(self_test())
