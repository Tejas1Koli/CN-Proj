"""Sandboxed execution of model-generated Python code (HumanEval / MBPP).

Untrusted code runs in a separate process, in its own temp directory, with a
CPU rlimit, an address-space rlimit and a wall-clock timeout. The process is
started in its own session so the whole process group can be killed on
timeout (a bare timeout only kills the direct child).

This is a best-effort sandbox, not a security boundary -- it is meant to stop
runaway generations (infinite loops, fork bombs, huge allocations) from taking
the benchmarking machine down. Run it on a disposable host if you are dealing
with untrusted code.
"""

import os
import subprocess
import sys
import tempfile

from .config import CODE_MEMORY_LIMIT_BYTES, CODE_TIMEOUT_S


def _limits(timeout: int):
    import resource

    def apply() -> None:
        # Each limit is set independently and defensively: on macOS,
        # setrlimit(RLIMIT_NPROC, ...) fails with EPERM when the new limit is
        # below the number of processes the user already owns, and that
        # failure must not abort the whole subprocess launch.
        for name, value in (
            ("RLIMIT_CPU", (timeout, timeout)),
            ("RLIMIT_AS", (CODE_MEMORY_LIMIT_BYTES,) * 2),
            ("RLIMIT_FSIZE", (64 * 1024 * 1024,) * 2),
            ("RLIMIT_CORE", (0, 0)),
        ):
            which = getattr(resource, name, None)
            if which is None:
                continue
            try:
                resource.setrlimit(which, value)
            except (ValueError, OSError):
                pass

    return apply


def run_python(code: str, timeout: int = CODE_TIMEOUT_S) -> tuple[bool, str]:
    """Run `code` in a subprocess. Returns (passed, error_text)."""
    with tempfile.TemporaryDirectory() as workdir:
        path = os.path.join(workdir, "candidate.py")
        with open(path, "w") as handle:
            handle.write(code)

        proc = subprocess.Popen(
            [sys.executable, path],
            cwd=workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            preexec_fn=_limits(timeout),
            start_new_session=True,
        )
        try:
            _, stderr = proc.communicate(timeout=timeout + 5)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            proc.communicate()
            return False, "TIMEOUT"

        if proc.returncode == 0:
            return True, ""
        return False, stderr.decode("utf-8", "replace")[-400:]


def _kill_group(proc: subprocess.Popen) -> None:
    """Kill the child's whole process group, but never our own.

    The child is started with start_new_session=True so it leads its own
    process group. If that ever failed to take effect, killpg on the child's
    pid would target *our* group and take the benchmark down with it, so the
    group id is checked against our own before anything is signalled.
    """
    try:
        child_group = os.getpgid(proc.pid)
    except (ProcessLookupError, PermissionError, OSError):
        child_group = None

    if child_group is not None and child_group != os.getpgid(0):
        try:
            os.killpg(child_group, 9)
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass

    try:
        proc.kill()
    except (ProcessLookupError, OSError):
        pass
