"""Regression guard: the find_vectors package must import on a machine WITHOUT
numba (no GPU / no numba wheel) and fall back to the CPU path.

numba is NOT a declared dependency — CI and many user machines won't have it.
The package splits the @cuda.jit kernels into kernels.py (defined only inside a
`try: from numba import cuda`); if any consumer does a top-level
`from .kernels import _some_kernel`, the import explodes when numba is absent and
the whole package (hence the CPU fallback) dies. This test simulates numba being
unavailable and asserts the package still imports with GPU kernels marked
unavailable.

Run in a SUBPROCESS so blocking numba can't pollute other tests' import cache.
"""
import subprocess
import sys
import textwrap


_PROBE = textwrap.dedent(
    """
    import builtins
    _real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name == "numba" or name.startswith("numba."):
            raise ImportError("numba blocked (simulated no-GPU machine)")
        return _real_import(name, *args, **kwargs)

    builtins.__import__ = _blocked

    import spyde.actions.find_vectors as fv
    # Must import; GPU kernels must report unavailable (CPU fallback path).
    assert fv.chunk._GPU_KERNELS_AVAILABLE is False, "kernels should be unavailable"
    # The CPU-side public surface must still be importable.
    from spyde.actions.find_vectors import (
        _do_compute_vectors, _find_peaks_single_frame, _auto_beamstop_from_signal,
    )
    print("NO_NUMBA_IMPORT_OK")
    """
)


class TestFindVectorsNoNumba:
    def test_package_imports_without_numba(self):
        """The package imports (CPU fallback) when numba is unavailable."""
        proc = subprocess.run(
            [sys.executable, "-c", _PROBE],
            capture_output=True, text=True, timeout=120,
        )
        assert "NO_NUMBA_IMPORT_OK" in proc.stdout, (
            f"package failed to import without numba.\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
        assert proc.returncode == 0, proc.stderr
