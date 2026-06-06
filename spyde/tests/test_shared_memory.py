"""
Tests for shared-memory IPC used in the navigation selector update path.

Scope: write_shared_array / read_shared_array correctness and the cross-process
       round-trip via a Dask distributed worker.  Does NOT test the full
       update_from_navigation_selection path because that requires a live
       PlotState and would race with the PlotUpdateWorker in CI.
"""
import threading
from multiprocessing import shared_memory

import numpy as np
import pytest

from spyde.drawing.update_functions import write_shared_array, read_shared_array


# ── Helpers ───────────────────────────────────────────────────────────────────

_HEADER = 4 + 7 + 4 + 2 * 8  # dtype_len + '<f4\x00...' + ndim + 2 dims
BUFFER_SIZE = (8192 * 8192 * 4) + 128  # same constant as Plot.shared_memory


def _make_shm(name: str) -> shared_memory.SharedMemory:
    try:
        shm = shared_memory.SharedMemory(name=name, create=False)
        shm.close()
        shm.unlink()
    except Exception:
        pass
    return shared_memory.SharedMemory(name=name, create=True, size=BUFFER_SIZE)


# ── Unit tests: write_shared_array / read_shared_array ───────────────────────

class TestWriteReadRoundTrip:
    """write then read must produce an identical array."""

    def _roundtrip(self, data: np.ndarray, shm_name: str) -> np.ndarray:
        shm = _make_shm(shm_name)
        try:
            write_shared_array(data, shm_name)
            result = read_shared_array(shm)
            return result.copy()
        finally:
            shm.close()
            shm.unlink()

    def test_2d_float32(self):
        data = np.random.rand(64, 64).astype(np.float32)
        out = self._roundtrip(data, "spyde_test_2d_f32")
        np.testing.assert_array_equal(out, data)

    def test_2d_uint16(self):
        data = np.random.randint(0, 65535, (32, 32), dtype=np.uint16)
        out = self._roundtrip(data, "spyde_test_2d_u16")
        np.testing.assert_array_equal(out, data)

    def test_shape_preserved(self):
        data = np.zeros((128, 256), dtype=np.float32)
        out = self._roundtrip(data, "spyde_test_shape")
        assert out.shape == (128, 256)

    def test_write_closes_handle(self):
        """write_shared_array must close its handle so the GUI can unlink cleanly."""
        shm = _make_shm("spyde_test_close")
        data = np.ones((16, 16), dtype=np.float32)
        try:
            write_shared_array(data, "spyde_test_close")
            # If the worker handle is still open, unlink would hang on Linux or
            # silently leave the segment on Windows.  We just verify it doesn't raise.
            shm.close()
            shm.unlink()
        except Exception as e:
            pytest.fail(f"unlink after write raised: {e}")

    def test_write_to_nonexistent_shm_is_silent(self):
        """write_shared_array must not raise if the segment doesn't exist."""
        data = np.ones((4, 4), dtype=np.float32)
        # No shm created — should silently return
        write_shared_array(data, "spyde_nonexistent_xyz_999")

    def test_concurrent_write_does_not_crash(self):
        """Multiple threads writing to different shm segments simultaneously is safe."""
        errors = []

        def _write(i):
            name = f"spyde_test_concurrent_{i}"
            shm = _make_shm(name)
            try:
                data = np.full((32, 32), float(i), dtype=np.float32)
                write_shared_array(data, name)
                result = read_shared_array(shm)
                if not np.allclose(result, float(i)):
                    errors.append(f"thread {i}: mismatch")
            except Exception as e:
                errors.append(str(e))
            finally:
                shm.close()
                shm.unlink()

        threads = [threading.Thread(target=_write, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors, f"Concurrent write errors: {errors}"


# ── Cross-process test: worker subprocess can write to GUI-created shm ────────

class TestCrossProcessWrite:
    """Verify that a Dask worker subprocess can open and write to shm created here."""

    @pytest.fixture(autouse=True)
    def client(self, stem_4d_dataset):
        self.client = stem_4d_dataset["window"].dask_manager.client

    def test_worker_can_write_to_gui_shm(self):
        """A distributed worker writes to shm created in the test process."""
        name = "spyde_xproc_test"
        shm = _make_shm(name)
        try:
            data = np.arange(64, dtype=np.float32).reshape(8, 8)
            fut = self.client.submit(write_shared_array, data, name)
            fut.result(timeout=10)

            result = read_shared_array(shm)
            np.testing.assert_array_equal(result, data)
        finally:
            shm.close()
            shm.unlink()

    def test_worker_write_is_safe_after_gui_closes_handle(self):
        """
        The worker's handle keeps the segment alive even after the GUI closes its handle.

        On Windows, shared memory is reference-counted by the kernel: the segment
        persists until ALL handles across ALL processes are closed.  So closing the
        GUI handle while the worker still holds its handle must not crash.
        """
        name = "spyde_xproc_close_test"
        shm = _make_shm(name)

        data = np.ones((16, 16), dtype=np.float32) * 42.0

        # Submit the write task (non-blocking)
        fut = self.client.submit(write_shared_array, data, name)

        # Close the GUI handle BEFORE the future finishes
        shm.close()

        # The worker should still complete without crashing
        fut.result(timeout=10)

        # We can't read from shm (GUI handle closed), but verifying no exception
        # is thrown is the key assertion.

    def test_multiple_sequential_writes(self):
        """Multiple writes to the same shm segment in sequence produce correct output."""
        name = "spyde_xproc_seq_test"
        shm = _make_shm(name)
        try:
            for val in [1.0, 2.0, 3.0]:
                data = np.full((8, 8), val, dtype=np.float32)
                fut = self.client.submit(write_shared_array, data, name)
                fut.result(timeout=10)
                result = read_shared_array(shm)
                np.testing.assert_allclose(result, val, atol=1e-6,
                                           err_msg=f"val={val}")
        finally:
            shm.close()
            shm.unlink()
