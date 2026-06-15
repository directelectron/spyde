"""
update_functions.py

Module containing functions to update a plot based on a selector.  These functions are
called on the move or change events of a selector.

"""

import sys
import numpy as np
import dask
import dask.array as da
import distributed
from distributed import Future

from scipy import fft

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spyde.drawing.selectors import BaseSelector
    from spyde.drawing.plots.plot import Plot
from multiprocessing import shared_memory

_SHARED_MEMORY_SUPPORTED = True

def write_shared_array(data, shared_arr_name):
    dtype_bytes = data.dtype.str.encode('utf-8')
    dtype_length = len(dtype_bytes)
    ndim = data.ndim
    shm = None
    try:
        shm = shared_memory.SharedMemory(name=shared_arr_name, create=False)
        buffer = shm.buf
        offset = 0
        buffer[offset:offset+4] = dtype_length.to_bytes(4, byteorder='little')
        offset += 4
        buffer[offset:offset+dtype_length] = dtype_bytes
        offset += dtype_length
        buffer[offset:offset+4] = ndim.to_bytes(4, byteorder='little')
        offset += 4
        for dim in data.shape:
            buffer[offset:offset+8] = dim.to_bytes(8, byteorder='little')
            offset += 8
        target_arr = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf[offset:])
        target_arr[:] = data
    except Exception:
        pass
    finally:
        if shm is not None:
            try:
                shm.close()
            except Exception:
                pass


def read_shared_array(shm):
    buffer = shm.buf
    offset = 0
    # Read dtype length
    dtype_length = int.from_bytes(buffer[offset:offset+4], byteorder='little')
    offset += 4
    # Read dtype
    dtype_str = bytes(buffer[offset:offset+dtype_length]).decode('utf-8')
    dtype = np.dtype(dtype_str)
    offset += dtype_length
    # Read ndim
    ndim = int.from_bytes(buffer[offset:offset+4], byteorder='little')
    offset += 4
    # Read shape
    shape = tuple(int.from_bytes(buffer[offset+i*8:offset+(i+1)*8], byteorder='little')
                  for i in range(ndim))
    offset += ndim * 8
    # Copy out of the shared buffer before returning — the caller's shm handle
    # may be closed (and the memoryview invalidated) before the array is used.
    arr = np.array(np.ndarray(shape, dtype=dtype, buffer=buffer[offset:]))
    return arr

def update_from_navigation_selection(
        selector: "BaseSelector",
        child: "Plot",
        indices,
        get_result: bool = False,
        cache_in_shared_memory: bool = True,
):
    """
    Update the plot based on the navigation selection. This is the most common update function for using some
    navigation selector (on a parent) and updating a child plot.

    Parameters
    ----------
    selector : BaseSelector
        The selector that triggered the update.
    child : Plot
        The child plot to update.
    indices : array-like
        The indices selected by the selector.
    get_result : bool
        Whether to compute the result immediately (for Dask arrays). Always False for using
        dask distributed futures.
    cache_in_shared_memory : bool
        Whether to cache the result in shared memory. This bypasses TCP transfer
        with distributed futures but requires that the child process can access the
        shared memory (e.g., same machine). Default is False.
    """
    # get the data from the signal tree based on the current indices

    current_signal = child.plot_state.current_signal

    if not selector.is_integrating:
        indices = np.mean(indices, axis=0).astype(int)

    # Clamp nav indices to the data's leading-axis sizes. The signal behind a
    # plot can change (e.g. set_signal_type, or swapping current_signal) while a
    # selector still holds positions from the previous, larger nav grid — the
    # subsequent data[...] index would raise IndexError. Clamping keeps the
    # display valid until the selector catches up to the new shape.
    try:
        data_shape = current_signal.data.shape
        idx_arr = np.asarray(indices)
        if idx_arr.size and len(data_shape):
            # Last axis holds the per-point coordinates (one per leading data
            # axis); clamp each coordinate to its axis size.
            ncoord = idx_arr.shape[-1] if idx_arr.ndim else 1
            limits = np.array(
                [data_shape[i] - 1 for i in range(min(ncoord, len(data_shape)))],
                dtype=idx_arr.dtype,
            )
            if limits.size == ncoord:
                indices = np.clip(idx_arr, 0, limits)
    except Exception:
        pass

    if current_signal._lazy:
        if isinstance(current_signal.data[0], Future):
            current_img = np.ones(current_signal.axes_manager.signal_shape, dtype=np.int8)
            if current_img.ndim == 2:
                #make checkerboard pattern to indicate loading
                current_img[::2, ::2] = 0
        else:
            # Cancel stale work from the previous position before requesting new data.
            # 1. Cancel the old shm-write future so workers aren't blocked on it.
            old_fut = getattr(child, "_pending_shm_future", None)
            if old_fut is not None and not old_fut.done():
                try:
                    old_fut.cancel()
                except Exception:
                    pass

            # 2. Cancel pending surrounding-block prefetch futures — they are
            #    low-priority but still consume worker slots.  The new core-block
            #    request will re-prefetch the right neighbours after it completes.
            cached_arr = getattr(current_signal, "cached_dask_array", None)
            if cached_arr is not None and hasattr(cached_arr, "cancel_surrounding"):
                cached_arr.cancel_surrounding()

            current_img = current_signal._get_cache_dask_chunk(
                indices, get_result=get_result, return_future=True,
            )
            if cache_in_shared_memory and _SHARED_MEMORY_SUPPORTED:
                # Pre-create the shm segment in the GUI process before submitting,
                # so the worker subprocess can open it by name.
                _ = child.shared_memory  # lazy creation
                shared_arr_name = f"plot_buffer{id(child)}"
                # priority=10 puts this ahead of surrounding-block prefetch tasks
                # (which are submitted at default priority=0 by CachedDaskArray).
                fut = child.main_window.dask_manager.client.submit(
                    write_shared_array, current_img, shared_arr_name,
                    priority=10,
                )
                child._pending_shm_future = fut
                current_img = fut
    else:
        tuple_inds = tuple(indices[ind] for ind in np.arange(len(indices)))
        if len(tuple_inds) == 1:
            current_img = current_signal.data[tuple_inds]
        else:
            current_img = np.mean(current_signal.data[tuple_inds], axis=0)
    return current_img


def get_fft(selector: "BaseSelector", child: "Plot", indices, get_result: bool = False):
    """
    Get the FFT of the image.

    Parameters
    ----------
    img : array-like
        The input image.

    Returns
    -------
    array-like
        The FFT of the input image.
    """
    # convert indices to image slice:
    max_x, max_y = np.max(indices, axis=0)
    min_x, min_y = np.min(indices, axis=0)

    img = selector.parent.image_item.image

    img_max_x = img.shape[0] - 1
    img_max_y = img.shape[1] - 1
    if max_x > img_max_x:
        max_x = img_max_x
    if max_y > img_max_y:
        max_y = img_max_y
    if min_x < 0:
        min_x = 0
    if min_y < 0:
        min_y = 0

    slice_x, slice_y = slice(min_x, max_x + 1), slice(min_y, max_y + 1)
    sliced_img = img[slice_x, slice_y]
    fft_img = fft.fftshift(fft.fft2(sliced_img))
    return fft_img.real


def compute_virtual_image_kernel(
    data: da.Array,
    mask: np.ndarray,
    client: distributed.Client,
    gpu_worker_address: "str | None",
) -> distributed.Future:
    """
    Compute a virtual image by masking and summing the last two (signal) axes.

    Equivalent to:
        np.sum(data * mask[np.newaxis, np.newaxis, ...], axis=(-1, -2))

    Works for any number of navigation axes (3D, 4D, 5D, 6D datasets).
    Signal axes must be the last two (HyperSpy convention).

    Broadcasting mask as a numpy array (not a dask array) means each worker
    multiplies its navigation chunk directly against the in-memory mask without
    any cross-chunk communication, then reduces over the last two axes within
    the chunk. This is O(n_nav_chunks) independent tasks with no shuffle.

    Parameters
    ----------
    data : dask array, shape (...nav..., ky, kx)
    mask : float32 numpy array, shape (ky, kx)
    client : dask distributed Client
    gpu_worker_address : str or None

    Returns
    -------
    distributed.Future resolving to np.ndarray of shape (...nav...)
    """
    mask = np.asarray(mask, dtype=np.float32)
    if gpu_worker_address:
        with dask.annotate(resources={"GPU": 1}):
            result = (data * mask).sum(axis=(-2, -1))
    else:
        result = (data * mask).sum(axis=(-2, -1))
    return client.compute(result)


def compute_with_live_buffer(
    result_array: da.Array,
    nav_shape: tuple,
    client: distributed.Client,
    shm_name: str,
    on_chunk_done=None,
) -> distributed.Future:
    """
    Progressive compute: submits one Future per nav chunk and calls
    ``on_chunk_done(chunk_result, nav_slices)`` from a Dask callback thread
    as each chunk completes.  The caller is responsible for marshalling
    back to the GUI thread (e.g. via a Qt Signal).

    Shared memory is written from the GUI process (not from worker
    subprocesses) to avoid Windows access-violation crashes when the shm
    segment is torn down during test teardown while a worker is mid-write.

    Parameters
    ----------
    result_array : dask array, nav-shaped (signal axes already reduced)
    nav_shape    : tuple — full navigation shape
    client       : dask distributed Client (or None for synchronous path)
    shm_name     : str — name of a pre-existing SharedMemory segment to
                   update from the GUI side (via on_chunk_done)
    on_chunk_done : callable(chunk_result, nav_slices) | None
                   Called from a Dask callback thread as each chunk finishes.
    """
    import itertools

    nav_ndim = len(nav_shape)
    nav_chunks = result_array.chunks

    if client is None:
        # Synchronous fallback: compute the whole array and write shm once
        result = result_array.compute()
        if shm_name:
            try:
                from multiprocessing import shared_memory as _shm_mod
                shm = _shm_mod.SharedMemory(name=shm_name, create=False)
                buf = np.ndarray(nav_shape, dtype=np.float32, buffer=shm.buf)
                buf[:] = result.astype(np.float32)
                shm.close()
            except Exception:
                pass

        class _SyncResult:
            def result(self): return result
            def done(self): return True
            def cancel(self): pass

        return _SyncResult()

    # Build per-nav-chunk futures
    axes_ranges = []
    for axis_chunks in nav_chunks:
        positions, start = [], 0
        for size in axis_chunks:
            positions.append((start, size))
            start += size
        axes_ranges.append(positions)

    chunk_futures = []
    chunk_slices = []
    for combo in itertools.product(*axes_ranges):
        slices = tuple(slice(s, s + n) for s, n in combo)
        full_slice = slices + (slice(None),) * (result_array.ndim - nav_ndim)
        chunk_da = result_array[full_slice]
        fut = client.compute(chunk_da)
        chunk_futures.append(fut)
        chunk_slices.append(slices)

    # Attach callbacks — run in Dask callback threads, must be thread-safe
    def _make_cb(fut, nav_slices):
        def _cb(f):
            try:
                chunk_result = f.result()
                if on_chunk_done is not None:
                    on_chunk_done(chunk_result, nav_slices)
            except Exception:
                pass
        return _cb

    if on_chunk_done is not None:
        for fut, nav_slices in zip(chunk_futures, chunk_slices):
            fut.add_done_callback(_make_cb(fut, nav_slices))

    # Return the whole-array future for progress indicator and commit path
    full_future = client.compute(result_array)
    return full_future


def ensure_live_buffer(nav_shape: tuple, shm_name: str) -> "shared_memory.SharedMemory":
    """
    Create (or recreate) a float32 shared memory segment for live display.

    Returns the SharedMemory object — the caller must keep a reference to
    prevent premature cleanup.  Call ``shm.unlink()`` when done.
    """
    from multiprocessing import shared_memory
    nbytes = int(np.prod(nav_shape)) * 4  # float32
    try:
        shm = shared_memory.SharedMemory(name=shm_name, create=False)
        if shm.size < nbytes:
            shm.close()
            shm.unlink()
            raise FileNotFoundError
        # Zero out existing buffer so old data doesn't show
        buf = np.ndarray(nav_shape, dtype=np.float32, buffer=shm.buf)
        buf[:] = np.nan
        return shm
    except (FileNotFoundError, Exception):
        try:
            shm = shared_memory.SharedMemory(name=shm_name, create=True, size=max(nbytes, 1))
            buf = np.ndarray(nav_shape, dtype=np.float32, buffer=shm.buf)
            buf[:] = np.nan
            return shm
        except FileExistsError:
            shm = shared_memory.SharedMemory(name=shm_name, create=False)
            buf = np.ndarray(nav_shape, dtype=np.float32, buffer=shm.buf)
            buf[:] = np.nan
            return shm


def read_live_buffer(nav_shape: tuple, shm_name: str) -> np.ndarray:
    """Read current contents of live shared-memory buffer into a new array."""
    from multiprocessing import shared_memory
    try:
        shm = shared_memory.SharedMemory(name=shm_name, create=False)
        arr = np.array(np.ndarray(nav_shape, dtype=np.float32, buffer=shm.buf))
        shm.close()
        return arr
    except Exception:
        return np.full(nav_shape, np.nan, dtype=np.float32)


def compute_line_profile_kernel(
    image: np.ndarray,
    roi,
    image_item,
    client: distributed.Client,
) -> distributed.Future:
    """Extract a 1D line profile from a 2D image via LineROI.getArrayRegion.

    Parameters
    ----------
    image : np.ndarray, shape (ny, nx)
        The currently displayed image (plot.image_item.image).
    roi : pyqtgraph.LineROI
    image_item : pyqtgraph.ImageItem
    client : dask distributed Client

    Returns
    -------
    distributed.Future resolving to np.ndarray shape (length_px,)

    Notes
    -----
    LineROI.getArrayRegion returns shape (length_px, width_px).
    nanmean over axis=1 collapses the perpendicular width to give the profile.
    """
    region = roi.getArrayRegion(image, image_item)   # (length_px, width_px)
    profile = np.nanmean(region, axis=1)             # (length_px,)
    return client.submit(lambda p=profile: p)


def compute_nav_line_sum_kernel(
    data: da.Array,
    ys: np.ndarray,
    xs: np.ndarray,
    client: distributed.Client,
    gpu_worker_address: "str | None",
) -> distributed.Future:
    """Compute the mean diffraction pattern over all nav pixels in a line strip.

    Parameters
    ----------
    data : dask array, shape (...nav..., nkx, nky)
        HyperSpy convention: last two axes are signal.
    ys : np.ndarray, shape (N,)
        Row (y) pixel indices of all nav pixels inside the strip.
    xs : np.ndarray, shape (N,)
        Column (x) pixel indices of all nav pixels inside the strip.
    client : dask distributed Client
    gpu_worker_address : str or None

    Returns
    -------
    distributed.Future resolving to np.ndarray shape (nkx, nky)
    """
    # Dask doesn't support multi-dimensional fancy indexing, so loop and vstack
    slices = [data[int(y), int(x)] for y, x in zip(ys, xs)]
    nav_slices = da.stack(slices, axis=0)  # (N, nkx, nky)
    resources = {"GPU": 1} if gpu_worker_address else {}
    with dask.annotate(resources=resources):
        result = da.mean(nav_slices, axis=0)
    return client.compute(result)
