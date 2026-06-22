"""
process_guard.py — guarantee Dask workers die with the backend process.

The Python backend is a subprocess of Electron and spawns a Dask LocalCluster
whose worker/nanny processes are *grandchildren*. If the backend is force-killed
(Task Manager), crashes, or Electron dies without sending a clean ``quit``, the
normal ``DaskManager.shutdown()`` never runs and those workers orphan — every
run leaks ``n_workers`` idle Python processes (observed: ~200 stale python.exe
holding tens of GB).

The OS-level fix on Windows is a **Job Object** with
``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``: assign this process to the job, and when
the last handle to the job closes (i.e. when THIS process dies, for ANY reason),
Windows terminates every process in the job — including all Dask workers spawned
afterwards, since child processes inherit the job by default.

On POSIX the equivalent is a new session/process-group plus ``prctl`` PDEATHSIG,
but Dask workers there are reaped by ``shutdown()`` reliably enough; this module
is a no-op off Windows (returns False) and callers keep the existing teardown.
"""
from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)

# Module-level so the job handle lives for the whole process lifetime. If it were
# a local it would be garbage-collected, closing the handle and (because
# KILL_ON_JOB_CLOSE) killing us immediately.
_job_handle = None


def install_kill_on_close() -> bool:
    """Assign the current process to a kill-on-close Job Object (Windows only).

    Returns True if the guard is active, False otherwise (non-Windows, or any
    failure — the caller should still keep its own ``shutdown()`` path).
    """
    global _job_handle
    if not sys.platform.startswith("win"):
        return False
    if _job_handle is not None:
        return True
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        # --- function prototypes ---
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD,
        ]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE

        # JOBOBJECT_EXTENDED_LIMIT_INFORMATION layout
        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_uint64),
                ("WriteOperationCount", ctypes.c_uint64),
                ("OtherOperationCount", ctypes.c_uint64),
                ("ReadTransferCount", ctypes.c_uint64),
                ("WriteTransferCount", ctypes.c_uint64),
                ("OtherTransferCount", ctypes.c_uint64),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
        JobObjectExtendedLimitInformation = 9  # JOBOBJECTINFOCLASS

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            raise ctypes.WinError(ctypes.get_last_error())

        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        ok = kernel32.SetInformationJobObject(
            job, JobObjectExtendedLimitInformation,
            ctypes.byref(info), ctypes.sizeof(info),
        )
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error())

        ok = kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess())
        if not ok:
            err = ctypes.get_last_error()
            # ERROR_ACCESS_DENIED (5): the process is ALREADY in a job that
            # disallows nesting (e.g. Electron already put us in one, or an older
            # Windows without nested-job support). In that case the parent's job
            # governs our lifetime anyway — treat as best-effort success.
            if err == 5:
                logger.info("process already in a job object; relying on parent "
                            "job for worker cleanup")
                return False
            raise ctypes.WinError(err)

        _job_handle = job  # keep alive for process lifetime
        logger.info("Dask workers guarded by kill-on-close Job Object")
        return True
    except Exception as e:
        logger.warning("could not install kill-on-close job object (%s); "
                       "relying on graceful shutdown for worker cleanup", e)
        return False
