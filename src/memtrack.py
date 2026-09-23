"""Process memory measurement - standard library only (ctypes + resource).

On macOS the number that matters is the *physical footprint* (what
Activity Monitor's "Memory" column shows): resident pages plus pages the
kernel has compressed or swapped out. `resource.getrusage().ru_maxrss`
only counts resident pages, so on a memory-constrained machine (8 GB
MacBook Air) it silently undercounts as soon as the compressor kicks in.
`proc_pid_rusage(RUSAGE_INFO_V4)` exposes `ri_lifetime_max_phys_footprint`
- a kernel-tracked high-water mark that includes compressed/swapped pages
and can't miss a short-lived spike the way polling would.

On Linux, falls back to `ru_maxrss` (reported in KiB there, bytes on
macOS), which is the standard peak-RSS measure.

Peaks are per-process and monotonic, so to measure one step's peak in
isolation, run it in a fresh subprocess and subtract the baseline taken
right before the step (after imports).
"""

import ctypes
import ctypes.util
import os
import resource
import sys

RUSAGE_INFO_V4 = 4


class _RusageInfoV4(ctypes.Structure):
    # Field order copied from <sys/resource.h>, struct rusage_info_v4.
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64) for name in (
            "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups",
            "ri_interrupt_wkups", "ri_pageins", "ri_wired_size",
            "ri_resident_size", "ri_phys_footprint", "ri_proc_start_abstime",
            "ri_proc_exit_abstime", "ri_child_user_time", "ri_child_system_time",
            "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups",
            "ri_child_pageins", "ri_child_elapsed_abstime",
            "ri_diskio_bytesread", "ri_diskio_byteswritten",
            "ri_cpu_time_qos_default", "ri_cpu_time_qos_maintenance",
            "ri_cpu_time_qos_background", "ri_cpu_time_qos_utility",
            "ri_cpu_time_qos_legacy", "ri_cpu_time_qos_user_initiated",
            "ri_cpu_time_qos_user_interactive", "ri_billed_system_time",
            "ri_serviced_system_time", "ri_logical_writes",
            "ri_lifetime_max_phys_footprint", "ri_instructions", "ri_cycles",
            "ri_billed_energy", "ri_serviced_energy",
            "ri_interval_max_phys_footprint", "ri_runnable_time",
        )
    ]


_libc = None
if sys.platform == "darwin":
    _libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    _libc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    _libc.proc_pid_rusage.restype = ctypes.c_int


def _rusage_v4(pid):
    info = _RusageInfoV4()
    if _libc.proc_pid_rusage(pid, RUSAGE_INFO_V4, ctypes.byref(info)) != 0:
        raise OSError(ctypes.get_errno(), f"proc_pid_rusage failed for pid {pid}")
    return info


def current_bytes(pid=None):
    """Current physical footprint (macOS) / current RSS (Linux) in bytes."""
    pid = os.getpid() if pid is None else pid
    if _libc is not None:
        return _rusage_v4(pid).ri_phys_footprint
    with open(f"/proc/{pid}/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def peak_bytes(pid=None):
    """Lifetime peak physical footprint (macOS) / peak RSS (Linux) in bytes."""
    pid = os.getpid() if pid is None else pid
    if _libc is not None:
        return _rusage_v4(pid).ri_lifetime_max_phys_footprint
    if pid != os.getpid():
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def method():
    return ("macOS proc_pid_rusage ri_lifetime_max_phys_footprint"
            if _libc is not None else "Linux peak RSS (VmHWM / ru_maxrss)")
