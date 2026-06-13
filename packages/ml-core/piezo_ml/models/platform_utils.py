"""
Cross-platform multiprocessing safety utilities.

macOS: n_jobs=1 to prevent OpenMP segfaults with tree-based models.
Windows/Linux: use all cores.
Also configures thread-pool limits and multiprocessing start method.
"""

from __future__ import annotations

import os
import platform


def get_safe_n_jobs() -> int:
    """Return safe n_jobs value. n_jobs=1 on macOS, -1 (all cores) elsewhere."""
    if platform.system() == "Darwin":
        return 1
    return -1


def configure_multiprocessing() -> None:
    """Set environment variables and start method for safe cross-platform ML."""
    if platform.system() == "Darwin":
        # Prevent OpenMP/BLAS thread-pool conflicts on macOS
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    # Use spawn on macOS to avoid fork-related issues; forkserver on Linux
    import multiprocessing
    system = platform.system()
    try:
        if system == "Darwin":
            multiprocessing.set_start_method("spawn", force=False)
        elif system == "Linux":
            multiprocessing.set_start_method("forkserver", force=False)
        # Windows defaults to spawn already
    except RuntimeError:
        pass  # Already set — safe to ignore


# Auto-configure on import
configure_multiprocessing()
