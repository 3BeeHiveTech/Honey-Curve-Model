"""Global configuration state and functions for management
"""
import os
import threading

_global_config = {
    "assume_finite": bool(os.environ.get("SKLEARN_ASSUME_FINITE", False)),
    "working_memory": int(os.environ.get("SKLEARN_WORKING_MEMORY", 1024)),
    "print_changed_only": True,
    "display": "diagram",
    "pairwise_dist_chunk_size": int(os.environ.get("SKLEARN_PAIRWISE_DIST_CHUNK_SIZE", 256)),
    "enable_cython_pairwise_dist": True,
}
_threadlocal = threading.local()


def _get_threadlocal_config():
    """Get a threadlocal **mutable** configuration. If the configuration
    does not exist, copy the default global configuration."""
    if not hasattr(_threadlocal, "global_config"):
        _threadlocal.global_config = _global_config.copy()
    return _threadlocal.global_config


def get_config():
    """Retrieve current values for configuration set by :func:`set_config`.

    Returns
    -------
    config : dict
        Keys are parameter names that can be passed to :func:`set_config`.

    See Also
    --------
    config_context : Context manager for global scikit-learn configuration.
    set_config : Set global scikit-learn configuration.
    """
    # Return a copy of the threadlocal configuration so that users will
    # not be able to modify the configuration with the returned dict.
    return _get_threadlocal_config().copy()
