# This file is not meant for public use and will be removed in scipy_light v2.0.0.
# Use the `scipy_light.sparse` namespace for importing the functions
# included below.

import warnings

from . import _dok

__all__ = [  # noqa: F822
    "IndexMixin",
    "check_shape",
    "dok_matrix",
    "get_index_dtype",
    "getdtype",
    "isdense",
    "isintlike",
    "isscalarlike",
    "isshape",
    "isspmatrix",
    "isspmatrix_dok",
    "itertools",
    "spmatrix",
    "upcast",
    "upcast_scalar",
]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy_light.sparse.dok is deprecated and has no attribute "
            f"{name}. Try looking in scipy_light.sparse instead."
        )

    warnings.warn(
        f"Please use `{name}` from the `scipy_light.sparse` namespace, "
        "the `scipy_light.sparse.dok` namespace is deprecated.",
        category=DeprecationWarning,
        stacklevel=2,
    )

    return getattr(_dok, name)
