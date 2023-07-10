# This file is not meant for public use and will be removed in scipy_light v2.0.0.
# Use the `scipy_light.sparse` namespace for importing the functions
# included below.

import warnings

from . import _coo

__all__ = [  # noqa: F822
    "SparseEfficiencyWarning",
    "check_reshape_kwargs",
    "check_shape",
    "coo_matrix",
    "coo_matvec",
    "coo_tocsr",
    "coo_todense",
    "downcast_intp_index",
    "get_index_dtype",
    "getdata",
    "getdtype",
    "isshape",
    "isspmatrix",
    "isspmatrix_coo",
    "operator",
    "spmatrix",
    "to_native",
    "upcast",
    "upcast_char",
    "warn",
]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy_light.sparse.coo is deprecated and has no attribute "
            f"{name}. Try looking in scipy_light.sparse instead."
        )

    warnings.warn(
        f"Please use `{name}` from the `scipy_light.sparse` namespace, "
        "the `scipy_light.sparse.coo` namespace is deprecated.",
        category=DeprecationWarning,
        stacklevel=2,
    )

    return getattr(_coo, name)
