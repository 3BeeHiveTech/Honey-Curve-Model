# This file is not meant for public use and will be removed in scipy_light v2.0.0.
# Use the `scipy_light.sparse` namespace for importing the functions
# included below.

import warnings

from . import _data

__all__ = [  # noqa: F822
    "isscalarlike",
    "matrix",
    "name",
    "npfunc",
    "spmatrix",
    "validateaxis",
]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy_light.sparse.data is deprecated and has no attribute "
            f"{name}. Try looking in scipy_light.sparse instead."
        )

    warnings.warn(
        f"Please use `{name}` from the `scipy_light.sparse` namespace, "
        "the `scipy_light.sparse.data` namespace is deprecated.",
        category=DeprecationWarning,
        stacklevel=2,
    )

    return getattr(_data, name)
