# This file is not meant for public use and will be removed in scipy_light v2.0.0.
# Use the `scipy_light.sparse` namespace for importing the functions
# included below.

import warnings

from . import _construct

__all__ = [  # noqa: F822
    "block_diag",
    "bmat",
    "bsr_matrix",
    "check_random_state",
    "coo_matrix",
    "csc_matrix",
    "csr_hstack",
    "csr_matrix",
    "dia_matrix",
    "diags",
    "eye",
    "get_index_dtype",
    "hstack",
    "identity",
    "isscalarlike",
    "issparse",
    "kron",
    "kronsum",
    "numbers",
    "partial",
    "rand",
    "random",
    "rng_integers",
    "spdiags",
    "upcast",
    "vstack",
]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy_light.sparse.construct is deprecated and has no attribute "
            f"{name}. Try looking in scipy_light.sparse instead."
        )

    warnings.warn(
        f"Please use `{name}` from the `scipy_light.sparse` namespace, "
        "the `scipy_light.sparse.construct` namespace is deprecated.",
        category=DeprecationWarning,
        stacklevel=2,
    )

    return getattr(_construct, name)
