"""
This module contains the BinMapper class.

BinMapper is used for mapping a real-valued dataset into integer-valued bins.
Bin thresholds are computed with the quantiles so that each bin contains
approximately the same number of samples.
"""
# Author: Nicolas Hug

import numpy as np

from ._bitset import set_bitset_memoryview
from .common import X_BITSET_INNER_DTYPE


class _BareboneBinMapper:
    """Barebone _BinMapper"""

    def __init__(
        self,
        n_bins=256,
        subsample=int(2e5),
        is_categorical=None,
        known_categories=None,
        random_state=None,
        n_threads=None,
        is_categorical_=None,
        bin_thresholds_=None,
    ):
        self.n_bins = n_bins
        self.subsample = subsample
        self.is_categorical = is_categorical
        self.known_categories = known_categories
        self.random_state = random_state
        self.n_threads = n_threads
        self.is_categorical_ = is_categorical_
        self.bin_thresholds_ = bin_thresholds_

    def make_known_categories_bitsets(self):
        """Create bitsets of known categories.

        Returns
        -------
        - known_cat_bitsets : ndarray of shape (n_categorical_features, 8)
            Array of bitsets of known categories, for each categorical feature.
        - f_idx_map : ndarray of shape (n_features,)
            Map from original feature index to the corresponding index in the
            known_cat_bitsets array.
        """

        categorical_features_indices = np.flatnonzero(self.is_categorical_)

        n_features = self.is_categorical_.size
        n_categorical_features = categorical_features_indices.size

        f_idx_map = np.zeros(n_features, dtype=np.uint32)
        f_idx_map[categorical_features_indices] = np.arange(
            n_categorical_features, dtype=np.uint32
        )

        known_categories = self.bin_thresholds_

        known_cat_bitsets = np.zeros((n_categorical_features, 8), dtype=X_BITSET_INNER_DTYPE)

        # TODO: complexity is O(n_categorical_features * 255). Maybe this is
        # worth cythonizing
        for mapped_f_idx, f_idx in enumerate(categorical_features_indices):
            for raw_cat_val in known_categories[f_idx]:
                set_bitset_memoryview(known_cat_bitsets[mapped_f_idx], raw_cat_val)

        return known_cat_bitsets, f_idx_map
