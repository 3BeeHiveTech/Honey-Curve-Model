"""Fast Gradient Boosting decision trees for classification and regression."""
# Author: Nicolas Hug

from abc import ABC, abstractmethod
from functools import partial
import numpy as np


from honey_curve.sklearn_light.validation import check_array, _check_y, check_X_y, _num_features

X_DTYPE = np.float64


class BareboneBaseEstimator:
    """Barebone version of the BaseEstimator in scikit-learn."""

    def _check_n_features(self, X, reset):
        """Set the `n_features_in_` attribute, or check against it.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The input samples.
        reset : bool
            If True, the `n_features_in_` attribute is set to `X.shape[1]`.
            If False and the attribute exists, then check that it is equal to
            `X.shape[1]`. If False and the attribute does *not* exist, then
            the check is skipped.
            .. note::
               It is recommended to call reset=True in `fit` and in the first
               call to `partial_fit`. All other methods that validate `X`
               should set `reset=False`.
        """
        try:
            n_features = _num_features(X)
        except TypeError as e:
            if not reset and hasattr(self, "n_features_in_"):
                raise ValueError(
                    "X does not contain any features, but "
                    f"{self.__class__.__name__} is expecting "
                    f"{self.n_features_in_} features"
                ) from e
            # If the number of features is not defined and reset=True,
            # then we skip this check
            return

        if reset:
            self.n_features_in_ = n_features
            return

        if not hasattr(self, "n_features_in_"):
            # Skip this check if the expected number of expected input features
            # was not recorded by calling fit first. This is typically the case
            # for stateless transformers.
            return

        if n_features != self.n_features_in_:
            raise ValueError(
                f"X has {n_features} features, but {self.__class__.__name__} "
                f"is expecting {self.n_features_in_} features as input."
            )

    def _validate_data(
        self,
        X="no_validation",
        y="no_validation",
        reset=True,
        validate_separately=False,
        **check_params,
    ):
        """Validate input data and set or check the `n_features_in_` attribute.

        Parameters
        ----------
        X : {array-like, sparse matrix, dataframe} of shape \
                (n_samples, n_features), default='no validation'
            The input samples.
            If `'no_validation'`, no validation is performed on `X`. This is
            useful for meta-estimator which can delegate input validation to
            their underlying estimator(s). In that case `y` must be passed and
            the only accepted `check_params` are `multi_output` and
            `y_numeric`.

        y : array-like of shape (n_samples,), default='no_validation'
            The targets.

            - If `None`, `check_array` is called on `X`. If the estimator's
              requires_y tag is True, then an error will be raised.
            - If `'no_validation'`, `check_array` is called on `X` and the
              estimator's requires_y tag is ignored. This is a default
              placeholder and is never meant to be explicitly set. In that case
              `X` must be passed.
            - Otherwise, only `y` with `_check_y` or both `X` and `y` are
              checked with either `check_array` or `check_X_y` depending on
              `validate_separately`.

        reset : bool, default=True
            Whether to reset the `n_features_in_` attribute.
            If False, the input will be checked for consistency with data
            provided when reset was last True.
            .. note::
               It is recommended to call reset=True in `fit` and in the first
               call to `partial_fit`. All other methods that validate `X`
               should set `reset=False`.

        validate_separately : False or tuple of dicts, default=False
            Only used if y is not None.
            If False, call validate_X_y(). Else, it must be a tuple of kwargs
            to be used for calling check_array() on X and y respectively.

            `estimator=self` is automatically added to these dicts to generate
            more informative error message in case of invalid input data.

        **check_params : kwargs
            Parameters passed to :func:`sklearn.utils.check_array` or
            :func:`sklearn.utils.check_X_y`. Ignored if validate_separately
            is not False.

            `estimator=self` is automatically added to these params to generate
            more informative error message in case of invalid input data.

        Returns
        -------
        out : {ndarray, sparse matrix} or tuple of these
            The validated input. A tuple is returned if both `X` and `y` are
            validated.
        """
        self._check_feature_names(X, reset=reset)

        if y is None and self._get_tags()["requires_y"]:
            raise ValueError(
                f"This {self.__class__.__name__} estimator "
                "requires y to be passed, but the target y is None."
            )

        no_val_X = isinstance(X, str) and X == "no_validation"
        no_val_y = y is None or isinstance(y, str) and y == "no_validation"

        default_check_params = {"estimator": self}
        check_params = {**default_check_params, **check_params}

        if no_val_X and no_val_y:
            raise ValueError("Validation should be done on X, y or both.")
        elif not no_val_X and no_val_y:
            X = check_array(X, input_name="X", **check_params)
            out = X
        elif no_val_X and not no_val_y:
            y = _check_y(y, **check_params)
            out = y
        else:
            if validate_separately:
                # We need this because some estimators validate X and y
                # separately, and in general, separately calling check_array()
                # on X and y isn't equivalent to just calling check_X_y()
                # :(
                check_X_params, check_y_params = validate_separately
                if "estimator" not in check_X_params:
                    check_X_params = {**default_check_params, **check_X_params}
                X = check_array(X, input_name="X", **check_X_params)
                if "estimator" not in check_y_params:
                    check_y_params = {**default_check_params, **check_y_params}
                y = check_array(y, input_name="y", **check_y_params)
            else:
                X, y = check_X_y(X, y, **check_params)
            out = X, y

        if not no_val_X and check_params.get("ensure_2d", True):
            self._check_n_features(X, reset=reset)

        return out


class BareboneBaseHistGradientBoosting(BareboneBaseEstimator, ABC):
    """Barebone version of the BaseHistGradientBoosting class in sklearn."""

    @abstractmethod
    def __init__(
        self,
        loss,
        *,
        learning_rate,
        max_iter,
        max_leaf_nodes,
        max_depth,
        min_samples_leaf,
        l2_regularization,
        max_bins,
        categorical_features,
        monotonic_cst,
        warm_start,
        early_stopping,
        scoring,
        validation_fraction,
        n_iter_no_change,
        tol,
        verbose,
        random_state,
    ):
        self.loss = loss
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.max_bins = max_bins
        self.monotonic_cst = monotonic_cst
        self.categorical_features = categorical_features
        self.warm_start = warm_start
        self.early_stopping = early_stopping
        self.scoring = scoring
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state

    def _raw_predict(self, X, n_threads=None):
        """Return the sum of the leaves values over all predictors.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        n_threads : int, default=None
            Number of OpenMP threads to use. `_openmp_effective_n_threads` is called
            to determine the effective number of threads use, which takes cgroups CPU
            quotes into account. See the docstring of `_openmp_effective_n_threads`
            for details.

        Returns
        -------
        raw_predictions : array, shape (n_samples, n_trees_per_iteration)
            The raw predicted values.
        """
        is_binned = getattr(self, "_in_fit", False)
        if not is_binned:
            X = self._validate_data(X, dtype=X_DTYPE, force_all_finite=False, reset=False)
        # Axel's NOTE: skip check_is_fitted
        # check_is_fitted(self)
        if X.shape[1] != self._n_features:
            raise ValueError(
                "X has {} features but this estimator was trained with "
                "{} features.".format(X.shape[1], self._n_features)
            )
        n_samples = X.shape[0]
        raw_predictions = np.zeros(
            shape=(n_samples, self.n_trees_per_iteration_),
            dtype=self._baseline_prediction.dtype,
            order="F",
        )
        raw_predictions += self._baseline_prediction

        # We intentionally decouple the number of threads used at prediction
        # time from the number of threads used at fit time because the model
        # can be deployed on a different machine for prediction purposes.
        # n_threads = _openmp_effective_n_threads(n_threads)
        n_threads = 6  # Axel's NOTE: fix this to avoid using _openmp_effective_n_threads
        self._predict_iterations(X, self._predictors, raw_predictions, is_binned, n_threads)
        return raw_predictions

    def _predict_iterations(self, X, predictors, raw_predictions, is_binned, n_threads):
        """Add the predictions of the predictors to raw_predictions."""
        if not is_binned:
            (
                known_cat_bitsets,
                f_idx_map,
            ) = self._bin_mapper.make_known_categories_bitsets()

        for predictors_of_ith_iteration in predictors:
            for k, predictor in enumerate(predictors_of_ith_iteration):
                if is_binned:
                    predict = partial(
                        predictor.predict_binned,
                        missing_values_bin_idx=self._bin_mapper.missing_values_bin_idx_,
                        n_threads=n_threads,
                    )
                else:
                    predict = partial(
                        predictor.predict,
                        known_cat_bitsets=known_cat_bitsets,
                        f_idx_map=f_idx_map,
                        n_threads=n_threads,
                    )
                raw_predictions[:, k] += predict(X)


class BareboneHistGradientBoostingClassifier(BareboneBaseHistGradientBoosting):
    """Barebone version of the Histogram-based Gradient Boosting Classification Tree."""

    def __init__(
        self,
        classes_,
        _loss,
        _n_features,
        n_trees_per_iteration_,
        _baseline_prediction,
        _predictors,
        _bin_mapper,
        _check_feature_names,
    ):
        self.classes_ = classes_
        self._loss = _loss
        self._n_features = _n_features
        self.n_trees_per_iteration_ = n_trees_per_iteration_
        self._baseline_prediction = _baseline_prediction
        self._predictors = _predictors
        self._bin_mapper = _bin_mapper
        self._check_feature_names = _check_feature_names

    def predict(self, X):
        """Predict classes for X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The predicted classes.
        """
        # TODO: This could be done in parallel
        encoded_classes = np.argmax(self.predict_proba(X), axis=1)
        return self.classes_[encoded_classes]

    def predict_proba(self, X):
        """Predict class probabilities for X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        p : ndarray, shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        raw_predictions = self._raw_predict(X)
        return self._loss.predict_proba(raw_predictions)
