"""
This module contains loss classes suitable for fitting.

It is not part of the public API.
Specific losses are used for regression, binary classification or multiclass
classification.
"""
# Goals:
# - Provide a common private module for loss functions/classes.
# - To be used in:
#   - LogisticRegression
#   - PoissonRegressor, GammaRegressor, TweedieRegressor
#   - HistGradientBoostingRegressor, HistGradientBoostingClassifier
#   - GradientBoostingRegressor, GradientBoostingClassifier
#   - SGDRegressor, SGDClassifier
# - Replace link module of GLMs.

from .link import MultinomialLogit


class BareboneHalfMultinomialLoss:
    """Barebone HalfMultinomialLoss"""

    is_multiclass = True

    def __init__(self):
        self.link = MultinomialLogit()

    def predict_proba(self, raw_prediction):
        """Predict probabilities.

        Parameters
        ----------
        raw_prediction : array of shape (n_samples, n_classes)
            Raw prediction values (in link space).

        Returns
        -------
        proba : array of shape (n_samples, n_classes)
            Element-wise class probabilities.
        """
        return self.link.inverse(raw_prediction)
