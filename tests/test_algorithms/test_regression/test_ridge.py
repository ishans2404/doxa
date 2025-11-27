"""Tests for Ridge Regression."""

import pytest
import numpy as np
from doxa.algorithms.regression import Ridge
from doxa import Tensor


class TestRidge:
    """Test cases for Ridge regression."""

    def test_initialization(self):
        model = Ridge()
        assert model.fit_intercept is True
        assert model.method == 'auto'
        assert model.alpha == 1.0
        assert not model._fitted

    def test_negative_alpha_raises(self):
        with pytest.raises(ValueError):
            Ridge(alpha=-0.1)

    @pytest.mark.parametrize("method", [
        'cholesky', 'svd', 'qr', 'pinv'
    ])
    def test_different_methods(self, sample_data, method):
        X, y = sample_data
        model = Ridge(method=method)
        model.fit(X, y)

        assert model._fitted
        assert model.coef_ is not None
        assert model.intercept_ is not None

        preds = model.predict(X)
        assert isinstance(preds, Tensor)
        assert preds.shape == y.shape

    def test_fit_without_intercept(self, sample_data):
        X, y = sample_data
        model = Ridge(fit_intercept=False)
        model.fit(X, y)

        assert model._fitted
        assert model.intercept_ == 0.0
        # coefficients length should match number of features
        coef = np.asarray(model.coef_)
        assert coef.shape[0] == X.shape[1]

    def test_prediction_before_fit(self, sample_data):
        X, y = sample_data
        model = Ridge()
        with pytest.raises(ValueError, match="Model must be fitted before making predictions."):
            model.predict(X)

    def test_alpha_effect_on_coefficients(self, sample_data):
        X, y = sample_data
        # small alpha (close to OLS)
        model_low = Ridge(alpha=1e-6)
        model_low.fit(X, y)
        coef_low = np.asarray(model_low.coef_)

        # large alpha (strong regularization)
        model_high = Ridge(alpha=1e3)
        model_high.fit(X, y)
        coef_high = np.asarray(model_high.coef_)

        # coefficients under strong regularization should have smaller norm
        norm_low = np.linalg.norm(coef_low)
        norm_high = np.linalg.norm(coef_high)
        assert norm_high <= norm_low

    def test_score_and_report(self, sample_data):
        X, y = sample_data
        model = Ridge()
        model.fit(X, y)

        r2 = model.score(X, y, metric='r2')
        assert isinstance(r2, float)

        report = model.report(X, y, return_dict=True)
        assert isinstance(report, dict)
        assert 'Summary' in report and 'Metrics' in report
