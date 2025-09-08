"""Test for Linear Regression."""

import pytest
import numpy as np
from doxa.algorithms.regression import LinearRegression
from doxa import Tensor, ones, zeros, randn

class TestLinearRegression:
    """Test cases for LinearRegression class."""

    def test_initialization(self):
        """Test model initialization"""
        model = LinearRegression()
        assert model.fit_intercept is True
        assert model.method == 'auto'
        assert not model._fitted

    @pytest.mark.parametrize("method", [
        'ols', 'svd', 'qr', 'pinv'
    ])
    def test_different_methods(self, sample_data, method):
        X, y = sample_data
        model = LinearRegression(method=method)
        model.fit(X, y)

        assert model._fitted
        assert model.coef_ is not None
        assert model.intercept_ is not None

        # Test prediction
        predictions = model.predict(X)
        assert isinstance(predictions, Tensor)
        assert predictions.shape == (X.shape,)

    def test_fit_without_intercept(self, sample_data):
        """Test fitting without intercept."""
        X, y = sample_data
        model = LinearRegression(fit_intercept=False)
        model.fit(X, y)

        assert model._fitted
        assert model.intercept_ == 0.0
        assert len(model.coef_) == X.shape
    
    def test_prediction_before_fit(self, sample_data):
        """Test that prediction fails before fitting."""
        X, y = sample_data
        model = LinearRegression()

        with pytest.raises(ValueError, match="Model must be fitted before making predictions."):
            model.predict(X)
        
    def test_score_method(self, sample_data):
        """Test scoring methods."""
        X, y = sample_data
        model = LinearRegression()
        model.fit(X, y)

        # Test R2 score
        r2 = model.score(X, y, metric='r2')
        assert isinstance(r2, float)
        assert r2 <= 1.0

        # Test MSE
        mse = model.score(X, y, metric='mse')
        assert isinstance(mse, float)
        assert mse >= 0

    def test_explain_method(self, small_data):
        """Test model explanation."""
        X, y = small_data
        model = LinearRegression()
        model.fit(X, y)

        explanation = model.explain(['feature1', 'feature2'])
        assert isinstance(explanation, str)
        assert 'feature1' in explanation
        assert 'feature2' in explanation

    def test_get_params(self, sample_data):
        """Test parameter retrieval."""
        X, y = sample_data
        model = LinearRegression()
        
        params = model.get_params()
        assert 'fit_intercept' in params
        assert 'method' in params

        model.fit(X, y)
        fitted_params = model.get_params()
        assert 'coefficients' in fitted_params
        assert 'intercept' in fitted_params

    @pytest.mark.slow
    def test_large_dataset(self):
        """Test linear regression on a large dataset with known parameters."""
        np.random.seed(331)
        
        # Create large input matrix and true parameters
        X = randn(10000, 50)
        true_weights = randn(50)
        noise = randn(10000) * 0.1
        
        # Generate target values: y = X @ w + noise
        y = (X @ true_weights) + noise
        
        # Fit and test the model
        model = LinearRegression()
        model.fit(X, y)
        
        # Verify the model is fitted
        assert model._fitted
        assert model.coef_.shape == (50,)
        
        # Test predictions
        predictions = model.predict(X)
        assert isinstance(predictions, Tensor)
        assert predictions.shape == (10000,)
        
        # Test score should be high since we used linear data
        r2_score = model.score(X, y)
        # Should be close to 1 due to low noise
        assert r2_score > 0.9

    def test_tensor_input(self, sample_data):
        """Test with Tensor inputs."""
        X, y = sample_data
        X_tensor = Tensor(X)
        y_tensor = Tensor(y)
        
        model = LinearRegression()
        model.fit(X_tensor, y_tensor)
        
        predictions = model.predict(X_tensor)
        assert isinstance(predictions, Tensor)
        assert predictions.shape == (X.shape,)
    
    def test_report_generation(self, sample_data):
        """Test report generation."""
        X, y = sample_data
        model = LinearRegression()
        model.fit(X, y)
        
        # Test dictionary return
        report = model.report(X, y, return_dict=True)
        assert isinstance(report, dict)
        assert 'Summary' in report
        assert 'Metrics' in report
        
        # Test metrics are present
        metrics = report['Metrics']
        assert 'MSE' in metrics
        assert 'R2 Score' in metrics
        