"""Base Classes for doxa components."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
import numpy as np


class DoxaBase(ABC):
    """Abstract base class for all Doxa components."""

    def __init__(self):
        self._fitted = False
        self._parameters = {}
        self._metadata = {}

    @abstractmethod
    def fit(self, X, y=None):
        """Fit the model to training data."""
        pass

    def get_params(self) -> Dict[str, Any]:
        """Get parameters of the model."""
        return self._parameters.copy()
    
    def set_params(self, **params) -> 'DoxaBase':
        """Set parameters of the model."""
        for key, val in params.items():
            if hasattr(self, key):
                setattr(self, key, val)
            self._parameters[key] = val
        return self
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata of the model."""
        return {
            'fitted' : self._fitted,
            'parameters': self._parameters,
            **self._metadata
        }
    
    @property
    def is_fitted(self) -> bool:
        """Check if the model has been fitted."""
        return self._fitted


class Algorithm(DoxaBase):
    """Base class for all algorithms in Doxa."""

    def __init__(self, **kwargs):
        super().__init__()
        self._parameters.update(kwargs)

    def predict(self, X):
        """Make predictions using the fitted model on input data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions.")
        return self._predict(X)
    
    @abstractmethod
    def _predict(self, X):
        """Internal method to be implemented by subclasses for prediction."""
        pass

    def fit_predict(self, X, y=None):
        """Fit the model and make predictions."""
        self.fit(X, y)
        return self.predict(X)
    

class Regressor(Algorithm):
    """Base Class for regression algorithms in Doxa."""

    def score(self, X, y):
        """Return the coefficient of determination R^2 of the prediction."""
        from ..utils.metrics import r2_score
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
    

class Classifier(Algorithm):
    """Base Class for classification algorithms in Doxa."""

    def fit_predict(self, X, y=None):
        """Fit the model and return cluster labels."""
        return super().fit_predict(X, y)
    
    