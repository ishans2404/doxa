import pytest
import numpy as np
from doxa.core.base import DoxaBase, Algorithm, Regressor, Classifier, Clusterer


class ConcreteDoxaBase(DoxaBase):
    """Concrete class for testing DoxaBase."""
    def fit(self, X, y=None):
        self._fitted = True
        return self


class ConcreteAlgorithm(Algorithm):
    """Concrete class for testing Algorithm."""
    def fit(self, X, y=None):
        self._fitted = True
        return self

    def _predict(self, X):
        return np.zeros(len(X))


class ConcreteRegressor(Regressor):
    """Concrete class for testing Regressor."""
    def fit(self, X, y=None):
        self._fitted = True
        return self

    def _predict(self, X):
        return np.zeros(len(X))


class ConcreteClassifier(Classifier):
    """Concrete class for testing Classifier."""
    def fit(self, X, y=None):
        self._fitted = True
        return self

    def _predict(self, X):
        return np.zeros(len(X))


class ConcreteClusterer(Clusterer):
    """Concrete class for testing Clusterer."""
    def fit(self, X, y=None):
        self._fitted = True
        return self

    def _predict(self, X):
        return np.zeros(len(X))


class TestDoxaBase:
    def test_init(self):
        model = ConcreteDoxaBase()
        assert not model.is_fitted
        assert model.get_params() == {}
        assert model.get_metadata() == {'fitted': False, 'parameters': {}}

    def test_set_get_params(self):
        model = ConcreteDoxaBase()
        model.set_params(param1=1, param2='test')
        assert model.get_params() == {'param1': 1, 'param2': 'test'}

    def test_metadata(self):
        model = ConcreteDoxaBase()
        model.fit(None)
        metadata = model.get_metadata()
        assert metadata['fitted'] is True


class TestAlgorithm:
    def test_init_with_params(self):
        algo = ConcreteAlgorithm(param1=1, param2='test')
        assert algo.get_params() == {'param1': 1, 'param2': 'test'}

    def test_predict_without_fit(self):
        algo = ConcreteAlgorithm()
        with pytest.raises(ValueError, match="Model must be fitted before making predictions"):
            algo.predict(np.array([1, 2, 3]))

    def test_fit_predict(self):
        algo = ConcreteAlgorithm()
        X = np.array([1, 2, 3])
        result = algo.fit_predict(X)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(X)


class TestRegressor:
    def test_score(self):
        reg = ConcreteRegressor()
        X = np.array([[1], [2], [3]])
        y = np.array([0, 0, 0])
        reg.fit(X)
        score = reg.score(X, y)
        assert isinstance(score, (float, np.floating))


class TestClassifier:
    def test_score(self):
        clf = ConcreteClassifier()
        X = np.array([[1], [2], [3]])
        y = np.array([0, 0, 0])
        clf.fit(X)
        score = clf.score(X, y)
        assert isinstance(score, (float, np.floating))


class TestClusterer:
    def test_fit_predict(self):
        cluster = ConcreteClusterer()
        X = np.array([[1], [2], [3]])
        labels = cluster.fit_predict(X)
        assert isinstance(labels, np.ndarray)
        assert len(labels) == len(X)