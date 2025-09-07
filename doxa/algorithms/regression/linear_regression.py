"""
Linear Regression Algorithm Implementation.
"""

from typing import Optional, List
import numpy as np
from doxa.core.base import Regressor
from doxa import Tensor, metrics
import matplotlib.pyplot as plt
from tabulate import tabulate

class LinearRegression(Regressor):
    """
    Linear Regression model with multiple solver bands.

    Args:
        fit_intercepr (bool): Whether to fit the intercept term.
        method (str): Solver method. Options:
            - 'auto': Default, tries SVD (stable)
            - 'ols': Normal equations
            - 'svd': Singular Value Decomposition
            - 'qr': QR Decomposition
            - 'pinv': Pseudo-inverse
    """

    def __init__(self, fit_intercept: bool = True, method: str = 'auto'):
        super().__init__()
        self.fit_intercept = fit_intercept
        self.method = method
        self.coef_ = None
        self.intercept_ = None
        self.device = 'cpu'  # default device
        self._parameters.update({
            'fit_intercept': fit_intercept,
            'method': method
        })

    def _get_backend(self, X_data):
        """Get appropriate backend (numpy or cupy) based on the data."""
        if hasattr(X_data, 'get'):  # CuPy array
            try:
                import cupy as cp
                return cp
            except ImportError:
                return np
        return np

    def fit(self, X, y):
        """
        Fit the linear regression model.

        Args:
            X: Training features
            y: Training targets

        Returns:
            self: Fitted estimator
        """
        # Tensor handling
        if not isinstance(X, Tensor):
            X = Tensor(X)
        if not isinstance(y, Tensor):
            y = Tensor(y)
        
        # Ensure same device
        if X.device != y.device:
            if X.device == 'cpu':
                y = y.to_cpu()
            else:
                X = X.to_gpu()
        
        self.device = X.device
        X_data, y_data = X.data, y.data
        backend = self._get_backend(X_data)

        # Add intercept column if needed
        if self.fit_intercept:
            ones = backend.ones(shape=(X_data.shape[0], 1))
            X_data = backend.hstack([ones, X_data])
        
        # Method selection
        method = self.method.lower()
        if method == 'auto':
            method = 'svd'
        
        try:
            if method == 'ols':
                XtX = X_data.T @ X_data
                Xty = X_data.T @ y_data
                try:
                    coef = backend.linalg.solve(XtX, Xty)
                except Exception:
                    coef = backend.linalg.pinv(XtX) @ Xty
            
            elif method == 'svd':
                U, S, Vt = backend.linalg.svd(X_data, full_matrices=False)
                coef = Vt.T @ (U.T @ y_data / S)
            
            elif method == 'qr':
                Q, R = backend.linalg.qr(X_data)
                coef = backend.linalg.solve(R, Q.T @ y_data)
            
            elif method == 'pinv':
                coef = backend.linalg.pinv(X_data) @ y_data
            
            else:
                raise ValueError(f"Unknown method '{self.method}'")
        
        except Exception as e:
            raise RuntimeError(f"Failed to fit with method '{self.method}': {str(e)}")
        
        # Extract intercet and coefficients
        if self.fit_intercept:
            self.intercept_ = float(coef[0])
            self.coef_ = coef[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = coef
        
        self._fitted = True
        return self
    
    def _predict(self, X):
        """
        Predict using the linear regression model.

        Args:
            X: Input features

        Returns:
            Predictions as a Tensor
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before making predictions.")
        
        if not isinstance(X, Tensor):
            X = Tensor(X)
        
        X_data = X.data
        coef, intercept = self._ensure_backend_match(X_data)
        preds = X_data @ coef + intercept
        return Tensor(preds, device=X.device)
    
    def _ensure_backend_match(self, X_data):
        """Ensure coefficients are on the same backend as input data."""
        backend = self._get_backend(X_data)

        # GPU data
        if hasattr(X_data, 'get'):
            try:
                import cupy as cp
                coef = cp.asarray(self.coef_)
                intercept = cp.asarray(self.intercept_)
            except ImportError:
                # Fallback to CPU
                coef = backend.asarray(self.coef_)
                intercept = backend.asarray(self.intercept_)
        # CPU data
        else:
            coef = backend.asarray(self.coef_)
            intercept = backend.asarray(self.intercept_)
        
        return coef, intercept
    
    def get_params(self):
        """Get model parameters."""
        params = super().get_params()
        if self._fitted:
            params.update({
                'coefficients': self.coef_,
                'intercept': self.intercept_,
                'method': self.method,
                'device': self.device
            })
        return params

    def report(self, X=None, y=None, return_dict=False):
        """
        Generate a report of model parameters and performance metrics.

        Args:
            X: Features for performance evaluation (optional)
            y: Targets for performance evaluation (optional)
            return_dict (bool): If True, return report as a dictionary.
        
        Returns:
            Report as printed output or dictionary.
        """
        summary = {
            "Method": self.method,
            "Fit Intercept": self.fit_intercept,
            "Device": self.device, 
            "Fitted": self._fitted
        }

        results = {}
        if X is not None and y is not None and self._fitted:
            try:
                preds = self._predict(X).data
                y_true = y.data if isinstance(y, Tensor) else np.asarray(y)
                results = {
                    "MSE": metrics.mse_loss(y_true=y_true, y_pred=preds),
                    "MAE": metrics.mae_loss(y_true=y_true, y_pred=preds),
                    "R2 Score": metrics.r2_score(y_true=y_true, y_pred=preds),
                    "Accuracy": metrics.accuracy_score(y_true=y_true, y_pred=preds)
                }
            except Exception as e:
                results = {
                    "Error": f"Failed to compute metrics: {str(e)}"
                }

        if return_dict:
            return {"Summary": summary, "Metrics": results}
        
        # Pretty printing
        print("=" * 55)
        print("Doxa Model Report")
        print(f"Model: LinearRegression (method={self.method}, fit_intercept={self.fit_intercept})\n")
        
        print("Training Summary:")
        print(tabulate(summary.items(), headers=["Key", "Value"], tablefmt="github"))
        
        if results:
            print("\nPerformance Metrics:")
            print(tabulate(results.items(), headers=["Metric", "Value"], tablefmt="github"))
        print("=" * 55)

    def viz(self, X, y, save=None):
        """
        Visualize the model fit.

        Args:
            X: Input features 
            y: True targets
            save (str): File path to save the plot (optional)
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before visualization.")
            return

        if not isinstance(X, Tensor):
            X = Tensor(X)
        if not isinstance(y, Tensor):
            y = Tensor(y)

        X_data, y_data = X.data, y.data
        preds = self._predict(X).data

        plt.figure(figsize=(10, 6))

        if X_data.shape[1] == 1:
            plt.scatter(X_data, y_data, alpha=0.6, label="Data")
            plt.plot(X_data, preds, color="red", label="Fit")
            plt.xlabel("X")
            plt.ylabel("y")
            plt.title("Data and Fit Plot")
        else:
            # Residuals plot for higher dimensions
            residuals = y_data - preds
            plt.scatter(preds, residuals, alpha=0.6)
            plt.axhline(0, color="red", linestyle="--")
            plt.xlabel("Predictions")
            plt.ylabel("Residuals")
            plt.title("Residuals Plot")
        
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save:
            plt.savefig(save, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
    def score(self, X, y, metric='r2'):
        """
        Evaluate model performance using specified metric.
        Args:
            X: Input features
            y: True targets
            metric (str): Metric to use. Options: 'mse', 'mae', 'r2', 'accuracy'
        Returns:
            Computed metric value
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before scoring")
        
        if not isinstance(X, Tensor):
            X = Tensor(X)
        if not isinstance(y, Tensor):
            y = Tensor(y)
        
        preds = self._predict(X).data
        y_true = y.data
        
        if metric == "r2":
            return metrics.r2_score(y_true, preds)
        elif metric == "mse":
            return metrics.mse_loss(y_true, preds)
        elif metric == "mae":
            return metrics.mae_loss(y_true, preds)
        elif metric == "rmse":
            return np.sqrt(metrics.mse_loss(y_true, preds))
        else:
            raise ValueError(f"Unknown metric '{metric}'. Available: r2, mse, mae, rmse")
        
    def explain(self, feature_names: Optional[List[str]] = None) -> str:
        """Generate human-readable equation explanation."""
        if not self._fitted:
            return "Model not fitted yet"
        
        if self.coef_ is None:
            raise ValueError("Coefficients are not available. Please fit the model first.")

        if feature_names is None:
            feature_names = [f"X{i}" for i in range(len(self.coef_))]
        
        if len(feature_names) != len(self.coef_):
            raise ValueError(f"feature_names length ({len(feature_names)}) "
                           f"must match number of features ({len(self.coef_)})")
        
        terms = [f"{coef:.3f}*{name}" for coef, name in zip(self.coef_, feature_names)]
        expr = " + ".join(terms)
        
        if self.fit_intercept:
            expr = f"y ≈ {self.intercept_:.3f} + " + expr
        else:
            expr = f"y ≈ {expr}"
        
        return expr