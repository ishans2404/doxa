"""
Ridge Regression Algorithm Implementation
"""

from typing import Optional, List
import numpy as np
from doxa.core.base import Regressor
from doxa import Tensor, metrics
import matplotlib.pyplot as plt
from tabulate import tabulate

class Ridge(Regressor):
    """
    Ridge Regression model with L2 Regularization and multiple solver methods.
    
    Ridge Regression adds a penalty term (alpha * sum of squared coefficients) to the loss function to prevent overfitting.

    Loss Function: MSE + alpha * ||w||^2

    Args:
        alpha (float): Regularization strength. Must be positive float.
            - alpha = 0: equivalent to Linear Regression.
            - Larger alpha: stronger regularization, simpler model.
        fit_intercept (bool): Whether to fit the intercept term.
        method (str): Solver method. Options:
            - 'auto': Default, uses Cholesky (fast and stable with regularization)
            - 'cholesky': Normal equations with Cholesky decomposition
            - 'svd': Singular Value Decomposition (most stable)
            - 'qr': QR Decomposition
            - 'pinv': Pseudo-inverse
    """

    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True, method: str = 'auto'):
        super().__init__()
        if alpha < 0:
            raise ValueError(f"alpha must be a non-negative float, got {alpha}")
        
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.method = method
        self.coef_ = None
        self.intercept_ = None
        self.device = 'cpu'
        self._parameters.update({
            'alpha': alpha,
            'fit_intercept': fit_intercept,
            'method': method
        })
    
    def _get_backend(self, X_data):
        """Get appropriate backend based on input data."""
        if hasattr(X_data, 'get'):  # CuPy array
            try:
                import cupy as cp
                return cp
            except ImportError:
                return np
        return np
    
    def fit(self, X, y):
        """Fit the ridge regression model.

        This implementation supports multiple solvers ('cholesky', 'svd', 'qr', 'pinv')
        and handles CPU/CuPy backends. When ``fit_intercept`` is True the data are
        centered so the intercept is not penalized; the intercept is computed as
        ``y_mean - X_mean @ coef`` after fitting on centered data.

        Args:
            X: Training features. Array-like or :class:`doxa.Tensor` with shape
                ``(n_samples, n_features)``.
            y: Training targets. Array-like or :class:`doxa.Tensor` with shape
                ``(n_samples,)`` or ``(n_samples, n_targets)``.

        Returns:
            self: The fitted estimator.
        """
        # Handle Tensor
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

        # ensure shapes
        n_samples, n_features = X_data.shape[0], X_data.shape[1]

        # Center data when fitting intercept to avoid penalizing it
        if self.fit_intercept:
            X_mean = backend.mean(X_data, axis=0)
            y_mean = backend.mean(y_data, axis=0)
            Xc = X_data - X_mean
            yc = y_data - y_mean
        else:
            Xc, yc = X_data, y_data
            X_mean, y_mean = None, None
        
        # Method selection
        method = self.method.lower()
        if method == 'auto':
            # Cholesky is fast and stable with regularization
            method = 'cholesky'
        
        try:
            alpha = float(self.alpha)

            if method == 'cholesky':
                # Equation: (X^T X + alpha I) w = X^T y
                XtX = Xc.T @ Xc
                reg_matrix = backend.eye(XtX.shape[0]) * alpha                
                XtX_reg = XtX + reg_matrix
                Xty = Xc.T @ yc

                try:
                    coef = backend.linalg.solve(XtX_reg, Xty)
                except Exception:
                    # Fallback to pseudo inverse if cholesky fails
                    coef = backend.linalg.pinv(XtX_reg) @ Xty

            elif method in ('svd', 'pinv'):
                # Tikhonov via SVD: w = V diag(S/(S^2 + alpha)) U^T y
                U, S, Vt = backend.linalg.svd(Xc, full_matrices=False)
                UTy = U.T @ yc
                denom = S ** 2 + alpha
                factors = S / denom
                # Handle 1D and 2D targets: UTy can be (k,) or (k, t)
                if getattr(UTy, 'ndim', 1) == 1:
                    coef = Vt.T @ (factors * UTy)
                else:
                    coef = Vt.T @ (factors[:, None] * UTy)
            
            elif method == 'qr':
                # Augmented system for ridge: [X; sqrt(alpha)*I] w = [y; 0]
                # Build column-shaped targets to avoid ambiguous stacking.
                sqrt_alpha = backend.sqrt(alpha)
                I = backend.eye(n_features)

                if getattr(yc, 'ndim', 1) == 1:
                    # make yc a column vector
                    yc_col = yc.reshape(-1, 1)
                    zeros = backend.zeros((n_features, 1))
                    X_aug = backend.vstack([Xc, sqrt_alpha * I])
                    y_aug = backend.vstack([yc_col, zeros])
                    Q, R = backend.linalg.qr(X_aug)
                    coef_col = backend.linalg.solve(R, Q.T @ y_aug)
                    coef = coef_col.reshape(-1)
                else:
                    # multi-output
                    zeros = backend.zeros((n_features, yc.shape[1]))
                    X_aug = backend.vstack([Xc, sqrt_alpha * I])
                    y_aug = backend.vstack([yc, zeros])
                    Q, R = backend.linalg.qr(X_aug)
                    coef = backend.linalg.solve(R, Q.T @ y_aug)
            
            else:
                raise ValueError(f"Unknown method '{self.method}'")

        except Exception as e:
            raise RuntimeError(f"Failed to fit Ridge with method '{self.method}': {str(e)}")
        
        # Normalize shapes for 1D targets and ensure coef is an array on the
        # Normalize 2D column results to 1D vector when appropriate
        try:
            if hasattr(coef, 'shape') and len(coef.shape) > 1 and coef.shape[1] == 1:
                coef = coef.reshape(-1)
        except Exception:
            # If shape inspection fails for non-standard array objects,
            # leave coef as-is and rely on backend.asarray below to raise if needed.
            pass

        coef_arr = backend.asarray(coef)
        
        # Set intercept and coefficients. Ensure X_mean/y_mean and coef are
        # backend arrays before performing matrix multiplication so '@' is supported
        if self.fit_intercept:
            self.coef_ = coef_arr
            # convert means to backend arrays if available
            if X_mean is not None and y_mean is not None:
                X_mean_arr = backend.asarray(X_mean)
                y_mean_arr = backend.asarray(y_mean)
                # compute intercept safely; handle scalar/multi-output
                try:
                    intercept = y_mean_arr - (X_mean_arr @ self.coef_)
                except Exception:
                    # fallback to dot when shapes require it
                    intercept = y_mean_arr - backend.dot(X_mean_arr, self.coef_)

                try:
                    if getattr(intercept, 'shape', ()) == ():
                        intercept = float(intercept)
                except Exception:
                    pass
                self.intercept_ = intercept
            else:
                # Defensive fallback (shouldn't occur when fit_intercept True)
                self.intercept_ = 0.0
        else:
            self.coef_ = coef_arr
            self.intercept_ = 0.0
        
        self._fitted = True
        return self
    
    def _predict(self, X):
        """Make predictions using the fitted Ridge model.

        Args:
            X: Array-like or :class:`doxa.Tensor` of shape ``(n_samples, n_features)``.

        Returns:
            doxa.Tensor: Predicted values with same device as input.
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
        """Ensure coefficients and intercept live on the same backend as input.

        Args:
            X_data: A backend array (NumPy or CuPy) representing features.

        Returns:
            tuple: ``(coef, intercept)`` as arrays on the same backend as ``X_data``.
        """
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
        """Return model parameters.

        Returns:
            dict: Dictionary of parameters including coefficients and intercept when fitted.
        """
        params = super().get_params()
        if self._fitted:
            params.update({
                'coefficients': self.coef_,
                'intercept': self.intercept_,
                'alpha': self.alpha,
                'method': self.method,
                'device': self.device
            })
        return params

    def report(self, X=None, y=None, return_dict=False):
        """Generate a human-readable report of model settings and performance.

        Args:
            X: Optional features to compute performance metrics.
            y: Optional targets to compute performance metrics.
            return_dict (bool): If True return the report as a dictionary
                instead of printing it.

        Returns:
            dict or None: If ``return_dict`` is True returns a dict with keys
            ``"Summary"`` and ``"Metrics"``; otherwise prints to stdout.
        """

        summary = {
            "Method": self.method,
            "Alpha": self.alpha,
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
                    "MSE": metrics.mse(y_true=y_true, y_pred=preds),
                    "MAE": metrics.mae(y_true=y_true, y_pred=preds),
                    "R2 Score": metrics.r2_score(y_true=y_true, y_pred=preds),
                    "Accuracy": metrics.accuracy(y_true=y_true, y_pred=preds),
                    "RMSE": metrics.rmse(y_true=y_true, y_pred=preds)
                }
            except Exception as e:
                results = {"Error": f"Failed to compute metrics: {str(e)}"}

        if return_dict:
            return {"Summary": summary, "Metrics": results}

        print("=" * 55)
        print("Doxa Model Report")
        print(f"Model: Ridge (alpha={self.alpha}, method={self.method}, fit_intercept={self.fit_intercept})\n")

        print("Training Summary:")
        print(tabulate(summary.items(), headers=["Key", "Value"], tablefmt="github"))

        if results:
            print("\nPerformance Metrics:")
            print(tabulate(results.items(), headers=["Metric", "Value"], tablefmt="github"))
        print("=" * 55)

    def viz(self, X, y, save=None):
        """Visualize the fit of the model.

        Produces a scatter + line plot for 1D features, or a residuals plot for
        higher-dimensional data.

        Args:
            X: Features used for plotting (array-like or :class:`doxa.Tensor`).
            y: True targets (array-like or :class:`doxa.Tensor`).
            save (str): Optional file path to save the figure. If not provided
                the plot is shown interactively.
        """

        if not self._fitted:
            raise ValueError("Model must be fitted before visualization.")

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
            residuals = y_data - preds
            plt.scatter(preds, residuals, alpha=0.6, label="Residuals")
            plt.axhline(0, color="red", linestyle="--", label="Zero Line")
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
        """Compute a performance metric for the fitted model.

        Args:
            X: Features for evaluation (array-like or :class:`doxa.Tensor`).
            y: True targets (array-like or :class:`doxa.Tensor`).
            metric (str): Metric to compute. Options: ``'r2'``, ``'mse'``,
                ``'mae'``, ``'rmse'``, ``'acc'``/``'accuracy'``.

        Returns:
            float: Computed metric value.
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
            return metrics.mse(y_true, preds)
        elif metric == "mae":
            return metrics.mae(y_true, preds)
        elif metric == "rmse":
            return metrics.rmse(y_true, preds)
        elif metric == "acc" or metric == "accuracy":
            return metrics.accuracy(y_true, preds)
        else:
            raise ValueError(f"Unknown metric '{metric}'. Available: r2, mse, mae, rmse")

    def explain(self, feature_names: Optional[List[str]] = None) -> str:
        """Return a human-readable equation explaining the fitted model.

        Args:
            feature_names: Optional list of feature names. If not provided a
                default ``['X0', 'X1', ...]`` is used.

        Returns:
            str: A short equation string describing the model.
        """

        if not self._fitted:
            return "Model not fitted yet"

        if self.coef_ is None:
            raise ValueError("Coefficients are not available. Please fit the model first.")

        # Convert coef_ to numpy for string formatting if needed
        try:
            coef = np.asarray(self.coef_)
            intercept = float(np.asarray(self.intercept_)) if self.fit_intercept else 0.0
        except Exception:
            coef = self.coef_
            intercept = self.intercept_

        if feature_names is None:
            feature_names = [f"X{i}" for i in range(len(coef))]

        if len(feature_names) != len(coef):
            raise ValueError(f"feature_names length ({len(feature_names)}) "
                             f"must match number of features ({len(coef)})")

        terms = [f"{c:.3f}*{name}" for c, name in zip(coef, feature_names)]
        expr = " + ".join(terms)

        if self.fit_intercept:
            expr = f"y ≈ {intercept:.3f} + " + expr
        else:
            expr = f"y ≈ {expr}"

        return expr

