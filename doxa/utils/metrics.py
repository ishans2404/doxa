"""Basic metrics for evaluating model."""

import numpy as np

def mean_squared_error(y_true, y_pred):
    """Calculate mean squared error between true and predicted values."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
    """Calculate R^2 (coefficient of determination) regression score."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_total) if ss_total else 0.0

def accuracy(y_true, y_pred):
    """Calculate accuracy classification score."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(y_true == y_pred)

def mse(y_true, y_pred):
    """Alias for mean squared error."""
    return mean_squared_error(y_true, y_pred)

def mae(y_true, y_pred):
    """Calculate mean absolute error between true and predicted values."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    """Alias for root mean squared error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

