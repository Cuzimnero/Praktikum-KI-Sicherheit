import numpy as np

def mean_absolute_percentage_error(y_true, y_predicted):
    return np.mean(np.abs(y_true - y_predicted))

def cumulative_score(y_true, y_predicted, tolerance = 1):
    return np.mean(np.abs(y_true - y_predicted) <= tolerance)