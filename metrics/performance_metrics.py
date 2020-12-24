import numpy as np


def calc_mean_squared_error(y_N, yhat_N):
    ''' Compute the mean squared error given true and predicted values

    Args
    ----
    y_N : 1D array, shape (N,)
        Each entry represents 'ground truth' numeric response for an example
    yhat_N : 1D array, shape (N,)
        Each entry representes predicted numeric response for an example

    Returns
    -------
    mse : scalar float
        Mean squared error performance metric'''

    return np.mean(np.square(y_N - yhat_N))


def calc_mean_absolute_error(y_N, yhat_N):
    ''' Compute the mean absolute error given true and predicted values

    Args
    ----
    y_N : 1D array, shape (N,)
        Each entry represents 'ground truth' numeric response for an example
    yhat_N : 1D array, shape (N,)
        Each entry representes predicted numeric response for an example

    Returns
    -------
    mae : scalar float
    '''
    return np.mean(abs(y_N - yhat_N))
