'''
logsumexp.py

Provides a numerically implementation of logsumexp function,
such that no matter what 1-dimensional input array is provided,
we return a finite floating point answer that does not overflow or underflow.

'''

import numpy as np

def my_logsumexp(scores_N):
    ''' Compute logsumexp on provided array in numerically stable way.

    This function only handles 1D arrays.
    The equivalent scipy function can handle arrays of many dimensions.

    Args
    ----
    scores_N : 1D NumPy array, shape (N,)
        An array of real values

    Returns
    -------
    a : float
        Result of the logsumexp computation

    '''
    scores_N = np.asarray(scores_N, dtype=np.float64)
    m = np.max(scores_N)
    logsumexp = m + np.log(np.sum(np.exp(scores_N - m)))
    return logsumexp
