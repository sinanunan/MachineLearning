'''
calc_performance_metrics_for_proba_predictions.py

Provides implementation of common metrics for assessing a binary classifier's
*probabilistic* predictions against true binary labels, including:

* binary cross entropy from probabilities
* binary cross entropy from scores (real-values, to be fed into sigmoid)
'''

import numpy as np

from scipy.special import logsumexp as scipy_logsumexp
from scipy.special import expit as sigmoid

def calc_mean_binary_cross_entropy_from_probas(ytrue_N, yproba1_N):
    ''' Compute average cross entropy for given binary classifier's predictions

    Args
    ----
    ytrue_N : 1D array, shape (n_examples,) = (N,)
        All values must be either 0 or 1. Will be cast to int dtype.
        Each entry represents the binary 'true' label of one example
        One entry per example in current dataset
    yproba1_N : 1D array, shape (n_examples,) = (N,)
        All values must be within the interval 0.0 - 1.0, inclusive.
        Will be truncated to (eps, 1 - eps) to keep log values from extremes,
        with small value eps equal to 10^{-14}.
        Each entry is probability that that example should have positive label.
        One entry per example in current dataset.
        Needs to be same size as ytrue_N.

    Returns
    -------
    bce : float
        Binary cross entropy, averaged over all N provided examples


    '''

    # Cast labels to integer just to be sure we're getting what's expected
    ytrue_N = np.asarray(ytrue_N, dtype=np.int32)
    N = int(ytrue_N.size)
    # Cast probas to float and be sure we're between zero and one
    yproba1_N = np.asarray(yproba1_N, dtype=np.float64)           
    yproba1_N = np.maximum(1e-14, np.minimum(1-1e-14, yproba1_N)) 
    BCE = (-ytrue_N * np.log2(yproba1_N) - (1 - ytrue_N) * np.log2(1 - yproba1_N))
    
    if (BCE.size == 0):
        return 0.0
    else:
        return np.mean(BCE)



def calc_mean_binary_cross_entropy_from_scores(ytrue_N, scores_N):
    ''' Compute average cross entropy for given binary classifier's predictions

    Args
    ----
    ytrue_N : 1D array, shape (n_examples,) = (N,)
        All values must be either 0 or 1. Will be cast to int dtype.
        Each entry represents the binary 'true' label of one example
        One entry per example in current dataset
    scores_N : 1D array, shape (n_examples,) = (N,)
        One entry per example in current dataset.
        Each entry is a real value, could be between -infinity and +infinity.
        Large negative values indicate strong probability of label y=0.
        Zero values indicate probability of 0.5.
        Large positive values indicate strong probability of label y=1.
        Needs to be same size as ytrue_N.

    Returns
    -------
    bce : float
        Binary cross entropy, averaged over all N provided examples

    
    '''
    
    ytrue_N = np.asarray(ytrue_N, dtype=np.int32)
    N = int(ytrue_N.size)
    if N == 0:
        return 0.0 # special case: empty array should be 0

    # Convert binary y values so 0 becomes +1 and 1 becomes -1
    
    yflippedsign_N = -1 * np.sign(ytrue_N - 0.001)
    arr = np.vstack((np.zeros(N), yflippedsign_N * scores_N))

    arr = np.apply_along_axis(lambda x : scipy_logsumexp(x) / np.log(2), 0, arr)
    
    return np.mean(arr)
    



