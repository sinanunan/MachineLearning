import numpy as np
import pandas as pd
import sklearn.metrics

def calc_confusion_matrix_for_probas_and_threshold(ytrue_N, yproba1_N, thr):
    ''' Compute confusion matrix for particular probabilistic predictions.

    User provides the predicted probabilities of the class labeled 1,
    as well as the specific threshold at which to predict class 1.

    Args
    ----
    ytrue_N : 1D array, shape (n_examples,) = (N,)
        Each entry is binary value (0 or 1) of 'true' label of one example
        One entry per example in current dataset
    yproba1_N : 1D array, shape (n_examples,) = (N,)
        One entry per example in current dataset.
        Each entry must be between 0.0 and 1.0 (inclusive).
        Each entry is a probability that correct label is positive (class 1).
        Needs to be same size as ytrue_N.
    thr : float, between 0.0 and 1.0 (inclusive)
        Scalar threshold for converting probabilities into hard decisions
        Calls an example "positive" if yproba1_N >= thr.

    Returns
    -------
    cm_df : Pandas DataFrame
        Can be printed like print(cm_df) to easily display results

    '''
    ytrue_N = np.asarray(ytrue_N, dtype=np.int32)
    
    # Determine hard predictions given probabilities
    yproba1_N = np.asarray(yproba1_N, dtype=np.float64)
    yhat_N = np.asarray(yproba1_N >= thr, dtype=np.int32)

    cm = sklearn.metrics.confusion_matrix(ytrue_N, yhat_N)
    cm_df = pd.DataFrame(data=cm, columns=[0, 1], index=[0, 1])
    cm_df.columns.name = 'Predicted'
    cm_df.index.name = 'True'
    return cm_df
