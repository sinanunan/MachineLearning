'''
calc_performance_metrics_for_binary_predictions

Provides implementation of common metrics for assessing a binary classifier's
hard decisions against true binary labels, including:
* accuracy
* true positive rate and true negative rate (TPR and TNR)
* positive predictive value and negative predictive value (PPV and NPV)
'''

import numpy as np

def calc_TP_TN_FP_FN(ytrue_N, yhat_N):
    ''' Count the four possible states of true and predicted binary values.
    
    Args
    ----
    ytrue_N : 1D array, shape (n_examples,) = (N,)
        All values must be either 0 or 1. Will be cast to int dtype.
        Each entry represents the binary 'true' label of one example
        One entry per example in current dataset
    yhat_N : 1D array, shape (n_examples,) = (N,)
        All values must be either 0 or 1. Will be cast to int dtype.
        Each entry represents a predicted label for one example
        One entry per example in current dataset.
        Needs to be same size as ytrue_N.

    Returns
    -------
    TP : int
        Number of true positives
    TN : int
        Number of true negatives
    FP : int
        Number of false positives
    FN : int
        Number of false negatives

    '''

    ytrue_N = np.asarray(ytrue_N, dtype=np.int32)
    yhat_N = np.asarray(yhat_N, dtype=np.int32)
    
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    
    for true, hat in zip(ytrue_N, yhat_N):
        if (true):
            if (hat):
                TP += 1
            else:
                FN += 1
        elif (hat):
            FP += 1
        else:
            TN += 1
            
    
    return TP, TN, FP, FN


def calc_ACC(ytrue_N, yhat_N):
    ''' Compute the accuracy of provided predicted binary values.
    
    Args
    ----
    ytrue_N : 1D array of floats, shape (n_examples,) = (N,)
        All values must be either 0.0 or 1.0.
        Each entry represents the binary 'true' label of one example
        One entry per example in current dataset
    yhat_N : 1D array of floats, shape (n_examples,) = (N,)
        All values must be either 0.0 or 1.0.
        Each entry represents a predicted label for one example
        One entry per example in current dataset.
        Needs to be same size as ytrue_N.

    Returns
    -------
    acc : float
        Accuracy = ratio of number correct over total number of examples

    '''

    # compute accuracy
    
    TP, TN, FP, FN = calc_TP_TN_FP_FN(ytrue_N, yhat_N)
    total = TP + TN + FP + FN
    if (total == 0 ):
        return 0.0
    else:
        return ((TP + TN) / total)


def calc_TPR(ytrue_N, yhat_N):
    ''' Compute the true positive rate of provided predicted binary values.

    Also known as the recall.

    Args
    ----
    ytrue_N : 1D array of floats, shape (n_examples,) = (N,)
        All values must be either 0.0 or 1.0.
        Each entry represents the binary 'true' label of one example
        One entry per example in current dataset
    yhat_N : 1D array of floats, shape (n_examples,) = (N,)
        All values must be either 0.0 or 1.0.
        Each entry represents a predicted label for one example
        One entry per example in current dataset.
        Needs to be same size as ytrue_N.

    Returns
    -------
    tpr : float
        TPR = ratio of true positives over total labeled positive

    '''
    
    TP, TN, FP, FN = calc_TP_TN_FP_FN(ytrue_N, yhat_N)
    total = TP + FN
    if (total == 0):
        return 0.0
    else: 
        return TP / total



def calc_TNR(ytrue_N, yhat_N):
    ''' Compute the true negative rate of provided predicted binary values.
    
    Args
    ----
    ytrue_N : 1D array of floats, shape (n_examples,) = (N,)
        All values must be either 0.0 or 1.0.
        Each entry represents the binary 'true' label of one example
        One entry per example in current dataset
    yhat_N : 1D array of floats, shape (n_examples,) = (N,)
        All values must be either 0.0 or 1.0.
        Each entry represents a predicted label for one example
        One entry per example in current dataset.
        Needs to be same size as ytrue_N.

    Returns
    -------
    tnr : float
        TNR = ratio of true negatives over total labeled negative.


    '''
    TP, TN, FP, FN = calc_TP_TN_FP_FN(ytrue_N, yhat_N)
    total = FP + TN
    if (total == 0):
        return 0.0
    else: 
        return TN / total




def calc_PPV(ytrue_N, yhat_N):
    ''' Compute positive predictive value of provided predicted binary values.

    Also known as the precision.
    
    Args
    ----
    ytrue_N : 1D array of floats, shape (n_examples,) = (N,)
        All values must be either 0.0 or 1.0.
        Each entry represents the binary 'true' label of one example
        One entry per example in current dataset
    yhat_N : 1D array of floats, shape (n_examples,) = (N,)
        All values must be either 0.0 or 1.0.
        Each entry represents a predicted label for one example
        One entry per example in current dataset.
        Needs to be same size as ytrue_N.

    Returns
    -------
    ppv : float
        PPV = ratio of true positives over total predicted positive.

    0.000
    '''

    TP, TN, FP, FN = calc_TP_TN_FP_FN(ytrue_N, yhat_N)
    total = TP + FP
    if (total == 0):
        return 0.0
    else: 
        return TP / total

def calc_NPV(ytrue_N, yhat_N):
    ''' Compute negative predictive value of provided predicted binary values.
    
    Args
    ----
    ytrue_N : 1D array of floats, shape (n_examples,) = (N,)
        All values must be either 0.0 or 1.0.
        Each entry represents the binary 'true' label of one example
        One entry per example in current dataset
    yhat_N : 1D array of floats, shape (n_examples,) = (N,)
        All values must be either 0.0 or 1.0.
        Each entry represents a predicted label for one example
        One entry per example in current dataset.
        Needs to be same size as ytrue_N.

    Returns
    -------
    npv : float
        NPV = ratio of true negative over total predicted negative.

    '''
    TP, TN, FP, FN = calc_TP_TN_FP_FN(ytrue_N, yhat_N)
    total = TN + FN
    if (total == 0):
        return 0.0
    else: 
        return TN / total


