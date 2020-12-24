import numpy as np

from performance_metrics import calc_mean_squared_error

''' Cross Validation fold creater and model trainer '''

def train_models_and_calc_scores_for_n_fold_cv(
        estimator, x_NF, y_N, n_folds=3, random_state=0):
    ''' Perform n-fold cross validation for a specific sklearn estimator object

    Args
    ----
    estimator : any regressor object with sklearn-like API
        Supports 'fit' and 'predict' methods.
    x_NF : 2D numpy array, shape (n_examples, n_features) = (N, F)
        Input measurements ("features") for all examples of interest.
        Each row is a feature vector for one example.
    y_N : 1D numpy array, shape (n_examples,)
        Output measurements ("responses") for all examples of interest.
        Each row is a scalar response for one example.
    n_folds : int
        Number of folds to divide provided dataset into.
    random_state : int or numpy.RandomState instance
        Allows reproducible random splits.

    Returns
    -------
    train_error_per_fold : 1D numpy array, size n_folds
        One entry per fold
        Entry f gives the error computed for train set for fold f
    test_error_per_fold : 1D numpy array, size n_folds
        One entry per fold
        Entry f gives the error computed for test set for fold f

    '''
    train_error_per_fold = np.zeros(n_folds, dtype=np.float64)
    test_error_per_fold = np.zeros(n_folds, dtype=np.float64)

    # defines the folds here by calling your function
    N,F = x_NF.shape
    train_ids_per_fold, test_ids_per_fold = make_train_and_test_row_ids_for_n_fold_cv(N, n_folds, random_state)
    

    index = 0
    for train, test in zip(train_ids_per_fold, test_ids_per_fold):
        estimator.fit(x_NF[train], y_N[train])
        train_yhat = estimator.predict(x_NF[train])
        test_yhat = estimator.predict(x_NF[test])
        train_error_per_fold[index] = calc_mean_squared_error(train_yhat, y_N[train])
        test_error_per_fold[index] = calc_mean_squared_error(test_yhat, y_N[test])
        index += 1


    return train_error_per_fold, test_error_per_fold

def make_train_and_test_row_ids_for_n_fold_cv(
        n_examples=0, n_folds=3, random_state=0):
    ''' Divide row ids into train and test sets for n-fold cross validation.

    Will *shuffle* the row ids via a pseudorandom number generator before
    dividing into folds.

    Args
    ----
    n_examples : int
        Total number of examples to allocate into train/test sets
    n_folds : int
        Number of folds requested
    random_state : int or numpy RandomState object
        Pseudorandom number generator (or seed) for reproducibility

    Returns
    -------
    train_ids_per_fold : list of 1D np.arrays
        One entry per fold
        Each entry is a 1-dim numpy array of unique integers between 0 to N
    test_ids_per_fold : list of 1D np.arrays
        One entry per fold
        Each entry is a 1-dim numpy array of unique integers between 0 to N

    '''
    if not hasattr(random_state, 'rand'):
        # Handle case where we pass "seed" for a PRNG as an integer
        random_state = np.random.RandomState(int(random_state))

    integer_list = (np.arange(n_examples))
    random_state.shuffle(integer_list)
    
    test_ids_per_fold = np.array_split(integer_list,n_folds)
        
    train_ids_per_fold = list(map(lambda cell:np.setdiff1d(integer_list,cell), 
                              test_ids_per_fold))

    return train_ids_per_fold, test_ids_per_fold
    
