'''
Squared Exponential Kernal Function Implementation
'''

import numpy as np

def calc_sqexp_kernel(x_QF, x_train_NF=None, length_scale=1.0):
    ''' Evaluate squared-exponential kernel matrix between two datasets.

    Will compute the kernel function for all possible pairs of feature vectors,
    one from the query dataset, one from the reference training dataset.

	Args
	----
    x_QF : 2D numpy array, shape (Q, F) = (n_query_examples, n_features)
		Feature array for *query* dataset
		Each row corresponds to the feature vector on example

    x_train_NF : 2D numpy array, shape (N, F) = (n_train_examples, n_features)
		Feature array for reference *training* dataset
		Each row corresponds to the feature vector on example
		
    Returns
    -------
    k_QN : 2D numpy array, shape (Q, N)
        Entry at index (q,n) corresponds to the kernel function evaluated
        at the feature vectors x_QF[q] and x_train_NF[n]
    '''
    assert x_QF.ndim == 2
    assert x_train_NF.ndim == 2

    Q, F = x_QF.shape
    N, F2 = x_train_NF.shape
    assert F == F2

    k_QN = np.zeros((Q, N))

    for i, z in enumerate(x_QF):
      for j, x in enumerate(x_train_NF):
        diff = x - z
        dot = np.dot(diff.T, diff)
        k_QN[i][j] = np.exp(-1 * dot / (length_scale ** 2))

    # Ensure the kernel matrix positive definite
    # By adding a small positive to the diagonal
    M = np.minimum(Q, N)
    k_QN[:M, :M] += 1e-08 * np.eye(M)
    return k_QN

