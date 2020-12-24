import numpy as np

from tree_utils import LeafNode, InternalDecisionNode

def select_best_binary_split(x_NF, y_N, MIN_SAMPLES_LEAF=1):
    ''' Determine best single feature binary split for provided dataset

    Args
    ----
    x_NF : 2D array, shape (N,F) = (n_examples, n_features)
        Training data features at current node we wish to find a split for.
    y_N : 1D array, shape (N,) = (n_examples,)
        Training labels at current node.
    min_samples_leaf : int
        Minimum number of samples allowed at any leaf.

    Returns
    -------
    feat_id : int or None, one of {0, 1, 2, .... F-1}
        Indicates which feature in provided x array is used for best split.
        If None, a binary split that improves the cost is not possible.
    thresh_val : float or None
        Value of x[feat_id] at which we threshold.
        If None, a binary split that improves the cost is not possible.
    x_LF : 2D array, shape (L, F)
        Training data features assigned to left child using best split.
    y_L : 1D array, shape (L,)
        Training labels assigned to left child using best split.
    x_RF : 2D array, shape (R, F)
        Training data features assigned to right child using best split.
    y_R : 1D array, shape (R,)
        Training labels assigned to right child using best split.

    
    '''
    N, F = x_NF.shape

    # Allocate space to store the cost and threshold of each feat's best split
    cost_F = np.inf * np.ones(F)
    thresh_val_F = np.zeros(F)
    for f in range(F):

        # Compute all possible x threshold values for current feature
        # possib_xthresh_V : 1D array of size V
        #    Each entry is a float
        #    Entries are in sorted order from smallest to largest
        #    Represents one possible distinct threshold for provided N examples
        xunique_U = np.unique(x_NF[:,f])
        possib_xthresh_V = 0.5 * (
            xunique_U[MIN_SAMPLES_LEAF:] + xunique_U[:-MIN_SAMPLES_LEAF])

        V = possib_xthresh_V.size
        if V == 0:
            # If all the x values for this feature are same, we can't split.
            # Keep cost as "infinite" and continue to next feature
            cost_F[f] = np.inf
            continue
        
        
        left_yhat_V = np.zeros(V) 
        right_yhat_V = np.ones(V) 
        left_cost_V = np.zeros(V) 
        right_cost_V = np.ones(V) 
        for index, thr in enumerate(possib_xthresh_V):
            left_mask = x_NF[:,f] < thr
            right_mask = np.logical_not(left_mask)

            left_yhat_V[index] = np.mean(y_N[left_mask])
            right_yhat_V[index] = np.mean(y_N[right_mask])

            left_cost_V[index]  = np.sum(np.square(y_N[left_mask] - left_yhat_V[index]))
            right_cost_V[index] = np.sum(np.square(y_N[right_mask] - right_yhat_V[index]))

        total_cost_V = left_cost_V + right_cost_V

        # Check if there is any split that improves our cost or predictions.
        # If not, all splits will have same cost and we should just not split.
        costs_all_the_same = np.allclose(total_cost_V, total_cost_V[0])
        yhat_all_the_same = np.allclose(left_yhat_V, right_yhat_V)
        if costs_all_the_same and yhat_all_the_same:
            # Keep cost as "infinite" and continue to next feature
            cost_F[f] = np.inf
            continue
        
        
        chosen_v_id = np.argmin(total_cost_V)
        cost_F[f] = total_cost_V[chosen_v_id]
        thresh_val_F[f] = possib_xthresh_V[chosen_v_id]

        
    # Determine single best feature to use
    best_feat_id = np.argmin(cost_F)
    best_thresh_val = thresh_val_F[best_feat_id]
    
    if not np.isfinite(cost_F[best_feat_id]):
        # Edge case: not possible to split further
        # Either all x values are the same, or all y values are the same
        return (None, None, None, None, None, None)

    left_mask = x_NF[:,best_feat_id] < best_thresh_val
    right_mask = np.logical_not(left_mask)

    x_LF, y_L = x_NF[left_mask], y_N[left_mask]
    x_RF, y_R = x_NF[right_mask], y_N[right_mask]
    
    left_cost = np.sum(np.square(y_L - np.mean(y_L)))
    right_cost = np.sum(np.square(y_R - np.mean(y_R)))
    assert np.allclose(cost_F[best_feat_id], left_cost + right_cost)

    return (int(best_feat_id), best_thresh_val, x_LF, y_L, x_RF, y_R)