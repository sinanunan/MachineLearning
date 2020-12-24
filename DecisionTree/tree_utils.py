"""
tree_utils.py

Defines two Python classes, one for each kind of nodes:
- InternalDecisionNode
- LeafNode
"""

import numpy as np

class InternalDecisionNode(object):

    '''
    Defines a single node used to make yes/no decisions within a binary tree.

    Attributes
    ----------
    x_NF : 2D array, shape (N,F)
        Feature vectors of the N training examples that have reached this node.
    y_N : 1D array, shape (N,)
        Labels of the N training examples that have reached this node.
    feat_id : int
        Which feature this node will split on.
    thresh_val : float
        The value of the threshold used to divide input examples to either the
        left or the right child of this node.
    left_child : instance of InternalDecisionNode or LeafNode class
        Use to make predictions for examples less than this node's threshold.
    right_child : instance of InternalDecisionNode or LeafNode class
        Use to make predictions for examples greater than this node's threshold.
    '''

    def __init__(self, x_NF, y_N, feat_id, thresh_val, left_child, right_child):
        self.x_NF = x_NF
        self.y_N = y_N
        self.feat_id = feat_id
        self.thresh_val = thresh_val
        self.left_child = left_child
        self.right_child = right_child


    def predict(self, x_TF):
        ''' Make prediction given provided feature array

        For an internal node, we assign each input example to either our
        left or right child to get its prediction.
        We then aggregate the results into one array to return.

        Args
        ----
        x_TF : 2D numpy array, shape (T, F)

        Returns
        -------
        yhat_T : 1D numpy array, shape (T,)
        '''
        T, F = x_TF.shape
        left_mask_N = x_TF[:, self.feat_id] < self.thresh_val
        right_mask_N = np.logical_not(left_mask_N)
        left_leaf = self.left_child.predict(x_TF[left_mask_N])
        right_leaf = self.right_child.predict(x_TF[right_mask_N])

        yhat_T = 1.2345 * np.ones(T, dtype=np.float64) 
        yhat_T[left_mask_N] = left_leaf
        yhat_T[right_mask_N] = right_leaf
        return yhat_T


    def __str__(self):
        ''' Pretty print a string representation of this node
        
        Returns
        -------
        s : string
        '''
        left_str = self.left_child.__str__()
        right_str = self.right_child.__str__()
        lines = [
            "Decision: X[%d] < %.3f?" % (self.feat_id, self.thresh_val),
            "  Y: " + left_str.replace("\n", "\n    "),
            "  N: " + right_str.replace("\n", "\n    "),
            ]
        return '\n'.join(lines)


class LeafNode(object):
    
    '''
    Defines a single node within a binary tree that makes constant predictions.

    We assume the objective function is to minimize squared error on the train
    set. This means the optimal thing to do is to predict the mean of all the
    train examples that reach the region defined by this leaf.

    Attributes
    ----------
    x_NF : 2D array, shape (N,F)
        Feature vectors of the N training examples that have reached this node.
    y_N : 1D array, shape (N,)
        Labels of the N training examples that have reached this node.
        This may be a subset of all the training examples.
    '''

    def __init__(self, x_NF, y_N):
        self.x_NF = x_NF
        self.y_N = y_N


    def predict(self, x_TF):
        ''' Make prediction given provided feature array
        
        For a leaf node, all input examples get the same predicted value,
        which is determined by the mean of the training set y values
        that reach this node.

        Args
        ----
        x_TF : 2D numpy array, shape (T, F)

        Returns
        -------
        yhat_T : 1D numpy array, shape (T,)
            Predicted y value for each provided example
        '''
        T = x_TF.shape[0]
        yhat_T = np.full((T,), np.mean(self.y_N))
        return yhat_T


    def __str__(self):
        ''' Pretty print a string representation of this node
        
        Returns
        -------
        s : string
        '''        
        return "Leaf: predict y = %.3f" % np.mean(self.y_N)
