import numpy as np

from train_tree import train_tree_greedy

class MyDecisionTreeRegressor(object):

    '''
    Prediction model object implementing an sklearn-like interface.

    Allows fitting and predicting from a decision tree.

    Attributes
    ----------
    max_depth
    min_samples_leaf
    min_samples_split

    Attributes available after calling fit
    --------------------------------------
    root_node : Instance of InternalDecisionNode or LeafNode from tree_utils
        Supports predict method.

    '''

    def __init__(self, max_depth=None, min_samples_leaf=1, min_samples_split=1):
        ''' Constructor for our prediction object

        Returns
        -------
        new MyDecisionTreeRegressor object
        '''
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def fit(self, x_NF, y_N):
        ''' Make prediction for each example in provided features array

        Args
        ----
        x_NF : 2D numpy array, shape (n_samples, n_features) = (N, F)
            Features for each training sample.
        y_N : 1D numpy array, shape (n_samples,) = (N,)
            Target outcome values for each training sample.

        Returns
        -------
        None.

        Notes
        --------------
        Attribute 'root_node' is set to a valid value.
        '''
        self.root_node = train_tree_greedy(
            x_NF, y_N, depth=0,
            MAX_DEPTH=self.max_depth,
            MIN_SAMPLES_INTERNAL=self.min_samples_split,
            MIN_SAMPLES_LEAF=self.min_samples_leaf)

    def predict(self, x_TF):
        ''' Make prediction for each example in provided features array.

        Args
        ----
        x_TF : 2D numpy array, shape (T, F)

        Returns
        -------
        yhat_T : 1D numpy array, shape (T,)
        '''
        return self.root_node.predict(x_TF)

    def to_string(self):
        return str(self.root_node)

    def __str__(self):
        return self.to_string()