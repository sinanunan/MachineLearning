'''
Collaberative Filtering System that uses only the general mean to calculate
predictions 

Notes
------
Not meant to be accurate
'''

import autograd.numpy as ag_np

# Use helper packages
from AbstractBaseCollabFilterSGD import AbstractBaseCollabFilterSGD
from train_valid_test_loader import load_train_valid_test_datasets


class CollabFilterMeanOnly(AbstractBaseCollabFilterSGD):
    ''' Simple baseline recommendation model.

    Always predicts same scalar no matter what user/movie.

    Attributes required in param_dict
    ---------------------------------
    mu : 1D array of size (1,)

    Notes
    -----
    Inherits *__init__** constructor from AbstractBaseCollabFilterSGD.
    Inherits *fit* method from AbstractBaseCollabFilterSGD.
    '''

    def init_parameter_dict(self, n_users, n_items, train_tuple):
        ''' Initialize parameter dictionary attribute for this instance.

        Post Condition
        --------------
        Updates the following attributes of this instance:
        * param_dict : dict
            Keys are string names of parameters (e.g. 'mu')
            Values are *numpy arrays* of parameter values
        '''

        self.param_dict = dict(mu=ag_np.zeros(1))

    def predict(self, user_id_N, item_id_N, mu=None):
        ''' Predict ratings at specific user_id, item_id pairs

        Args
        ----
        user_id_N : 1D array, size n_examples
            Specific user_id values to use to make predictions
        item_id_N : 1D array, size n_examples
            Specific item_id values to use to make predictions
            Each entry is paired with the corresponding entry of user_id_N

        Returns
        -------
        yhat_N : 1D array, size n_examples
            Scalar predicted ratings, one per provided example.
            Entry n is for the n-th pair of user_id, item_id values provided.
        '''

        N = user_id_N.size
        yhat_N = ag_np.full(N, mu)
        return yhat_N

    def calc_loss_wrt_parameter_dict(self, param_dict, data_tuple):
        ''' Compute loss at given parameters

        Args
        ----
        param_dict : dict
            Keys are string names of parameters
            Values are *numpy arrays* of parameter values

        Returns
        -------
        loss : float scalar
        '''
        y_N = data_tuple[2]
        yhat_N = self.predict(data_tuple[0], data_tuple[1], **param_dict)
        diff_sq = (y_N - yhat_N) ** 2
        loss_total = ag_np.sum(diff_sq)
        return loss_total

    

if __name__ == '__main__':

    # Load the dataset
    train_tuple, valid_tuple, test_tuple, n_users, n_items = \
        load_train_valid_test_datasets()
    # Create the model and initialize its parameters
    # to have right scale as the dataset (right num users and items)
    model = CollabFilterMeanOnly(
        n_epochs=10, batch_size=10000, step_size=0.1)
    model.init_parameter_dict(n_users, n_items, train_tuple)

    # Fit the model with SGD
    model.fit(train_tuple, valid_tuple)

