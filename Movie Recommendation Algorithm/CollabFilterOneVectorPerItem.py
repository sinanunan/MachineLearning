'''

Collaberative Filtering System that uses a 1d Vector Bias for each movie
and for each user (factors) to make predictions
'''

import autograd.numpy as ag_np

from AbstractBaseCollabFilterSGD import AbstractBaseCollabFilterSGD
from train_valid_test_loader import load_train_valid_test_datasets

class CollabFilterOneVectorPerItem(AbstractBaseCollabFilterSGD):
    ''' One-vector-per-user, one-vector-per-item recommendation model.

    Assumes each user, each item has learned vector of size `n_factors`.

    Attributes required in param_dict
    ---------------------------------
    mu : 1D array of size (1,)
    b_per_user : 1D array, size n_users
    c_per_item : 1D array, size n_items
    U : 2D array, size n_users x n_factors
    V : 2D array, size n_items x n_factors

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
            Keys are string names of parameters
            Values are *numpy arrays* of parameter values
        '''
        random_state = self.random_state # inherited RandomState object

        self.param_dict = dict(
            mu=ag_np.ones(1),
            b_per_user=ag_np.ones(n_users), 
            c_per_item=ag_np.ones(n_items), 
            U=0.001 * random_state.randn(n_users, self.n_factors), 
            V=0.001 * random_state.randn(n_items, self.n_factors), 
            )


    def predict(self, user_id_N, item_id_N,
                mu=None, b_per_user=None, c_per_item=None, U=None, V=None):
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
        s = ag_np.sum(U[user_id_N] * V[item_id_N], axis=1)
        yhat_N = ag_np.full(N, mu) + b_per_user[user_id_N] + c_per_item[item_id_N] + s
            

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
        diff_sq = ag_np.sum(ag_np.square(y_N - yhat_N))
        U = param_dict['U']
        V = param_dict['V']
        loss_total = diff_sq +  self.alpha * (ag_np.sum(ag_np.square(U)) +  ag_np.sum(ag_np.square(V)))
        return loss_total    


if __name__ == '__main__':

    # Load the dataset
    train_tuple, valid_tuple, test_tuple, n_users, n_items = \
        load_train_valid_test_datasets()
    # Create the model and initialize its parameters
    # to have right scale as the dataset (right num users and items)
    model = CollabFilterOneVectorPerItem(
        n_epochs=10, batch_size=10000, step_size=0.1,
        n_factors=2, alpha=0.0)
    model.init_parameter_dict(n_users, n_items, train_tuple)

    # Fit the model with SGD
    model.fit(train_tuple, valid_tuple)
