import numpy as np

class LeastSquaresLinearRegressor(object):
    ''' A linear regression model with sklearn-like API

    Fit by solving the "least squares" optimization problem.

    Attributes
    ----------
    * self.w_F : 1D numpy array, size n_features (= F)
        vector of weights, one value for each feature
    * self.b : float
        scalar real-valued bias or "intercept"
    '''

    def __init__(self):
        pass

    def fit(self, x_NF, y_N):
        ''' Compute and store weights that solve least-squares problem.

        Args
        ----
        x_NF : 2D numpy array, shape (n_examples, n_features) = (N, F)
            Input measurements ("features") for all examples in train set.
            Each row is a feature vector for one example.
        y_N : 1D numpy array, shape (n_examples,) = (N,)
            Response measurements for all examples in train set.
            Each row is a feature vector for one example.

        Returns
        -------
        N/A

        Notes
        --------------
        self.w_F and self.b will be updated

        '''
        N,F = x_NF.shape

        # print(F)
        xtilde = np.hstack([x_NF, np.ones((N, 1))])
        inv_xtilde = np.linalg.inv(np.dot(xtilde.T, xtilde))
        theta = np.dot(inv_xtilde, np.dot(xtilde.T, y_N))
        # print(theta_G)
        self.b = theta[F]
        self.w_F = theta[:F]

    def predict(self, x_MF):
        ''' Make predictions given input features for M examples

        Args
        ----
        x_NF : 2D numpy array, shape (n_examples, n_features) (M, F)
            Input measurements ("features") for all examples of interest.
            Each row is a feature vector for one example.

        Returns
        -------
        yhat_N : 1D array, size M
            Each value is the predicted scalar for one example
        '''
        return np.sum(x_MF * self.w_F + self.b, axis=1)
