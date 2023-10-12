import numpy as np
from scipy.optimize import minimize_scalar
from pyearth import Earth
from time import time
from IPython.display import SVG, display
def show_svg(name):
    display(SVG(name))
from copy import deepcopy


from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
from joblib import Parallel, delayed
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pyearth import Earth


class RandomMarsMSE:
    def __init__(
        self, n_estimators, max_terms=None, max_degree=1, random_twist=False, feature_subsample_size=None,
        **mars_parameters
    ):
        """
        n_estimators : int
            The number of models in the forest.

        max_terms : int
            The maximum number of base functions 

        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self._n_estimators = n_estimators
        self._max_terms = max_terms
        self._max_degree = max_degree
        self._random_twist = random_twist

        if feature_subsample_size is None:
            self._feature_subsample_size = 1 / 3
        else:
            self._feature_subsample_size = feature_subsample_size
        
        self._estimators = [Earth(max_terms=max_terms, max_degree=self._max_degree, **mars_parameters)
                            for _ in range(n_estimators)]

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects

        X_val : numpy ndarray
            Array of size n_val_objects, n_features

        y_val : numpy ndarray
            Array of size n_val_objects
        """
        def calc_rmse(history, rmse, y):
            history[rmse] = (
                np.cumsum(history[rmse], axis=0)
                / np.arange(1, self._n_estimators + 1)[..., np.newaxis])
            history[rmse] -= y[np.newaxis, ...]
            history[rmse] = (np.mean(history[rmse]**2, axis=1))**(1/2)

        rng = np.random.default_rng()
        n_objects, n_features = X.shape
        self._twists = [1]*self._n_estimators

        history = {'RMSE_train': np.zeros((self._n_estimators, n_objects)),
                      'RMSE_val': [], 'time': np.zeros(self._n_estimators)}
        is_val = X_val is not None and y_val is not None
        if is_val:
            history['RMSE_val'] = np.zeros((self._n_estimators, X_val.shape[0]))

        n_selected_features = int(n_features * self._feature_subsample_size)
        used_features = np.zeros((self._n_estimators, n_selected_features), dtype=int)
        
        for i in range(len(self._estimators)):
            t = time()

            obj_inds = rng.choice(n_objects, size=n_objects, replace=True)
            used_features[i] = rng.choice(n_features, size=n_selected_features, replace=False)

            if self._random_twist:
                twist = rng.normal(size=(n_selected_features, n_selected_features))
                self._estimators[i].fit(X[obj_inds][:, used_features[i]] @ twist, y[obj_inds])
                self._twists[i] = twist
            else:
                self._estimators[i].fit(X[obj_inds][:, used_features[i]], y[obj_inds])

            history['RMSE_train'][i] = (self._estimators[i].predict(X[:, used_features[i]] @ twist) 
                                        if self._random_twist else self._estimators[i].predict(X[:, used_features[i]]))
            history['time'][i] = time() - t
            if is_val:
                history['RMSE_val'][i] = (self._estimators[i].predict(X_val[:, used_features[i]] @ twist) 
                                          if self._random_twist else self._estimators[i].predict(X_val[:, used_features[i]]))
            
        self._used_features = used_features
        calc_rmse(history, 'RMSE_train', y)
        if is_val:
            calc_rmse(history, 'RMSE_val', y_val)
        self.history = history

        return self

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        ans = None
        if self._random_twist:
            ans = np.array([estimator.predict(X[:, self._used_features[i]] @ self._twists[i])
                         for i, estimator in enumerate(self._estimators)]).mean(axis=0)
        else:
            ans = np.array([estimator.predict(X[:, self._used_features[i]])
                         for i, estimator in enumerate(self._estimators)]).mean(axis=0)    
        return ans

class GradientBoostingMSE:
    def __init__(
        self, n_estimators, learning_rate=0.1, max_terms=None, max_degree=1,
        random_twist=False, feature_subsample_size=None, **mars_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.

        learning_rate : float
            Use alpha * learning_rate instead of alpha

        max_terms : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self._n_estimators = n_estimators
        self._max_terms = max_terms
        self._max_degree = max_degree
        self._random_twist = random_twist
        self._lr = learning_rate

        if feature_subsample_size is None:
            self._feature_ratio = 1 / 3
        else:
            self._feature_ratio = feature_subsample_size

        self._estimators = [0] + [Earth(max_terms=max_terms,
                                        max_degree=self._max_degree, **mars_parameters)
                            for _ in range(n_estimators)]
        self._coefs = np.zeros(n_estimators + 1)

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects
        """
        def MSE(alpha, s, new_pred):
            return ((s - alpha * new_pred)**2).mean()

        def calc_rmse(history, rmse, y):
            history[rmse] = np.cumsum(history[rmse], axis=0)
            history[rmse] -= y[np.newaxis, ...]
            history[rmse] = (np.mean(history[rmse]**2, axis=1))**(1/2)


        rng = np.random.default_rng()
        n_objects, n_features = X.shape
        n_selected_features = int(n_features * self._feature_ratio)
        used_features = np.zeros((self._n_estimators + 1, n_selected_features), dtype=int)
        self._twists = [1]*(self._n_estimators + 1)

        history = {'RMSE_train': np.zeros((self._n_estimators + 1, n_objects)),
                      'RMSE_val': [], 'time': np.zeros(self._n_estimators + 1)}
        is_val = X_val is not None and y_val is not None
        if is_val:
            history['RMSE_val'] = np.zeros((self._n_estimators + 1, X_val.shape[0]))


        self._estimators[0] = y.mean()
        s = y - self._estimators[0]
        history['RMSE_train'][0, :] = self._estimators[0]
        if is_val:
            history['RMSE_val'][0, :] = self._estimators[0]

        for i in range(1, len(self._estimators)):
            t = time()

            estimator = self._estimators[i]
            used_features[i] = rng.choice(n_features, size=n_selected_features, replace=False)

            if self._random_twist:
                twist = rng.normal(size=(n_selected_features, n_selected_features))
                estimator.fit(X[:, used_features[i]] @ twist, s)
                self._twists[i] = twist
                new_pred = estimator.predict(X[:, used_features[i]] @ twist)
            else:
                estimator.fit(X[:, used_features[i]], s)
                new_pred = estimator.predict(X[:, used_features[i]])

            alpha = minimize_scalar(MSE, args=(s, new_pred)).x
            self._coefs[i] = self._lr * alpha
            s -= self._lr * alpha * new_pred

            history['RMSE_train'][i] = self._lr * alpha * new_pred
            history['time'][i] = time() - t
            if is_val:
                history['RMSE_val'][i] = self._lr * alpha * (self._estimators[i].predict(X_val[:, used_features[i]] @ twist) 
                                          if self._random_twist else self._estimators[i].predict(X_val[:, used_features[i]]))


        self._used_features = used_features

        calc_rmse(history, 'RMSE_train', y)
        if is_val:
            calc_rmse(history, 'RMSE_val', y_val)
        self.history = history
        return self


    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        result = np.zeros(X.shape[0])
        for i in range(1, len(self._estimators)):
            coef, estimator = self._coefs[i], self._estimators[i]
            if self._random_twist:
                preds = estimator.predict(X[:, self._used_features[i]] @ self._twists[i])
            else:
                preds = estimator.predict(X[:, self._used_features[i]])

            result += coef * preds
        return result + self._estimators[0]



def mse(y1, y2):
    return np.mean((y1-y2)**2)
def rmse(y1, y2):
    return np.mean((y1-y2)**2)**0.5

def split_params(params):
    keys, values = zip(*params.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return permutations_dicts


def grid_search(cls, params, X_train, y_train, X_val, y_val):
    history = {'score': np.zeros(len(params))}
    for i , par in enumerate(params):
        model = cls(par).fit(X_train, y_train)
        history['score'][i] = mse(model.predict(X_val), y_val)
    return history


def preprocess(data, target_name, drop=True, index_col=0):
    if isinstance(data, (str, )):
        if index_col is not None:
            df = pd.read_csv(data, index_col=0).reset_index()
        else:
            df = pd.read_csv(data)
        #df = pd.read_csv(data, index_col=0).reset_index()
        if drop:
            df = df.drop(columns='index')
        X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=target_name), df[target_name], train_size=0.8)

        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, train_size=0.5)

        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()
        y_val = y_val.to_numpy()

        scaler = StandardScaler().fit(X_train)

        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        X_val = scaler.transform(X_val)
    else:
        X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, train_size=0.8)

        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, train_size=0.5)

        #y_train = y_train.to_numpy()
        #y_test = y_test.to_numpy()
        #y_val = y_val.to_numpy()

        scaler = StandardScaler().fit(X_train)

        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        X_val = scaler.transform(X_val)

    return X_train, y_train, X_val, y_val, X_test, y_test


def get_data(data, start, end, num, n_estimators, max_terms, random_twist, enable_pruning=False, max_degree=1):
    subsamples = np.linspace(start, end, num)
    samples_hist_twist = []

    X_train, y_train, X_val, y_val, X_test, y_test = data

    for i, rate in enumerate(subsamples):
        bagging = RandomMarsMSE(n_estimators, max_terms=max_terms, random_twist=random_twist,
                                feature_subsample_size=rate, enable_pruning=enable_pruning,
                                max_degree=max_degree)
        bagging.fit(X_train, y_train, X_val, y_val)
        print(rmse(bagging.predict(X_train), y_train), rmse(bagging.predict(X_val), y_val), rmse(bagging.predict(X_test), y_test))
        samples_hist_twist.append(deepcopy(bagging.history))
    return samples_hist_twist





    
