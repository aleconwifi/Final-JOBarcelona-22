import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import neighbors
from sklearn import tree
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
import pickle

from utils import Utils


class ModelHandler(Utils):

    models = {'XGB': XGBRegressor, 'XGBtest': XGBRegressor}
    hyperparams = {'XGB': {
        'n_estimators' : [1000],
        'learning_rate': [1e-1, 1e-3],
        'subsample': [0.5, 1.0],
        'colsample_bytree': [0.5, 1.0],
        'max_depth': [6, 10]
    }, 'XGBtest': {
        'n_estimators' : [300],
        'learning_rate' : [0.1],
        'max_depth': [1, 2, 4]
    }}

    def __init__(self, X, Y, model: str, scale = False, **kwargs):

        super().__init__()

        self.n_splits = kwargs['n_splits'] if 'n_splits' in kwargs else 5
        if 'n_splits' in kwargs:
            del kwargs['n_splits']
        self.hyperparam = self.hyperparams[model]
        self.hyperparam.update(kwargs)
        self.scale = scale
        self.model = self.models[model](
            **{x: kwargs[x][0] if type(kwargs[x]) == list else kwargs[x] for x in kwargs})
        if self.scale:
            self.Scaler, self.X = self.do_scaling(self._ensure_dimensionalit(X))
        self.Y = Y
        assert(len(Y.shape) == 1)
        self.grid = self._gen_gridSearch(
            self.model, self.hyperparam, self.n_splits)
        self.grid_flag = False

    def fit(self, with_score=True, with_grid=True):
        if with_grid:
            self.grid_flag = True
            self.grid.fit(self.X, self.Y)
            print(f"[INFO] The best parameters are {self.grid.best_params_}")
            print(f"[INFO] The best score is {self.grid.best_score_:.4f}")
 
        else:
            if self.scale:
                self.model = self.model.fit(self.Scaler(self.X), self.Y)
            else:
                self.model = self.model.fit(self.X, self.Y)

        if with_score:
            pred = self.predict(self.X)
            print(f"[INFO] Train acc  is : {self.mse(pred, self.Y):.4f}")

    def predict(self, X):
        X = self._ensure_dimensionalit(X)
        if self.scale:
            X = self.Scaler.transform(X)
        return self.model.predict(X)

    def available_models(self):
        return self.models.keys()
    
    def save(self, name = False):
        if not name:
            name = str(self.model.__class__).split('.')[-1][:-2]
        pickle.dump(self.model, open(name + '.pickle','wb'))
    
    def load(self,path):
        self.model = pickle.load(open(path,'rb'))

        
