import numpy as np
import pandas as pd
from xgboost import XGBClassifier

import config


class XGboostrapping(object):
    def __init__(self, n_iter=10):
        self.models = []
        self.n_iter = n_iter

    def fit(self, x,y):
        x.reset_index(drop=True)
        self.models = []
        for i in range(self.n_iter):
            model = XGBClassifier(nthread=-1)
            bootstrapping_x = x.sample(frac=0.8)
            bootstrapping_y = y.loc[bootstrapping_x.index]
            model.fit(bootstrapping_x, bootstrapping_y)
            self.models.append(model)
        return self.models

    def predict_proba(self,X):
        predictions = []
        for i in range(self.n_iter):
            predictions.append(self.models[i].predict_proba(X)[:,1])
        return np.mean(predictions)
    
    def predict_interval(self,X):
        predictions = []
        for i in range(self.n_iter):
            predictions.append(self.models[i].predict_proba(X)[:,1])
        predictions = np.array(predictions)
        means, std = np.mean(predictions,axis=0), np.std(predictions, axis=0)
        return means, std

    def uncertainty_aware_risk_stratification(self, train_X, train_y, test_X, test_y, feature_names):
        means, std = self.predict_interval(train_X[feature_names].values)
        return means, std

    # compute uncertainty based on width of std
    def method1(self, X, feature_names):
        means, std = self.predict_interval(X[feature_names].values)
        uncertainty = pd.qcut(std, 10, labels=False)
        return uncertainty 

    # compute uncertainty based on which threshold is in prediction interval
    def method2(self, X, feature_names):
        means, std = self.predict_interval(X[feature_names].values)
        indices = (config.threshold < means + std*1.96) and (means - std < config.threshold)
        certain_group = X.iloc[~indices]['RID']
        uncertain_group = X.iloc[indices]['RID']
        return certain_group, uncertain_group

    # integration of method 1 and 2
    def method3(self, X, feature_names):
        pass
