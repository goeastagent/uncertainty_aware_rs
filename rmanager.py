import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import pickle
import config

class ResultManager(object):
    def __init__(self, X, y, feature_names):
        self.X, self.y = X, y
        self.aurocs = []
        self.auprcs = []
        self.models = []
        self.xtrain_indices = []
        self.xtest_indices = []
        self.feature_names = feature_names
        
    def append(self, auroc, auprc, model, xtrain_index, xtest_index):
        self.aurocs.append(auroc)
        self.auprcs.append(auprc)
        self.models.append(model)
        self.xtrain_indices.append(xtrain_index)
        self.xtest_indices.append(xtest_index)

    @staticmethod
    def load(filename):
        with open(config.out_filename_prefix + filename, 'rb') as inp:
            rm = pickle.load(inp)
        return rm

    def save(self, filename):
        with open(config.out_filename_prefix + filename, 'wb') as outp:
            pickle.dump(self, outp)

class StratifiedAnalysis(object):
    def __init__(self, X, y, feature_names, label_names, n_cut):
        self.label_names = label_names
        self.feature_names = feature_names
        self.X = X
        self.y = y
        self.n_cut = n_cut
        
        self.clfs = []
        self.cutoffs = []
        self.vals = []
        self.tests = []

    @staticmethod
    def load(filename):
        with open(config.out_filename_prefix + filename, 'rb') as inp:
            rm = pickle.load(inp)
        return rm

    def save(self, filename):
        with open(config.out_filename_prefix + filename, 'wb') as outp:
            pickle.dump(self, outp)
    def append(self, clf, cutoff, val, test):
        self.clfs.append(clf)
        self.cutoffs.append(cutoff)
        self.vals.append(val)
        self.tests.append(test)

class ResultAnalyzer(object):
    def __init__(self, filename):
        self.rm = ResultManager.load(filename)
        self.BASE_INDEX = 0
        self.ADORS_INDEX = 1
        
    def pick_best_model(self):
        aurocs = np.array(self.rm.aurocs)
        indices = aurocs.argmax(axis=0)
        baseline_model = self.rm.models[indices[self.BASE_INDEX]][self.BASE_INDEX]
        adors_model = self.rm.models[indices[self.ADORS_INDEX]][self.ADORS_INDEX]
        return baseline_model, adors_model

    def pick_average_model(self):
        aurocs = np.array(self.rm.aurocs)
        auroc_mean = aurocs.mean(axis=0)
        residual = (aurocs - auroc_mean)**2
        indices = residual.argmin(axis=0)
        baseline_model = self.rm.models[indices[self.BASE_INDEX]][self.BASE_INDEX]
        adors_model = self.rm.models[indices[self.ADORS_INDEX]][self.ADORS_INDEX]
        return baseline_model, adors_model    

    def average_performance(self):
        aurocs = np.array(self.rm.aurocs)
        auprcs = np.array(self.rm.auprcs)
        
        adors_auroc_mean = aurocs.mean(axis=0)[1]
        adors_auroc_std = aurocs.std(axis=0)[1]
        adors_auprc_mean = auprcs.mean(axis=0)[1]
        adors_auprc_std = auprcs.std(axis=0)[1]

        baseline_auroc_mean = aurocs.mean(axis=0)[0]
        baseline_auroc_std = aurocs.std(axis=0)[0]
        baseline_auprc_mean = auprcs.mean(axis=0)[0]
        baseline_auprc_std = auprcs.std(axis=0)[0]
    
        print("adors: {:.3f}({:.3f}), {:.3f}({:.3f}), baseline: {:.3f}({:.3f}), {:.3f}({:.3f})".format(adors_auroc_mean, adors_auroc_std, adors_auprc_mean, adors_auprc_std, baseline_auroc_mean, baseline_auroc_std, baseline_auprc_mean, baseline_auprc_std))

