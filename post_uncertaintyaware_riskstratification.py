import pandas as pd
import numpy as np

from scipy import stats
from scipy.stats import beta

import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn import metrics


import shap

import dataloader
import config
import rmanager

def jan24th2023():
    filename = 'risk_stratification_classificationCMCI_labeladors-all-integrated'
    rm = load_rm(filename)
    cmanager = dataloader.CohortManager('data/ts_gene_merged.csv')
    ts_X, ts_y = cmanager.generate_baseline_MCI_cohort(config.feature_names_ts) 

    performance_table(rm)
    

def load_rm(filename):
    rm = rmanager.StratifiedAnalysis.load(filename)
    return rm

def performance_table(rm):
    get_regular_performance(rm, 'all') # biomakers
    get_regular_performance(rm, 'integrated') # adors + biomarkers
    get_integrated_performance(rm,'adors')            # uncertainty-aware risk stratification
    get_integrated_performance(rm, 'prs')             # prs-based uncertainty-aware risk stratification


def a():
    for i in range(len(rm.cutofs)):
        pass

def evaluate(y_true, y_pred):
    auprc = metrics.average_precision_score(y_true, y_pred)
    auroc = metrics.roc_auc_score(y_true, y_pred)
    return auroc, auprc

def get_regular_performance(rm, name):
    aurocs, auprcs = [], []    
    for i in range(len(rm.cutoffs)):
        # y_pred = rm.clfs[i][name+'_clf'].predict_proba(rm.tests[i]['test_X'][feature_names])[:,1]
        auroc, auprc = evaluate(rm.tests[i]['test_y'], rm.tests[i]['test_X'][name+'_pred'])
        aurocs.append(auroc)
        auprcs.append(auprc)
    print("AUROC: {}({}), AUPRC: {}({})".format(np.mean(aurocs), np.std(aurocs), np.mean(auprcs), np.std(auprcs)))
    
def get_integrated_performance(rm, name):
    val_aurocs, val_auprcs = [], []
    test_aurocs, test_auprcs = [], []
    cut=0
    for i in range(len(rm.cutoffs)):
        adors_val = rm.vals[i]['val_X'][rm.vals[i]['val_X'][name + '_cut']<=cut]
        biomarker_val = rm.vals[i]['val_X'][rm.vals[i]['val_X'][name + '_cut']>cut]
        
        val_y_pred = pd.concat([adors_val[name + '_pred'],biomarker_val['all_pred']], axis=0)
        val_y_true = pd.concat([adors_val['y'], biomarker_val['y']], axis=0)
        val_p = evaluate(val_y_true, val_y_pred)
        val_aurocs.append(val_p[0])
        val_auprcs.append(val_p[1])

        adors_test = rm.tests[i]['test_X'][rm.tests[i]['test_X'][name + '_cut']<=cut]
        biomarker_test = rm.tests[i]['test_X'][rm.tests[i]['test_X'][name + '_cut']>cut]

        test_y_pred = pd.concat([adors_test[name + '_pred'],biomarker_test['all_pred']], axis=0)
        test_y_true = pd.concat([adors_test['y'], biomarker_test['y']], axis=0)
        test_p = evaluate(test_y_true, test_y_pred)
        test_aurocs.append(test_p[0])
        test_auprcs.append(test_p[1])
    print("AUROC: {}({}), AUPRC: {}({})".format(np.mean(test_aurocs), np.std(test_aurocs), np.mean(test_auprcs), np.std(test_auprcs)))
