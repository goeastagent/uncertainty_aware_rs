import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, ShuffleSplit

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


import matplotlib.pyplot as plt
import seaborn as sns

import config
import dataloader
import rmanager
import gene_selection
import pilearn


#cmanager = dataloader.CohortManager('data/gene_level_directed_rebuilt_merge_pupdated_apoe.csv')
cmanager = dataloader.CohortManager('data/ts_gene_merged.csv')


classification_cohort = 'MCI'
cohort_adors_learning = 'ADCN'
n_gene=20
feature_names = config.feature_names_ts
adors_target_columns=['FDG','AV45']
filename = 'risk_stratification_adorsC{}_classificationC{}_k{}'.format(cohort_adors_learning, classification_cohort, n_gene)
    
def adors_learning():
    if cohort_adors_learning == 'ADCN':
        gselection_cohort, _ = cmanager.generate_baseline_ADCN_cohort(feature_names)
        X, y = cmanager.generate_baseline_ADCN_cohort(feature_names)
    elif cohort_adors_learning == 'MCI':
        gselection_cohort, _ = cmanager.generate_baseline_MCI_cohort(feature_names)
        X, y = cmanager.generate_baseline_MCI_cohort(feature_names)
        
    genelist = select_gene(gselection_cohort, adors_target_columns, n_gene)
    genelist += ['APOE_e2e4']
    
    def adors_regular_training(x,y):
        clf = LogisticRegression(solver='newton-cg', random_state=0)
        clf.fit(x, y)
        return clf
    def adors_pi_training(x,y):
        clf = pilearn.XGboostrapping()
        clf.fit(x,y)
        return clf

    clfs = {}
    X, y = cmanager.generate_baseline_cohort(X,y)
    x = X[genelist]
    clfs[config.REGULAR_MODEL_KEY] = adors_regular_training(x,y)
    clfs[config.PI_MODEL_KEY] = adors_pi_training(x,y)    
    return clfs, genelist

def evaluate(model, test_X, test_y):
    y_pred = model.predict_proba(test_X.values)[:,1]
    auprc = metrics.average_precision_score(test_y, y_pred)
    auroc = metrics.roc_auc_score(test_y, y_pred)
    return auroc, auprc
    
def select_gene(cohort, target_columns, k):
    gsl = gene_selection.GeneSelectionLoader()
    return gsl.discover_gene(cohort, target_columns, k)

def integration_performance(train_X, train_y, test_X, test_y, genelist):
    # proposed
    proposed_feature_names = feature_names + ['adors']
    proposed_model = XGBClassifier(nthread=-1)
    proposed_model.fit(train_X[proposed_feature_names].values, train_y)
    proposed_acc = evaluate(proposed_model, test_X[proposed_feature_names], test_y)

    # baseline
    baseline_feature_names = feature_names
    baseline_model = XGBClassifier(nthread=-1)
    baseline_model.fit(train_X[baseline_feature_names].values, train_y)
    baseline_acc = evaluate(baseline_model, test_X[baseline_feature_names], test_y)

    # prs
    prs_feature_names = feature_names + ['SCOREp1e5']
    prs_model = XGBClassifier(nthread=-1)
    prs_model.fit(train_X[prs_feature_names].values, train_y)
    prs_acc = evaluate(prs_model, test_X[prs_feature_names], test_y)
    return proposed_model, proposed_acc, baseline_model, baseline_acc, prs_model, prs_acc

def single_run(train_X, train_y, test_X, test_y, adors, genelist):
    proposed_model, proposed_acc, baseline_model, baseline_acc, prs_model, prs_acc = integration_performance(train_X, train_y, test_X, test_y, genelist)
    regular_u, pi_u = uncertainty_aware_risk_stratification(train_X, train_y, test_X, test_y, adors, genelist)

def uncertainty_aware_risk_stratification(train_X, train_y, test_X, test_y, adors, genelist):
    regular_uncertainty = adors[config.REGULAR_MODEL_KEY].predict_proba(train_X[genelist])[:,1]
    pi_uncertainty = adors[config.PI_MODEL_KEY].uncertainty_aware_risk_stratification(train_X, train_y, test_X, test_y, genelist)
    return regular_uncertainty, pi_uncertainty
    
def main(X, y):        
    adors, genelist = adors_learning()
    X['adors'] = adors[config.REGULAR_MODEL_KEY].predict_proba(X[genelist])[:,1]
    n_iter = 10
    aurocs, auprcs = [], []
    for i in range(n_iter):
        print("Iteration: {}".format(i))
        skf = StratifiedKFold(5, shuffle=True, random_state =i)
        for train_index, test_index in skf.split(X,y):
            train_X, train_y = X.iloc[train_index, :], y.iloc[train_index]
            test_X, test_y = X.iloc[test_index, :], y.iloc[test_index]
            single_run(train_X, train_y, test_X, test_y, adors, genelist)           

if __name__ == '__main__':
    if classification_cohort == 'ADCN':
        X, y = cmanager.generate_baseline_ADCN_cohort(feature_names)
    elif classification_cohort == 'MCI':
        X, y = cmanager.generate_baseline_MCI_cohort(feature_names) 
    else:
        exit(0)
        
    X, y = cmanager.generate_baseline_cohort(X,y)
    main(X, y)


    # record:
    # genelist,
    # train IDs, test IDs
    # models
    # every risk index classified by each model in a iteration
