import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, ShuffleSplit

from xgboost import XGBClassifier
from sklearn import metrics

import pickle


import config
import dataloader
import resultmanager
import gene_selection


cmanager = dataloader.CohortManager('data/gene_level_directed_merge_pupdated_apoe_prsnorm.csv')
feature_names_baseline = config.feature_names


def single_run(train_X, train_y, test_X, test_y, genelist):
    # baseline
    baseline_model = XGBClassifier(nthread=-1)
    baseline_model.fit(train_X[feature_names_baseline].values, train_y.values)
    bauroc, bauprc = evaluate(baseline_model, test_X[feature_names_baseline], test_y)
    
    # adors    
    adors_model = XGBClassifier(nthread=-1)
    adors_model.fit(train_X[feature_names_baseline + genelist].values, train_y.values)
    aauroc, aauprc = evaluate(adors_model, test_X[feature_names_baseline + genelist], test_y)
    print("baseline: {}, {}, adors: {}, {}".format(bauroc, bauprc, aauroc, aauprc))
    return (bauroc, aauroc), (bauprc, aauprc), (baseline_model, adors_model)

def evaluate(model, test_X, test_y):
    y_pred = model.predict_proba(test_X.values)[:,1]
    auprc = metrics.average_precision_score(test_y.values, y_pred)
    auroc = metrics.roc_auc_score(test_y.values, y_pred)
    return auroc, auprc
    
def define_cohort(feature_names):
    X, y = cmanager.generate_baseline_MCI_cohort(feature_names)
    return X, y

def select_gene(cohort, target_columns, k):
    gsl = gene_selection.GeneSelectionLoader()
    return gsl.discover_gene(cohort, target_columns, k)


def main(filename, X, y, genelist):
    rm = resultmanager.ResultManager(X, y, feature_names_baseline + genelist)
    n_iter = 100
    for i in range(n_iter):
        skf = StratifiedKFold(5, shuffle=True, random_state =i)
        for train_index, test_index in skf.split(X,y):
            train_X, train_y = X.iloc[train_index, :], y.iloc[train_index]
            test_X, test_y = X.iloc[test_index, :], y.iloc[test_index]
            auroc, auprc, models = single_run(train_X, train_y, test_X, test_y, genelist)
            rm.append(auroc, auprc, models, train_index, test_index)
    rm.save(filename)

if __name__ == '__main__':
    k = 20
    
    filename = 'MCI_FDG-AV45k' + str(k) + '.with.APOE'
    #filename = 'PRS.without.APOE'
    target_columns = ['FDG','AV45']
    gselection_cohort, _ = cmanager.generate_baseline_MCI_cohort(feature_names_baseline)
    #gselection_cohort, _ = cmanager.generate_baseline_ADCN_cohort(feature_names_baseline)
    #gselection_cohort, _ = cmanager.generate_baseline_cohort(feature_names_baseline)

    genelist = select_gene(gselection_cohort, target_columns, k)
    #genelist = ['SCORE']
    genelist += ['APOE_e2e4']
    #genelist = ['SCORE', 'APOE_e2e4']

    X, y = define_cohort(feature_names_baseline + genelist)
    main(filename, X, y, genelist)
