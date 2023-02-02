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
cut=10
genelist=config.adors_genelist + ['APOE_e2e4']
adors_target_columns=['FDG','AV45']

feature_names = [genelist, config.feature_names_ts, genelist + config.feature_names_ts]
label_names = ['adors','all','integrated']
filename = 'risk_stratification_classificationC{}_label{}'.format(classification_cohort, '-'.join(label_names))


def adors_learning(train_X, train_y, val_X, val_y, test_X, test_y):
    clf = LogisticRegression(solver='newton-cg', random_state=0)
    if classification_cohort == 'ADCN':
        clf.fit(train_X[genelist], train_y)
    else:
        clf.fit(X[genelist],y)

    val_pred = clf.predict_proba(val_X[genelist])[:,1]
    cutoff = find_optimal_cutoff(val_y, val_pred)

    performance_basedon_uncertainty(clf, val_X, val_y, genelist, 'adors',cutoff)
    performance_basedon_uncertainty(clf, test_X, test_y, genelist, 'adors',cutoff)
    return clf, cutoff

def evaluate(y_true, y_pred):
    auprc = metrics.average_precision_score(y_true, y_pred)
    auroc = metrics.roc_auc_score(y_true, y_pred)
    return auroc, auprc
    
def find_optimal_cutoff(target, predicted):
    fpr, tpr, threshold = metrics.roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr-(1-fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort().iloc[0]]
    return roc_t['threshold']

def performance_basedon_uncertainty(clf, X, y, feature_names, labelname, cutoff):
    y_pred = clf.predict_proba(X[feature_names])[:,1]

    X[labelname+'_pred'] = y_pred

    negative_indices = y_pred <= cutoff
    positive_indices = y_pred > cutoff
    negative_risk = y_pred[negative_indices]
    positive_risk = y_pred[positive_indices]
    negative_y = y[negative_indices]
    positive_y = y[positive_indices]
    negative_labels = pd.qcut(negative_risk-cutoff, cut, labels=False)
    positive_labels = pd.qcut(cutoff - positive_risk, cut, labels=False)

    X.loc[negative_indices, labelname+'_cut'] = negative_labels
    X.loc[positive_indices, labelname+'_cut'] = positive_labels
    # for c in range(cut):
    #     nindices = negative_labels <= c
    #     pindices = positive_labels <= c
    #     subrisk = np.concatenate((negative_risk[nindices],positive_risk[pindices]), axis=0)
    #     suby = np.concatenate((negative_y[nindices], positive_y[pindices]), axis=0)
    #     auroc, auprc = evaluate(suby, subrisk)
    #     print(auroc, auprc, len(suby))
    


def learn_and_uestimate(train_X, train_y, test_X, test_y, feature_names, labelname):
    clf = XGBClassifier(nthread=-1)
    clf.fit(train_X[feature_names], train_y)

    val_pred = clf.predict_proba(val_X[feature_names])[:,1]
    cutoff = find_optimal_cutoff(val_y, val_pred)

    performance_basedon_uncertainty(clf, val_X, val_y, feature_names, labelname, cutoff)
    performance_basedon_uncertainty(clf, test_X, test_y, feature_names, labelname, cutoff)
    return clf, cutoff
    
def single_run(train_X, train_y, test_X, test_y):
    adors_clf, adors_cutoff = adors_learning(train_X, train_y, test_X, test_y)
    all_clf, all_cutoff = learn_and_uestimate(train_X, train_y, test_X, test_y, config.feature_names_ts, 'all')
    integrated_clf, integrated_cutoff = learn_and_uestimate(train_X, train_y, test_X, test_y, config.feature_names_ts + genelist, 'integrated')

    test_X['y'] = test_y

    clf = {'adors_clf': adors_clf, 'all_clf': all_clf, 'integrated_clf': integrated_clf}
    cutoff = {'adors_cutoff': adors_cutoff, 'all_cutoff': all_cutoff, 'integrated_cutoff': integrated_cutoff}
    val = {'val_X': val_X[['RID','adors_pred', 'adors_cut','all_pred','all_cut','integrated_pred','integrated_cut','y']], 'val_y': val_y}
    test = {'test_X': test_X[['RID','adors_pred', 'adors_cut','all_pred','all_cut','integrated_pred','integrated_cut','y']], 'test_y': test_y}
    return clf, cutoff, None, test



    
def uncertainty_aware_risk_stratification(train_X, train_y, test_X, test_y, adors, genelist):
    regular_uncertainty = adors[config.REGULAR_MODEL_KEY].predict_proba(train_X[genelist])[:,1]
    pi_uncertainty = adors[config.PI_MODEL_KEY].uncertainty_aware_risk_stratification(train_X, train_y, test_X, test_y, genelist)
    return regular_uncertainty, pi_uncertainty
    
def main(X, y, rm):
    #adors, genelist = adors_learning()
    #X['adors'] = adors[config.REGULAR_MODEL_KEY].predict_proba(X[genelist])[:,1]
    n_iter = 10
    aurocs, auprcs = [], []
    for i in range(n_iter):
        print("Iteration: {}".format(i))
        skf = StratifiedKFold(5, shuffle=True, random_state =i)
        for train_index, test_index in skf.split(X,y):
            train_X, train_y = X.iloc[train_index, :], y.iloc[train_index]
            test_X, test_y = X.iloc[test_index, :], y.iloc[test_index]

            clf, cutoff, val, test = single_run(train_X, train_y, test_X, test_y)
            rm.append(clf, cutoff, val, test)
    rm.save(filename)

if __name__ == '__main__':
    if classification_cohort == 'ADCN':
        X, y = cmanager.generate_baseline_ADCN_cohort(config.feature_names_ts)
    elif classification_cohort == 'MCI':
        X, y = cmanager.generate_baseline_MCI_cohort(config.feature_names_ts) 
    else:
        exit(0)
        
    X, y = cmanager.generate_baseline_cohort(X,y)
    rm = rmanager.StratifiedAnalysis(X, y, feature_names, label_names, cut)
    main(X, y, rm)


    # record:
    # genelist,
    # train IDs, test IDs
    # models
    # every risk index classified by each model in a iteration
