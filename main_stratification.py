import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, ShuffleSplit

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

import pickle

import matplotlib.pyplot as plt
import seaborn as sns


import config
import dataloader
import resultmanager
import gene_selection


#cmanager = dataloader.CohortManager('data/gene_level_directed_merge_pupdated_apoe_prsnorm.csv')
cmanager = dataloader.CohortManager('data/gene_level_directed_rebuilt_merge_pupdated_apoe.csv')

feature_names_baseline = config.feature_names
    
def adors_learning(train_X, train_y, genelist):
    clf = LogisticRegression(solver='newton-cg', random_state=0)
    clf.fit(train_X[genelist].values, train_y)
    return clf

def run(X, y, feature_names):
    n_iter = 10

    aurocs, auprcs = [], []
    for i in range(n_iter):
        skf = StratifiedKFold(5, shuffle=True, random_state =i)
        for train_index, test_index in skf.split(X,y):
            train_X, train_y = X.iloc[train_index, :], y[train_index]
            test_X, test_y = X.iloc[test_index, :], y[test_index]
            clf = XGBClassifier(nthread=-1)
            clf.fit(train_X[feature_names].values, train_y)
            y_pred = clf.predict_proba(test_X[feature_names].values)[:,1]
            auroc, auprc = evaluate(test_y, y_pred)
            aurocs.append(auroc)
            auprcs.append(auprc)
    print(np.mean(aurocs), np.mean(auprcs))

# def evaluate(model, test_X, test_y):
#     y_pred = model.predict_proba(test_X.values)[:,1]
#     auprc = metrics.average_precision_score(test_y, y_pred)
#     auroc = metrics.roc_auc_score(test_y, y_pred)
#     return auroc, auprc

def evaluate(y, y_pred):
    auprc = metrics.average_precision_score(y, y_pred)
    auroc = metrics.roc_auc_score(y, y_pred)
    return auroc, auprc        


def define_cohort(feature_names):
    X, y = cmanager.generate_baseline_MCI_cohort(feature_names)
    #X, y = cmanager.generate_baseline_ADCN_cohort(feature_names)
    #X, y = cmanager.generate_baseline_cohort(feature_names)
    return X, y

def select_gene(cohort, target_columns, k):
    gsl = gene_selection.GeneSelectionLoader()
    return gsl.discover_gene(cohort, target_columns, k)

def risk_uncertainty_performance():
    ratio = 0.311
    negative_indices = X['adors'] <= ratio
    positive_indices = X['adors'] > ratio
    negative_risk = X['adors'][negative_indices]
    positive_risk = X['adors'][positive_indices]
    negative_y = y.values[negative_indices]
    positive_y = y.values[positive_indices]
    cut = 10
    negative_labels = pd.qcut(negative_risk-ratio, cut, labels=False)
    positive_labels = pd.qcut(ratio  - positive_risk, cut, labels=False)

    uncertainty, samples, aurocs, auprcs = [], [], [], []

    for c in range(cut):
        nindices = negative_labels <= c
        pindices = positive_labels <= c
        subrisk = np.concatenate((negative_risk[nindices],positive_risk[pindices]), axis=0)
        suby = np.concatenate((negative_y[nindices], positive_y[pindices]), axis=0)
        auroc, auprc = evaluate(suby, subrisk)
        print(auroc, auprc, len(suby))
        uncertainty.append(c/cut)
        samples.append(len(suby))
        aurocs.append(auroc)
        auprcs.append(auprc)

    plt.plot(uncertainty, aurocs, color='#CC503E', label='proposed')
    plt.xlabel("Uncertainty")
    plt.ylabel("AUROC")
    plt.legend()
    plt.show()
    
    plt.plot(uncertainty, auprcs, color='#CC503E', label='proposed')
    plt.ylabel("AUPRC")
    plt.xlabel("Uncertainty")
    plt.legend()
    plt.show()

    plt.plot(uncertainty, samples,color='#CC503E', label='proposed')
    plt.ylabel("#samples")
    plt.xlabel("Uncertainty")
    plt.legend()
    plt.show()


        

def main(filename, X, y, genelist):
    clf = adors_learning(X, y, genelist)
    risk_index = clf.predict_proba(X[genelist].values)[:,1]
    X['adors'] = risk_index

    colname = 'adors'
    #y = y.values

    # sample size across cutoff
    for cutoff in np.arange(0.1,0.4,0.01):
        certain_group, certain_group_y = X[(X[colname] < cutoff) | (X[colname] > 1-cutoff)].reset_index(drop=True), y[(X[colname] < cutoff) | (X[colname] > 1-cutoff)].values
        uncertain_group, uncertain_group_y = X[(cutoff <= X[colname]) & (X[colname] <= 1-cutoff)].reset_index(drop=True), y[(cutoff <= X[colname]) & (X[colname] <= 1-cutoff)].values
        print('cutoff: {:.2f}, #certain_group: {}(+{}), #uncertain_group: {}(+{})'.format(cutoff, len(certain_group), sum(certain_group_y==1), len(uncertain_group), sum(uncertain_group_y==1)))

        if (sum(uncertain_group_y==1) < 5) or (sum(certain_group_y == 1) < 5):
            continue

        run(certain_group, certain_group_y, feature_names_baseline)
        run(uncertain_group, uncertain_group_y, feature_names_baseline)
           

if __name__ == '__main__':
    k = 20
    filename = 'FDG-AV45k' + str(k) + '.without.APOE'
    target_columns = ['FDG','AV45']
    #gselection_cohort, _ = cmanager.generate_baseline_MCI_cohort(feature_names_baseline)
    gselection_cohort, _ = cmanager.generate_baseline_ADCN_cohort(feature_names_baseline)
    #gselection_cohort, _ = cmanager.generate_baseline_cohort(feature_names_baseline)
    genelist = select_gene(gselection_cohort, target_columns, k)
    #genelist = ['SCORE','APOE_e2e4']
    genelist += ['APOE_e2e4']
    #genelist = ['SCORE', 'APOE_e2e4']
    X, y = define_cohort(feature_names_baseline + genelist)

    main(filename, X, y, genelist)
