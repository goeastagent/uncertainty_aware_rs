import pandas as pd
import numpy as np


def concat_gene_tsdata_oct30th2022():
    data = pd.read_csv('data/adni_all_garam.csv', sep=',')
    data = data.drop(['PTGENDER','AGE','PTEDUCAT','APOE4'],axis=1)
    genedata = pd.read_csv('data/gene_level_directed_rebuilt_merge_pupdated_apoe.csv',sep=',').drop('VISCODE',axis=1)
    #genedata = pd.read_csv('data/gene_level_directed_merge_pupdated_apoe_prsnorm.csv',sep=',').drop('VISCODE',axis=1)
    df = pd.merge(genedata, data, on='RID')
    df.to_csv('data/ts_gene_merged.csv',sep=',',index=False)
    
