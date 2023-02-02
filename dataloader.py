import pandas as pd


class CohortManager(object):
    def __init__(self, filename):
        df = pd.read_csv(filename, sep=',')
        self.genenames = pd.read_csv('data/gene_level_directed_rebuilt',sep='\t').drop('count',axis=1)['gene']

        # preprocessing
        df.loc[df['ABETA'] == '1700', 'ABETA'] = 1700
        df.loc[df['PTAU'] == '<8', 'PTAU'] = 7
        df.loc[df['ABETA'] == '<200', 'ABETA'] = 199
        df.loc[df['TAU'] == '<80', 'TAU'] = 79
        df['ABETA'] = df['ABETA'].astype('float64')
        df['PTAU'] = df['PTAU'].astype('float64')
        df['TAU'] = df['TAU'].astype('float64')
        
        self.df = df
                
    def generate_baseline_ADCN_cohort(self, feature_names):
        data = self.df[self.df['DX_bl'].isin(['CN','AD'])]
        data.loc[:,'AD'] = data['DX_bl'].map({'AD':1, 'CN':0})
        data = data.dropna(subset=feature_names,axis=0)
        return data, data['AD']

    def generate_baseline_MCI_cohort(self, feature_names):
        data = self.df[self.df['DX_bl'].isin(['LMCI','EMCI'])]
        data = data[data['MCIc'] != -9]
        data = data.dropna(subset=feature_names,axis=0)
        return data, data['MCIc']

    def generate_baseline_cohort(self, X, y):
        indices = X['VISCODE'] == 0
        return X.loc[indices], y.loc[indices]
        
    def filterout_GO1_cohort(self, X, y):
        index = X['RID'] > 3000
        return X[index], y[index]

    
    
    
def define_phenotype(df):
    data1 = df[df['DX_bl'].isin(['CN','AD'])]
    data1.loc[:,'AD'] = data1['DX_bl'].map({'AD':1, 'CN':0})
    data2 = df[df['MCIc'] == 1]
    data2.loc[:,'AD'] = data2['MCIc']
    data1 = data1[~data1['RID'].isin(data2['RID'])]
    data = pd.concat([data1, data2],axis=0)
    return data

