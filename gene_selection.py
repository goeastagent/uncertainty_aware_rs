import pandas as pd
import numpy as np

from scipy.stats.stats import pearsonr

class GeneSelectionLoader(object):
    def __init__(self):
        self.gene_map = pd.read_csv('data/snp_gene_mapping.csv',sep=',')
        self.genenames = pd.read_csv('data/gene_level_directed_rebuilt',sep='\t').drop('count',axis=1)['gene']

    def feature_screen(self, data):
        gene_var = data[self.genenames].var(axis=0)
        genelist = gene_var[gene_var<1].index
        data = data.drop(genelist, axis=1)
        return data

    def feature_selection(self, dmatrix, targets, k):
        genelist = []
        dmatrix_copy = dmatrix.copy()
        for target in targets:
            dmatrix_copy[target+'_abs'] = dmatrix_copy[target].abs()
            dmatrix_copy = dmatrix_copy.sort_values(target+'_abs',ascending=False)
            genelist += dmatrix_copy[dmatrix_copy[target+'_pvalue'] < 0.05].index[:k].tolist()
        return genelist

    def build_snp_list_from_gene(self, genelist):
        indices = []
        for idx, record in self.gene_map.iterrows():
            for gene in  record['Gene.refGene'].split(','):
                if gene in genelist:
                    indices.append(idx)
                    break
        return self.gene_map.iloc[indices]

    # main
    def discover_gene(self, cohort, target_colnames, k):
        data = cohort
        data = self.feature_screen(data)

        dmatrix = self.generate_dmatrix(data, target_colnames)
        genelist = self.feature_selection(dmatrix, target_colnames, k)
        genelist = pd.Series(genelist).unique().tolist()

        df_saved = dmatrix.loc[genelist]
        df_saved['gene'] = genelist
        df_saved.to_csv('correlation_pvalue_k'+str(k) + '.csv',sep=',',index=False)
        snplist = self.build_snp_list_from_gene(genelist)
        snplist.to_csv('SNP_discovery_' + '_'.join(target_colnames) + '_k'+str(k)+'.csv', sep=',',index=False)
        return genelist

    def discover_gene_disentangle(self, cohort, target_colnames, k):
        data = cohort
        data = self.feature_screen(data)

        dmatrix = self.generate_dmatrix(data, target_colnames)
        genelist = self.feature_selection(dmatrix, target_colnames, k)
        genelist = pd.Series(genelist).unique().tolist()


    def generate_dmatrix(self, data, target_colnames):
        target_genes = data.columns[data.columns.isin(self.genenames)]
        data.index = data['ADNI_genotype2']
        data = data.dropna(subset=target_colnames)
        
        dmatrix = {}
        pvalues = {}
        for target_colname in target_colnames:
            s1 = data.loc[:, target_colname]
            similarities = []
            ps = []
            for gene in target_genes:
                s2 = data.loc[:, gene]
                r, p = pearsonr(s1,s2)
                similarities.append(r)
                ps.append(p)
            dmatrix[target_colname] = similarities
            dmatrix[target_colname+'_pvalue'] = ps
        return pd.DataFrame(dmatrix, index=target_genes).dropna()

    
