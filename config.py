sample_ratio=0.312

feature_names=['AGE','PTGENDER','PTEDUCAT','ABETA','TAU','PTAU','ADAS11','ADAS13','Hippocampus','Entorhinal']
feature_names_ts = ['AGE','PTGENDER','PTEDUCAT','ABETA','TAU','PTAU','ADNI_MEM','ADNI_EF','Hippocampus','Entorhinal']

ID_colname = 'RID'

REGULAR_MODEL_KEY = 'regular'
PI_MODEL_KEY = 'pi'
model_keys = [REGULAR_MODEL_KEY, PI_MODEL_KEY]

adors_genelist = ['LINC01364', 'MRGPRX2', 'SYNJ1', 'CDK20', 'EHD3', 'LINC00536', 'SRC', 'DLGAP2-AS1', 'LOC440173', 'KCNIP1', 'CP', 'UBE2F-SCLY', 'KRTAP26-1', 'C16orf46', 'SHANK2', 'LINC002481', 'GSTA5', 'LINC01578', 'PYY', 'EMB', 'FAM30A', 'PLCE1-AS2', 'MSH2', 'ACSF2', 'GLYATL1', 'SP7', 'XPO7', 'LPIN3', 'CEP250', 'TEX48', 'CRYM', 'EPS8L2', 'NRG1-IT1', 'LL22NC03-63E9.3', 'GYG1', 'SCOC', 'HPS3', 'LINC00459', 'KLHL42']



out_filename_prefix = 'result.bin/'
