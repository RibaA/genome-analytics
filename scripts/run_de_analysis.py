"""
A pipeline to show the Immuno-Oncology (IO) association analysis 
===========================
We consider the IMvigor210 bladder cancer patients and assess the association of genes with IO outcomes.
We apply logistic regression, and DESeq method for binary response and Cox model for overall survival analyses. 
We begin by importing the necessary libraries and optionally defining a path to save the results.

"""
import os
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LogisticRegression
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats

# function rem to remove low expressed genes

def rem(x, missing_perc, const_int):
    # Ensure x is a numpy array of floats
    x = np.array(x, dtype=float)
    # Calculate log2(const_int)
    log_const = np.log2(const_int)
    # Count how many entries per row are equal to log2(const_int) (with rounding)
    r = np.sum(np.round(x, 6) == round(log_const, 6), axis=1)
    # Find rows where count > number of columns * missing_perc
    remove = np.where(r < x.shape[1] * missing_perc)[0]
    return remove

# Replace this with the path to directory where you would like results to be saved
#OUTPUT_PATH = "C:/genome-analytics/results/synthetic_example"
OUTPUT_PATH = "C:/genome-analytics/results/IO_IMvigor210"
os.makedirs(OUTPUT_PATH, exist_ok=True)  # Create path if it doesn't exist

# load expression, clinical and annotation data
# expression data includes 195 samples and 17,993 genes
expr = pd.read_csv('C:/genome-analytics/data/Mariathasan__Bladder__PD-L1_expr.csv')
clin = pd.read_csv('C:/genome-analytics/data/Mariathasan__Bladder__PD-L1_clin.csv')
annot = pd.read_csv('C:/genome-analytics/data/Gencode.v19.annotation.csv')

# top 5 data
expr.set_index(expr.columns[0], inplace=True)
expr.index.name = None
print('\n top 5 count data: \n', expr.head(5))
print('\n number of genes and samples: \n', expr.shape)

clin.set_index('patientid', inplace=True)
clin.index.name = None
print('\n top 5 meta-data: \n', clin.head(5))
print('\n number of samples and clinical variables: \n', clin.shape)

print('\n top 5 gene annotation data: \n', annot.head(5))
print('\n number of genes and annotation variables: \n', annot.shape)

# data pre-processing
# 1. include protein-coding genes in analysis
label = annot['gene_type'] == 'protein_coding'
annot[label].shape
annot_pc = annot[label]

# 2. subset expression data to protein-coding
label = annot_pc['gene_name']
expr_pc = expr.loc[expr.index.intersection(label)]
expr_pc.iloc[:3, :4]
expr_tpm = (2 ** expr_pc) - 0.001
expr_tpm = expr_tpm.round(0).astype(int)

# 3. remove NA's in response value
samples_to_keep = ~clin.response.isna()
expr_tpm = expr_tpm.loc[:, samples_to_keep]
expr_tpm.shape
clin = clin.loc[samples_to_keep]
clin.shape

# 4. remove low/zero-expressed genes
r = np.sum(expr_tpm == 0, axis=1)
expr_tpm_filtered = expr_tpm[r < expr_tpm.shape[1] * 0.5]
expr_tpm_filtered = expr_tpm_filtered.T 

# 5. log2(TPM+1) data
expr_logtpm_filtered = np.log2(expr_tpm_filtered + 1)
expr_logtpm_filtered = expr_logtpm_filtered

# Part 1: assess association of genes across two groups of IO response (NR vs R)
# create 'DeseqDataSet' object
fit_deseq = DeseqDataSet(
    counts=expr_tpm_filtered,
    metadata=clin,
    design="~response",
    refit_cooks=True,
    n_cpus=8 
)

# run the deseq2() method to fit dispersions and log-fold changes (LFCs)
fit_deseq.deseq2()
print(fit_deseq)
print(fit_deseq.varm["dispersions"])
print(fit_deseq.varm["LFC"])

# statistical analysis with the DeseqStats class
# response R vs NR
ds = DeseqStats(fit_deseq, contrast=["response", "NR", "R"])
ds.summary()

# Part 2: assess association of genes across two groups of IO response (NR vs R)
#data = pd.concat([clin.loc[:, "response"], expr_logtpm_filtered], axis=1)
y = clin.loc[:, "response"].map({"R": 1, "NR": 0})
X = expr_logtpm_filtered

results = []

for gene in X.columns:
    X_gene = sm.add_constant(X[[gene]])  # add intercept
    model = sm.Logit(y, X_gene)
    
    try:
        res = model.fit(disp=0)  # disp=0 suppresses output
        coef = res.params[gene]
        pval = res.pvalues[gene]
        results.append((gene, coef, pval))
    except:
        results.append((gene, None, None))  # Handle failed fits

# apply multiple test correction
results_df = pd.DataFrame(results, columns=["gene_name", "coefficient", "p_value"])
results_df.dropna(inplace=True)

# Add FDR using Benjamini-Hochberg
results_df["FDR"] = multipletests(results_df["p_value"], method="fdr_bh")[1]

# Sort by FDR or p-value
results_df = results_df.sort_values("p_value")