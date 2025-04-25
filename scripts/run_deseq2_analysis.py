"""
A simple PyDESeq2 workflow
===========================
This example demonstrates how to run a basic differential expression analysis on bulk RNA-seq data using PyDESeq2. 
We begin by importing the necessary libraries and optionally defining a path to save the results.

"""
import os
import pickle as pkl
import matplotlib.pyplot as plt

from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
from pydeseq2.utils import load_example_data


# Replace this with the path to directory where you would like results to be saved
OUTPUT_PATH = "C:/genome-analytics/results/synthetic_example"
os.makedirs(OUTPUT_PATH, exist_ok=True)  # Create path if it doesn't exist

# load expression count data
# expression data includes 100 samples and 10 genes
counts_df = load_example_data(
    modality="raw_counts",
    dataset="synthetic",
    debug=False,
)

print('\n top 5 count data: \n', counts_df.head(5))
print('\n number of genes and samples: \n', counts_df.shape)

# load meta-data data
# clinical data includes 100 samples and 2 variables named 'condition' and 'group'
metadata = load_example_data(
    modality="metadata",
    dataset="synthetic",
    debug=False,
)

print('\n top 5 meta-data: \n', metadata.head(5))
print('\n number of samples and metadata variables: \n', metadata.shape)

# data pre-processing
metadata['condition'].unique()
metadata['condition'].value_counts()
metadata['condition'].isna().sum()
samples_to_keep = ~metadata.condition.isna()


counts_df = counts_df.loc[samples_to_keep]
counts_df.shape
metadata = metadata.loc[samples_to_keep]
metadata.shape

genes_to_keep = counts_df.columns[counts_df.sum(axis=0) >= 10]
counts_df = counts_df[genes_to_keep]

min_val = counts_df.min().min()
max_val = counts_df.max().max()
print(f"Data range: {min_val} to {max_val}")

# single factor analysis
# assess association of genes across two groups of condition (A vs B)
inference = DefaultInference(n_cpus=8)

# create 'DeseqDataSet' object
dds = DeseqDataSet(
    counts=counts_df,
    metadata=metadata,
    design="~condition",
    refit_cooks=True,
    inference=inference,
    # n_cpus=8, # n_cpus can be specified here or in the inference object
)

# run the deseq2() method to fit dispersions and log-fold changes (LFCs)
dds.deseq2()
print(dds)
print(dds.varm["dispersions"])
print(dds.varm["LFC"])

# statistical analysis with the DeseqStats class
# condition B vs A
ds = DeseqStats(dds, contrast=["condition", "B", "A"], inference=inference)
ds.summary()
# condition A vs B
# ds = DeseqStats(dds, contrast=["condition", "A", "B"], inference=inference)
# ds.summary()

# visualization
ds.summary(lfc_null=0.1, alt_hypothesis="greaterAbs")
ds.plot_MA(s=20)
plt.savefig("C:/genome-analytics/results/synthetic_example/MA_plot_condition.png", dpi=300, bbox_inches='tight')

# LFC shrinkage
ds.lfc_shrink(coeff="condition[T.B]")

# multifactor analysis
# compare expressions across condition A and B, while the impact of group variable (X nad Y) is taken into account
# or compare expressions across group X and Y, while the impact of condition variable (A nad B) is taken into account
dds = DeseqDataSet(
    counts=counts_df,
    metadata=metadata,
    design="~group + condition",
    refit_cooks=True,
    inference=inference,
)

dds.deseq2()
print(dds)
print(dds.varm['LFC'])

# compare the condition B to the condition A
ds_B_vs_A = DeseqStats(dds, contrast=["condition", "B", "A"], inference=inference)
ds_B_vs_A.summary()

# compare the group Y to the group X
ds_Y_vs_X = DeseqStats(dds, contrast=["group", "Y", "X"], inference=inference)
ds_Y_vs_X.summary()

ds_Y_vs_X.summary(lfc_null=0.1, alt_hypothesis="greaterAbs")
ds.plot_MA(s=20)
plt.savefig("C:/genome-analytics/results/synthetic_example/MA_plot_group.png", dpi=300, bbox_inches='tight')


"""
steps to do DE analysis using DESeq2 pipeline:
===========================
1. Load data (count expression & meta-data)
2. Data pre-processing
3. Run DeseqDataSet function to create an object
4. Run deseq2() function to estimate dispersion and compute LFCs
5. Run statistical analysis to compare groups ---> extract Wald test p-value and adjusted p-values
"""
