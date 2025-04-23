"""
A simple PyDESeq2 workflow
===========================
This example demonstrates how to run a basic differential expression analysis on bulk RNA-seq data using PyDESeq2. 
We begin by importing the necessary libraries and optionally defining a path to save the results.

"""
import os
import pickle as pkl

from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
from pydeseq2.utils import load_example_data


# Replace this with the path to directory where you would like results to be saved
OUTPUT_PATH = "C:/genome-analytics/results/synthetic_example"
os.makedirs(OUTPUT_PATH, exist_ok=True)  # Create path if it doesn't exist

# load data
counts_df = load_example_data(
    modality="raw_counts",
    dataset="synthetic",
    debug=False,
)

metadata = load_example_data(
    modality="metadata",
    dataset="synthetic",
    debug=False,
)

print('\n top 5 count data: \n', counts_df.head(5))
print('\n top 5 meta-data: \n', metadata.head(5))
print('\n number of genes and samples: \n', counts_df.shape)
print('\n number of samples and metadata variables: \n', metadata.shape)

# data pre-processing
metadata['condition'].unique()
metadata['condition'].value_counts()
metadata['condition'].isna().sum()
samples_to_keep = ~metadata.condition.isna()
#print(samples_to_keep.value_counts())
#num_true = samples_to_keep.sum() 
#num_true = (samples_to_keep == True).sum()

counts_df = counts_df.loc[samples_to_keep]
counts_df.shape
metadata = metadata.loc[samples_to_keep]
metadata.shape

genes_to_keep = counts_df.columns[counts_df.sum(axis=0) >= 10]
counts_df = counts_df[genes_to_keep]

min_val = counts_df.min().min()
max_val = counts_df.max().max()
print(f"Data range: {min_val} to {max_val}")

# association analysis
inference = DefaultInference(n_cpus=8)
dds = DeseqDataSet(
    counts=counts_df,
    metadata=metadata,
    design="~condition",
    refit_cooks=True,
    inference=inference,
    # n_cpus=8, # n_cpus can be specified here or in the inference object
)



dds.deseq2()
print(dds)


print(dds.varm["dispersions"])
print(dds.varm["LFC"])

ds = DeseqStats(dds, contrast=["condition", "B", "A"], inference=inference)
ds.summary()

ds = DeseqStats(dds, contrast=["condition", "A", "B"], inference=inference)
ds.summary()