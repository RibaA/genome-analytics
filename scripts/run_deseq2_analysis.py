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

print(counts_df)
print(metadata)

# data pre-processing
samples_to_keep = ~metadata.condition.isna()
print(samples_to_keep.value_counts())
num_true = samples_to_keep.sum() 
num_true = (samples_to_keep == True).sum()

