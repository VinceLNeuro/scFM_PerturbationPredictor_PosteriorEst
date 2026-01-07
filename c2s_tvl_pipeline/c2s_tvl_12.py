######## Load modules ########
from __future__ import annotations #default now for name.error issue
import os
## ensure model trains on one GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
os.environ["WORLD_SIZE"] = "1"
import pickle
from datetime import datetime
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
import scipy
from tqdm import tqdm

# Cell2Sentence imports
import cell2sentence as cs
# from cell2sentence.utils import benchmark_expression_conversion, reconstruct_expression_from_cell_sentence
from cell2sentence import utils
from cell2sentence.tasks import embed_cells, predict_cell_types_of_data
from cell2sentence.prompt_formatter import get_cell_sentence_str, PromptFormatter #for custom prompt

# Hugging Face
from transformers import TrainingArguments, AutoModelForCausalLM
from datasets import Dataset # Arrow

# Single-cell libraries
import scanpy as sc
import anndata as ad
from collections import Counter, defaultdict #count table

sc.set_figure_params(dpi=300, color_map="viridis_r", facecolor="white", )
sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()

# Define device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


######## Re-load Perturbation Data ########
DATA_PATH = "/ix/ccdg/storage3/til177/ParkLab/Project_C2S_scFM/code/tutorials/data_PerturbSeq/GSE264667_jurkat_processed.h5ad"
adata = ad.read_h5ad(DATA_PATH)
print(adata)             # check dimension
print(adata.X.max())     # check max value (log10 transformation expects a maximum value somewhere around 3 or 4)
print(adata.var_names)   # will be used to create the __cell sentences__
# print(adata.var)

# target_gene_counter = Counter(adata.obs['target_gene'])
# print(f"{len(target_gene_counter)} unique perturbations")
# print(target_gene_counter.most_common(20)) #('non-targeting', 11742)!


#### recreate `vocab_list` used in `c2s_tvl_11v3_fullLength_perturbation_PosteriorEst.py` > `posterior_sentences_to_expression` ####
adata_obs_cols_to_keep = ['batch_var','cell_type','target_gene','gene_id','mitopercent','UMI_count']
arrow_ds, vocabulary = cs.CSData.adata_to_arrow(
    adata=adata, 
    random_state=SEED, 
    sentence_delimiter=' ',
    label_col_names=adata_obs_cols_to_keep
)
print(arrow_ds)
vocab_list = list(vocabulary.keys())

## check if the same
print(vocab_list[:10])
print(adata.var_names)


######## Load Posterior Estimations ######## 
# Load the sampling data
benchmarking_wd="/ix/ccdg/storage3/til177/ParkLab/Project_C2S_scFM/code/tutorials/data_PerturbSeq/perturbation_predictor_finetuned_final_benchmarking"
with open(os.path.join(benchmarking_wd, "posterior_samples_meta_script11v2.pkl"), "rb") as f:
    meta = pickle.load(f)

# Access results
print(meta["high_temp"].keys())
print(meta["low_temp"].keys())
print()

# same `n_samples`
print(meta["high_temp"]['n_samples'] == meta["low_temp"]['n_samples'])
print("'n_samples':", meta["high_temp"]['n_samples'])
print()
# same `prompt`
print(meta["low_temp"]['prompt'] == meta["high_temp"]['prompt'])
print("prompt:", meta["low_temp"]['prompt'][:1000])
print()

# `temperature`
print("'temperature':")
print(meta["high_temp"]['temperature'])
print(meta["low_temp"]['temperature'])
print()

#### Check cell sentences ####
# meta["high_temp"]['posterior_samples']
# meta["low_temp"]['posterior_samples']

#### Check expression matrix ####
expr_samples_high = meta["high_temp"]["expr_samples"]
expr_samples_low  = meta["low_temp"]["expr_samples"]

print(expr_samples_high.shape)
print(expr_samples_high)
print(expr_samples_high.max())

print(expr_samples_low.shape)

#### this is un-post-processed!!! Have duplicates ####
print(len(meta["high_temp"]['posterior_samples'][4].split(" ")) == len(set(meta["high_temp"]['posterior_samples'][4].split(" "))))

print(len(meta["high_temp"]['posterior_samples'][4].split(" ")))
print(len(set(meta["high_temp"]['posterior_samples'][4].split(" "))))


######## Create a merged anndata for visualization to _quality check_ if temperature difference during generation will create a different distribution of the samples (i.e., higher temperature would have a more spread distribution on the shared UMAP) ########

# top_k_genes=200

# Try 1: Use full reconstructed expression matrix
# Try 2 (later): Use union set of the two temperature samples
#     - Union mask keeps any gene that appears in either temperature at least once, so you preserve temperature-specific support changes while removing genes that are always zero in both groups.

print(expr_samples_low.shape == expr_samples_high.shape)
n_samples, n_genes = expr_samples_low.shape

# Check the length of vocab_list and posteroir expression matrix
print(n_samples, n_genes)
print(len(vocab_list))

# 1. obs with temperature labels
obs_low = pd.DataFrame(
    {"temperature": ["temp=0.3"] * n_samples},
    index=[f"low_{i}" for i in range(n_samples)],
)
obs_high = pd.DataFrame(
    {"temperature": ["temp=0.8"] * n_samples},
    index=[f"high_{i}" for i in range(n_samples)],
)
# display(obs_low)
# display(obs_high)

# 2. var copied from your reference adata (or built from vocab_list)
var = pd.DataFrame(index=pd.Index(vocab_list, name="gene"))

adata_low = ad.AnnData(
    X=expr_samples_low.astype(np.float32),
    obs=obs_low,
    var=var,
)
adata_high = ad.AnnData(
    X=expr_samples_high.astype(np.float32),
    obs=obs_high,
    var=var,
)
# print(adata_low)
# print(adata_high)

# 3. concatenate and run PCA/UMAP jointly
concat_adata = ad.concat([adata_low, adata_high], axis=0)
concat_adata.var = var  # ensure var is consistent
print(concat_adata)
print(concat_adata.X.max())
print(concat_adata.obs)

# 4. Visualize (same as in script1)
sc.pp.scale(concat_adata, zero_center=True, max_value=5) #ensure two datasets are transformed correctly before PCA
sc.tl.pca(concat_adata)

def format_plot(ax, xlabel, ylabel,
                draw_axhline = True, draw_axvline = True,
                x_set_size_inches = 8,
                y_set_size_inches = 4,
                xticks_lb = -8, xticks_hb = 9, xticks_step = 2, #-8 to 8, step=2
                yticks_lb = -8, yticks_hb = 9, yticks_step = 2, #-8 to 8, step=2
               ):
    
    # Force ticks/labels to display
    ax.tick_params(axis="both", which="both", bottom=True, left=True, labelbottom=True, labelleft=True)
    ax.set_xticks(np.arange(xticks_lb, xticks_hb, xticks_step))
    ax.set_yticks(np.arange(yticks_lb, yticks_hb, yticks_step))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Remove all gridlines (Scanpy/rcParams may enable them)
    ax.grid(False)
    # Draw ONLY the x=0 and y=0 lines
    if draw_axhline:
        ax.axhline(0, color="black", linewidth=0.3, alpha=0.6, zorder=3) #control the visual order
    if draw_axvline:
        ax.axvline(0, color="black", linewidth=0.3, alpha=0.6, zorder=3)
    
    plt.tight_layout()

    fig = ax.figure
    fig.set_size_inches(x_set_size_inches, y_set_size_inches)
    print(fig.get_size_inches())
    
    plt.show()

    return fig

# PCA
ax = sc.pl.pca(
    concat_adata,
    color="temperature",
    size=20,
    title="Posterior samples by temperature",
    frameon=True, show=False,
)
fig = format_plot(ax, xlabel = "PC1", ylabel = "PC2",
                  xticks_lb = -15, xticks_hb = 26, xticks_step = 5,
                  yticks_lb = -15, yticks_hb = 26, yticks_step = 5)
# fig.savefig(
#     benchmarking_wd+"/PCA__top_k_genes200_PosteriorSamplesByTemp.png",
#     dpi=500,
#     bbox_inches="tight"
# )

## UMAP
sc.pp.neighbors(concat_adata, n_neighbors=15, n_pcs=50)
sc.tl.umap(concat_adata)
ax = sc.pl.umap(
    concat_adata,
    color="temperature",
    size=30,
    title="Posterior samples by temperature",
    # edges = True,
    frameon=True, show=False,
)
fig = format_plot(ax, xlabel = "UMAP1", ylabel = "UMAP2",
                  draw_axvline = False, draw_axhline = False,
                  xticks_lb = 0, xticks_hb = 16, xticks_step = 2,
                  yticks_lb = -2, yticks_hb = 10, yticks_step = 2)
# fig.savefig(
#     benchmarking_wd+"/umap__top_k_genes200_PosteriorSamplesByTemp.png",
#     dpi=500,
#     bbox_inches="tight"
# )
