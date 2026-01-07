######## Load modules ########
from __future__ import annotations #default now for name.error issue
import os
import pickle
from datetime import datetime
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
import scipy
from tqdm import tqdm

# Cell2Sentence imports
import cell2sentence as cs
from cell2sentence.utils import benchmark_expression_conversion, reconstruct_expression_from_cell_sentence
from cell2sentence.tasks import embed_cells, predict_cell_types_of_data

# Hugging Face
from transformers import TrainingArguments

# Single-cell libraries
import scanpy as sc
import anndata as ad
from collections import Counter #count table

sc.set_figure_params(dpi=300, color_map="viridis_r", facecolor="white", )
sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()


######## Reload preprocessed and filtered data ########
DATA_PATH = "/ix/ccdg/storage3/til177/ParkLab/Project_C2S_scFM/code/tutorials/dominguez_conde_immune_tissue_two_donors_preprocessed_tutorial_0_PYFILE.h5ad"
adata = ad.read_h5ad(DATA_PATH)
print(adata)             # check dimension
print(adata.X.max())     # check max value (log10 transformation expects a maximum value somewhere around 3 or 4)
print(adata.obs.columns) # check colnames (for next step)


######## C2S conversion (AnnData -> Arrow -> CSData) ########
# Keep meta data information (from adata.obs)
adata_obs_cols_to_keep = ["cell_type", "tissue", "batch_condition", "organism", "sex"]

# Create PyArrow dataset
arrow_ds, vocabulary = cs.CSData.adata_to_arrow(
    adata=adata, 
    random_state=SEED, 
    sentence_delimiter=' ',
    label_col_names=adata_obs_cols_to_keep
)
print(arrow_ds) #check data type
print(arrow_ds.shape)

# Check cell info
sample_idx = 0
print(arrow_ds[sample_idx]) # each row is a dict/json, containing info for that cell 
# ##   Check cell sentence length
# len(arrow_ds[sample_idx]["cell_sentence"].split(" "))  # Cell 0 has 2191 nonzero expressed genes, yielding a sentence of 2191 gene names separated by spaces.

# # Check feature info
# print(type(vocabulary))
# print(len(vocabulary))
# print(list(vocabulary.items())[:10]) #fist 10, also contains the number of cells 'that gene' was expressed in.


######## Load CSModel first (FT, training_task = "cell_type_prediction") 
#        (bc `data_split_indices_dict` is associated with the FT model)

#### Load a fine-tuned model __checkpoint__ (best)
cell_type_prediction_model_path = "/ix/ccdg/storage3/til177/ParkLab/Project_C2S_scFM/code/tutorials/csmodel_tutorial_3/2025-11-24-18_42_05_finetune_cell_type_prediction/checkpoint-3725" #smallest eval_loss
save_dir = "/ix/ccdg/storage3/til177/ParkLab/Project_C2S_scFM/code/tutorials/csmodel_tutorial_3"
save_name = "cell_type_pred_pythia_410M_inference"

csmodel = cs.CSModel(
    model_name_or_path=cell_type_prediction_model_path,
    save_dir=save_dir,
    save_name=save_name
)
print(csmodel)

#### Load the `data_split_indices_dict` associated with the fine-tuned model
base_path = "/".join(cell_type_prediction_model_path.split("/")[:-1])
print(cell_type_prediction_model_path)
print(base_path)
with open(os.path.join(base_path, 'data_split_indices_dict.pkl'), 'rb') as f:
    data_split_indices_dict = pickle.load(f)
print(data_split_indices_dict.keys())

# This would be: train/val/test split (80/10/10) 
print(len(data_split_indices_dict["train"]))
print(len(data_split_indices_dict["val"]))
print(len(data_split_indices_dict["test"]))


######## C2S conversion CONT. ########
#### Select test set
test_ds = arrow_ds.select(data_split_indices_dict["test"])
print(test_ds)

#### Convert test set to CSData
c2s_save_dir = "/ix/ccdg/storage3/til177/ParkLab/Project_C2S_scFM/code/tutorials"
c2s_save_name = "dominguez_immune_tissue_tutorial4"
csdata = cs.CSData.csdata_from_arrow(
    arrow_dataset=test_ds, 
    vocabulary=vocabulary,
    save_dir=c2s_save_dir,
    save_name=c2s_save_name,
    dataset_backend="arrow"
)
print(csdata) #obj; path; format

# cell_sentences_list = csdata.get_sentence_strings()
# def print_first_N_genes(cell_sentence_str: str, top_k_genes: int, delimiter: str = " "):
#     """Helper function to print K genes of a cell sentence."""
#     print(delimiter.join(cell_sentence_str.split(delimiter)[:top_k_genes]))
# print_first_N_genes(cell_sentences_list[0], top_k_genes=200)


######## INFERENCE: Predict cell types ########
predicted_cell_types = predict_cell_types_of_data(
    csdata=csdata,
    csmodel=csmodel,
    n_genes=200
)
print(len(predicted_cell_types))
print(predicted_cell_types[:3])


######## Evalutaion Metrics - Accuracy ########
print(test_ds)

total_correct = 0.0
for model_pred, gt_label in zip(predicted_cell_types, test_ds["cell_type"]):
    # C2S might predict a period at the end of the cell type, which we remove (see the prompt response)
    if model_pred[-1] == ".":
        model_pred = model_pred[:-1]
    
    if model_pred == gt_label:
        total_correct += 1

accuracy = total_correct / len(predicted_cell_types)
print("Accuracy:", accuracy)

# Sample some cells for comparison between
for idx in range(0, 100, 10):
    print("Model pred: {}\nGround truth label: {}\n".format(predicted_cell_types[idx], test_ds[idx]["cell_type"]))


