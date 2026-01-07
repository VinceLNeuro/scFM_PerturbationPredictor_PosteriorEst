######## Load modules ########
from __future__ import annotations #default now for name.error issue
import os
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
from cell2sentence.tasks import embed_cells

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


c2s_save_dir = "/ix/ccdg/storage3/til177/ParkLab/Project_C2S_scFM/code/tutorials"
c2s_save_name = "dominguez_immune_tissue_tutorial3"
csdata = cs.CSData.csdata_from_arrow(
    arrow_dataset=arrow_ds, 
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


######## Load CSModel (FM) ########
#### Load a pretrained C2S model (Define CSModel object)
cell_type_prediction_model_path = "/ix/ccdg/storage3/til177/ParkLab/Project_C2S_scFM/C2S_models/C2S-Pythia-410m-diverse-single-and-multi-cell-tasks/snapshots/51f7c9d46776273ea4732ddaf494d1db733ca5d6"
save_dir = "/ix/ccdg/storage3/til177/ParkLab/Project_C2S_scFM/code/tutorials/csmodel_tutorial_3"
save_name = "cell_type_pred_pythia_410M_2"
csmodel = cs.CSModel(
    model_name_or_path=cell_type_prediction_model_path,
    save_dir=save_dir,
    save_name=save_name
)
print(csmodel)


######## FT on new dataset ########
training_task = "cell_type_prediction" 
""" # available tasks 
    SUPPORTED_TASKS = [
                "cell_type_prediction",
                "cell_type_generation",
            ]
    MULTICELL_SUPPORTED_TASKS = [
        "tissue_prediction",
        "tissue_conditional_generation",
        "natural_language_interpretation",
    ]
"""
# Create a datetimestamp to mark our training session:
datetimestamp = datetime.now().strftime('%Y-%m-%d-%H_%M_%S') #2025-11-24-17_04_58
# Create output dir
output_dir = os.path.join(csmodel.save_dir, datetimestamp + f"_finetune_{training_task}")
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
print(output_dir)

#### Define TrainingArguments ####
"""
# Key concepts 

1. effective batch size = 32

2.  learning_rate=1e-5, #controlling the magnitude of weight updates
    lr_scheduler _reduce_ the learning rate over time based on predefined rules to improve convergence
    - lr_scheduler_type = "cosine": smoothly decrease (after warmup) as a cosine curve
    
    www.geeksforgeeks.org/machine-learning/impact-of-learning-rate-on-a-model/

3. Warmup
During warmup:
    LR starts near 0.
    Increases _linearly (by default in HF)_ up to learning_rate.
    After warmup ends, the scheduler (e.g., cosine) takes over and starts decaying it.

Why warmup?
    At the very beginning, gradients can be unstable (random init of your head, random batches).
    A sudden high LR can cause divergence or large, bad updates.
    Warmup gently introduces the full LR, stabilizing early training.
"""
train_args = TrainingArguments(
    bf16=True, #more numerically stable   than fp16 in many LLM (if GPU A100, H100, recent RTX, etc.)
    fp16=False, #use when want mixed precision but without hardware above
                #default fp32
    
    #effective batch size = 32 (If training is very noisy (loss bounces a lot), try larger effective batch)
    per_device_train_batch_size=8, #default
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4, #increase this for EBS
    gradient_checkpointing=False,
    
    # Step 0 → step 5% of total (warmup): LR ramps linearly from ~0 → 1e-5
    # Step 5% → 100%: LR starts at 1e-5 and then cosine‑decays down toward 0 by the end
    learning_rate=1e-5,         #controlling the magnitude of weight updates
    lr_scheduler_type="cosine", #LR slowly decrease after warmup
    warmup_ratio=0.05,          #5% straining steps are warmup steps
    load_best_model_at_end=True, ##save the found best model at the end of training 
    
    num_train_epochs=5, #5 complete passes through the entire training dataset 
                        #(determine by: when validation loss stops improving)

    # total steps = epoch * N_train / effect_batch
    #             = 5*30k/32 := 4690
    logging_steps=50, #logging training loss per 50 steps
    logging_strategy="steps",
    eval_steps=50,    #evaluate validation loss per 50 steps
    eval_strategy="steps",
    save_steps=100,   #must be a mulitple of `eval_steps`
    save_strategy="steps",
    save_total_limit=3, #limit total amount of checkpoints, keep 3
    
    output_dir=output_dir
)

#### Fine Tune ####
csmodel.fine_tune(
    csdata=csdata,
    task=training_task,
    train_args=train_args,
    loss_on_response_only=False, #whether to take loss only on model’s answer 
                                 #(if True: You don’t care if the model can generate the question text; you care that it outputs the right cell type.)
    top_k_genes=200, #number of top/first genes to use for each cell sentence. 
                     #(Ignored if prompt_formatter is not None)
    max_eval_samples=500, # number of samples (2% here) to use for validation
)

