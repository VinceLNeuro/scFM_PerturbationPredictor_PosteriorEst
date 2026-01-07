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
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
import scipy
from tqdm import tqdm
import torch

# Cell2Sentence imports
import cell2sentence as cs
from cell2sentence.utils import benchmark_expression_conversion, reconstruct_expression_from_cell_sentence
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


# ######## Load Perturbation Data ########
# # wget "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE264nnn/GSE264667/suppl/GSE264667%5Fjurkat%5Fraw%5Fsinglecell%5F01%2Eh5ad" -O GSE264667_jurkat.h5ad
# DATA_PATH = "/ix/ccdg/storage3/til177/ParkLab/Project_C2S_scFM/code/tutorials/data_PerturbSeq/GSE264667_jurkat.h5ad"
# adata = ad.read_h5ad(DATA_PATH)
# print(adata)
# print(adata.obs.head())

# #### Preprocess to match the tutorial ####
# # obs
# adata.obs['cell_type'] = "jurkat"
# adata.obs = adata.obs[['gem_group','cell_type','gene','gene_id','mitopercent','UMI_count']]
# adata.obs = adata.obs.rename(columns={'gem_group': 'batch_var', # change colnames 
#                                       'gene'     : 'target_gene'})
# adata.obs['batch_var'] = 'jurkat'+adata.obs['batch_var'].astype(str)
# print(adata.obs.head())

# # var
# # Set gene_name as the index for adata.var
# adata.var_names = adata.var['gene_name'].astype(str)
# adata.var = pd.DataFrame(index=adata.var_names)
# adata.var_names_make_unique()
# print(adata.var.head())

# #### Check before normalization ####
# print(Counter(adata.obs["batch_var"]))
# # will be used to create the __cell sentences__
# print(adata.var_names)
# # Need normalization
# print(adata.X.max())

# #### Preprocessing & Normalization ####
# # Basic filtering
# print(adata)
# sc.pp.filter_cells(adata, min_genes=200)
# sc.pp.filter_genes(adata, min_cells=3)
# print(adata) # check number decreased?

# # Annotate the group of mitochondrial genes as "mt"
# adata.var["mt"] = adata.var_names.str.startswith("MT-")
# sc.pp.calculate_qc_metrics(
#     adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
# )

# print(f"Median UMI: {np.median(adata.obs['UMI_count']):.0f}") #Median UMI: 10160

# # More filterings based on visualization (Jurkat Perturb-seq with median UMI ~10k)
# min_umi = 1000    # ~10% of median
# max_umi = 40000   # ~4x median (doublet filter)
# min_genes = 500
# max_genes = 6000
# max_mito = 15
# #   Apply filters
# adata = adata[
#     (adata.obs['UMI_count'] > min_umi) &
#     (adata.obs['UMI_count'] < max_umi) &
#     (adata.obs['n_genes'] > min_genes) &
#     (adata.obs['n_genes'] < max_genes) &
#     (adata.obs['mitopercent'] < max_mito)
# ].copy()
# print(adata)


# #### Normalization ####
# # Count normalization
# sc.pp.normalize_total(adata)
# # Lop1p transformation with base 10 - base 10 is important for C2S transformation!!!
# sc.pp.log1p(adata, base=10)
# # check --> ~3.4, which is expected for a base-10 log transformation.
# print(adata.X.max())


# #### Visualization ####
# sc.tl.pca(adata)
# sc.pp.neighbors(adata)
# sc.tl.umap(adata)
# # UMAP showed little community structure
# # > Should be expected, given this is scRNA-seq data from __cell-line__ with perturbation (minimal global affect).

# # # save data (NOT from this py file)
# # SAVE_PATH = "/ix/ccdg/storage3/til177/ParkLab/Project_C2S_scFM/code/tutorials/data_PerturbSeq/GSE264667_jurkat_processed.h5ad"
# # adata.write_h5ad(SAVE_PATH)


######## Re-load Perturbation Data ########
DATA_PATH = "/ix/ccdg/storage3/til177/ParkLab/Project_C2S_scFM/code/tutorials/data_PerturbSeq/GSE264667_jurkat_processed.h5ad"
adata = ad.read_h5ad(DATA_PATH)
print(adata)             # check dimension
print(adata.var_names)   # will be used to create the __cell sentences__
print(adata.obs.columns) # check colnames (for next step)
print(adata.X.max())     # check max value (log10 transformation expects a maximum value somewhere around 3 or 4)

target_gene_counter = Counter(adata.obs['target_gene'])
# print(len(target_gene_counter))
print(target_gene_counter.most_common(20)) #('non-targeting', 11742)!


######## AnnData -> Arrow ########
# We'll keep all relevant columns for our new task
adata_obs_cols_to_keep = ['batch_var','cell_type','target_gene','gene_id','mitopercent','UMI_count']

# Create Arrow dataset and vocabulary
arrow_ds, vocabulary = cs.CSData.adata_to_arrow(
    adata=adata, 
    random_state=SEED, 
    sentence_delimiter=' ',
    label_col_names=adata_obs_cols_to_keep
)
print(arrow_ds)

# Check single-cell info
sample_idx = 0
print(arrow_ds[sample_idx])
##   Check cell sentence length
print(len(arrow_ds[sample_idx]["cell_sentence"].split(" ")))  # Cell 0 has 4016 nonzero expressed genes
##   Check feature info
print(type(vocabulary))
print(len(vocabulary))
print(list(vocabulary.items())[:10]) #fist 10, also contains the number of cells 'that gene' was expressed in.


######## Custom Prompt Formatting for Perturbation Prediction ########

#  The input provides the {control cell} and the {perturbation}, asking for the {perturbed result}.
custom_input_prompt_template = """Given the following cell sentence of {num_genes} expressed genes representing a cell's basal state, predict the cell sentence after applying the perturbation: {perturbation_name}.
Control cell sentence: {control_cell_sentence}.

Perturbed cell sentence:"""

# The answer is simply the target cell sentence.
answer_template = "{perturbed_cell_sentence}."


#### Create PerturbationPromptFormatter (format_hf_ds outputs: formatted HF Arrow DataSet) ####
class PerturbationPromptFormatter(PromptFormatter):
    def __init__(self,
        task_name,
        input_prompt,
        answer_template,
        top_k_genes, 
        perturbation_col='target_gene',
        control_label='non-targeting'
    ):
        """
        Initializes the custom prompt formatter.

        Args:
            task_name (str): The name for this task.
            input_prompt (str): The template for the model's input.
            answer_template (str): The template for the model's expected response.
            top_k_genes (int): The number of top genes to include in the cell sentence.
            perturbation_col (str): The column name in the dataset that contains perturbation info.
            control_label (str): The label used to identify control cells in the perturbation_col.
        """
        super().__init__()
        self.task_name = task_name
        self.input_prompt = input_prompt
        self.answer_template = answer_template
        self.top_k_genes = top_k_genes
        self.perturbation_col = perturbation_col
        self.control_label = control_label
        assert top_k_genes > 0, "'top_k_genes' must be an integer > 0"

    def format_hf_ds(self, hf_ds):
        """
        Custom formatting function for perturbation prediction. This function creates pairs of
        (control, perturbed) cells by sampling from a global pool of control cells.
        """
        model_inputs_list = []
        responses_list = []
        
        # 1. Separate all cells into a global control pool and a dict of perturbed cells
        control_indices = []
        pert_to_indices = defaultdict(list)

        print("Grouping cells by perturbation and identifying global controls...")
        for i, sample in enumerate(hf_ds):
            if sample[self.perturbation_col] == self.control_label:
                control_indices.append(i)
            else:
                pert_to_indices[sample[self.perturbation_col]].append(i)

            # For each cell (sample) in the dataset:
            # If it's a control cell (e.g., target_gene == 'non-targeting'): add its index to control_indices
            # If it's perturbed (e.g., target_gene == 'BRCA1'): add its index to the pert_to_indices dictionary under that perturbation name
        
        assert len(control_indices) > 0, "No control cells found. Cannot create pairs."
        print(f"Found {len(control_indices)} control cells.")
        print(f"Found {len(pert_to_indices)} unique perturbations.")

        # 2. Create prompt-response pairs by iterating through perturbed cells
        print("Creating control-perturbed pairs...")
        for pert_name, perturbed_indices in tqdm(pert_to_indices.items()):
            for perturbed_idx in perturbed_indices:
                # Pair each perturbed cell with a random control cell from the global pool
                control_idx = random.choice(control_indices)
                
                control_sample = hf_ds[control_idx]
                perturbed_sample = hf_ds[perturbed_idx]

                # Format control cell sentence
                control_sentence, num_genes_str = get_cell_sentence_str(#https://github.com/vandijklab/cell2sentence/blob/a6efaf079f98491d4723ced44b929936b94368aa/src/cell2sentence/prompt_formatter.py#L31
                    control_sample,
                    num_genes=self.top_k_genes
                )
                # Format perturbed cell sentence
                perturbed_sentence, _ = get_cell_sentence_str(
                    perturbed_sample,
                    num_genes=self.top_k_genes
                )
                
                #### Matches the template fstring ####
                # Format the model input string using the perturbation name
                model_input_str = self.input_prompt.format(
                    num_genes=num_genes_str,
                    perturbation_name=pert_name,
                    control_cell_sentence=control_sentence
                )
                # Format the response string
                response_str = self.answer_template.format(
                    perturbed_cell_sentence=perturbed_sentence
                )

                model_inputs_list.append(model_input_str)
                responses_list.append(response_str)

        # Create the final Hugging Face Dataset
        ds_split_dict = {
            "sample_type": [self.task_name] * len(model_inputs_list),
            "model_input": model_inputs_list,
            "response": responses_list,
        }
        ds = Dataset.from_dict(ds_split_dict)
        return ds

#### Test run to see the results (this will automatically done in `csmodel.fine_tune()`)
# Initiate the formatter
task_name = "perturbation_prediction"
prompt_formatter = PerturbationPromptFormatter(
    task_name=task_name,
    input_prompt=custom_input_prompt_template,
    answer_template=answer_template,
    top_k_genes=200 # Using top 200 genes for this example. For real applications, ideal to use all nonzero expressed genes if possible.
)
# Format the dataset
formatted_ds = prompt_formatter.format_hf_ds(arrow_ds)
print(formatted_ds)
print(type(formatted_ds)) #datasets.arrow_dataset.Dataset
# Inspect a formatted sample
print("--- Formatted Sample ---")
print("#----Model input:----#")
print(formatted_ds[0]["model_input"], "\n")
print("#----Response:----#")
print(formatted_ds[0]["response"])


######## C2S conversion: formatted arrow_ds -> CSData ########
# Save-directory for Huggingface dataset
c2s_save_dir = "/ix/ccdg/storage3/til177/ParkLab/Project_C2S_scFM/code/tutorials/data_PerturbSeq"
c2s_save_name = "jurkat_perturbation_c2s"

csdata = cs.CSData.csdata_from_arrow(
    arrow_dataset=arrow_ds,  # Regular cell sentence dataset put here, finetune() function will repeat the formatting with the prompt formatter
    vocabulary=vocabulary,
    save_dir=c2s_save_dir,
    save_name=c2s_save_name,
    dataset_backend="arrow"
)
print(csdata)


######## Load and Finetune a pretrained C2S-Scale model ########
# hf download vandijklab/C2S-Scale-Gemma-2-27B (39.1G) (incomplete)
# srun --mem=16G --time=2:00:00 --pty bash
# hf download vandijklab/C2S-Scale-Pythia-1b-pt --local-dir /ix/ccdg/storage3/til177/ParkLab/Project_C2S_scFM/models/pythia-1b
model_name_or_path = "/ix/ccdg/storage3/til177/ParkLab/Project_C2S_scFM/C2S_models/pythia-1b"
save_dir = "/ix/ccdg/storage3/til177/ParkLab/Project_C2S_scFM/code/tutorials/data_PerturbSeq"
save_name = "perturbation_pythia_1B"

csmodel = cs.CSModel(
    model_name_or_path=model_name_or_path,
    save_dir=save_dir,
    save_name=save_name
)
print(csmodel)

#### Fine-tune: Perturbation Prediction ####
#   For this tutorial, we'll run for a small number of steps (max_steps=500). 
#   For a full finetuning run, you would typically train for several epochs.

datetimestamp = datetime.now().strftime('%Y-%m-%d-%H_%M_%S')
output_dir = os.path.join(csmodel.save_dir, f"finetunedModel_{datetimestamp}_testFinetune_{task_name}")
print(output_dir)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

train_args = TrainingArguments( #see more in `c2s_tvl_3_FT.py`
    bf16=True,
    fp16=False,
    
    # need to modify in full run (effective batch size = 8 for now)
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    gradient_checkpointing=False, #default

    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    # warmup_ratio=0.05,          #5% straining steps are warmup steps
    load_best_model_at_end=True,

    num_train_epochs=1, # change epochs in full run

    # total steps = 1*N_train/8 ~= 24,567
    logging_steps=50,
    logging_strategy="steps",
    eval_steps=50,
    eval_strategy="steps",
    save_steps=100,
    save_strategy="steps",
    output_dir=output_dir,
    # save_total_limit=3, #limit total amount of checkpoints, keep 3
    max_steps=500  # Shortened for tutorial purposes
)

# Run Fine-tuning
csmodel.fine_tune(
    csdata=csdata,
    task=task_name,
    train_args=train_args,
    loss_on_response_only=True, # We only want to calculate loss on the predicted cell-sentence
    top_k_genes=200,  # Use top 200 genes for this example, normally would use full cell sentence (all nonzero expressed genes) if possible
    prompt_formatter=prompt_formatter  # Pass in our custom prompt formatter
)

