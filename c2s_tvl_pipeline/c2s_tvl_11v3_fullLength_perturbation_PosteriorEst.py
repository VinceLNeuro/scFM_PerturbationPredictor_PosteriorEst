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


# ######## Re-load Perturbation Data ########
# DATA_PATH = "/ix/ccdg/storage3/til177/ParkLab/Project_C2S_scFM/code/tutorials/data_PerturbSeq/GSE264667_jurkat_processed.h5ad"
# adata = ad.read_h5ad(DATA_PATH)
# print(adata)             # check dimension
# # print(adata.var_names)   # will be used to create the __cell sentences__
# # print(adata.obs.columns) # check colnames (for next step)
# print(adata.X.max())     # check max value (log10 transformation expects a maximum value somewhere around 3 or 4)

# target_gene_counter = Counter(adata.obs['target_gene'])
# print(f"{len(target_gene_counter)} unique perturbations")
# print(target_gene_counter.most_common(20)) #('non-targeting', 11742)!


# ######## AnnData -> Arrow ########
# # We'll keep all relevant columns for our new task
# adata_obs_cols_to_keep = ['batch_var','cell_type','target_gene','gene_id','mitopercent','UMI_count']

# # Create Arrow dataset and vocabulary
# arrow_ds, vocabulary = cs.CSData.adata_to_arrow(
#     adata=adata, 
#     random_state=SEED, 
#     sentence_delimiter=' ',
#     label_col_names=adata_obs_cols_to_keep
# )
# print(arrow_ds)

# # # Check single-cell info
# # sample_idx = 0
# # print(arrow_ds[sample_idx])
# # ##   Check cell sentence length
# # print(len(arrow_ds[sample_idx]["cell_sentence"].split(" ")))  # Cell 0 has 4016 nonzero expressed genes
# # ##   Check feature info
# # print(type(vocabulary))
# # print(len(vocabulary))
# print(list(vocabulary.items())[:10]) #fist 10, also contains the number of cells 'that gene' was expressed in.


# ######## Custom Prompt Formatting for Perturbation Prediction ########

# #  The input provides the {control cell} and the {perturbation}, asking for the {perturbed result}.
# custom_input_prompt_template = """Given the following cell sentence of {num_genes} expressed genes representing a cell's basal state, predict the cell sentence after applying the perturbation: {perturbation_name}.
# Control cell sentence: {control_cell_sentence}.

# Perturbed cell sentence:"""

# # The answer is simply the target cell sentence.
# answer_template = "{perturbed_cell_sentence}."


# #### Create PerturbationPromptFormatter (format_hf_ds outputs: formatted HF Arrow DataSet) ####
# class PerturbationPromptFormatter(PromptFormatter):
#     def __init__(self,
#         task_name,
#         input_prompt,
#         answer_template,
#         top_k_genes, 
#         perturbation_col='target_gene',
#         control_label='non-targeting'
#     ):
#         """
#         Initializes the custom prompt formatter.

#         Args:
#             task_name (str): The name for this task.
#             input_prompt (str): The template for the model's input.
#             answer_template (str): The template for the model's expected response.
#             top_k_genes (int): The number of top genes to include in the cell sentence.
#             perturbation_col (str): The column name in the dataset that contains perturbation info.
#             control_label (str): The label used to identify control cells in the perturbation_col.
#         """
#         super().__init__()
#         self.task_name = task_name
#         self.input_prompt = input_prompt
#         self.answer_template = answer_template
#         self.top_k_genes = top_k_genes
#         self.perturbation_col = perturbation_col
#         self.control_label = control_label
#         assert top_k_genes > 0, "'top_k_genes' must be an integer > 0"

#     def format_hf_ds(self, hf_ds):
#         """
#         Custom formatting function for perturbation prediction. This function creates pairs of
#         (control, perturbed) cells by sampling from a global pool of control cells.
#         """
#         model_inputs_list = []
#         responses_list = []
        
#         # 1. Separate all cells into a global control pool and a dict of perturbed cells
#         control_indices = []
#         pert_to_indices = defaultdict(list)

#         print("Grouping cells by perturbation and identifying global controls...")
#         for i, sample in enumerate(hf_ds):
#             if sample[self.perturbation_col] == self.control_label:
#                 control_indices.append(i)
#             else:
#                 pert_to_indices[sample[self.perturbation_col]].append(i)

#             # For each cell (sample) in the dataset:
#             # If it's a control cell (e.g., target_gene == 'non-targeting'): add its index to control_indices
#             # If it's perturbed (e.g., target_gene == 'BRCA1'): add its index to the pert_to_indices dictionary under that perturbation name
        
#         assert len(control_indices) > 0, "No control cells found. Cannot create pairs."
#         print(f"Found {len(control_indices)} control cells.")
#         print(f"Found {len(pert_to_indices)} unique perturbations.")

#         # 2. Create prompt-response pairs by iterating through perturbed cells
#         print("Creating control-perturbed pairs...")
#         for pert_name, perturbed_indices in tqdm(pert_to_indices.items()):
#             for perturbed_idx in perturbed_indices:
#                 # Pair each perturbed cell with a random control cell from the global pool
#                 control_idx = random.choice(control_indices)
                
#                 control_sample = hf_ds[control_idx]
#                 perturbed_sample = hf_ds[perturbed_idx]

#                 # Format control cell sentence
#                 control_sentence, num_genes_str = get_cell_sentence_str(#https://github.com/vandijklab/cell2sentence/blob/a6efaf079f98491d4723ced44b929936b94368aa/src/cell2sentence/prompt_formatter.py#L31
#                     control_sample,
#                     num_genes=self.top_k_genes  #list slicing would NOT give out-of-bounds error -> can use total-gene-length from adata
#                                                 #cap, but may be > actual nnz genes
#                 )
#                 # Compute the true number of genes actually used in the sentence
#                 num_genes_str = str(len(control_sentence.split(" ")))

#                 # Format perturbed cell sentence
#                 perturbed_sentence, _ = get_cell_sentence_str(
#                     perturbed_sample,
#                     num_genes=self.top_k_genes
#                 )
                
#                 #### Matches the template fstring ####
#                 # Format the model input string using the perturbation name
#                 model_input_str = self.input_prompt.format(
#                     num_genes=num_genes_str,
#                     perturbation_name=pert_name,
#                     control_cell_sentence=control_sentence
#                 )
#                 # Format the response string
#                 response_str = self.answer_template.format(
#                     perturbed_cell_sentence=perturbed_sentence
#                 )

#                 model_inputs_list.append(model_input_str)
#                 responses_list.append(response_str)

#         # Create the final Hugging Face Dataset
#         ds_split_dict = {
#             "sample_type": [self.task_name] * len(model_inputs_list),
#             "model_input": model_inputs_list,
#             "response": responses_list,
#         }
#         ds = Dataset.from_dict(ds_split_dict)
#         return ds

# #### Test run to see the results (this will automatically done in `csmodel.fine_tune()`)
# # Initiate the formatter
# task_name = "perturbation_prediction"
# prompt_formatter = PerturbationPromptFormatter(
#     task_name=task_name,
#     input_prompt=custom_input_prompt_template,
#     answer_template=answer_template,
#     top_k_genes=2048 # For real applications, ideal to use all nonzero expressed genes if possible.
#                      #  (real `num_genes` figured out inside formatter)
#                      # Due to the lack of (V)RAM + NA eval_loss if using full length -> use top 2048 genes
# )
# # Format the dataset
# formatted_ds = prompt_formatter.format_hf_ds(arrow_ds)
# print(formatted_ds)
# # print(type(formatted_ds)) #datasets.arrow_dataset.Dataset
# # Inspect a formatted sample
# print("--- Formatted Sample ---")
# print("#----Model input:----#")
# print(formatted_ds[0]["model_input"], "\n")
# print("#----Response:----#")
# print(formatted_ds[0]["response"])


# ######## C2S conversion: formatted arrow_ds -> CSData ########
# # Save-directory for Huggingface dataset
# c2s_save_dir = "/ix/ccdg/storage3/til177/ParkLab/Project_C2S_scFM/code/tutorials/data_PerturbSeq"
# c2s_save_name = "jurkat_perturbation_c2s"

# csdata = cs.CSData.csdata_from_arrow(
#     arrow_dataset=arrow_ds,  # Regular cell sentence dataset put here, finetune() function will repeat the formatting with the prompt formatter
#     vocabulary=vocabulary,
#     save_dir=c2s_save_dir,
#     save_name=c2s_save_name,
#     dataset_backend="arrow"
# )
# print(csdata)


# ######## Load and Finetune a pretrained C2S-Scale model ########
# # hf download vandijklab/C2S-Scale-Gemma-2-27B (39.1G) (incomplete)
# # srun --mem=16G --time=2:00:00 --pty bash
# # hf download vandijklab/C2S-Scale-Pythia-1b-pt --local-dir /ix/ccdg/storage3/til177/ParkLab/Project_C2S_scFM/models/pythia-1b
# model_name_or_path = "/ix/ccdg/storage3/til177/ParkLab/Project_C2S_scFM/C2S_models/pythia-1b"
# save_dir = "/ix/ccdg/storage3/til177/ParkLab/Project_C2S_scFM/code/tutorials/data_PerturbSeq"
# save_name = "perturbation_pythia_1B"

# csmodel = cs.CSModel(
#     model_name_or_path=model_name_or_path,
#     save_dir=save_dir,
#     save_name=save_name
# )
# print(csmodel)

# #### Fine-tune: Perturbation Prediction ####

# datetimestamp = datetime.now().strftime('%Y-%m-%d-%H_%M_%S')
# output_dir = os.path.join(csmodel.save_dir, f"finetunedModel_{datetimestamp}_FullLengthFinetune_{task_name}")
# print(output_dir)
# if not os.path.exists(output_dir):
#     os.mkdir(output_dir)


# """
# Resources: A100 80GB (limit per-device batch size & token length) + AMD CPUs (768G, num_proc=1)
# Model    : Pythia-1B with 8192-token context limit (-> 2048 genes)

# - eff batch size = 16
# - learning_rate=1e-5  #controlling the magnitude of weight updates
#     - lr_scheduler_type = "cosine": smoothly decrease lr (after warmup) as a cosine curve
# - num_train_epochs=3

# Q1: OOM
# A1: per_device_*_batch_size=1 && a100_nvlink (80GB VRAM + 768G RAM) 

# Q_new: eval_loss: nan
# A_new: Truncation issue? Use top 2048 genes (Pythia-1B with 8192-token context limit -> 2048 genes)

# Q2: Check if too noisy loss curve (if so, need larger eff batch size)
# Q3: Check if eval_loss has no improvements over the epochs (if so, lower num_train_epochs = early stop)
# """
# train_args = TrainingArguments( #see more in `c2s_tvl_3_FT.py`
#     bf16=True,
#     fp16=False,
    
#     # eff batch size = 16
#     per_device_train_batch_size=1,
#     per_device_eval_batch_size=1,
#     gradient_accumulation_steps=16,
#     gradient_checkpointing=True, #slower computation time, but save VRAM (for longer sequence)

#     # Step 0 → step 5% of total (=warmup): LR ramps linearly from ~0 → 1e-5
#     # Step 5% → 100%: LR starts at 1e-5 and then cosine‑decays down toward 0 by the end
#     learning_rate=1e-5,
#     lr_scheduler_type="cosine",
#     warmup_ratio=0.05,          #5% straining steps are warmup steps
#     load_best_model_at_end=True,

#     num_train_epochs=3, 
#     # max_steps=500  # Shortened for tutorial purposes (not use here)

#     # total steps = epoch * N_train / effect_batch = 3*196781/16 ~= 36,897 steps
#     logging_steps=100, #much more total steps now
#     logging_strategy="steps",
#     eval_steps=500,
#     eval_strategy="steps",
#     save_steps=500,
#     save_strategy="steps",
#     output_dir=output_dir,
#     save_total_limit=3, #limit total amount of checkpoints, keep 3
# )

# # Run Fine-tuning
# csmodel.fine_tune(
#     csdata=csdata,
#     task=task_name,
#     train_args=train_args,
#     loss_on_response_only=True, # We only want to calculate loss on the predicted cell-sentence
    
#     top_k_genes=2048,                   # full length (limited VRAM)
#     prompt_formatter=prompt_formatter,  # Pass in our custom prompt formatter

#     max_eval_samples=1000, # 1000 validation samples each eval (downsample from 24k -> increase speed)
#     num_proc=1, #reduces duplicated memory (RAM) from multiprocessing
# )



# ######## Run2: Model Selection - Visualize Train/Val loss curves for the last checkpoint ####
# import json

# # 0. Setup the `best_model_checkpoint` from `trainer_state.json`
# output_dir = "/ix/ccdg/storage3/til177/ParkLab/Project_C2S_scFM/code/tutorials/data_PerturbSeq/finetunedModel_2025-12-26-04_18_42_FullLengthFinetune_perturbation_prediction/checkpoint-36500"
# state_file = os.path.join(output_dir, "trainer_state.json")

# # 1. Load the state file
# with open(state_file, "r") as f:
#     state = json.load(f)
# # Check best model
# print(state["best_model_checkpoint"])
# # Load history
# log_history = state["log_history"]

# # 2. Extract train and eval losses
# train_steps, train_losses = [], []
# eval_steps, eval_losses = [], []

# for entry in log_history:
#     # print(entry)
#     if "loss" in entry and "step" in entry: #print(entry["step"])
#         train_steps.append(entry["step"])
#         train_losses.append(entry["loss"])
#     if "eval_loss" in entry and "step" in entry:
#         eval_steps.append(entry["step"])
#         eval_losses.append(entry["eval_loss"])

# # print(train_steps == eval_steps)
# # print(train_losses)

# # 3. Plot
# plt.figure(figsize=(7, 5))
# plt.plot(train_steps, train_losses, '--.', label="train loss")
# plt.plot(eval_steps,  eval_losses,  '--.', label="eval loss")
# plt.xlabel("Steps")
# plt.ylabel("Loss: avg cross‑entropy per unmasked token/position")
# plt.title("Training and Evaluation Loss")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()

# save_path = os.path.join(output_dir, "loss_curves.png")
# plt.savefig(save_path, dpi=500)

# plt.show()
# print(f"Saved figure to {save_path}")



######## Run3: Generating predictions with the Finetuned Model ########
final_ckpt_path = "/ix/ccdg/storage3/til177/ParkLab/Project_C2S_scFM/code/tutorials/data_PerturbSeq/finetunedModel_2025-12-26-04_18_42_FullLengthFinetune_perturbation_prediction/checkpoint-36500"
save_dir        = "/ix/ccdg/storage3/til177/ParkLab/Project_C2S_scFM/code/tutorials/data_PerturbSeq"
print(final_ckpt_path)
print(save_dir)

######## Re-load Perturbation Data ########
DATA_PATH = "/ix/ccdg/storage3/til177/ParkLab/Project_C2S_scFM/code/tutorials/data_PerturbSeq/GSE264667_jurkat_processed.h5ad"
adata = ad.read_h5ad(DATA_PATH)
print(adata)             # check dimension
# print(adata.var_names)   # will be used to create the __cell sentences__
# print(adata.obs.columns) # check colnames (for next step)
print(adata.X.max())     # check max value (log10 transformation expects a maximum value somewhere around 3 or 4)

target_gene_counter = Counter(adata.obs['target_gene'])
print(f"{len(target_gene_counter)} unique perturbations")
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

# # Check single-cell info
# sample_idx = 0
# print(arrow_ds[sample_idx])
# ##   Check cell sentence length
# print(len(arrow_ds[sample_idx]["cell_sentence"].split(" ")))  # Cell 0 has 4016 nonzero expressed genes
# ##   Check feature info
# print(type(vocabulary))
# print(len(vocabulary))
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
                    num_genes=self.top_k_genes  #list slicing would NOT give out-of-bounds error -> can use total-gene-length from adata
                                                #cap, but may be > actual nnz genes
                )
                # Compute the true number of genes actually used in the sentence
                num_genes_str = str(len(control_sentence.split(" ")))

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
    top_k_genes=2048 # For real applications, ideal to use all nonzero expressed genes if possible.
                     #  (real `num_genes` figured out inside formatter)
                     # Due to the lack of (V)RAM + NA eval_loss if using full length -> use top 2048 genes
)
# Format the dataset
formatted_ds = prompt_formatter.format_hf_ds(arrow_ds)
print(formatted_ds)
# print(type(formatted_ds)) #datasets.arrow_dataset.Dataset
# Inspect a formatted sample
print("--- Formatted Sample ---")
print("#----Model input:----#")
print(formatted_ds[0]["model_input"], "\n")
print("#----Response:----#")
print(formatted_ds[0]["response"])


#### Load the finetuned model & Save the best checkpoint as new CSModel ####
finetuned_model = cs.CSModel(
    model_name_or_path=final_ckpt_path, # Path is updated after finetuning
    save_dir=save_dir,
    save_name="perturbation_predictor_FullLengthFinetuned_final"
)
print(finetuned_model.save_path)

# Loading the final finetuned model checkpoint into a regular Hugging Face AutoModelForCausalLM and moving it to GPU/CPU
final_model = AutoModelForCausalLM.from_pretrained(
    finetuned_model.save_path,
    cache_dir=os.path.join(save_dir, ".cache"), #where to store / reuse model files
    trust_remote_code=True
).to(device)
print(final_model)

# Load dataset split (done in finetune() function, saved to output directory)
ft_dir="/ix/ccdg/storage3/til177/ParkLab/Project_C2S_scFM/code/tutorials/data_PerturbSeq/finetunedModel_2025-12-26-04_18_42_FullLengthFinetune_perturbation_prediction/"
with open(os.path.join(ft_dir, 'data_split_indices_dict.pkl'), 'rb') as f:
    data_split_indices_dict = pickle.load(f)
print(data_split_indices_dict.keys())
print(len(data_split_indices_dict['train']))
print(len(data_split_indices_dict['val']))
print(len(data_split_indices_dict['test']))

# Select a few unseen samples
formatted_test_ds = formatted_ds.select(data_split_indices_dict['test'][:10])
formatted_test_ds

# Select a sample from the test set for inference
sample_idx = 0
inference_prompt      = formatted_test_ds[sample_idx]['model_input']
ground_truth_response = formatted_test_ds[sample_idx]['response']

print("--- Inference Prompt ---")
print(inference_prompt)

#### Generate the prediction ####
predicted_response = finetuned_model.generate_from_prompt(
    model=final_model, # a c2s model
    prompt=inference_prompt,
    max_num_tokens=8192 # max number of tokens to generate, ~4 tokens per gene
)
print("\n--- Ground Truth Perturbed Cell ---")
print(ground_truth_response)
print("\n--- Predicted Perturbed Cell ---")
print(predicted_response)



################ Posterior Over Responses Instead of Point Estimation ################
# Helper function
def sample_perturbation_posterior_sentences( 
    csmodel, 
    model, 
    prompt: str, 
    n_samples: int = 100, 
    max_num_tokens: int = 1024, #generate_from_prompt default
    temperature: float = 0.8, 
    top_p: float = 0.9, 
)-> list[str]: 
    """ 
    Draw multiple stochastic generations from the C2S model for a single 
    perturbation prompt. This approximates the posterior over cell sentences 
    p(cell_sentence | prompt). 
    """ 
    samples = [] 
    for _ in tqdm(range(n_samples)): 
        gen = csmodel.generate_from_prompt( 
            model=model, 
            prompt=prompt, 
            max_num_tokens=max_num_tokens, 
            # https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationConfig.temperature
            do_sample=True, # <-- sampling instead of 'greedy' or 'beam' search (Multinomial sampling)
                            #  * Sample the next token at random according to its probability.
                            #  * Stochastic, used to get many different generations from the same prompt
            temperature=temperature, # soften/sharpen distribution (default=1)
                                     # T<1 = sharper = highProb tokens have more mass = disterministic
                                     # T>1 = flatter = lowProb tokens can be sampled = diverse
            top_p=top_p, # keep smallest set with cum.prob >= top_p (0.9)
        )
        samples.append(gen) 
    
    return samples


def posterior_sentences_to_expression( 
    sentences: list[str], 
    vocab_list: list[str], 
    slope: float,     #from benchmark_expression_conversion()
    intercept: float, #from benchmark_expression_conversion()
    delimiter: str = " ", 
) -> np.ndarray: 
    """ 
    Convert a list of generated cell sentences into a matrix of reconstructed 
    expression vectors using the standard C2S inverse transform. 
    """ 
    expr_list = [] 
    for cell_sentence in sentences: 
        
        # Post-processing (tut5): clean up non-gene tokens and duplicates 
        processed_genes, _ = utils.post_process_generated_cell_sentences( 
            cell_sentence, 
            vocab_list=vocab_list, 
            replace_nonsense_string="NOT_A_GENE", 
        ) 
        processed_sentence = delimiter.join(processed_genes)
        processed_sentence = processed_sentence.replace(" NOT_A_GENE", "")  # replace nonsense string
        
        # Reconstruct
        expr_vec = utils.reconstruct_expression_from_cell_sentence( 
            cell_sentence_str=processed_sentence, 
            delimiter=delimiter, 
            vocab_list=vocab_list, 
            slope=slope, 
            intercept=intercept, 
        )
        expr_list.append(expr_vec) 
 
    return np.vstack(expr_list)  # shape: (n_samples, n_genes) 
    

######## Sampling 100 times (with two temperatures) ########
n_samples = 100
low_temp  = 0.3
high_temp = 0.8 #as v1
max_num_tokens = 2048 * 4

posterior_sentences_high_temp = sample_perturbation_posterior_sentences( 
    csmodel=finetuned_model, 
    model=final_model, 
    prompt=inference_prompt, 
    n_samples=n_samples, 
    max_num_tokens=max_num_tokens, # max number of tokens to generate, ~4 tokens per gene, 
    temperature=high_temp, 
    top_p=0.9, 
)

posterior_sentences_low_temp = sample_perturbation_posterior_sentences( 
    csmodel=finetuned_model, 
    model=final_model, 
    prompt=inference_prompt, 
    n_samples=n_samples, 
    max_num_tokens=max_num_tokens, # max number of tokens to generate, ~4 tokens per gene, 
    temperature=low_temp, 
    top_p=0.9, 
)


######## Post-processing and Reconstruction ########

vocab_list = list(vocabulary.keys())
## Benchmarking -> Obtain slope and intercept  for Reconstruction
benchmarking_save_dir = "/ix/ccdg/storage3/til177/ParkLab/Project_C2S_scFM/code/tutorials/data_PerturbSeq"
benchmarking_save_name = "perturbation_predictor_FullLengthFinetuned_final_benchmarking"
benchmarking_output_path = os.path.join(benchmarking_save_dir, benchmarking_save_name)
print(benchmarking_output_path)
os.makedirs(benchmarking_output_path, exist_ok=True)
transformation_benchmarking_save_name = "inverse_transformation_script11v3Run3"

utils.benchmark_expression_conversion(
    benchmark_output_dir = benchmarking_output_path,
    save_name = transformation_benchmarking_save_name,
    normalized_expression_matrix = scipy.sparse.csr_matrix(adata.X),
    sample_size = 1024, #number of cells to sample for computing metrics and plots
)

## Get metrics
metrics_df = pd.read_csv(os.path.join(benchmarking_output_path, transformation_benchmarking_save_name+"_benchmark", "c2s_transformation_metrics.csv"))
print(metrics_df)

slope = metrics_df.iloc[0]["slope"]
intercept = metrics_df.iloc[0]["intercept"]
print("slope:", slope)
print("intercept:", intercept)


######## Reconstructed expression vectors & calculate posterior estimation ########

print(f"#### high temp (temperature={high_temp}) expression matrix ####")
expr_samples_high_temp = posterior_sentences_to_expression( 
    sentences=posterior_sentences_high_temp, 
    vocab_list=vocab_list, 
    slope=slope,
    intercept=intercept, 
)
print(expr_samples_high_temp.shape)
# expr_samples: (n_samples, n_genes)

posterior_mean = expr_samples_high_temp.mean(axis=0)
posterior_low  = np.quantile(expr_samples_high_temp, 0.025, axis=0)
posterior_high = np.quantile(expr_samples_high_temp, 0.975, axis=0)
print(f"empirical mean = {posterior_mean}")
print(f"empirical 2.5% quantile  = {posterior_low}")
print(f"empirical 97.5% quantile = {posterior_high}")


print(f"#### low temp (temperature={low_temp}) expression matrix ####")
expr_samples_low_temp = posterior_sentences_to_expression( 
    sentences=posterior_sentences_low_temp, 
    vocab_list=vocab_list, 
    slope=slope,
    intercept=intercept, 
)
print(expr_samples_low_temp.shape)
# expr_samples: (n_samples, n_genes)

posterior_mean = expr_samples_low_temp.mean(axis=0)
posterior_low  = np.quantile(expr_samples_low_temp, 0.025, axis=0)
posterior_high = np.quantile(expr_samples_low_temp, 0.975, axis=0)
print(f"empirical mean = {posterior_mean}")
print(f"empirical 2.5% quantile  = {posterior_low}")
print(f"empirical 97.5% quantile = {posterior_high}")



######## Save the results ########
with open(os.path.join(benchmarking_output_path, "posterior_samples_meta_script11v3Run3.pkl"), "wb") as f:
    pickle.dump(
        {
            "high_temp":{
                "temperature": high_temp,
                "posterior_samples": posterior_sentences_high_temp,  # list[str]
                "expr_samples": expr_samples_high_temp,              # np.ndarray
                "prompt": inference_prompt,
                "n_samples": len(posterior_sentences_high_temp),
            },
            "low_temp":{
                "temperature": low_temp,
                "posterior_samples": posterior_sentences_low_temp,  # list[str]
                "expr_samples": expr_samples_low_temp,              # np.ndarray
                "prompt": inference_prompt,
                "n_samples": len(posterior_sentences_low_temp),
            }
        },
        f,
    )

# # To load and access later:
# import os
# import pickle
# benchmarking_wd="/ix/ccdg/storage3/til177/ParkLab/Project_C2S_scFM/code/tutorials/data_PerturbSeq/perturbation_predictor_FullLengthFinetuned_final_benchmarking"
# with open(os.path.join(benchmarking_wd, "posterior_samples_meta_script11v3Run3.pkl"), "rb") as f:
#     meta = pickle.load(f)
# # Access results
# high_temp_expr = meta["high_temp"]["expr_samples"]
# low_temp_posterior_sentences = meta["low_temp"]["posterior_samples"]

