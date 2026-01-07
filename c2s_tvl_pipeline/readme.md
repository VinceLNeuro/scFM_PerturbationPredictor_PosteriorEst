# C2S

Author: Tianze (Vincent) Luo \
Updated: 2025-12-04

__Table of Contents__

- [C2S](#c2s)
    - [0. preprocessing data](#0-preprocessing-data)
    - [1. Cell Sentence Conversion \& Reconstruction](#1-cell-sentence-conversion--reconstruction)
        - [1.1. Conversion Workflow](#11-conversion-workflow)
        - [1.2. Cell Sentence Conversion Benchmarking](#12-cell-sentence-conversion-benchmarking)
        - [1.3. Reconstruct Cell Expression Matrix From Cell Sentences](#13-reconstruct-cell-expression-matrix-from-cell-sentences)
    - [2. Cell Embedding with C2S Foundation Models _without finetuning_](#2-cell-embedding-with-c2s-foundation-models-without-finetuning)
        - [Rationale](#rationale)
        - [Workflow](#workflow)
    - [3. \[key\] Finetuning on a New Single-Cell Dataset](#3-key-finetuning-on-a-new-single-cell-dataset)
        - [Question: Why do we do cell embedding on a Pre-trained model (given that we are fine-tuning the model, why not just do cell embedding on the fine-tuned model)?](#question-why-do-we-do-cell-embedding-on-a-pre-trained-model-given-that-we-are-fine-tuning-the-model-why-not-just-do-cell-embedding-on-the-fine-tuned-model)
        - [Workflow](#workflow-1)
        - [Output](#output)
    - [4. Cell Type Prediction/Annotation (Using Fine-Tuned Model)](#4-cell-type-predictionannotation-using-fine-tuned-model)
        - [Workflow](#workflow-2)
    - [7. \[key\] Custom prompt templates (for PromptEngineering)](#7-key-custom-prompt-templates-for-promptengineering)
    - [10. \[key\] Finetuning for Perturbation Response Prediction](#10-key-finetuning-for-perturbation-response-prediction)
        - [Rationale](#rationale-1)
        - [Workflow](#workflow-3)
    - [11. \[key\] Posterior Over Responses Instead of Point Estimation](#11-key-posterior-over-responses-instead-of-point-estimation)
    - [11v2/v3. \[key\] Sampling 100 times for 1 inference prompt (comparing with two temperatures)](#11v2v3-key-sampling-100-times-for-1-inference-prompt-comparing-with-two-temperatures)
        - [Caveat](#caveat)
        - [Use top\_k\_genes=200 model for visualization to quality check if temperature difference during generation will create a different distribution of the samples (i.e., higher temperature would have a more spread distribution on the shared UMAP)](#use-top_k_genes200-model-for-visualization-to-quality-check-if-temperature-difference-during-generation-will-create-a-different-distribution-of-the-samples-ie-higher-temperature-would-have-a-more-spread-distribution-on-the-shared-umap)
    - [Other Optional Tasks](#other-optional-tasks)
        - [5. Cell Generation](#5-cell-generation)
        - [6. Cell Type Annotation with C2S Foundation Model](#6-cell-type-annotation-with-c2s-foundation-model)
        - [8 \& 9 \[not yet implemented\] Multi-cell](#8--9-not-yet-implemented-multi-cell)

<br>

## 0. preprocessing data

- __~30k cells (2 donors), ~36k genes__ from the Domínguez Conde et al. study
- 5'-capture (TCR/BCR)

- gene names in `adata.var_names` --> will be used to create the __cell sentences__ by the C2S code base functions later on
- should start from __raw counts__ (counts, not continuous (normalized)).

<br>

Step1. Data processing + Normalization: C2S only deviates from the standard preprocessing and normalization pipeline in that the __log transformation is done with a base of 10__ rather than natural logarithm

Step2. PCA > gKNN (nPC=50) > calcUMAP > plotUMAP

<!-- `md5sum dominguez_conde_immune_tissue_two_donors_preprocessed_tutorial_0.h5ad dominguez_conde_immune_tissue_two_donors_preprocessed_tutorial_0_PYFILE.h5ad`: same two files -->

<br>


## 1. Cell Sentence Conversion & Reconstruction

### 1.1. Conversion Workflow

1. `AnnData` object: containing our single-cell dataset 
2. `Huggingface PyArrow dataset` (`arrow_ds`: cell meta, cell sentence) && `vocabulary` (vocabulary/features/genes)
   - cs.CSData.adata_to_arrow
3. `CSData` object: wraper of arrow dataset; data for inference or finetuning
   - cs.CSData.csdata_from_arrow
   - `/ix/ccdg/storage3/til177/ParkLab/Project_C2S_scFM/code/tutorials/dominguez_immune_tissue_tutorial1`

<br>

`arrow_ds`

- 29,773 cells have now been converted into __rows__ of a Dataset object with an additional `cell_sentence` column 
    - The cell sentence contains a sentence of gene names ordered by __descending expression level__, giving a <mark>rank-based gene name representation of the cell.
    - Each row is a dict for a cell
    - e.g.
      ```
      {'cell_name': 'Pan_T7935490_AAACCTGCAAATTGCC',
       'cell_sentence': 'RPLP1 ACTB EEF1A1 HSP90AA1 TMSB4X B2M FTH1 KLF6 HSPA1B MALAT1 RPS12 HSPA8 RPL13 MT-CO1 ATF3 MT-CO2 RPL41 TPT1 MT-CO3 ..., ...}
      ```
    
`vocabulary`

- vocabulary is an OrderedDict of __gene features__, corresponding to the original 23944 genes in our adata object. The OrderedDict denotes the gene features present in our single-cell dataset, and also stores the number of cells that gene was expressed in.
- e.g.
  ```
  [('RP11-34P13', 38),
   ('RP11-34P13-3', 106),
    ...
  ]
  ```

<br>

### 1.2. Cell Sentence Conversion Benchmarking

Aim: Know how well the conversion did, and how much expression information was lost when we switched to a rank ordering of genes rather than exact expression values.

- Paper Fig 10: __linear relationship__ was found between the Log-Rank of a gene and its Log-Norm expression value

- `benchmark_expression_conversion()`
    - Fit a <mark>linear model</mark> on the ranks and expression of the original data, which can be used to <mark>reconstruct expression from rank
    - Save plots of (1) log rank vs log expression and (2) log expression vs reconstructed expression from rank

    <div style="display: flex; align-items: flex-start; gap: 10px;">
    <img src="../dominguez_immune_tissue_tutorial1/inverse_transformation_testing_tutorial_1_benchmark/normalized_expression_vs_log_rank.png" 
        alt="logNorm vs logRank" 
        style="width: 49%;">
    <img src="../dominguez_immune_tissue_tutorial1/inverse_transformation_testing_tutorial_1_benchmark/reconstructed_expression.png" 
        alt="Reconstructed Expression (from cs) vs Original Expression" 
        style="width: 49%;">
    </div>

<br>

### 1.3. Reconstruct Cell Expression Matrix From Cell Sentences

`reconstruct_expression_from_cell_sentence()`
- Need 
    - cell_sentences_list (from csdata)
    - vocab_list
    - benchmarking slope & intercept

- Predict logNorm expression vector && Convert back to `Anndata`
    - predicted_expression = intercept + (slope * log(rank_of_gene))

- Very successful inverse transformtion, in terms of rebuilding expression_vector -> rebuilding anndata -> UMAP comparison

![umap_comparison.png](../dominguez_immune_tissue_tutorial1/inverse_transformation_testing_tutorial_1_benchmark/umap_comparison.png)

<br>


## 2. Cell Embedding with C2S Foundation Models _without finetuning_

### Rationale

- By loading/defining a `CSModel` object (pretrained model), the model parameters are __completely frozen__ during embedding extraction. The model simply performs <mark> a full forward pass through the pretrained transformer to obtain the __last-layer learned latent hidden states__ </mark>, which are then __average pooled the latents__ into a single embedding vector per cell.
    
    - manuscript BioRxiv L448
    - 1024-d

- Note: given the pretraining process is using large-scale data across batches/datasets with many tissues, donors, cell types --> the forward pass will show biological structure and <mark>implicitly reduce batch effect</mark>. 

- By converting cells into learned embeddings, we create a compact representation that captures the essential information from the cell sentences. Cell embeddings are crucial for downstream tasks such as clustering, visualization, and classification

### Workflow

1. Reload preprocessed immune tissue single-cell dataset (preprocessed in tutorial notebook 0, two sample donors) -> create a CSData() wrapper around it

2. Load a __pretrained C2S model__ (preferably, C2S models which have been trained to do cell type or tissue prediction) to create a `CSModel` object.
    
    - Check here for models to use (not include the 2-27B Gemma model): https://github.com/vandijklab/cell2sentence?tab=readme-ov-file#model-zoo
        - `vandijklab/C2S-Pythia-410m-diverse-single-and-multi-cell-tasks`
        - downloaded @ /ihome/hpark/til177/.cache/huggingface/hub/models--vandijklab--C2S-Pythia-410m-diverse-single-and-multi-cell-tasks/snapshots/51f7c9d46776273ea4732ddaf494d1db733ca5d6/README.md

3. Embed the cells using the specific C2S model
    - `embed_cells()`
    - Input: `CSData`, `CSModel`,number of genes to use per cell sentence
    - How it worked ([details here](https://github.com/vandijklab/cell2sentence/blob/master/src/cell2sentence/tasks.py)):
        - Load the C2S model and data
        - Format the cell sentences into prompts for task="cell_type_prediction" (same as a later task)
        - Run the prompts through the C2S model
        - <mark> Uses the internal hidden states (via `csmodel.embed_cells_batched`) as cell embeddings, instead of sampling generated text (normal task="cell_type_prediction")


4. Visualize the cell embeddings to gain insights into the data

    - Steps: 
        - __SKIP PCA__ (given that this latent space is already a kind of nonlinear dimension reduction -- low‑dimensional, model-learned representation)
        
        - Do gKNN construction -> UMAP  

    - >Results:
      >  - Retain distinct clusters separated by cell type and tissue (i.e., cells with similar expression programs and similar cell types should end up close to each other in this refined-learned-representation space) --> _all over the place is a warning message_
      >  - Similar cell-type clustering patterns in cell embedding UMAP, when comparing with original UMAP
      >  - <mark>Batch being corrected

        <br>

        <div style="display: flex; align-items: flex-start; gap: 10px;">
            <img src="../csmodel_tutorial_2/umap_CSModel_pythia_410M_1__cell_embeddings_UMAP_Batch.png" 
                alt="Cell embedding colored by batch" 
                style="width: 49%;">
            <img src="../csmodel_tutorial_2/umap_CSModel_pythia_410M_1__cell_embeddings_UMAP_Tissue.png" 
                alt="Cell embedding colored by tissue" 
                style="width: 49%;">
        </div>

        <br>

        Original UMAP
        ![](../output_tut0_preprocessing_umap/umap_cell_type.png)

        Cell Embedding UMAP
        ![](../csmodel_tutorial_2/umap_CSModel_pythia_410M_1__cell_embeddings_UMAP_CellType.png)

<br>


## 3. [key] Finetuning on a New Single-Cell Dataset

### Question: Why do we do cell embedding on a Pre-trained model (given that we are fine-tuning the model, why not just do cell embedding on the fine-tuned model)?

<mark> Use Pre-trained model (FM) as "Initialization" / __"feature extractor"__ (better than PCA) </mark>

- `M_pre` has already learned a good representation of immune cells (basic cell types, marker relationships, etc.).

- Those embeddings are richer and more biologically meaningful __than raw gene counts or PCA__, especially if `M_pre` saw a huge amount of data.

- So, embeddings from `M_pre` may be used as a <mark>fixed reference space:
    - For QC (outlier detection, batch inspection)
    - For mapping new datasets into the same space later.

<br>

Fine‑tuning

- <mark>Next-token prediction objective (as pretraining) with __prompts formatted to match each task__</mark> 
  --> so, even if the cells are from the same dataset, the loss function and goal are different.

- Code-wise: same underlying loss function for all tasks (Hugging Face causal LM cross‑entropy on tokens).
    - <mark>What changes by task is:</mark>
        - What you ask the model to do in the prompt (model_input).
        - What you treat as the “ground‑truth” response (response).
        - Whether you take loss on response only vs prompt + response.

<br>


### Workflow

1. Load an preprocessed immune tissue single-cell dataset (two sample donors)

2. (Optional) Custom Prompt Formatting: Format the dataset using a `CustomPromptFormatter` object, which prepares the data for the fine-tuning process.

3. (get CSData & CSModel) Load a pretrained C2S model.

4. Fine-tune the C2S model to improve its performance on cell type prediction.

    - Specify `training_task` (Possible values for the training task parameter can be found in the `prompt_formatter.py` file in the source code, under `SUPPORTED_TASKS`)
        - [Click here for github code](https://github.com/vandijklab/cell2sentence/blob/master/src/cell2sentence/prompt_formatter.py)
            
            ```python
            SUPPORTED_TASKS = [
                "cell_type_prediction",
                "cell_type_generation",
            ]
            MULTICELL_SUPPORTED_TASKS = [
                "tissue_prediction",
                "tissue_conditional_generation",
                "natural_language_interpretation",
            ]
            ```
    - `TrainingArguments`: https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments
        - see the codes for details

    - `csmodel.fine_tune()` (https://github.com/vandijklab/cell2sentence/blob/master/src/cell2sentence/csmodel.py > L77)
        
        - Format prompt from custom or pre-defined (`C2SPromptFormatter(task=task, top_k_genes=top_k_genes)`)
            - https://github.com/vandijklab/cell2sentence/blob/master/src/cell2sentence/prompt_formatter.py#L69
            - https://github.com/vandijklab/cell2sentence/blob/master/src/cell2sentence/prompts/single_cell_cell_type_prediction_prompts.json
        - `.format_hf_ds`
            - https://github.com/vandijklab/cell2sentence/blob/a6efaf079f98491d4723ced44b929936b94368aa/src/cell2sentence/prompt_formatter.py#L113
                ```py
                # output {task, prompt+cellsentence(input), response} for finetuning
                ds_split_dict = {
                    "sample_type": [self.task] * hf_ds.num_rows,
                    "model_input": model_inputs_list,
                    "response": responses_list,
                }
                ds = Dataset.from_dict(ds_split_dict)
                return ds
                ```

        - Tokenize

        - Perform internal train/val/test split (80/10/10) [@L187](https://github.com/vandijklab/cell2sentence/blob/a6efaf079f98491d4723ced44b929936b94368aa/src/cell2sentence/csmodel.py#L187)
            - helper function `train_test_split_arrow_ds` [Click here](https://github.com/vandijklab/cell2sentence/blob/a6efaf079f98491d4723ced44b929936b94368aa/src/cell2sentence/utils.py#L331)

        - <mark>__Training scheme__: a causal LM using __cross‑entropy on the next token__</mark>
            - <span style="color:red"> (Tutorial 4) Final testing / evaluation metrics : __accuracy__ of the predicted __response tokens__

### Output

Training time (GPU): 4296 seconds

- Model_step3600: {'eval_loss': 1.392207384109497}
- Model_step3700: {'eval_loss': 1.3921808004379272}
- \*__Model_step3725: {'eval_loss': 1.3921139240264893}__

    <img src="../csmodel_tutorial_3/2025-11-24-18_42_05_finetune_cell_type_prediction/checkpoint-3725/loss_curves.png" alt="see ../csmodel_tutorial_3/2025-11-24-18_42_05_finetune_cell_type_prediction/checkpoint-3725/loss_curves.png" style="width: 80%;">

<br>


## 4. Cell Type Prediction/Annotation (Using Fine-Tuned Model)

### Workflow

1. Load an preprocessed immune tissue single-cell dataset (two sample donors)

2. Load __fine-tuned CSModel (training_task = "cell_type_prediction")__ CHECKPOINT
    - Load the `data_split_indices_dict.pkl` associated with the fine-tuned model 
        - Containing indices
        - train/val/test split = 80/10/10

3. C2S conversion (AnnData [full] -> Arrow [full] -> CSData [__test only__])

4. INFERENCE: Predict Cell Types
    - Input: finetuned cell type prediction model `CSModel` and have our test set `CSData`
    - `predict_cell_types_of_data()`

5. Calculate accuracy
    - _accuracy = 0.81_

<br>


## 7. [key] Custom prompt templates (for PromptEngineering)

1. Load an preprocessed immune tissue single-cell dataset (two sample donors)

2. AnnData -> Arrow

3. <mark>Custom Prompt Formatting</mark>
    
    - Create a subclass of the [`PromptFormatter` class](https://github.com/vandijklab/cell2sentence/blob/master/src/cell2sentence/prompt_formatter.py), which has to define a `format_hf_ds` method that takes in a cell sentence __arrow dataset__ and returns a __formatted dataset__ 

    - `prompt_formatter = CustomPromptFormatter()`

    - > It can be beneficial to provide __variations of prompt templates to provide some diversity__ - simply create several templates and choose one when formatting each sample in the formatting function!

    - <font color="red">Note: `csmodel.fine_tune(prompt_formatter=prompt_formatter)` function will do the formatting on the full dataset for us</font> --> no need to do formatting here

        ```
        # Example Arrow
        Dataset({
            features: ['cell_name', 'cell_sentence', 'cell_type', 'tissue', 'batch_condition', 'organism', 'sex'],
            num_rows: 10
        })
        
        # Example Formatted Arrow
        Dataset({
            features: ['sample_type', 'model_input', 'response'],
            num_rows: 10
        })
        ```

4. Arrow -> CSData

5. Load C2S FM (https://huggingface.co/collections/vandijklab/cell2sentence-models)

6. Fine-tune on new task!

<br>


## 10. [key] Finetuning for Perturbation Response Prediction

### Rationale

__Perturbation Response__: 

> how a cell's gene expression profile changes in response to a specific perturbation (e.g., a genetic knockout or a drug treatment)

> We will treat this as a "translation" task in natural language: __translating a cell__ (in cell sentence format) from its basal <mark>(control) state to its perturbed state, conditioned on the perturbation applied</mark>.

### Workflow

At a high level, we will:

1. Load a public single-cell __perturbation dataset__.
    - Data requirement
        - AnnData object
        - `.obs` dataframe must contain: 
            - A column that distinguishes control cells from perturbed cells, e.g., adata.obs['condition']
    
    - __Data used in this analysis__
        - Original Paper: https://www-nature-com.pitt.idm.oclc.org/articles/s41588-025-02169-3#Sec10
            
            - >"A second CRISPRi Jurkat cell line expressing the optimized UCOE-EF1α-Zim3-dCas9-P2A-mCherry CRISPRi construct was generated as previously described and was used for Perturb-seq."
                - Same cell line
            - >"For the Jurkat Perturb-seq experiment, Jurkat cells expressing Zim3-dCas9-P2A-mCherry were transduced with dJR092 library lentivirus by spinfection (1,000g) with polybrene (8 µg ml−1; Sigma-Aldrich) with a targeted low infection rate of ~10%. __This low rate was chosen to reduce the chances of a single cell being infected by several viruses__."
                - Controlled the doses intentionally
        
        - Perturb-seq data/experiments in Jurkat cells: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE264667
            
            - Different cells get different sgRNAs (targeting different genes or control sgRNAs) --> Pooled all the edited cells together to do scRNA-seq
            
            - <span style="color:red">The downloaded data (`/ix/ccdg/storage3/til177/ParkLab/Project_C2S_scFM/code/tutorials/data_PerturbSeq/GSE264667_jurkat.h5ad`) already passed filtering used in original paper</span>
                
                - Filtering in original paper -> 262,956 cells retained
                    1. min_umi >1,750
                    2. max_mito = 14

    - Our basic filtering did not remove cells/features; main concern is sequencing depth -> used advanced filtering (obs: 262956 -> 257412)
        ```py
        # For Jurkat Perturb-seq with median UMI ~10k
        min_umi = 1000    # ~10% of median
        max_umi = 40000   # ~4x median (doublet filter)
        min_genes = 500
        max_genes = 6000
        max_mito = 15

        # Apply filters
        adata = adata[
            (adata.obs['UMI_count'] > min_umi) &
            (adata.obs['UMI_count'] < max_umi) &
            (adata.obs['n_genes'] > min_genes) &
            (adata.obs['n_genes'] < max_genes) &
            (adata.obs['mitopercent'] < max_mito)
        ].copy()
        ```

        - <span style="color:red">[Added 12/11/2025] Filtering functioning here are: 
            - max_umi = 40000
            - max_genes = 6000

    - Normalization


2. Write a __custom prompt template__ for perturbation prediction.

    - Subclass the `PromptFormatter` class (ABC) to create pairs of control and perturbed cells.
        - `format_hf_ds` method (__will be applied automatically in `csmodel.fine_tune()`__)
        - Output: _formatted HF Dataset_
        - <mark>Note</mark>: Using __top 200 genes__ for this example. For real applications, ideal to use __all nonzero expressed genes__ if possible.


3. Load and __Finetune__ a pretrained C2S-Scale model on this new task.
    - <mark>Note</mark>: For this tutorial, we'll run for a small number of steps (max_steps=500). For a full finetuning run, you would typically train for __several epochs__.
    
    - `loss_on_response_only=True`: use input 'control cell' & 'purturbation gene name' as _condition_ (`p( perturbed_cell | control_cell, perturbation )`)
        - __We only want to compute loss on the predicted perturbed cell sentence__
        - Do not need to waste resources on the control cell-sentence prediction

4. [`c2s_tvl_11`] Generate a __prediction__ with our new finetuned model to see it in action. 


<br>


## 11. [key] Posterior Over Responses Instead of Point Estimation

See `c2s_tvl_11_perturbation_ftEval_PosteriorEst.ipynb` for details

Created two helper functions for empirical statistics calculation

- sample_perturbation_posterior_sentences()
- posterior_sentences_to_expression()

<br>

## 11v2/v3. [key] Sampling 100 times for 1 inference prompt (comparing with two temperatures)

Data stored at `/ix/ccdg/storage3/til177/ParkLab/Project_C2S_scFM/code/tutorials/data_PerturbSeq/perturbation_predictor_finetuned_final_benchmarking/posterior_samples_meta_script11v2.pkl`

- > (see 'c2s_tvl_11v2_perturbation_PosteriorEst.py')

### Caveat

If using `top_k_genes=200` but not the whole list of genes (e.g., _in v2 `PerturbationPromptFormatter`_), then `reconstruct_expression_from_cell_sentence()` will [create zeros for those genes NOT among the top_k_genes](https://github.com/vandijklab/cell2sentence/blob/a6efaf079f98491d4723ced44b929936b94368aa/src/cell2sentence/utils.py#L446C5-L446C46). 

### Use top_k_genes=200 model for visualization to quality check if temperature difference during generation will create a different distribution of the samples (i.e., higher temperature would have a more spread distribution on the shared UMAP)

<img src="/ix/ccdg/storage3/til177/ParkLab/Project_C2S_scFM/code/tutorials/data_PerturbSeq/perturbation_predictor_finetuned_final_benchmarking/PCA__top_k_genes200_PosteriorSamplesByTemp.png" alt="see perturbation_predictor_finetuned_final_benchmarking/PCA__top_k_genes200_PosteriorSamplesByTemp.png" width=80%>
<img src="/ix/ccdg/storage3/til177/ParkLab/Project_C2S_scFM/code/tutorials/data_PerturbSeq/perturbation_predictor_finetuned_final_benchmarking/umap__top_k_genes200_PosteriorSamplesByTemp.png" alt = "see perturbation_predictor_finetuned_final_benchmarking/umap__top_k_genes200_PosteriorSamplesByTemp.png" width=80%>

<br>

__In v3, we switched to (full-length) top 2048 genes__ (Pythia-1B with 8192-token context limit -> 2048 genes)

<!-- - Expected: some inference prompts have <8882 expressed genes (in cell sentence)

- Outcome :
    - \*2569\* expressed genes in `inference_prompt`
    - >`print(inference_prompt)`\
      >\>"Given the following cell sentence of \*8882\* expressed genes representing a cell's basal state, predict the cell sentence after applying the perturbation: NELFE." -->

- best_model_checkpoint: `/ix/ccdg/storage3/til177/ParkLab/Project_C2S_scFM/code/tutorials/data_PerturbSeq/finetunedModel_2025-12-26-04_18_42_FullLengthFinetune_perturbation_prediction/checkpoint-36500`

    - No overfitting
    - Seems that the loss curve is too noisy            --> might need larger eff. batch size
    - The loss curve decreases slower after 15k steps   --> may consider early stop
    <img src="/ix/ccdg/storage3/til177/ParkLab/Project_C2S_scFM/code/tutorials/data_PerturbSeq/finetunedModel_2025-12-26-04_18_42_FullLengthFinetune_perturbation_prediction/checkpoint-36500/loss_curves.png" alt="see loss_curves_FullLength.png">

- <mark>Problematic in the generation step </mark>
    - The top_k_genes=2048 model did not exceed the 8192 context length. However, the generation length is limited given the context length cap. 
    
    - >prompt_len_tokens: 7015 model_max: 8192 max_new_tokens: 1113
        
        - The current generation script is beyond the `model_max` (total=7015+8192) --> The `posterior_samples_meta_script11v3Run3.pkl` is low quality from the PCA/UMAP (expected degraded beyond model limit) 
        
        - Truncation (to within the model limit) improves the result but still invalid... -> __should regenerate if want to use the top_k_genes=2048 model__

<!-- - <mark>TODO: model evaluation</mark>
    - Benchmarking: https://www-nature-com.pitt.idm.oclc.org/articles/s41592-025-02980-0.pdf
    - GEARS (Roohani et al., Nature Biotechnology 2024) 
        - DEGs pearson's corr -->



<br>

## Other Optional Tasks

### 5. Cell Generation

Workflow

- Similar pipeline
- Need a model __finetuned to do cell generation__ (from tutorial 3)
- `generate_cells_conditioned_on_cell_type()`
- Post-processing and Reconstruction
    - Remove words that are not gene names in the dataset (vocabulary)
    - Deal with duplicated genes
    - `post_processed_sentence, num_genes_replaced = post_process_generated_cell_sentences()`
    - Reconstruction (see tutorial 1)
- Visualization
    - Compare the UMAP seperately
    - Compare heatmap
    - Plot in shared space --> should be very similar in the shared embedding space

### 6. Cell Type Annotation with C2S Foundation Model

Workflow very similar to tutorial 4


### 8 & 9 [not yet implemented] Multi-cell


