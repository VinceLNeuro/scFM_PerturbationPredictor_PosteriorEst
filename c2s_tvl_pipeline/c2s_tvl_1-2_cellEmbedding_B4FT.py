######## Load modules ########
from __future__ import annotations #default now for name.error issue
import os
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

# Single-cell libraries
import scanpy as sc
import anndata as ad
from collections import Counter #count table

sc.set_figure_params(dpi=300, color_map="viridis_r", facecolor="white", )
sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()

# ######## Tutorial 0 ########

# #### Load data ####
# DATA_PATH = "/ix/ccdg/storage3/til177/ParkLab/Project_C2S_scFM/code/tutorials/dominguez_conde_immune_tissue_two_donors.h5ad"
# # read the data
# adata = ad.read_h5ad(DATA_PATH)
# print(adata)
# print(Counter(adata.obs["batch_condition"]))
# print(adata.obs.head())   # observation meta-data (cells)
# print(adata.var.head())   # variable meta-data (genes)
# print(Counter(adata.obs["tissue"]))
# print(Counter(adata.obs["cell_type"]))

# # will be used to create the __cell sentences__
# print(adata.var_names)

# # check non-zero values (first 10) of the sparse matrix
# print(adata.X.data[:10])

# #### Preprocessing ####
# # basic filtering
# print(adata)
# sc.pp.filter_cells(adata, min_genes=200)
# sc.pp.filter_genes(adata, min_cells=3)
# print(adata) # check number decreased

# # annotate the group of mitochondrial genes as "mt"
# adata.var["mt"] = adata.var_names.str.startswith("MT-")
# sc.pp.calculate_qc_metrics(
#     adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
# )
# # adata = adata[adata.obs.n_genes_by_counts < 6000, :]
# # adata = adata[adata.obs.pct_counts_mt < 50, :].copy()

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

# # create a folder to store plots
# TUT0_PLOT_DIR = "/ix/ccdg/storage3/til177/ParkLab/Project_C2S_scFM/code/tutorials/output_tut0_preprocessing_umap"
# os.makedirs(TUT0_PLOT_DIR, exist_ok=True)

# # set Scanpy figure directory
# sc.settings.figdir = TUT0_PLOT_DIR
# sc.pl.umap(
#     adata,
#     color="batch_condition",
#     size=8,
#     title="Human Immune Tissue UMAP",
#     save="_batch_condition.png",# will save in current working directory unless sc.settings.figdir is set
# )
# sc.pl.umap(
#     adata,
#     color="cell_type",
#     size=8,
#     title="Human Immune Tissue UMAP",
#     save="_cell_type.png"
# )
# sc.pl.umap(
#     adata,
#     color="tissue",
#     size=8,
#     title="Human Immune Tissue UMAP",
#     save="_tissue.png"
# )

# # save data (re-done using this py file, first round using diff virtual env)
# SAVE_PATH = "/ix/ccdg/storage3/til177/ParkLab/Project_C2S_scFM/code/tutorials/dominguez_conde_immune_tissue_two_donors_preprocessed_tutorial_0_PYFILE.h5ad"
# adata.write_h5ad(SAVE_PATH)


######## Tutorial 1: Cell Sentence Conversion & Reconstruction ########
# Reload preprocessed and filtered data
DATA_PATH = "/ix/ccdg/storage3/til177/ParkLab/Project_C2S_scFM/code/tutorials/dominguez_conde_immune_tissue_two_donors_preprocessed_tutorial_0_PYFILE.h5ad"
adata = ad.read_h5ad(DATA_PATH)
print(adata)             # check dimension
print(adata.X.max())     # check max value (log10 transformation expects a maximum value somewhere around 3 or 4)
print(adata.obs.columns) # check colnames (for next step)

#### C2S conversion (AnnData -> Arrow -> CSData) ####
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
##   Check cell sentence length
len(arrow_ds[sample_idx]["cell_sentence"].split(" "))  # Cell 0 has 2191 nonzero expressed genes, yielding a sentence of 2191 gene names separated by spaces.

# Check feature info
print(type(vocabulary))
print(len(vocabulary))
print(list(vocabulary.items())[:10]) #fist 10, also contains the number of cells 'that gene' was expressed in.

# # Create CSData object (wrap around arrow dataset)
# c2s_save_dir = "/ix/ccdg/storage3/til177/ParkLab/Project_C2S_scFM/code/tutorials"  # C2S dataset will be saved into this directory
# c2s_save_name = "dominguez_immune_tissue_tutorial1"  # This will be the name of our C2S dataset on disk
# csdata = cs.CSData.csdata_from_arrow(
#     arrow_dataset=arrow_ds, 
#     vocabulary=vocabulary,
#     save_dir=c2s_save_dir,
#     save_name=c2s_save_name,
#     dataset_backend="arrow"
# )
# print(csdata)

# # View cell sentences from csdata obj
# cell_sentences_list = csdata.get_sentence_strings()
# print(len(cell_sentences_list)) # 29773
# # print(cell_sentences_list[1]) # each element is a cell_sentence
# def print_first_N_genes(cell_sentence_str: str, top_k_genes: int, delimiter: str = " "):
#     """Helper function to print K genes of a cell sentence."""
#     print(delimiter.join(cell_sentence_str.split(delimiter)[:top_k_genes]))
# print_first_N_genes(cell_sentences_list[0], top_k_genes=100)

# #### C2S benchmarking/evaluation
# output_path = os.path.join(c2s_save_dir, c2s_save_name)
# output_path
# transformation_benchmarking_save_name = "inverse_transformation_testing_tutorial_1"
# benchmark_expression_conversion(
#     benchmark_output_dir=output_path,
#     save_name=transformation_benchmarking_save_name,
#     normalized_expression_matrix=adata.X,
#     sample_size=1024, #subset
# )
# # load the metrics
# metrics_df = pd.read_csv(os.path.join(output_path, transformation_benchmarking_save_name + "_benchmark", "c2s_transformation_metrics.csv"))
# metrics_df.shape

# #### Reconstruct Cell Expression From Cell Sentences
# vocab_list = list(vocabulary.keys())
# print(len(vocab_list))
# print(len(cell_sentences_list))

# # Test for 1 cell
# expression_vector = reconstruct_expression_from_cell_sentence(
#     cell_sentence_str=cell_sentences_list[0],
#     delimiter=" ",
#     vocab_list=vocab_list,
#     slope=slope,
#     intercept=intercept,
# )
# print(type(expression_vector))
# print(expression_vector.shape)
# print(expression_vector)
# print(expression_vector.sum())

# # Reconstruct the whole data
# all_reconstructed_expression_vectors = []
# for idx in tqdm(range(len(cell_sentences_list))):
#     expression_vector = reconstruct_expression_from_cell_sentence(
#         cell_sentence_str=cell_sentences_list[idx],
#         delimiter=" ",
#         vocab_list=vocab_list,
#         slope=slope,
#         intercept=intercept,
#     )
#     all_reconstructed_expression_vectors.append(expression_vector)
# all_reconstructed_expression_vectors = np.stack(all_reconstructed_expression_vectors)
# print(all_reconstructed_expression_vectors.shape)
# ## convert to Anndata
# all_reconstructed_expression_vectors = scipy.sparse.csr_array(all_reconstructed_expression_vectors)
# print(all_reconstructed_expression_vectors)
# reconstructed_adata = ad.AnnData(
#     X=all_reconstructed_expression_vectors,
#     obs=adata.obs.copy(),
#     var=adata.var.copy()
# )
# print(reconstructed_adata)

# #### Plotting Reconstructed Expression Vectors && compare with original data
# del adata.uns
# del adata.obsm
# del adata.varm
# del adata.obsp
# # Assign column specifying data origin before rowbind
# adata.obs["c2s_data_label"]               = ["Original Data"] * adata.obs.shape[0]
# reconstructed_adata.obs["c2s_data_label"] = ["Reconstructed From Cell Sentences"] * reconstructed_adata.obs.shape[0]
# # Combine
# combined_adata = ad.concat([adata, reconstructed_adata], axis=0)
# combined_adata.obs_names_make_unique()
# # Assign var & obs
# combined_adata.var = adata.var.copy()
# combined_adata.obs = combined_adata.obs[["cell_type", "tissue", "batch_condition", "organism", "sex", "c2s_data_label"]]
# print(combined_adata)
# print(combined_adata.obs.head()) #from origin
# print(combined_adata.obs.tail()) #from c2s
# # PCA, UMAP
# sc.tl.pca(combined_adata)
# sc.pp.neighbors(combined_adata)
# sc.tl.umap(combined_adata)
# print(combined_adata) # check if successfully performed
# # # Plot side-by-side
# # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4.5))
# # sc.pl.umap(
# #     combined_adata[combined_adata.obs["c2s_data_label"] == "Original Data", :],
# #     color="tissue",
# #     size=8,
# #     title="Original Human Immune Tissue Data",
# #     show=False,
# #     ax=ax1
# # )
# # sc.pl.umap(
# #     combined_adata[combined_adata.obs["c2s_data_label"] == "Reconstructed From Cell Sentences", :],
# #     color="tissue",
# #     size=8,
# #     title="Reconstructed From Cell Sentences",
# #     show=False,
# #     ax=ax2
# # )
# # plt.tight_layout()
# # plt.show()
# # plt.close()


######## Tutorial 2: Cell Embedding with C2S Models ########
# Reload preprocessed and filtered data
# DATA_PATH = "/ix/ccdg/storage3/til177/ParkLab/Project_C2S_scFM/code/tutorials/dominguez_conde_immune_tissue_two_donors_preprocessed_tutorial_0_PYFILE.h5ad"
# adata = ad.read_h5ad(DATA_PATH)
# print(adata)             # check dimension
# adata.obs = adata.obs[["cell_type", "tissue", "batch_condition", "organism", "sex"]]

# adata_obs_cols_to_keep = ["cell_type", "tissue", "batch_condition", "organism", "sex"]
# arrow_ds, vocabulary = cs.CSData.adata_to_arrow(
#     adata=adata, 
#     random_state=SEED, 
#     sentence_delimiter=' ',
#     label_col_names=adata_obs_cols_to_keep
# )
c2s_save_dir = "/ix/ccdg/storage3/til177/ParkLab/Project_C2S_scFM/code/tutorials"
c2s_save_name = "dominguez_immune_tissue_tutorial2"
csdata = cs.CSData.csdata_from_arrow(
    arrow_dataset=arrow_ds, 
    vocabulary=vocabulary,
    save_dir=c2s_save_dir,
    save_name=c2s_save_name,
    dataset_backend="arrow"
)
print(csdata)

cell_sentences_list = csdata.get_sentence_strings()
def print_first_N_genes(cell_sentence_str: str, top_k_genes: int, delimiter: str = " "):
    """Helper function to print K genes of a cell sentence."""
    print(delimiter.join(cell_sentence_str.split(delimiter)[:top_k_genes]))
print_first_N_genes(cell_sentences_list[0], top_k_genes=200)

#### *Cells' learned embedding from pretrained transformer
#### Load a pretrained C2S model (Define CSModel object)
cell_type_prediction_model_path = "/ix/ccdg/storage3/til177/ParkLab/Project_C2S_scFM/C2S_models/C2S-Pythia-410m-diverse-single-and-multi-cell-tasks/snapshots/51f7c9d46776273ea4732ddaf494d1db733ca5d6"
save_dir = "/ix/ccdg/storage3/til177/ParkLab/Project_C2S_scFM/code/tutorials/csmodel_tutorial_2"
save_name = "cell_embedding_prediction_pythia_410M_1"
csmodel = cs.CSModel(
    model_name_or_path=cell_type_prediction_model_path,
    save_dir=save_dir,
    save_name=save_name
)

# Check inputs of the next function
print(csmodel)
print(csdata)
adata.shape[0] #29773 cells

#### Embed cells
embedded_cells = embed_cells(
    csdata=csdata,
    csmodel=csmodel,
    n_genes=200,
)
print(embedded_cells.shape) #check embedding space -> should be (29773, 1024) = (all cells, dim hidden)

# Save it for later easy loading
# Save
np.save(save_dir+"/CSModel_pythia_410M_1__cell_embeddings.npy", embedded_cells)
# # Later, load
# embedded_cells = np.load()


#### Visualize the C2S cell embeddings (UMAP)
# We will add the cell embeddings to our adata object, and rerun the neighbors and UMAP algorithms in Scanpy using the cell embeddings.
del adata.uns
del adata.obsm
del adata.varm
del adata.obsp
print(adata)
adata.obsm["c2s_cell_embeddings"] = embedded_cells #add to adata
print(adata)

# SKIP PCA (given that this latent space is already a kind of nonlinear dimension reduction -- low‑dimensional, model-learned representation)
sc.pp.neighbors(adata, use_rep="c2s_cell_embeddings")  # calculate neighbors using cell embeddings
sc.tl.umap(adata)

sc.settings.figdir = save_dir
sc.pl.umap(
    adata,
    color="cell_type",
    size=8,
    title="C2S Cell Embeddings Colored By Cell Type",
    save="_CSModel_pythia_410M_1__cell_embeddings_UMAP_CellType.png"
)
sc.pl.umap(
    adata,
    color="tissue",
    size=8,
    title="C2S Cell Embeddings Colored By Tissue",
    save="_CSModel_pythia_410M_1__cell_embeddings_UMAP_Tissue.png"
)
sc.pl.umap(
    adata,
    color="batch_condition",
    size=8,
    title="C2S Cell Embeddings Colored By Donor",
    save="_CSModel_pythia_410M_1__cell_embeddings_UMAP_Batch.png"
)


