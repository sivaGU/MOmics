# MOmics Multi-Omics DIABLO Integration
#
# NOTE: data paths below (data/metabolome_pnnl.v4.0.tsv, data/all_subtypes.v5.1.tsv,
# data/metabolites/..., data/msea_ora_result.csv) predate the repo reorg and need to
# be pointed at data/discovery/ (and wherever the metabolite MSEA/ORA results live)
# before this will run end-to-end.


# ----------------------------------------------------
# Multi-omics Integration
# ----------------------------------------------------
# ----------------------------------------------------
# SECTION 0: Setup and Environment
# ----------------------------------------------------
# Core R packages for data handling and plotting
library(tidyverse)
library(readr)
library(dplyr)
library(tidyr)
library(caret)
library(corrplot)
library(ggplot2)

# Core packages for Multi-omics Integration and gene mapping
library(mixOmics) 
library(biomaRt) 
library(pbapply) 

# --- NEW: Packages for R-based KEGG/GO Enrichment---
library(org.Hs.eg.db) 
library(clusterProfiler)
library(AnnotationDbi) 

options(pboptions = list(type="timer"))


# --- CRITICAL: Set seed for reproducible results (e.g., in tuning and validation) ---
set.seed(42) 

cat("Running final publication-standard pipeline. Max iterations set to 1500.\n")

# ----------------------------------------------------
# SECTION 1: Data Loading and Preprocessing & IMPUTATION
# ----------------------------------------------------
cat("\n--- 1. Loading and Preprocessing Data ---\n")

# Load raw data 
metabolomics_data <- read_tsv("data/metabolome_pnnl.v4.0.tsv")
proteomics_data <- read_tsv("data/proteome_mssm_per_gene_imputed.v4.0.tsv")
transcriptomics_data <- read_tsv("data/rnaseq_washu_readcount.v4.0.tsv")
sample_info_data <- read_tsv("data/all_subtypes.v5.1.tsv")

# --- 1.1 Metabolomics: Filter, Wide Format, Impute ---
metabolomics_df_long <- metabolomics_data %>%
  rename(feature = Metabolite) %>%
  pivot_longer(cols = -feature, names_to = "case_id", values_to = "value")

metabolomics_df_wide <- metabolomics_df_long %>%
  pivot_wider(names_from = feature, values_from = value, id_cols = case_id)

met_data_matrix <- as.matrix(metabolomics_df_wide[, -1])
met_variances <- apply(met_data_matrix, 2, var, na.rm=TRUE) 
N_select_met <- min(500, length(met_variances)) 
top_met_features <- names(sort(met_variances, decreasing = TRUE))[1:N_select_met]

metabolomics_df_final <- metabolomics_df_wide %>% dplyr::select(case_id, all_of(top_met_features))

# Median Imputation
metabolomics_df_final[-1] <- lapply(metabolomics_df_final[-1], function(col) {
  col <- as.numeric(col)
  if (all(is.na(col))) return(rep(0, length(col))) 
  col[is.na(col)] <- median(col, na.rm = TRUE)
  return(col)
})

# --- 1.2 Proteomics: Transpose, Impute, Filter (Top 2000) ---
gene_names <- proteomics_data[[1]]
prot_mat <- t(as.matrix(proteomics_data[, -1]))
colnames(prot_mat) <- gene_names
rownames(prot_mat) <- gsub("\\.", "-", colnames(proteomics_data)[-1])
proteomics_data_final <- as_tibble(prot_mat, rownames = "case_id")
proteomics_data_final[-1] <- lapply(proteomics_data_final[-1], function(col) {
  col <- as.numeric(col)
  col[is.na(col)] <- median(col, na.rm = TRUE)
  return(col)
})
prot_data_matrix <- as.matrix(proteomics_data_final[, -1])
prot_variances <- apply(prot_data_matrix, 2, var)
top_prot_features <- names(sort(prot_variances, decreasing = TRUE))[1:2000]
proteomics_data_final <- proteomics_data_final %>% dplyr::select(case_id, all_of(top_prot_features))

# --- 1.3 Transcriptomics: Transpose, Impute, Log-Transform, Filter (Top 5000) ---
cat("\nProcessing Transcriptomics (with Log2 Transformation)...\n")

gene_names_rna <- transcriptomics_data[[1]]
rna_mat <- t(as.matrix(transcriptomics_data[, -1]))
colnames(rna_mat) <- gene_names_rna
rownames(rna_mat) <- gsub("\\.", "-", colnames(transcriptomics_data)[-1])

transcriptomics_data_final <- as_tibble(rna_mat, rownames = "case_id")

# 1. Median Imputation first
transcriptomics_data_final[-1] <- lapply(transcriptomics_data_final[-1], function(col) {
  col <- as.numeric(col)
  if (all(is.na(col))) col <- rep(0, length(col))
  col[is.na(col)] <- median(col, na.rm = TRUE)
  return(col)
})

# 2. CRITICAL: Log2 Transformation (Fixes the 3.3 million max value issue)
# We add +1 to avoid errors with zero values (log(0) is infinite)
transcriptomics_data_final[-1] <- log2(transcriptomics_data_final[-1] + 1)
cat("Log2 Transformation applied. Data range is now suitable for DIABLO.\n")

# 3. Variance Filtering (On the Logged Data)
rna_data_matrix <- as.matrix(transcriptomics_data_final[, -1])
rna_variances <- apply(rna_data_matrix, 2, var)
N_select_rna <- min(5000, length(rna_variances)) 
top_rna_features <- names(sort(rna_variances, decreasing = TRUE))[1:N_select_rna]
transcriptomics_data_final <- transcriptomics_data_final %>% dplyr::select(case_id, all_of(top_rna_features))

# 4. Remove nearZeroVar (Final cleanup)
nzv_cols <- caret::nearZeroVar(transcriptomics_data_final[-1], names = TRUE)
if(length(nzv_cols) > 0) {
  transcriptomics_data_final <- dplyr::select(transcriptomics_data_final, -any_of(nzv_cols))
}
cat("Transcriptomics processing complete.\n")


# ----------------------------------------------------
# SECTION 2: Data Alignment, Blocks, and Stabilization
# ----------------------------------------------------

# Align all data blocks by 'case_id'
aligned_data <- sample_info_data %>%
  rename(case_id = case) %>%
  dplyr::select(case_id, sample_type) %>%
  inner_join(metabolomics_df_final, by = "case_id") %>%
  inner_join(proteomics_data_final, by = "case_id") %>%
  inner_join(transcriptomics_data_final, by = "case_id")

# Create omics data blocks (matrices for DIABLO)
data_blocks <- list(
  metabolomics = as.matrix(aligned_data[, colnames(metabolomics_df_final)[-1]]),
  proteomics = as.matrix(aligned_data[, colnames(proteomics_data_final)[-1]]),
  transcriptomics = as.matrix(aligned_data[, colnames(transcriptomics_data_final)[-1]])
)

for (i in seq_along(data_blocks)) {
  rownames(data_blocks[[i]]) <- aligned_data$case_id
}

# Outcome variable (Y): Relabel and set as factor
Y_outcome_relabeled <- gsub("GTEx normal", "normal", aligned_data$sample_type) 
Y_outcome <- as.factor(Y_outcome_relabeled) 
names(Y_outcome) <- aligned_data$case_id

# --- FINAL STABILIZATION AND ZERO-VARIANCE REMOVAL (CRITICAL FIX) ---
cat("\n--- Running Final Stabilization and Feature Cleanup ---\n")
data_blocks_final_stable <- list()

for (name in names(data_blocks)) {
  #  CRITICAL STABILIZATION STEP (Guaranteed Zero-Variance Removal) 
  Y_dummy <- factor(rep(c("A", "B"), length.out = nrow(data_blocks[[name]])))
  
  cleaned_result <- plsda(
    X = data_blocks[[name]], 
    Y = Y_dummy, 
    ncomp = 1,
    near.zero.var = TRUE # Forces the internal cleaning
  )
  data_blocks_final_stable[[name]] <- cleaned_result$X
  
  # Final Variance Test
  if (sum(apply(data_blocks_final_stable[[name]], 2, var, na.rm=TRUE) == 0) > 0) {
    stop(paste("ERROR: Block", name, "still contains zero-variance features AFTER stabilization."))
  } else {
    cat(paste(" PASS: Block", name, "is fully stabilized and clean. Dimensions:", dim(data_blocks_final_stable[[name]])[1], "x", dim(data_blocks_final_stable[[name]])[2], "\n"))
  }
}

data_blocks <- data_blocks_final_stable 
Y_outcome <- Y_outcome[names(Y_outcome) %in% rownames(data_blocks[[1]])] 

# ----------------------------------------------------
# SECTION 3: DIABLO Model Setup (High Correlation Design)
# ----------------------------------------------------

# HIGH CORRELATION FIX: Set DIABLO Design Matrix to 0.9
integration_weight <- 0.9 
design <- matrix(integration_weight, ncol = length(data_blocks), nrow = length(data_blocks))
diag(design) <- 0
colnames(design) <- rownames(design) <- names(data_blocks)

cat(paste("\n--- Design Matrix Set for High Integration (Weight =", integration_weight, ") ---\n"))
print(design)

# ----------------------------------------------------
# SECTION 4: Model Tuning (Optimal Feature Selection)
# ----------------------------------------------------
cat("Starting 20 repeats (sequential run) to choose optimal features.\n")

# Grid of features to test
list.keepX <- list(
  metabolomics = c(5, 10, 25, 50, 100),
  proteomics = c(5, 10, 25, 50, 100),
  transcriptomics = c(5, 10, 25, 50, 100)
)

# Run tuning 
tune_diablo <- tune.block.splsda(
  X = data_blocks,
  Y = Y_outcome,
  ncomp = 2,
  test.keepX = list.keepX,
  design = design,
  validation = 'Mfold',
  folds = 5,
  nrepeat = 20, 
  dist = 'max.dist',
  max.iter = 1500,
  tol = 1e-06 
)

optimal.keepX <- tune_diablo$optimal.keepX
cat("\n Optimal features (keepX) selected after tuning:\n")
print(optimal.keepX)

# Output: Tuning Plot (Figure for Methods/Supplementary)
pdf("diablo_tuning_results.pdf", width = 8, height = 6)
plot(tune_diablo)
dev.off()
cat(" Tuning plot saved as: diablo_tuning_results.pdf\n")

# ----------------------------------------------------
# SECTION 5: Final DIABLO Model Run
# ----------------------------------------------------
cat("\n--- 5. Running Final DIABLO Model ---\n")

diablo_model <- block.splsda(
  X = data_blocks,
  Y = Y_outcome,
  ncomp = 2,
  keepX = optimal.keepX, 
  design = design,
  scale = TRUE 
)

# ----------------------------------------------------
# SECTION 6: Model Performance, AUROC & FDR (Fixed)
# ----------------------------------------------------

# --- 6.1 Performance & AUROC ---
cat("Starting 50-repeat cross-validation for robust performance...\n")

perf_diablo <- perf(
  diablo_model,
  validation = 'Mfold',
  folds = 5,
  nrepeat = 50,
  dist = 'max.dist',
  auc = TRUE
)

# Error Rate Plot
pdf("diablo_performance_error_rate.pdf", width = 8, height = 6)
plot(perf_diablo, sd = TRUE, legend.title = 'Distance')
dev.off()
cat("Performance error rate plot saved.\n")

# AUROC per block
if (!is.null(perf_diablo$auc)) {
  cat("\nAUROC Summary per Omics Block:\n")
  for (block in names(perf_diablo$auc)) {
    cat("\nBlock:", block, "\n")
    print(perf_diablo$auc[[block]])
  }
}

# --- 6.2 FDR Calculation for DIABLO-selected Features ---

calculate_fdr_safe <- function(X_block, Y_factor, feature_names) {
  # Only keep columns that exist
  feature_names <- feature_names[feature_names %in% colnames(X_block)]
  if (length(feature_names) == 0) return(NULL)
  
  p_values <- sapply(feature_names, function(feature) {
    feature_data <- X_block[, feature]
    fit <- aov(feature_data ~ Y_factor)
    if (length(summary(fit)) > 0 && nrow(summary(fit)[[1]]) > 0) {
      summary(fit)[[1]][1, "Pr(>F)"]
    } else NA
  })
  
  p_adj <- p.adjust(p_values, method = "BH")
  data.frame(
    feature = feature_names,
    P_value_raw = unname(p_values),
    P_value_adj_FDR = unname(p_adj),
    row.names = NULL
  )
}

# Selected features from components 1 & 2
features_comp1 <- selectVar(diablo_model, comp = 1)
features_comp2 <- selectVar(diablo_model, comp = 2)

list_features_to_test <- list(
  metabolomics  = unique(c(features_comp1$metabolomics$name, features_comp2$metabolomics$name)),
  proteomics    = unique(c(features_comp1$proteomics$name, features_comp2$proteomics$name)),
  transcriptomics = unique(c(features_comp1$transcriptomics$name, features_comp2$transcriptomics$name))
)

fdr_results_list <- list()
for (block_name in names(data_blocks)) {
  sel_features <- list_features_to_test[[block_name]]
  
  if (length(sel_features) == 0) {
    cat(paste("WARNING: No DIABLO-selected features for block", block_name, "\n"))
    next
  }
  
  X_block <- data_blocks[[block_name]]
  fdr_res <- calculate_fdr_safe(X_block, Y_outcome[rownames(X_block)], sel_features)
  
  if (!is.null(fdr_res)) {
    fdr_results_list[[block_name]] <- fdr_res %>% mutate(block = block_name)
    cat(paste("FDR calculated for", nrow(fdr_res), "features in block", block_name, "\n"))
  } else {
    cat(paste("WARNING: No valid features after alignment for block", block_name, "\n"))
  }
}

all_fdr_results <- bind_rows(fdr_results_list)
cat(paste("\nSUCCESS: FDR results created for", nrow(all_fdr_results), "features across blocks.\n"))

# ----------------------------------------------------
# SECTION 7: Visualization and Biomarker Ranking
# ----------------------------------------------------

# --- 7.1 Sample Clustering Plot ---
pdf("diablo_sample_clustering.pdf", width = 8, height = 8)
plotIndiv(
  diablo_model, 
  group = Y_outcome, 
  legend = TRUE, 
  title = "DIABLO - Sample Clustering (Comp 1 & 2)"
)
dev.off()
cat("Sample clustering plot saved as 'diablo_sample_clustering.pdf'\n")

# --- 7.2 Heatmap of Top Features ---
pdf("diablo_heatmap_3omics_cim.pdf", width = 18, height = 18)
cimDiablo(diablo_model, comp = 1, legend = TRUE, title = "Top Features - DIABLO Heatmap")
dev.off()
cat("Heatmap saved as 'diablo_heatmap_3omics_cim.pdf'\n")

# --- 7.3 Correlation Plot (Combined Loadings) ---
loadings_comp1 <- lapply(diablo_model$loadings, function(x) x[, 1])
loadings_df <- bind_rows(
  lapply(names(loadings_comp1), function(block_name) {
    data.frame(
      feature = names(loadings_comp1[[block_name]]),
      loading = loadings_comp1[[block_name]],
      block = block_name
    )
  })
)

# Optional: keep only strong loadings for correlation
top_loadings <- loadings_df %>% filter(abs(loading) > 0.05)
cor_matrix <- cor(
  do.call(cbind, lapply(names(data_blocks), function(block_name) data_blocks[[block_name]][, top_loadings$feature[top_loadings$block == block_name], drop = FALSE])),
  use = "pairwise.complete.obs"
)

pdf("diablo_correlation_plot.pdf", width = 12, height = 12)
corrplot(cor_matrix, method = "color", type = "upper", tl.cex = 0.7, tl.col = "black", number.cex = 0.7)
dev.off()
cat("Correlation plot saved as 'diablo_correlation_plot.pdf'\n")
# --- 7.4 Map Transcriptomics to Gene Symbols ---
ensembl <- useMart("ensembl", dataset = "hsapiens_gene_ensembl")
trans_indices <- which(loadings_df$block == "transcriptomics")
trans_features_clean <- sub("\\..*", "", loadings_df$feature[trans_indices])

gene_map <- tryCatch({
  getBM(
    attributes = c("ensembl_gene_id", "hgnc_symbol"), 
    filters = "ensembl_gene_id", 
    values = trans_features_clean, 
    mart = ensembl
  )
}, error = function(e) {
  message("Biomart connection failed. Using ENSEMBL IDs for transcriptomics.")
  return(data.frame(ensembl_gene_id = trans_features_clean, hgnc_symbol = trans_features_clean))
})

# Correct assignment
loadings_df$gene_symbol <- loadings_df$feature
loadings_df$gene_symbol[trans_indices] <- gene_map$hgnc_symbol[match(trans_features_clean, gene_map$ensembl_gene_id)]
# --- 7.5 Merge Loadings with FDR Results ---
loadings_df_merged <- loadings_df %>%
  left_join(all_fdr_results, by = c("feature", "block"))

# --- 7.6 Compute Combined Weight Score and Rank Features ---
combined_scores_ranked <- loadings_df_merged %>%
  group_by(gene_symbol) %>%
  summarise(
    combined_weight = sum(abs(loading), na.rm = TRUE),
    min_P_value_adj_FDR = min(P_value_adj_FDR, na.rm = TRUE),
    contributing_blocks = paste(unique(block), collapse = ";"),
    .groups = "drop"
  ) %>%
  filter(
    !is.na(gene_symbol),
    gene_symbol != "",
    !grepl("normal|tumor|GTEx|NA", gene_symbol, ignore.case = TRUE),
    !grepl("unknown|NA\\.|\\d+", gene_symbol, ignore.case = TRUE)
  ) %>%
  arrange(desc(combined_weight))

# Save ranked targets
write_csv(combined_scores_ranked, "diablo_multiomics_ranked_features_FDR_CLEAN.csv")
cat("\nTop 10 multi-omics targets (highest combined weight):\n")
print(head(combined_scores_ranked, 10))

# ----------------------------------------------------
# SECTION 8: KEGG Pathway Enrichment for Proteins & Genes
# ----------------------------------------------------
# --- 8.0 Prepare Results Folder ---
results_dir <- here::here("results")
if (!dir.exists(results_dir)) dir.create(results_dir)
cat("Results folder set to:", results_dir, "\n")

# --- 8.1 Map Genes/Proteins to ENTREZ IDs ---
# Use your ranked protein/gene list from DIABLO
prot_gene_features <- combined_scores_ranked %>%
  filter(grepl("proteomics|transcriptomics", contributing_blocks)) %>%
  pull(gene_symbol) %>%
  unique()

# Map to ENTREZ IDs using org.Hs.eg.db
entrez_ids <- mapIds(
  org.Hs.eg.db,
  keys = prot_gene_features,
  column = "ENTREZID",
  keytype = "SYMBOL",
  multiVals = "first"
)
entrez_ids <- na.omit(entrez_ids)
cat("Mapped", length(entrez_ids), "proteins/genes to ENTREZ IDs.\n")

# --- 8.2 KEGG Pathway Enrichment ---
kegg_res <- enrichKEGG(
  gene = entrez_ids,
  organism = "hsa",
  pAdjustMethod = "BH",
  qvalueCutoff = 0.05
)

# --- 8.3 Save Results ---
kegg_file <- file.path(results_dir, "diablo_prot_gene_kegg_enrichment.csv")
write_csv(as.data.frame(kegg_res), kegg_file)
cat("KEGG enrichment results saved as:", kegg_file, "\n")

# --- 8.4 Optional: Save a smaller table with top pathways ---
top_kegg <- kegg_res@result %>%
  arrange(p.adjust) %>%
  dplyr::select(ID, Description, GeneRatio, BgRatio, pvalue, p.adjust, qvalue, geneID) %>%
  head(20)

top_kegg_file <- file.path(results_dir, "diablo_prot_gene_kegg_top20.csv")
write_csv(top_kegg, top_kegg_file)
cat("Top 20 KEGG pathways saved as:", top_kegg_file, "\n")

# =====================================================
# SECTION 8: KEGG Pathway Enrichment for Metabolites
# =====================================================
# Path to your Documents folder
metabo_path <- here::here("data", "metabolites")

# Read metabolite enrichment results
msea_ora_results <- read_csv(file.path(metabo_path, "msea_ora_result.csv"))

# Quick check
head(msea_ora_results)

# Select top 10 pathways
top10_pathways <- prot_gene_kegg %>%
  arrange(p.adjust) %>%
  slice(1:10)

# Save as PDF
pdf(file.path(metabo_path, "top10_kegg_prot_gene_bubble.pdf"), width = 10, height = 6)

# =====================================================
# SECTION 9: Integrated Visualization (Shared Pathways)
# =====================================================

# --- 0. Load libraries ---
library(ggplot2)
library(stringr)

# --- 1. Load KEGG enrichment results for proteins/genes ---
prot_gene_kegg <- read_csv("results/diablo_prot_gene_kegg_enrichment.csv")

# --- 2. Prepare metabolites table ---
metabo_df <- msea_ora %>%
  rename(
    Pathway = `...1`,      # first column has pathway names
    p_value = `Raw p`,
    Hits = hits
  ) %>%
  select(Pathway, p_value, Hits) %>%
  mutate(Omics = "Metabolites")

# --- 3. Prepare proteins/genes table ---
protgene_df <- prot_gene_kegg %>%
  rename(
    Pathway = Description,
    p_value = pvalue,
    Feature = geneID  # this is a string of gene symbols separated by ";"
  ) %>%
  mutate(
    Hits = str_count(Feature, ";") + 1,  # count number of genes per pathway
    Omics = "Proteins/Genes"
  ) %>%
  select(Pathway, p_value, Hits, Omics)

# --- 4. Combine datasets ---
combined_df <- bind_rows(metabo_df, protgene_df)

# --- 5. Filter for overlapping pathways only ---
overlap_df <- combined_df %>%
  group_by(Pathway) %>%
  filter(n() > 1) %>%   # pathways present in both omics
  ungroup()

# --- 6. Define colors for plotting ---
omic_colors <- c("Metabolites" = "#1f78b4", "Proteins/Genes" = "#e31a1c")

# --- 7. Plot shared KEGG pathways as a bubble plot ---
pdf("results/overlap_pathways_bubble_plot_colored.pdf", width = 12, height = 8)
ggplot(overlap_df, aes(x = reorder(Pathway, -Hits), y = Omics)) +
  geom_point(aes(size = Hits, color = Omics)) +
  scale_color_manual(values = omic_colors) +
  scale_size_continuous(range = c(3, 10)) +
  coord_flip() +
  labs(
    x = "KEGG Pathways",
    y = "Omics Type",
    size = "Number of Hits",
    color = "Omics",
    title = "Shared KEGG Pathways: Metabolites vs Proteins/Genes"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    axis.text.y = element_text(size = 10),  # better for long pathway names
    plot.title = element_text(face = "bold", hjust = 0.5)
  )
dev.off()

cat(" Bubble plot saved as 'results/overlap_pathways_bubble_plot_colored.pdf'\n")
# =====================================================
# Define shared_pathways for downstream use
# =====================================================

# From the processed tables in Section 9:
metabo_pathways <- metabo_df$Pathway
protgene_pathways <- protgene_df$Pathway

# Intersection = pathways present in both omics
shared_pathways <- intersect(metabo_pathways, protgene_pathways)
length(shared_pathways)  # check how many overlapping pathways

# =====================================================
# 1. Load Required Libraries
# =====================================================
library(here)
library(biomaRt)
library(gridExtra)
library(grid)

# =====================================================
# 2. Load Data
# =====================================================
metabo_file <- here::here("data", "msea_ora_result.csv")
protgene_file <- here::here("results", "diablo_prot_gene_kegg_enrichment.csv")

# Inspect column names
colnames(metabo_df)
colnames(protgene_df)

# Standardize column names
colnames(metabo_df) <- c("Index", "Total", "Expected", "Hits", "Raw_p", "Holm_p", "FDR")
colnames(protgene_df) <- c("Category", "Subcategory", "ID", "Description", "GeneRatio", "BgRatio",
                           "RichFactor", "FoldEnrichment", "zScore", "pvalue", "p_adjust",
                           "qvalue", "geneID", "Count")


# =====================================================
# 4. Map Entrez IDs to HGNC Gene Symbols
# =====================================================
ensembl <- useMart("ensembl", dataset = "hsapiens_gene_ensembl")

# Split Entrez IDs by "/" if multiple IDs per pathway
all_entrez_ids <- unique(unlist(strsplit(protgene_df$geneID, "/")))
gene_map <- getBM(
  attributes = c("entrezgene_id", "hgnc_symbol"),
  filters = "entrezgene_id",
  values = all_entrez_ids,
  mart = ensembl
)

# Replace Entrez IDs with gene symbols
protgene_df <- protgene_df %>%
  rowwise() %>%
  mutate(
    geneID = paste(
      gene_map$hgnc_symbol[match(unlist(strsplit(geneID, "/")), gene_map$entrezgene_id)],
      collapse = ";"
    )
  ) %>%
  ungroup()

# =====================================================
# 5. Extract Overlapping Pathway Makeup
# =====================================================
# Metabolites per pathway
metabo_names <- metabo_df %>%
  filter(Index %in% shared_pathways) %>%
  select(Pathway = Index, Metabolites = Hits) %>%
  mutate(Metabolites = as.character(Metabolites))

# Genes/proteins per pathway
protgene_names <- protgene_df %>%
  filter(Description %in% shared_pathways) %>%
  group_by(Pathway = Description) %>%
  summarise(Genes_Proteins = paste(geneID, collapse = ";"), .groups = "drop")

# Combine into a single detailed table
pathway_makeup <- full_join(protgene_names, metabo_names, by = "Pathway")

# Save as CSV
write_csv(pathway_makeup, here::here("diablo_overlapping_pathways_detailed.csv"))
cat("Detailed overlapping pathways saved as 'diablo_overlapping_pathways_detailed.csv'\n")

# =====================================================
# 6. Manual Target Validation
# =====================================================
manual_targets <- unique(c(
  "GLDC","AMT","PDHB","OGDH","GATM","MAOA","PSPH",
  "CBS","PKLR","PC","PKM","ACACA","ACACB",
  "DHFR","TYMS","MTR"
))

manual_target_support <- loadings_df_merged %>%
  filter(gene_symbol %in% manual_targets) %>%
  group_by(gene_symbol) %>%
  summarise(
    Proteomics = any(block == "proteomics"),
    Transcriptomics = any(block == "transcriptomics"),
    Metabolomics = any(block == "metabolomics"),
    Blocks_Present = paste(sort(unique(block)), collapse = ", "),
    Combined_Loading = sum(abs(loading), na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    Classification = case_when(
      Proteomics & Transcriptomics ~ "Protein (RNA-supported)",
      Proteomics ~ "Protein (proteomics only)",
      Transcriptomics ~ "Gene (RNA only)",
      Metabolomics ~ "Metabolite-linked",
      TRUE ~ "Unclassified"
    ),
    Drug_Target_Eligible = ifelse(Proteomics, "YES", "NO")
  ) %>%
  arrange(desc(Proteomics), desc(Combined_Loading))

# Save Validation PDF
pdf("DIABLO_manual_target_validation.pdf", width = 11, height = 8)
grid.newpage()
grid.text(
  "Manual Validation of DIABLO Targets\n(Proteomic vs Transcriptomic Evidence)",
  y = 0.95,
  gp = gpar(fontsize = 16, fontface = "bold")
)
table_grob <- tableGrob(
  manual_target_support,
  rows = NULL,
  theme = ttheme_minimal(base_size = 9)
)
grid.draw(table_grob)
dev.off()

sessionInfo()
