# GBM Single-Omics Proteomics: Differential Expression Analysis (DESeq2)
#
# Kept separate from single_omics_proteomics_KEGG.R by design (DEG output feeds the KEGG script).
# NOTE: paths below are still the original author's local machine paths
# (~/Desktop/..., /Users/yeroabketemasamuel/...) and need to be pointed at
# data/discovery/ before this will run against the repo's data.

﻿# Installing BiocManager
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
# Installing DESeq2 
if (!require("DESeq2", quietly = TRUE))
    BiocManager::install("DESeq2")
library(DESeq2)
library(ggplot2)
library(pheatmap)
# ============================================================================
# 2. DATA LOADING
# ============================================================================
cat("=== LOADING DATA ===\n")
# Load the expression data (normalized_data_clean.csv)
data <- read.csv("~/Desktop/normalized_data_clean.csv", header = TRUE, stringsAsFactors = FALSE)
# Load the metadata (Mapping case IDs to sample types)
metadata <- read.csv("/Users/yeroabketemasamuel/Desktop/LCCC 2025/S048_CPTAC_GBM METADATACLEANED_Discovery_Cohort_TMT11_CaseID_SampleID_AliquotID_Map_Dec2019_r1.csv", header = TRUE, stringsAsFactors = FALSE)
# Print initial data stats
cat(sprintf("Total genes: %d\n", nrow(data)))
cat(sprintf("Total sample columns: %d\n", ncol(data) - 1))
# ============================================================================
# 3. SAMPLE MATCHING & ALIGNMENT
# ============================================================================
cat("\n=== MATCHING SAMPLES ===\n")
# Identify sample IDs from expression data (columns) and metadata (row entries)
sample_ids_data <- colnames(data)[-1]
sample_ids_metadata <- metadata[, 7]
sample_types_metadata <- metadata[, 8]
# Subset and reorder metadata to perfectly match the order of columns in 'data'
matched_metadata <- metadata[metadata[, 7] %in% sample_ids_data, ]
matched_metadata <- matched_metadata[match(sample_ids_data, matched_metadata[, 7]), ]
# Print matching results
cat(sprintf("Matched samples: %d\n", nrow(matched_metadata)))
cat("Sample types:\n")
print(table(matched_metadata[, 8]))
# ============================================================================
# 4. DATA CLEANING & NA HANDLING
# ============================================================================
cat("\n=== PREPARING DATA (HANDLING NAs) ===\n")
# Extract gene names and create a numeric matrix of expression values
gene_names <- data$Gene
expression_matrix <- as.matrix(data[, -1])
rownames(expression_matrix) <- gene_names
# Identify and count missing values (NAs)
n_na_values <- sum(is.na(expression_matrix))
cat(sprintf("Total NA values found: %d\n", n_na_values))
# Identify which rows (genes) contain at least one NA
genes_with_na <- rowSums(is.na(expression_matrix)) > 0
cat(sprintf("Genes with NA values: %d\n", sum(genes_with_na)))
# Filter out the genes containing NAs
expression_matrix_clean <- expression_matrix[!genes_with_na, ]
gene_names_clean <- gene_names[!genes_with_na]
cat(sprintf("Genes after removing NAs: %d\n", nrow(expression_matrix_clean)))
# --- Back-transformation to Counts ---
# Convert log-normalized data back to linear scale, scale up, and round for DESeq2
count_matrix <- 2^expression_matrix_clean
count_matrix <- round(count_matrix * 1000)
count_matrix[count_matrix < 1] <- 1  # Replace zeros/negative values with 1
# ============================================================================
# 5. DESEQ2 DATASET PREPARATION
# ============================================================================
sample_types <- matched_metadata[, 8]
# Create the colData object required by DESeq2
colData <- data.frame(
    sample = sample_ids_data,
    condition = factor(sample_types, levels = c("normal", "tumor")),
    row.names = sample_ids_data
)
# Final cleanup: Remove samples with no condition and align count matrix
colData <- colData[!is.na(colData$condition), ]
count_matrix <- count_matrix[, rownames(colData)]
cat("Sample counts:\n")
print(table(colData$condition))
# Checking for remaining NAs before creating DESeq object
cat("\n=== CREATING DESEQ2 DATASET ===\n")
if(any(is.na(count_matrix))) {
    stop("ERROR: NAs detected in count matrix before DESeq2")
}
if(any(is.na(colData$condition))) {
    stop("ERROR: NAs detected in sample conditions")
}
# Construct the DESeqDataSet object
dds <- DESeqDataSetFromMatrix(
    countData = count_matrix,
    colData = colData,
    design = ~ condition
)
# Keep genes with at least 10 total reads
keep <- rowSums(counts(dds)) >= 10
dds <- dds[keep, ]
cat(sprintf("Genes after low-count filtering: %d\n", nrow(dds)))
# ============================================================================
# 6. RUN DIFFERENTIAL EXPRESSION
# ============================================================================
cat("\n=== RUNNING DESEQ2 ===\n")
# Execute the DESeq2 pipeline
dds <- DESeq(dds)
cat("✓ Analysis complete!\n")
# Extract the results comparing tumor to normal
res <- results(dds, contrast = c("condition", "tumor", "normal"))
cat("\nResults summary:\n")
summary(res)
# ============================================================================
# 7. IDENTIFY SIGNIFICANT DEGS
# ============================================================================
cat("\n=== IDENTIFYING DEGS ===\n")
# Set significance thresholds
padj_threshold <- 0.05
log2fc_threshold <- 1
# Convert results to dataframe and remove genes with NA p-values
res_df <- as.data.frame(res)
res_df$gene <- rownames(res_df)
res_df <- res_df[!is.na(res_df$padj), ]
# Categorize genes based on thresholds
res_df$regulation <- "Not Significant"
res_df$regulation[res_df$padj < padj_threshold & res_df$log2FoldChange > log2fc_threshold] <- "Upregulated in Tumor"
res_df$regulation[res_df$padj < padj_threshold & res_df$log2FoldChange < -log2fc_threshold] <- "Downregulated in Tumor"
# Print DEG statistics
n_up <- sum(res_df$regulation == "Upregulated in Tumor")
n_down <- sum(res_df$regulation == "Downregulated in Tumor")
n_total_deg <- n_up + n_down
cat(sprintf("\nDEGs (padj < %.2f, |log2FC| > %.1f):\n", padj_threshold, log2fc_threshold))
cat(sprintf("  Upregulated: %d\n", n_up))
cat(sprintf("  Downregulated: %d\n", n_down))
cat(sprintf("  Total: %d\n", n_total_deg))
# Sort results by adjusted p-value
res_df <- res_df[order(res_df$padj), ]
# ============================================================================
# 8. SAVE RESULTS
# ============================================================================
cat("\n=== SAVING RESULTS ===\n")
degs <- res_df[res_df$regulation != "Not Significant", ]
write.csv(degs, "~/Desktop/DESeq2_significant_DEGs.csv", row.names = FALSE)
cat(sprintf("✓ Significant DEGs saved (%d genes)\n", nrow(degs)))