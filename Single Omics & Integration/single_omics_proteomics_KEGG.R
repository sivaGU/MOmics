# GBM Single-Omics Proteomics: KEGG Pathway Enrichment
#
# Kept separate from single_omics_proteomics_DEG.R by design (consumes its DEG output).
# NOTE: path below is still the original author's local machine path
# (~/Desktop/DESeq2_significant_DEGs.csv) and needs to be pointed at this
# repo's results/single_omics/Proteomics/ output before this will run standalone.

﻿# We have already installed the packages so we loaded the libraries directly 
library(clusterProfiler)
library(org.Hs.eg.db)
library(enrichplot)
library(ggplot2)
# ============================================================================
# 2. Load DEP file from the previous differential expression analysis 
# ============================================================================
cat("=== LOADING DEG FILE ===\n")
degs <- read.csv("~/Desktop/DESeq2_significant_DEGs.csv", header = TRUE, stringsAsFactors = FALSE)
cat(sprintf("Total DEGs loaded: %d\n", nrow(degs)))
# Inspect column structure
cat("\nColumn names in DEG file:\n")
print(colnames(degs))
# Show preview of data
cat("\nFirst few rows:\n")
print(head(degs))
# ============================================================================
# Extraction of the Gene list (Protein list) from the previous file
# ============================================================================
# Extract gene names based on common column naming conventions
if("gene" %in% colnames(degs)) {
    gene_list <- degs$gene
} else if("Gene" %in% colnames(degs)) {
    gene_list <- degs$Gene
} else {
    # Use the first column 
    gene_list <- degs[, 1]
}
cat(sprintf("\nGenes to analyze: %d\n", length(gene_list)))
# ============================================================================
# Converting Gene names to Entrez ID
# ============================================================================
cat("\n=== CONVERTING TO ENTREZ IDs ===\n")
gene_mapping <- bitr(gene_list, 
                     fromType = "SYMBOL", 
                     toType = "ENTREZID", 
                     OrgDb = org.Hs.eg.db)
cat(sprintf("Successfully mapped: %d out of %d genes\n", nrow(gene_mapping), length(gene_list)))
# Identify and report genes that could not be mapped
unmapped <- setdiff(gene_list, gene_mapping$SYMBOL)
if(length(unmapped) > 0) {
    cat(sprintf("\nWarning: %d genes could not be mapped:\n", length(unmapped)))
    print(head(unmapped, 20))
}
# Store the final vector of Entrez IDs
entrez_ids <- gene_mapping$ENTREZID
# ============================================================================
# KEGG pathways analysis 
# ============================================================================
cat("\n=== RUNNING KEGG PATHWAY ENRICHMENT ===\n")
# Perform Hypergeometric test for pathway over-representation
kegg_result <- enrichKEGG(gene = entrez_ids, 
                          organism = 'hsa', # Homo sapiens
                          pvalueCutoff = 0.05, 
                          qvalueCutoff = 0.2)
cat(sprintf("\nEnriched pathways found: %d\n", nrow(kegg_result@result)))
# ============================================================================
# Result Pathways 
# ============================================================================
if(nrow(kegg_result@result) > 0) {
    cat("\n=== TOP 10 ENRICHED KEGG PATHWAYS ===\n")
      # Convert result object to data frame for printing
    kegg_df <- as.data.frame(kegg_result)
     # Loop through and print top 10 pathways
    top10 <- head(kegg_df, 10)
    for(i in 1:nrow(top10)) {
        cat(sprintf("\n%d. %s\n", i, top10$Description[i]))
        cat(sprintf("  Pathway ID: %s\n", top10$ID[i]))
        cat(sprintf("  P-value: %.2e\n", top10$pvalue[i]))
        cat(sprintf("  Adjusted P-value: %.2e\n", top10$p.adjust[i]))
        cat(sprintf("  Gene count: %s\n", top10$Count[i]))
        cat(sprintf("  Gene ratio: %s\n", top10$GeneRatio[i]))
    }
}
# ============================================================================
# Saving the result
# ============================================================================
cat("\n=== SAVING RESULTS ===\n")
# Export full enrichment table to CSV
write.csv(kegg_df, "~/Desktop/KEGG_pathway_enrichment.csv", row.names = FALSE)
cat("✓ Results saved: ~/Desktop/KEGG_pathway_enrichment.csv\n")