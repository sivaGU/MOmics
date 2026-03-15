# GBM Single-omics Transcriptomics 

set.seed(123)

# 0. Libraries and directories
library(DESeq2)
library(apeglm)
library(clusterProfiler)
library(org.Hs.eg.db)
library(AnnotationDbi)
library(enrichplot)
library(ggplot2)
library(pheatmap)
library(RColorBrewer)
library(tidyverse)
library(ggrepel)
library(STRINGdb)
library(googlesheets4)

# 1. Global settings
data_dir <- "data"
results_dir <- "results"
if(!dir.exists(results_dir)) dir.create(results_dir)

group_colors <- c(Normal = "#2C7BB6", Tumor = "#D7191C")
options(stringsAsFactors = FALSE)

# Thresholds
log2fc_cutoff <- 1
padj_cutoff <- 0.05
volcano_top_n <- 6
ppi_top_n <- 200

# 2.  Import data sets for analysis 
counts <- read.table(
  file.path(data_dir, "rnaseq_washu_readcount.v4.0.tsv"),
  header = TRUE, sep = "\t", row.names = 1, check.names = FALSE
)
count_data <- counts[, 8:ncol(counts)]

# 3. Sample metadata
sample_ids <- colnames(count_data)
condition <- case_when(
  grepl("^C3N", sample_ids) ~ "Normal",
  grepl("^C3L|^PT-", sample_ids) ~ "Tumor",
  TRUE ~ NA_character_
)
keep <- !grepl("^exo", sample_ids)
count_data <- count_data[, keep]
condition <- condition[keep]
sample_ids <- sample_ids[keep]

colData <- data.frame(
  row.names = sample_ids,
  condition = factor(condition, levels = c("Normal","Tumor"))
)

# 4. DESeq2 differential expression
dds <- DESeqDataSetFromMatrix(countData = count_data, colData = colData, design = ~ condition)
dds <- dds[rowSums(counts(dds)) >= 10, ]
dds <- DESeq(dds)
res <- lfcShrink(dds, coef = "condition_Tumor_vs_Normal", type = "apeglm")
res_df <- as.data.frame(res) %>% rownames_to_column("ensembl") %>% filter(!is.na(padj))

# 5. Gene annotation
res_df$ensembl_clean <- sub("\\..*$", "", res_df$ensembl)
res_df$symbol <- mapIds(org.Hs.eg.db, keys = res_df$ensembl_clean,
                        column = "SYMBOL", keytype = "ENSEMBL", multiVals = "first")
sig_genes <- res_df %>% filter(padj < padj_cutoff & abs(log2FoldChange) > log2fc_cutoff)

# 6. Create Principal Component Analysis (PCA)
vsd <- vst(dds, blind = FALSE)
pca <- plotPCA(vsd, intgroup = "condition", returnData = TRUE)
percentVar <- round(100 * attr(pca, "percentVar"))

pca_plot <- ggplot(pca, aes(PC1, PC2, color = condition)) +
  geom_point(size = 4, alpha = 0.9) +
  scale_color_manual(values = group_colors) +
  xlab(paste0("PC1 (", percentVar[1], "%)")) +
  ylab(paste0("PC2 (", percentVar[2], "%)")) +
  theme_classic(base_size = 14) +
  ggtitle("PCA of Transcriptomic Profiles")

ggsave(file.path(results_dir, "PCA_Transcriptomics.png"), plot = pca_plot, width = 7, height = 6, dpi = 300)

# 7. Create a Volcano plot: displays most up- and down- regulated genes 
top_up <- sig_genes %>% arrange(desc(log2FoldChange)) %>% slice_head(n = volcano_top_n)
top_down <- sig_genes %>% arrange(log2FoldChange) %>% slice_head(n = volcano_top_n)

volcano_plot <- ggplot(res_df, aes(log2FoldChange, -log10(padj))) +
  geom_point(color = "grey75", alpha = 0.5) +
  geom_point(data = sig_genes, aes(color = log2FoldChange > 0), alpha = 0.8) +
  geom_text_repel(data = bind_rows(top_up, top_down), aes(label = symbol), size = 5, max.overlaps = 20) +
  scale_color_manual(values = c("#2C7BB6","#D7191C")) +
  theme_classic(base_size = 18) +
  xlab("log2 Fold Change (Tumor vs Normal)") +
  ylab("-log10 Adjusted p-value") +
  ggtitle("Volcano Plot of Differential Expression") +
  theme(legend.position = "none")

ggsave(file.path(results_dir, "Volcano_DEGs.png"), plot = volcano_plot, width = 7, height = 7, dpi = 300)

# 8. KEGG pathway enrichment
entrez_sig <- mapIds(org.Hs.eg.db, keys = sig_genes$ensembl_clean, column = "ENTREZID",
                     keytype = "ENSEMBL", multiVals = "first") %>% na.omit() %>% unique()
entrez_bg <- mapIds(org.Hs.eg.db, keys = sub("\\..*$", "", rownames(dds)), column = "ENTREZID",
                    keytype = "ENSEMBL", multiVals = "first") %>% na.omit() %>% unique()

kk <- enrichKEGG(gene = entrez_sig, universe = entrez_bg, organism = "hsa",
                 keyType = "ncbi-geneid", pvalueCutoff = 0.01)
kegg_df <- as.data.frame(kk)
kegg_df$log10p_raw <- -log10(kegg_df$pvalue)
top_kegg <- kegg_df[1:10, ]

kegg_plot <- ggplot(top_kegg, aes(x = reorder(Description, log10p_raw), y = log10p_raw)) +
  geom_col(fill = "#4682B4") +
  coord_flip() +
  theme_classic(base_size = 18) +
  xlab("") +
  ylab("-log10 raw p-value") +
  ggtitle("Top KEGG Pathways Enriched in Tumor")

ggsave(file.path(results_dir, "Figure4_KEGG_Pathway_Enrichment.png"), plot = kegg_plot, width = 8, height = 6, dpi = 300)

# 9. Biomarker analysis (Z-score) and Graph 
vsd_mat <- assay(vsd)
rownames(vsd_mat) <- sub("\\..*$", "", rownames(vsd_mat))
genes_use <- sig_genes$ensembl_clean[sig_genes$ensembl_clean %in% rownames(vsd_mat)]
vsd_mat_subset <- vsd_mat[genes_use, , drop = FALSE]
gene_symbols <- make.unique(ifelse(is.na(sig_genes$symbol[match(genes_use, sig_genes$ensembl_clean)]),
                                   genes_use, sig_genes$symbol[match(genes_use, sig_genes$ensembl_clean)]))
rownames(vsd_mat_subset) <- gene_symbols
zscore_mat <- t(scale(t(vsd_mat_subset)))

biomarker_df <- data.frame(
  Gene = rownames(zscore_mat),
  Mean_Z_Tumor = rowMeans(zscore_mat[, colData$condition == "Tumor", drop = FALSE]),
  Mean_Z_Normal = rowMeans(zscore_mat[, colData$condition == "Normal", drop = FALSE])
) %>% mutate(Zscore_Difference = Mean_Z_Tumor - Mean_Z_Normal,
             Direction = ifelse(Zscore_Difference > 0, "Up in Tumor", "Down in Tumor")) %>%
  arrange(desc(abs(Zscore_Difference)))

top_biomarkers <- bind_rows(
  biomarker_df %>% filter(Direction == "Up in Tumor") %>% slice_head(n = 5),
  biomarker_df %>% filter(Direction == "Down in Tumor") %>% slice_head(n = 5)
)

biomarker_plot <- ggplot(top_biomarkers, aes(x = reorder(Gene, Zscore_Difference), y = Zscore_Difference, fill = Direction)) +
  geom_col() + coord_flip() +
  scale_fill_manual(values = c("Up in Tumor" = "#D7191C", "Down in Tumor" = "#2C7BB6")) +
  theme_classic(base_size = 18) +
  xlab("Biomarkers") + ylab("Z-score Difference") + ggtitle("Top 10 Candidate Biomarkers")

ggsave(file.path(results_dir, "Top_10_Zscore_Biomarkers.png"), plot = biomarker_plot, width = 8, height = 6, dpi = 300)

# 10. PPI network (STRINGdb) and Figure 
string_db <- STRINGdb$new(version="12", species=9606, score_threshold=400)
genes_mapped <- string_db$map(sig_genes, "symbol", removeUnmappedRows = TRUE)
top_hits <- genes_mapped %>% arrange(padj) %>% slice_head(n = ppi_top_n) %>% pull(STRING_id)
interactions <- string_db$get_interactions(top_hits)
hub_counts <- as.data.frame(table(c(interactions$from, interactions$to)))
colnames(hub_counts) <- c("STRING_id", "Degree")
hubs_with_data <- merge(hub_counts, genes_mapped[, c("STRING_id", "symbol", "log2FoldChange", "padj")], by = "STRING_id") %>% arrange(desc(Degree))
top_targets <- head(hubs_with_data %>% filter(log2FoldChange > 0), 10)

png(file.path(results_dir, "PPI_Drug_Targets_Network.png"), width = 3*300, height = 3*300, res = 300)
par(mar = c(0, 0, 0, 0))
string_db$plot_network(top_targets$STRING_id)
dev.off()

# 11. Export results to Google Sheets

# Authenticate (will open browser first time)
gs4_auth(cache = ".secrets", email = "woubes24@gmail.com")

# Create Google Sheet and write all results
ss <- gs4_create("GBM_Transcriptomics_Results")  # empty sheet

sheet_write(sig_genes, ss = ss, sheet = "Significant_Genes")
sheet_write(kegg_df, ss = ss, sheet = "KEGG_Pathways")
sheet_write(hubs_with_data, ss = ss, sheet = "Full_Network_Hubs")
sheet_write(top_targets, ss = ss, sheet = "Top_Drug_Targets")
sheet_write(biomarker_df, ss = ss, sheet = "Full_Biomarker_List")
sheet_write(top_biomarkers, ss = ss, sheet = "Top_10_Biomarkers")

cat("All results successfully saved to Google Sheets:\n")

# 12. Session info
sessionInfo()
