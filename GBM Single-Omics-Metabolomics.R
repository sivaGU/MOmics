#GBM Single-Omics Metabolomics Analysis

set.seed(123)

# 0. Libraries and directories

library(readr)
library(dplyr)
library(ggplot2)
library(ggrepel)
library(openxlsx)
library(pheatmap)
library(tidyr)
library(tibble)
library(limma)
library(clusterProfiler)
library(KEGGREST)
library(stringr)
library(mixOmics)

data_dir <- "data"
results_dir <- "results"

if(!dir.exists(results_dir)) dir.create(results_dir)

# 1. Import data sets for analysis 

expr_raw <- read_tsv(file.path(data_dir,"metabolome_pnnl.v4.0.tsv"), show_col_types = FALSE)
metadata <- read_tsv(file.path(data_dir,"metabolome_sample_info.v4.0.tsv"), show_col_types = FALSE)

# Remove unknown metabolites and duplicates
expr_clean <- expr_raw %>%
  filter(!grepl("unknown", Metabolite, ignore.case = TRUE)) %>%
  distinct(Metabolite, .keep_all = TRUE)

expr_mat <- as.data.frame(expr_clean[,-1])
rownames(expr_mat) <- expr_clean$Metabolite

# Match metadata: Determine tumor or normal 
metadata <- metadata %>%
  filter(case_id %in% colnames(expr_mat)) %>%
  mutate(tumor_normal = factor(tumor_normal, levels=c("normal","tumor")))

expr_mat <- expr_mat[, metadata$case_id]

# 2. Quality control filtering

present_count <- rowSums(expr_mat > 0, na.rm=TRUE)

expr_mat <- expr_mat[present_count >= (ncol(expr_mat)*0.8), ]

# Imputation + log transform
min_val <- min(expr_mat[expr_mat > 0], na.rm=TRUE) / 2
expr_mat[expr_mat == 0 | is.na(expr_mat)] <- min_val

expr_log2 <- log2(expr_mat)

# 3. Create Principal Component Analysis (PCA)

pca_res <- prcomp(t(expr_log2), scale.=TRUE)

pca_df <- as.data.frame(pca_res$x) %>%
  mutate(Group = metadata$tumor_normal)

pca_plot <- ggplot(pca_df, aes(PC1, PC2, color=Group)) +
  geom_point(size=4, alpha=0.8) +
  stat_ellipse(level=0.95) +
  scale_color_manual(values=c("normal"="#2C7BB6","tumor"="#D7191C")) +
  theme_classic(base_size=14) +
  labs(
    title="PCA: Metabolic Distribution",
    x=paste0("PC1 (", round(summary(pca_res)$importance[2,1]*100,1), "%)"),
    y=paste0("PC2 (", round(summary(pca_res)$importance[2,2]*100,1), "%)")
  )

ggsave(file.path(results_dir,"Fig1_PCA.png"), pca_plot, width=7, height=5, dpi=300)

# 4. Generate PLS-DA

pls_res <- plsda(t(expr_log2), metadata$tumor_normal, ncomp=2)

vip_vals <- vip(pls_res)[,1]

# Cross-validation plot for PLS-DA
set.seed(123)
pls_perf <- perf(pls_res, validation="Mfold", folds=5, nrepeat=50)

# Save CV error plot as PNG
png(filename = file.path(results_dir, "Fig_PLSDA_CV_Error.png"),
    width = 800, height = 600, res = 120)
plot(pls_perf, main = "PLS-DA Cross-Validation Error")
dev.off()

# Check overall error rate
pls_perf$error.rate

# Plot CV error rate
plot(pls_perf)

# 5. Preform differential analysis
design <- model.matrix(~ tumor_normal, data=metadata)

fit <- lmFit(expr_log2, design)
fit <- eBayes(fit, trend=TRUE, robust=TRUE)

results_df <- topTable(fit, coef=2, number=Inf) %>%
  rownames_to_column("Metabolite") %>%
  mutate(
    VIP = vip_vals[Metabolite],
    adj.P.Val = p.adjust(P.Value, method="BH"),
    Status = case_when(
      adj.P.Val < 0.05 & logFC > 0 ~ "Upregulated",
      adj.P.Val < 0.05 & logFC < 0 ~ "Downregulated",
      TRUE ~ "Non-significant"
    )
  )

# 6. Create a Volcano plot: displays most up- and down- regulated metabolites 
sig_hits <- results_df %>% filter(adj.P.Val < 0.05)

plot_labels <- bind_rows(
  sig_hits %>% filter(logFC < 0) %>% arrange(logFC) %>% head(5),
  sig_hits %>% filter(logFC > 0) %>% arrange(desc(logFC)) %>% head(5)
)
volcano_plot <- ggplot(results_df, 
                       aes(x=logFC, y=-log10(adj.P.Val), color=Status)) +
  geom_point(alpha=0.4, size=2) +
  scale_color_manual(values=c("Upregulated"="#D7191C","Downregulated"="#2C7BB6","Non-significant"="grey80")) +
  geom_hline(yintercept=-log10(0.05), linetype="dashed", color="black") +
  geom_text_repel(data=plot_labels, aes(label=Metabolite),
                  color="black", size=4, fontface="bold", box.padding=0.5) +
  theme_classic(base_size=18) +
  labs(title="Fig 2. Differential Expression (Volcano Plot, Adjusted P-Values)",
       x="Log2 Fold Change", y="-log10(Adjusted P-Value)")

ggsave(file.path(results_dir,"Fig2_Volcano.png"),
       volcano_plot, width=8, height=6, dpi=300)

# 7. Identify the top metabolic biomarkers and create a graph 

top10_biomarkers <- results_df %>%
  arrange(desc(VIP)) %>%
  head(10)

biomarker_plot <- ggplot(top10_biomarkers,
                         aes(VIP, reorder(Metabolite, VIP), fill=Status)) +
  geom_col() +
  scale_fill_manual(values=c(
    "Upregulated"="#D7191C",
    "Downregulated"="#2C7BB6"
  )) +
  theme_classic(base_size=18) +
  labs(
    title="Top 10 Discriminatory Biomarkers",
    x="VIP Score",
    y=NULL
  )

ggsave(file.path(results_dir,"Fig3_Biomarkers.png"),
       biomarker_plot, width=8, height=5, dpi=300)

# KEGG Pathways pipline 
# 1. Re-map IDs to ensure perfect size matching with results_df
all_mapped_list <- sapply(results_df$Metabolite, function(x) {
  query <- tryCatch(keggFind("compound", x), error=function(e) NULL)
  if(!is.null(query) && length(query)>0) return(names(query)[1]) else return(NA)
})
all_mapped_clean <- gsub("cpd:", "", all_mapped_list)

# 2. Extract Significant IDs (P < 0.05 )
# We use the clean mapping to ensure we have KEGG-compatible IDs
results_with_ids <- results_df %>% mutate(KEGG_ID = all_mapped_clean)

sig_ids <- results_with_ids %>%
  filter(P.Value < 0.05 & !is.na(KEGG_ID)) %>%
  pull(KEGG_ID) %>% unique()

univ_ids <- unique(all_mapped_clean[!is.na(all_mapped_clean)])

# 3. Build Human (HSA) Pathway Mapping
cpd_to_path_raw <- keggLink("pathway", "cpd")
pathway_mapping <- data.frame(
  Compound = gsub("cpd:", "", names(cpd_to_path_raw)),
  Pathway = gsub("path:map", "hsa", cpd_to_path_raw)
)

hsa_pathway_list <- keggList("pathway", "hsa")
valid_hsa_ids <- gsub("path:", "", names(hsa_pathway_list))

pathway_dict <- data.frame(
  Pathway = valid_hsa_ids,
  Name = as.character(hsa_pathway_list)
) %>% mutate(Name = str_remove(Name, " - Homo sapiens \\(human\\)"))

# 4. Run Enrichment (ORA)
kegg_enrich <- enricher(
  gene = sig_ids, 
  universe = univ_ids,
  TERM2GENE = pathway_mapping %>% 
    filter(Pathway %in% valid_hsa_ids) %>% 
    dplyr::select(Pathway, Compound), 
  TERM2NAME = pathway_dict,
  pvalueCutoff = 1, 
  qvalueCutoff = 1
)

# 5. Plotting Top 10 as Blue Bars
if(!is.null(kegg_enrich)){
  
  kegg_plot_data <- as.data.frame(kegg_enrich) %>%
    mutate(neg_log10_p = -log10(pvalue)) %>%
    arrange(desc(neg_log10_p)) %>%
    head(10)
  
  kegg_plot <- ggplot(kegg_plot_data, aes(x = neg_log10_p, y = reorder(Description, neg_log10_p))) +
    geom_col(fill = "dodgerblue4", width = 0.7) + 
    # Vertical red line at p=0.05
    geom_vline(xintercept = -log10(0.05), linetype = "dashed", color = "red", linewidth = 0.8) + 
    theme_classic(base_size = 14) +
    labs(
      title = "Enriched KEGG Pathways (Human)",
      subtitle = "KEGG Pathway | Red Line: p = 0.05",
      x = expression(-log[10](Raw~P-Value)), 
      y = NULL
    ) +
    # Add a small label for the red line
    annotate("text", x = -log10(0.05) + 0.05, y = 1.5, label = "p=0.05", color = "red", hjust = 0)
  
  print(kegg_plot)
  ggsave(file.path(results_dir, "Fig4_KEGG_Pathways.png"), width = 8, height = 5, dpi = 300)
}
# Export Results to google sheets
library(googlesheets4)

# Authenticate your Google account (will open browser)
gs4_auth()

# Prepare tables
sig_table <- results_with_ids %>%
  filter(P.Value < 0.05) %>%
  select(Metabolite, logFC, P.Value, adj.P.Val, VIP)

up_table <- results_with_ids %>%
  filter(Status == "Upregulated") %>%
  select(Metabolite, logFC, VIP)

down_table <- results_with_ids %>%
  filter(Status == "Downregulated") %>%
  select(Metabolite, logFC, VIP)

top10_table <- top10_biomarkers %>%
  select(Metabolite, VIP, logFC, P.Value)

kegg_table <- as.data.frame(kegg_enrich) %>%
  mutate(Metabolite_Names = sapply(geneID, function(ids) {
    paste(results_with_ids$Metabolite[match(unlist(strsplit(ids, "/")), results_with_ids$KEGG_ID)],
          collapse = ", ")
  })) %>%
  select(Description, pvalue, Count, Metabolite_Names)

# Create Google Sheet with multiple tabs
sheet_url <- gs4_create(
  name = "GBM_Metabolomics_Results",
  sheets = list(
    Significant = sig_table,
    Upregulated = up_table,
    Downregulated = down_table,
    Top10_Biomarkers = top10_table,
    KEGG_Enrichment = kegg_table
  )
)
cat("Results successfully saved to Google Sheets:\n", sheet_url, "\n")
# Save session info once
writeLines(capture.output(sessionInfo()), file.path(results_dir, "SessionInfo.txt"))