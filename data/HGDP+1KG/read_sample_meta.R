# Load libraries
library(tidyverse)

# Read samples files
samples <- read.table(file = 'hgdp_1kg_subset_sample_meta.tsv', sep = '\t', header = TRUE)

# Samples ID
ind <- samples$s

# Imputed sex
sex_karyo <- sapply(str_split(samples$gnomad_sex_imputation, ","), function(x) x[8]) %>%
  str_replace("sex_karyotype:", "")

gender <- ifelse(sex_karyo == "XX", "female", ifelse(sex_karyo == "XY", "male", NA))

# Sample filters
high_quality <- samples$high_quality

# Relatedness inference
related_samples <- sapply(str_match_all(samples$relatedness_inference, "s:\\s*(.*?)\\s*,"), function(x) str_replace(x[, 2], "\\[\\{s:", ""))
related <- rep("false", length(related_samples))
related[which(related_samples != "[]")] <- "true"

related_exclude <- rep("false", length(related_samples))
related_exclude[which(related_samples != "[]")] <- "true"

for (i in which(related_samples != "[]")){
  rel_ids <- related_samples[i][related_samples[i] %in% ind]
  if (length(rel_ids) >= 1){
    keep <- unlist(c(ind[i], rel_ids)) %>% sort() %>% first()
    related_exclude[which(ind==keep)] <- "false"
  } else{
    related[i] <- "false"
    related_exclude[i] <- "false"
  }
}

# Sample metadata
project <- sapply(strsplit(samples$hgdp_tgp_meta, ","), function(x) x[1]) %>%
           str_replace("\\{project:", "")

pop <- sapply(strsplit(samples$hgdp_tgp_meta, ","), function(x) x[3]) %>%
  str_replace("population:", "")

super_pop <- sapply(strsplit(samples$hgdp_tgp_meta, ","), function(x) x[4]) %>%
  str_replace("genetic_region:", "")

# Population inference
global_pca_scores <- sapply(str_extract_all(samples$hgdp_tgp_meta, "(?<=\\[).+?(?=\\])"), function(x) strsplit(x[1], ",")) %>%
  do.call(rbind, .)
colnames(global_pca_scores) <- paste0("PC", 1:20)

# Create final data frame
final <- data.frame(ind, project, pop, super_pop, gender, related, related_exclude, global_pca_scores) %>%
         filter(high_quality == "true")

# Write data frame to csv file
write.csv(final, "covars.csv", row.names = FALSE, quote = FALSE)
