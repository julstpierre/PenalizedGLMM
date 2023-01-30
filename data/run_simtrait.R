# ------------------------------------------------------------------------
# Initialize parameters
# ------------------------------------------------------------------------
# Assign default command-line arguments
args <- commandArgs(TRUE)
# args = c("0.2", "0.1", "0.4", "0.2", "0.1", "10000", "0.01", "0.1", "20", "true", "NONE", "1d", "")

# Fraction of variance due to fixed polygenic additive effect (logit scale)
h2_g <- as.numeric(args[1])

# Fraction of variance due to fixed GEI effect (logit scale)
h2_GEI <- as.numeric(args[2])

# Fraction of variance due to random polygenic additive effect (logit scale)
h2_b <- as.numeric(args[3])

# Fraction of variance due to unobserved shared environmental effect (logit scale)
h2_d <- as.numeric(args[4])

# Prevalence
pi0 <- as.numeric(args[5])

# Number of snps to randomly select accros genome
p_design <- as.numeric(args[6])

# Percentage of causal SNPs
percent_causal <- as.numeric(args[7])

# Fraction of GEI effects among causal SNPs
percent_causal_GEI <- as.numeric(args[8])

# Number of populations
K <- as.numeric(args[9])

# Hierarchical GEI effects
hier = ifelse(args[10] == "true", TRUE, FALSE)

# Overlap between design and kinship SNPs
percent_overlap <- ifelse(args[11] == "ALL", "100", ifelse(args[11] == "NONE", "0"))

# Geography
geography <- args[12]

# Number of SNPs to use for kinship estimation
p_kinship <- 50000

# Number of subjects to simulate
n <- 2500

# ------------------------------------------------------------------------
# Source R function to simulate genotypes
# ------------------------------------------------------------------------
#source(paste0(dirname(rstudioapi::getSourceEditorContext()$path), '/simtrait.R'))
source('simtrait.R')
admixed <- gen_structured_model(n = n,
                                p_design = p_design,
                                p_kinship = p_kinship,
                                geography = geography,
                                percent_causal = percent_causal,
                                percent_causal_GEI = percent_causal_GEI,
                                percent_overlap = percent_overlap,
                                k = K, s = 0.5, Fst = NULL,
                                b0 = pi0, nPC = 10,
                                h2_g = h2_g, h2_b = h2_b,
                                h2_GEI = h2_GEI, h2_d = h2_d,
                                hier = hier,
                                train_tune_test = c(0.8, 0, 0.2)
)

# #-----------------------
# # PCA plot
# #----------------------
# library(ggplot2)
# ggplot(data.frame(admixed$PC, pop = as.character(admixed$subpops)),
#        aes(PC1, PC2, col = pop)) +
#    geom_point(show.legend = FALSE) +
#    xlab("PC1") + ylab("PC2") +
#   theme_bw()+
#   theme(plot.background = element_blank(),
#         panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank())+
#   #theme(panel.border= element_blank())+
#   theme(axis.line.x = element_line(color="black", size = 0.5),
#         axis.line.y = element_line(color="black", size = 0.5),
#         plot.title = element_text(hjust = 0.5)) +
#   theme(legend.title=element_blank())

# #-----------------------
# #Create a heatmap
# #-----------------------
# library(dplyr)
# library(tidyr)
# des <- model.matrix(~-1+factor((admixed$subpops)))
# colnames(des) <- 1:20
# df <- add_rownames(data.frame(cor(des, admixed$PC)), var = "Pop") %>%
#         pivot_longer(!Pop, names_to = "PC") %>%
#         mutate(PC = factor(PC, levels=paste0("PC", 20:1))) %>%
#         mutate(Pop = factor(Pop, levels=c(1:7, 11, 8:10, 12:20))) %>%
#         mutate(value_ = ifelse(abs(value) > 0.2, signif(value, 3), NA))
# 
# ggplot(data = df, aes(x=Pop, y=PC, fill=abs(value), label = value_)) +
#   geom_tile() +
#   geom_text(color = "black", size = 3) +
#   scale_fill_distiller(palette = "RdPu") +
#   labs(x="Population", y = "", fill = expression("|"~r^2~"|")) +
#   theme_bw()+
#   theme(plot.background = element_blank(),
#         panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank())+
#   theme(panel.border= element_blank())+
#   theme(axis.line.x = element_blank(),
#         axis.line.y = element_blank(),
#         plot.title = element_text(hjust = 0.5))

#-----------------------------------------
# Write kinship to txt.gz compressed file
#-----------------------------------------
gz <- gzfile(paste0(args[13], "grm.txt.gz"), "w")
write.csv(admixed$kin, gz, row.names = FALSE)
close(gz)

#----------------------
# Write csv files
#----------------------
# Phenotype
y <- vector("numeric", length = n)
y[admixed$train_ind] <- admixed$ytrain
y[admixed$tune_ind] <- admixed$ytune
y[admixed$test_ind] <- admixed$ytest

# Covariates
X <- array(dim = c(n, ncol(admixed$xtrain_lasso) - p_design))
colnames(X) <- colnames(admixed$xtrain_lasso[ ,-c(1:p_design)])
X[admixed$train_ind, ] <- admixed$xtrain_lasso[ ,-c(1:p_design)]
X[admixed$tune_ind, ] <- admixed$xtune_lasso[ ,-c(1:p_design)]
X[admixed$test_ind, ] <- admixed$xtest_lasso[ ,-c(1:p_design)]

# Genetic predictors
G <- array(dim = c(n, p_design))
colnames(G) <- colnames(admixed$xtrain_lasso[ , 1:p_design])
G[admixed$train_ind, ] <- admixed$xtrain_lasso[ ,1:p_design]
G[admixed$tune_ind, ] <- admixed$xtune_lasso[ , 1:p_design]
G[admixed$test_ind, ] <- admixed$xtest_lasso[ , 1:p_design]

# CSV file containing covariates
final_dat <- cbind(IID = paste0("ID", 1:n), X, set = sapply(1:n, function(i) ifelse(i %in% admixed$train_ind, "train", ifelse(i %in% admixed$tune_ind, "tune", "test"))), y)
write.csv(final_dat, paste0(args[13], "covariate.txt"), quote = FALSE, row.names = FALSE)

# CSV file containing genetic predictors
write.csv(G, paste0(args[13], "snps.txt"), quote = FALSE, row.names = FALSE)

# CSV file containing beta (on original genotype scale) for each SNP
beta <- admixed$beta / admixed$std
write.csv(cbind(beta = beta), paste0(args[13], "betas.txt"), quote = FALSE, row.names = FALSE)

# CSV file containing beta (on original genotype scale) for each SNP
gamma <- admixed$gamma / admixed$std
write.csv(cbind(gamma = gamma), paste0(args[13], "gammas.txt"), quote = FALSE, row.names = FALSE)
