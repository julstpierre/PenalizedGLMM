# ------------------------------------------------------------------------
# Initialize parameters
# ------------------------------------------------------------------------
# Assign default command-line arguments
args <- commandArgs(TRUE)
# args = c("0.5", "0.2", "0", "0.1", "10000", "0.005", "10", "NONE", "")

# Fraction of variance due to fixed polygenic additive effect (logit scale)
h2_g <- as.numeric(args[1])

# Fraction of variance due to random polygenic additive effect (logit scale)
h2_b <- as.numeric(args[2])

# Fraction of variance due to unobserved shared environmental effect (logit scale)
h2_d <- as.numeric(args[3])

# Prevalence
pi0 <- as.numeric(args[4])

# Number of snps to randomly select accros genome
p_design <- as.numeric(args[5])

# Percentage of causal SNPs
percent_causal <- as.numeric(args[6])

# Number of populations
K <- as.numeric(args[7])

# Overlap between design and kinship SNPs
percent_overlap <- ifelse(args[8] == "ALL", "100", ifelse(args[8] == "NONE", "0"))

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
                                geography = "1d",
                                percent_causal = percent_causal,
                                percent_overlap = percent_overlap,
                                k = K, s = 0.5, Fst = 1/K,
                                b0 = pi0, nPC = 10,
                                h2_g = h2_g, h2_b = h2_b,
                                train_tune_test = c(0.8, 0, 0.2)
)

#-----------------------
# PCA plot
#----------------------
#library(ggplot2)
#ggplot(data.frame(admixed$PC, pop = as.character(admixed$subpops)), aes(PC1, PC2, col = pop)) + 
#  geom_point(size = 3, show.legend = FALSE) +
#  xlab("PC1") + ylab("PC2")

#-----------------------------------------
# Write kinship to txt.gz compressed file
#-----------------------------------------
gz <- gzfile(paste0(args[9], "grm.txt.gz"), "w")
write.csv(admixed$kin, gz, row.names = FALSE)
close(gz)

#----------------------
# Write csv files
#----------------------
# Phenotype
y <- vector("numeric", length = n)
y[admixed$train_ind] <- admixed$ytrain
y[admixed$test_ind] <- admixed$ytest

# Covariates
X <- array(dim = c(n, ncol(admixed$xtrain_lasso) - p_design))
colnames(X) <- colnames(admixed$xtrain_lasso[ ,-c(1:p_design)])
X[admixed$train_ind, ] <- admixed$xtrain_lasso[ ,-c(1:p_design)]
X[admixed$test_ind, ] <- admixed$xtest_lasso[ ,-c(1:p_design)]

# Genetic predictors
G <- array(dim = c(n, p_design))
colnames(G) <- colnames(admixed$xtrain_lasso[ , 1:p_design])
G[admixed$train_ind, ] <- admixed$xtrain_lasso[ ,1:p_design]
G[admixed$test_ind, ] <- admixed$xtest_lasso[ , 1:p_design]

# CSV file containing covariates
final_dat <- cbind(IID = paste0("ID", 1:n), X, train = c(1:n) %in% admixed$train_ind, y)
write.csv(final_dat, paste0(args[9], "covariate.txt"), quote = FALSE, row.names = FALSE)

# CSV file containing genetic predictors
write.csv(G, paste0(args[9], "snps.txt"), quote = FALSE, row.names = FALSE)

# CSV file containing beta (on original genotype scale) for each SNP
beta = admixed$beta / admixed$std
write.csv(cbind(beta = beta), paste0(args[9], "betas.txt"), quote = FALSE, row.names = FALSE)
