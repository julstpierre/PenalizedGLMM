rm(list=ls())

#======================================================================
# Load packages
#======================================================================
library(dplyr)
library(bigsnpr)
library(ggmix)
library(data.table)

#=======================================================================
# Load the phenotype data
#=======================================================================
#Read phenotype and covariates file
pheno.cov <- read.table("covariate.txt", sep=",", header = T)
trainrowinds <- which(pheno.cov$train %in% c(TRUE, "true"))
n <- length(trainrowinds)

#=======================================================================
# Load the genotype data
#=======================================================================
if (file.exists("geno.bed")){
  tmpfile <- tempfile()
  snp_readBed("geno.bed",backingfile = tmpfile)
  
  # Attach the "bigSNP" object in R session
  obj.bigSNP <- snp_attach(paste0(tmpfile, ".rds"))
  p <- nrow(obj.bigSNP$map)
  G <- bigsnpr::snp_fastImputeSimple(obj.bigSNP$genotypes)[,1:p]
  
} else if (file.exists("snps.txt")){
  G <- read.csv("snps.txt")
  p <- ncol(G)
}

# Standardize genotype matrix
s <- apply(G, 2, sd)
G <- scale(G)
Gtrain <- G[trainrowinds,]

#Read GRM matrix
GRM <- as.matrix(fread("grm.txt.gz"))
colnames(GRM) <- pheno.cov$IID
rownames(GRM) <- pheno.cov$IID

#=======================================================================
# GGMIX
#=======================================================================
Xtrain <- pheno.cov[trainrowinds, c("AGE","SEX")]
ytrain <- as.matrix(pheno.cov$y)[trainrowinds]

fit_ggmix <- ggmix(x = as.matrix(cbind(Xtrain, Gtrain)),
                   y = ytrain,
                   standardize = TRUE,		
                   kinship=GRM[trainrowinds, trainrowinds],
                   penalty.factor = c(rep(0,ncol(Xtrain)),rep(1,p))
)

# Find lambda that gives minimum GIC				  
aic <- ggmix::gic(fit_ggmix, an = 2)
bic <- ggmix::gic(fit_ggmix, an = log(n))

# Save betas for ggmix with different GIC criteria
ggmixAIC_beta <- 1/s * coef(aic)[setdiff(rownames(coef(aic)), c("(Intercept)","AGE","SEX","eta","sigma2")),]
ggmixBIC_beta <- 1/s * coef(bic)[setdiff(rownames(coef(bic)), c("(Intercept)","AGE","SEX","eta","sigma2")),]

# Read file with real values
true_betas = read.csv("betas.txt")$beta
ggmix_betas = 1/s * fit_ggmix$beta[-(1:ncol(Xtrain)),]

# Predict phenotype on test set
Xtest <- pheno.cov[-trainrowinds, c("AGE","SEX")]
Gtest <- G[-trainrowinds,]
ggmixAIC_yhat <- predict(aic, as.matrix(cbind(Xtest, Gtest)), covariance = GRM[-trainrowinds, trainrowinds])
ggmixBIC_yhat <- predict(bic, as.matrix(cbind(Xtest, Gtest)), covariance = GRM[-trainrowinds, trainrowinds])

#Save results
write.csv(cbind(ggmixAIC = ggmixAIC_beta, ggmixBIC = ggmixBIC_beta), "ggmix_results.txt", quote=FALSE, row.names = FALSE)
write.csv(cbind(ggmixAIC = ggmixAIC_yhat, ggmixBIC = ggmixBIC_yhat), "ggmix_fitted_values.txt", quote=FALSE, row.names = FALSE)
write.csv(t(coef(aic, type = "nonzero")[c("eta", "sigma2"),]), "ggmix_tau.txt", quote=FALSE, row.names = FALSE)

# False positive rate (FPR)
for (fpr in seq(0,0.01,0.001)){
  
  #Predict y at a given FPR
  v <- apply((ggmix_betas != 0) & (true_betas == 0), 2, sum)/sum(true_betas == 0) <= fpr
  ggmixFPR_beta <- ggmix_betas[,tapply(seq_along(v), v, max)["TRUE"]]
  ggmixFPR_yhat <- predict(fit_ggmix, as.matrix(cbind(Xtest, Gtest)), covariance = GRM[-trainrowinds, trainrowinds])[,tapply(seq_along(v), v, max)["TRUE"]]
  
  #Save results
  write.csv(cbind(ggmixFPR = ggmixFPR_beta), paste0("ggmix_results_fpr", fpr, ".txt"), quote=FALSE, row.names = FALSE)
  write.csv(cbind(ggmixFPR = ggmixFPR_yhat), paste0("ggmix_fitted_values_fpr", fpr, ".txt"), quote=FALSE, row.names = FALSE)
}