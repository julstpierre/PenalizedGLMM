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
pheno.cov <- read.table("covariate.txt", sep=",", header = T) %>%
			 mutate(ID=paste(FID,":",IID,sep=""))
			 
n <- nrow(pheno.cov)	
		 
#=======================================================================
# Load the genotype data
#=======================================================================
tmpfile <- tempfile()
snp_readBed("geno.bed",backingfile = tmpfile)

# Attach the "bigSNP" object in R session
obj.bigSNP <- snp_attach(paste0(tmpfile, ".rds"))
p <- nrow(obj.bigSNP$map)
G <- bigsnpr::snp_fastImputeSimple(obj.bigSNP$genotypes)[,1:p] %>% scale()

#Read GRM matrix
GRM <- as.matrix(fread("grm.txt.gz"))
colnames(GRM) <- pheno.cov$ID
rownames(GRM) <- pheno.cov$ID

#=======================================================================
# GGMIX
#=======================================================================
fit_ggmix <- ggmix(x = as.matrix(cbind(pheno.cov[,c("AGE","SEX")],G)),
					 y = as.matrix(pheno.cov$y),
					 standardize = T,		
					 kinship=GRM,
					 penalty.factor = c(rep(0,2),rep(1,p))
				  )

# Find lambda that gives minimum GIC				  
aic <- ggmix::gic(fit_ggmix, an = 2)
bic <- ggmix::gic(fit_ggmix, an = log(n))

# Save betas for ggmix with different GIC criteria
ggmixAIC_beta <- coef(aic)[setdiff(rownames(coef(aic)), c("(Intercept)","AGE","SEX","eta","sigma2")),]
ggmixBIC_beta <- coef(bic)[setdiff(rownames(coef(bic)), c("(Intercept)","AGE","SEX","eta","sigma2")),]

# Read file with real values
true_betas = read.csv("betas.txt")$beta
ggmix_betas = fit_ggmix$beta[-c(1,2),]

# False positive rate (FPR) at 1%
v <- apply((ggmix_betas != 0) & (true_betas == 0), 2, sum)/sum(true_betas == 0) < 0.01
ggmixFPR_beta <- ggmix_betas[,tapply(seq_along(v), v, max)["TRUE"]]

#Save results
write.csv(cbind(ggmixAIC_beta, ggmixBIC_beta, ggmixFPR_beta), "ggmix_results.txt", quote=FALSE, row.names = FALSE)
write.csv(t(coef(aic, type = "nonzero")[c("eta", "sigma2"),]), "ggmix_tau.txt", quote=FALSE, row.names = FALSE)