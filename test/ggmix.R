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
hdbic <- ggmix::gic(fit_ggmix, an = log(log(n)) * log(p))

# Save betas for ggmix with HDBIC criteria
ggmixHDBIC_beta <- coef(hdbic)[setdiff(rownames(coef(hdbic)), c("(Intercept)","AGE","SEX","eta","sigma2")),]

# Read file with real values
true_betas = read.csv("betas.txt")$beta
ggmix_betas = fit_ggmix$beta[-c(1,2),]

# False positive rate (FPR) at 5%
v <- apply((ggmix_betas != 0) & (true_betas == 0), 2, mean) < 0.05
ggmixFPR5_beta <- ggmix_betas[,tapply(seq_along(v), v, max)["TRUE"]]

#Save results
write.csv(cbind(ggmixHDBIC_beta, ggmixFPR5_beta), "ggmix_results.txt", quote=FALSE, row.names = FALSE)
write.csv(t(coef(hdbic, type = "nonzero")[c("eta", "sigma2"),]), "ggmix_tau.txt", quote=FALSE, row.names = FALSE)