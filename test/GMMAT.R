#Load libraries
library(dplyr)
library(GMMAT)
library(data.table)

#Read phenotype and covariates file
pheno.cov <- read.table("covariate.txt", sep=",", header = T) %>%
			 mutate(ID=paste(FID,":",IID,sep=""))

#Read GRM matrix
GRM <- as.matrix(fread("~/projects/def-bhatnaga/gf591137/PenalizedGLMM/data/grm.txt.gz"))
colnames(GRM) <- pheno.cov$ID
rownames(GRM) <- pheno.cov$ID

#Fit null GLMM
model0 <- glmmkin(y ~ SEX + AGE
					 ,data = pheno.cov
					 ,kins = GRM
					 ,id = "ID"
					 ,family = binomial(link = "logit")
					 )
					 
#Save variance components estimates
write.table(model0$theta["kins1"], "GMMAT_kins.txt", quote=FALSE, row.names = FALSE, col.names=FALSE)