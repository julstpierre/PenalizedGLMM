#Load libraries
library(dplyr)
library(GMMAT)
library(data.table)

#Read phenotype and covariates file
pheno.cov <- read.table("covariate.txt", sep=",", header = T) %>%
  mutate(ID=paste(FID,":",IID,sep="")) %>%
  filter(train == "true")

#Read GRM matrix
traininds = which(pheno.cov$train == "true")
GRM <- as.matrix(fread("grm.txt.gz"))[traininds, traininds]
colnames(GRM) <- pheno.cov$ID
rownames(GRM) <- pheno.cov$ID

#Fit null GLMM with GRM only
model0 <- glmmkin(y ~ SEX + AGE
                  ,data = pheno.cov
                  ,kins = GRM
                  ,id = "ID"
                  ,family = binomial(link = "logit")
)

#Save variance components estimates
write.table(cbind(tau=model0$theta["kins1"]), "GMMAT_kins.txt", quote=FALSE, row.names = FALSE)

