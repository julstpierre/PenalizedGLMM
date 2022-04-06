#Load libraries
library(dplyr)
library(GMMAT)
library(data.table)

#Read phenotype and covariates file
pheno.cov <- read.table("covariate.txt", sep=",", header = T)

#Read GRM matrix
trainrowinds <- which(pheno.cov$set %in% c("train"))
GRM <- as.matrix(fread("grm.txt.gz"))[trainrowinds, trainrowinds]
colnames(GRM) <- pheno.cov[trainrowinds,"IID"]
rownames(GRM) <- pheno.cov[trainrowinds,"IID"]

#Fit null GLMM with GRM only
model0 <- glmmkin(y ~ SEX + AGE
                  ,data = pheno.cov[trainrowinds,]
                  ,kins = GRM
                  ,id = "IID"
                  ,family = binomial(link = "logit")
)

#Save variance components estimates
write.table(cbind(tau=model0$theta["kins1"]), "GMMAT_kins.txt", quote=FALSE, row.names = FALSE)
