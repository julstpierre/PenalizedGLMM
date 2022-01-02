#Load libraries
library(dplyr)
library(GMMAT)
library(data.table)

#Command-line argument
args<-commandArgs(TRUE)
method <- args[1]

#Read phenotype and covariates file
pheno.cov <- read.table("covariate.txt", sep=",", header = T) %>%
			 mutate(ID=paste(FID,":",IID,sep=""))

#Read GRM matrix
GRM <- as.matrix(fread("~/projects/def-bhatnaga/gf591137/PenalizedGLMM/data/grm.txt.gz"))
colnames(GRM) <- pheno.cov$ID
rownames(GRM) <- pheno.cov$ID

if (method == "1RE"){
  #Fit null GLMM with GRM only
  model0 <- glmmkin(y ~ SEX + AGE
  					 ,data = pheno.cov
  					 ,kins = GRM
  					 ,id = "ID"
  					 ,family = binomial(link = "logit")
  					 )
  
  #Save variance components estimates
  write.table(cbind(tau=model0$theta["kins1"]), "GMMAT_kins.txt", quote=FALSE, row.names = FALSE)
  
} else if (method == "2REs"){
  # Environment relatedness matrix
  n = nrow(pheno.cov)
  K_D = matrix(nrow=n, ncol=n)
  for (i in 1:n){ 
    for (j in i:n){
      K_D[i, j] = ifelse(pheno.cov$Exposed[i] == pheno.cov$Exposed[j], 1, 0)
    }
  }
  K_D[lower.tri(K_D)] <- t(K_D)[lower.tri(K_D)]
  colnames(K_D) <- pheno.cov$ID
  rownames(K_D) <- pheno.cov$ID
  
  #Fit null GLMM with GRM and one random effect
  model0 <- glmmkin(y ~ SEX + AGE
                    ,data = pheno.cov
                    ,kins = list(GRM, K_D)
                    ,id = "ID"
                    ,family = binomial(link = "logit")
  )
  
  #Save variance components estimates
  write.table(cbind(tau = model0$theta[c("kins1", "kins2")]), "GMMAT_kins.txt", quote=FALSE, row.names = FALSE)
}