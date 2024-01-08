# Load libraries
library(softImpute)
library(dplyr)

# Import GRM
GRM_full <- read.csv("grm.txt.gz", header = TRUE) %>% as.matrix()

# Find largest singular value
fit0 = svd.als(GRM_full, rank = 1)
lam0 <- fit0$d
lamseq=exp(seq(from=log(lam0),to=log(1),length=10))

# Create a sequence of values for lambda
fits=as.list(lamseq)
ranks=as.integer(lamseq)
rank.max=2
warm=NULL

for( i in seq(along=lamseq)){
  fiti=svd.als(GRM_full,lambda=lamseq[i],rank=rank.max,warm=warm)
  ranks[i]=sum(round(fiti$d,4)>0)
  warm=fiti
  fits[[i]]=fiti
  cat(i,"lambda=",lamseq[i],"rank.max",rank.max,"rank",ranks[i],"\n")
  rank.max=ranks[i]+100
}
