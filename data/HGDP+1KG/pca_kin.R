# load packages
library(GENESIS)
library(SNPRelate)
library(GWASTools)
library(SeqArray)
library(dplyr)
library(RColorBrewer)
library(GGally)

# Convert plink files to gds
# snpgdsBED2GDS(bed.fn = "HGDP+1KG/HGDP+1KG.bed", 
#               bim.fn = "HGDP+1KG/HGDP+1KG.bim", 
#               fam.fn = "HGDP+1KG/HGDP+1KG.fam", 
#               out.gdsfn = "HGDP+1KG/HGDP+1KG.gds")

gds <- snpgdsOpen("HGDP+1KG.gds")

# run LD pruning
# set.seed(100) # LD pruning has a random element; so make this reproducible
# snpset <- snpgdsLDpruning(gds,
#                           method="corr", 
#                           slide.max.bp=10e6, 
#                           ld.threshold=sqrt(0.1)
#                          )
# pruned <- unlist(snpset, use.names=FALSE)
# saveRDS(pruned, "pruned_snps.rds")

pruned <- readRDS("pruned_snps.rds")

# King robust estimation
king <- snpgdsIBDKING(gds, snp.id=pruned)
kingMat <- king$kinship
colnames(kingMat) <- rownames(kingMat) <- king$sample.id

# PC-Air
gdsfmt::showfile.gds(closeall=TRUE)
geno <- GdsGenotypeReader(filename = "HGDP+1KG.gds")
genoData <- GenotypeData(geno)

pca <- pcair(genoData,
             snp.include = pruned,
             kinobj = kingMat,
             kin.thresh=2^(-9/2),
             divobj = kingMat,
             div.thresh=-2^(-9/2)
             )

# Make a plot with the PCs
pcs <- data.frame(pca$vectors[,1:20])
colnames(pcs) <- paste0('PC', 1:20)
pcs$sample.id <- pca$sample.id

pc.df <- read.csv("covars.csv") %>%
  select(super_pop, ind) %>%
  rename(sample.id = ind, Population = super_pop) %>%
  left_join(pcs, by = "sample.id")

pop.cols <- setNames(brewer.pal(7, "Paired"),
                     c("AFR", "AMR", "CSA", "EAS", "EUR", "MID", "OCE"))
ggparcoord(pc.df, columns=3:22, groupColumn="Population", scale="uniminmax") +
  scale_color_manual(values=pop.cols) +
  xlab("PC") + ylab("")

# PC-Relate
genoData_it <- GenotypeBlockIterator(genoData, snpInclude=pruned)
mypcrelate <- pcrelate(genoData_it, pcs = pca$vectors[, 1:10, drop = FALSE], 
                       training.set = pca$unrels,
                       BPPARAM = BiocParallel::SerialParam())

# Convert to GRM
pcrelMat <- pcrelateToMatrix(mypcrelate, 
                             scaleKin=1, 
                             verbose=FALSE
                             )

# Write matrix with sample ids
data.table::fwrite(as.matrix(pcrelMat), file = "grm.txt")
system("gzip grm.txt")
write.table(rownames(pcrelMat), file="grm_ids.txt", quote = F, row.names = F, col.names = F)

# Convert to sparse GRM
pcrelMatsp <- pcrelateToMatrix(mypcrelate, 
                             scaleKin=1, 
                             verbose=FALSE,
                             thresh = 2^(-9/2)
)
saveRDS(pcrelMatsp, "sparse_grm.rds")
