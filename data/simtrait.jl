# ========================================================================
# Code for simulating binary traits from related UKBB subjects
# ========================================================================
using CSV, DataFrames, SnpArrays, DataFramesMeta, StatsBase, LinearAlgebra, Distributions, CodecZlib

# ------------------------------------------------------------------------
# Initialize parameters
# ------------------------------------------------------------------------
# Assign default command-line arguments
const ARGS_ = isempty(ARGS) ? ["0.5", "0.4", "0", "0.1", "5000", "0.01", "NONE", ""] : ARGS

# Fraction of variance due to fixed polygenic additive effect (logit scale)
h2_g = parse(Float64, ARGS_[1])

# Fraction of variance due to random polygenic additive effect (logit scale)
h2_b = parse(Float64, ARGS_[2])

# Fraction of variance due to unobserved shared environmental effect (logit scale)
h2_d = parse(Float64, ARGS_[3])

# Prevalence
pi0 = parse(Float64, ARGS_[4])

# Number of snps to randomly select accros genome
p = parse(Int, ARGS_[5])

# Percentage of causal SNPs
c = parse(Float64, ARGS_[6])

# Number of snps to use for GRM estimation
p_kin = 50000

# ------------------------------------------------------------------------
# Load the covariate file
# ------------------------------------------------------------------------
# Read plink fam file
samples = @chain CSV.read("UKBB/UKBB.fam", DataFrame; header = false) begin  
    @select!(:FID = :Column1, :IID = :Column2)
end

# Combine into a DataFrame
dat = @chain CSV.read("UKBB/covariate.txt", DataFrame) begin
	rightjoin(samples, on = [:IID, :FID])
    @select!(:FID, :IID, :SEX, :AGE, :PCA1, :PCA2, :PCA3, :PCA4, :PCA5, :PCA6, :PCA7, :PCA8, :PCA9, :PCA10, :famid)
    rename!(:PCA1 => :PC1, :PCA2 => :PC2, :PCA3 => :PC3, :PCA4 => :PC4, :PCA5 => :PC5, :PCA6 => :PC6, :PCA7 => :PC7, :PCA8 => :PC8, :PCA9 => :PC9, :PCA10 => :PC10)
end
n = nrow(dat)

# Randomly sample subjects by family for training, tune and test sets
nfams = length(unique(dat.famid))
train_ids = [19; sample(unique(dat.famid), Int(floor(nfams * 0.4)); replace = false)]
tune_ids = sample(setdiff(unique(dat.famid), train_ids), Int(ceil(length(setdiff(unique(dat.famid), train_ids)) * 0.5)); replace = false)
test_ids = setdiff(unique(dat.famid), [train_ids; tune_ids])

# Add indicator variable for training subjects
dat.set = Array{String, 1}(undef, n)
dat.set[[dat.famid[i] in train_ids for i in 1:size(dat, 1)]] .= "train"
dat.set[[dat.famid[i] in tune_ids for i in 1:size(dat, 1)]] .= "tune"
dat.set[[dat.famid[i] in test_ids for i in 1:size(dat, 1)]] .= "test"

train = [dat.famid[i] in train_ids for i in 1:size(dat, 1)]

sum(dat.set .== "train") / n
sum(dat.set .== "tune") / n
sum(dat.set .== "test") / n

#-------------------------------------------------------------------------
# Load genotype Data
#-------------------------------------------------------------------------
# Read plink bim file
UKBB = SnpArray("UKBB/UKBB.bed")

# Remove SNPs with MAF = 0 or 0.5 either in the train set or in the train+test set
_maf = DataFrame()
_maf.ALL = maf(@view(UKBB[:, :]))
_maf.ALLtrain = maf(@view(UKBB[train, :]))
snps = findall((_maf.ALL .!= 0) .& (_maf.ALL .!= 0.5) .& (_maf.ALLtrain .!= 0) .& (_maf.ALLtrain .!= 0.5))

# Sample p candidate SNPs randomly accross genome, convert to additive model and impute
snp_inds = sample(snps, p, replace = false, ordered = true)
G = convert(Matrix{Float64}, @view(UKBB[:, snp_inds]), impute = true)

# Save filtered plink file
rowmask, colmask = trues(n), [col in snp_inds for col in 1:size(UKBB, 2)]
SnpArrays.filter("UKBB/UKBB", rowmask, colmask, des = ARGS_[8] * "geno")

if ARGS_[7] == "ALL"
    # Causal SNPs are included in the GRM
    grm_inds = sample(setdiff(snps, snp_inds), p_kin - p, replace = false, ordered = true) |>
               x -> [x; snp_inds] |>
               sort
elseif ARGS_[7] == "NONE"
    # Causal SNPs are excluded from the GRM
    grm_inds = sample(setdiff(snps, snp_inds), p_kin, replace = false, ordered = true)
end

# Estimated GRM
GRM = 2 * grm(UKBB, cinds = grm_inds)

# Make sure GRM is posdef
function posdef(K, n = size(K, 1), xi = 1e-4)
    while !isposdef(K)
        K = K + xi * Diagonal(ones(n))
        xi = 10 * xi
    end
    return(K)
end
GRM = posdef(GRM)
    
# Save GRM in compressed file
open(GzipCompressorStream, ARGS_[8] * "grm.txt.gz", "w") do stream
    CSV.write(stream, DataFrame(GRM, :auto))
end

# #--------------------------------------------------------------------
# # Calculate PCs on the training subjects and project them on test set
# #--------------------------------------------------------------------
# # Compute singular value decomposition on training set only
# X_grm = convert(Matrix{Float64}, @view(UKBB[:, grm_inds]), impute = true, center = true, scale = true)
# U, S, V = svd(X_grm[train,:])

# # PCs for training set are columns of U
# PCs = Array{Float64}(undef, n, 10)
# PCs[train,:] = U[:,1:10]

# # PCs for tune and test sets are obtained as projection from training PCs
# PCs[train .!= 1,:] = (X_grm[train .!= 1,:] * V * inv(Diagonal(S)))[:,1:10]

# ------------------------------------------------------------------------
# Simulate phenotypes
# ------------------------------------------------------------------------
# Variance components
sigma2_e = pi^2 / 3 + log(1.3)^2 * var(dat.SEX) + log(1.05)^2 * var(dat.AGE / 10)
sigma2_g = h2_g / (1 - h2_g - h2_b - h2_d) * sigma2_e
sigma2_b = h2_b / (1 - h2_g - h2_b - h2_d) * sigma2_e
sigma2_d = h2_d / (1 - h2_g - h2_b - h2_d) * sigma2_e

# Simulate fixed effects for randomly sampled causal snps
W = zeros(p)
s = sample(1:p, Integer(round(p*c)), replace = false)
W[s] .= sigma2_g/length(s)
beta = rand.([Normal(0, sqrt(W[i])) for i in 1:p])

# Simulate random effects
b = h2_b > 0 ? rand(MvNormal(sigma2_b * GRM)) : zeros(n)

# Standardize G
mu = mean(G, dims = 1) 
s = std(G, dims = 1, corrected = false)
G = (G .- mu) ./ s

# Simulate binary traits
logit(x) = log(x / (1 - x))
expit(x) = exp(x) / (1 + exp(x))
final_dat = @chain dat begin
	@transform!(:logit_pi = logit(pi0) .- log(1.3) * :SEX + log(1.05) * (:AGE / 10) + G * beta + b)
    @transform!(:pi = expit.(:logit_pi))
    @transform(:y = rand.([Binomial(1, :pi[i]) for i in 1:nrow(dat)]))
    select!(Not([:pi, :logit_pi]))
end

# Prevalence
println("Prevalence is ", round(mean(final_dat.y), digits = 3))

#----------------------
# Write csv files
#---------------------
# CSV file containing covariates
CSV.write(ARGS_[8] * "covariate.txt", final_dat)

# Convert simulated effect for each SNP on original genotype scale
df = SnpData("UKBB/UKBB").snp_info[snp_inds, [1,4]]
df.beta = [ beta[i] / s[i] for i in 1:p ]

# CSV file containing MAFs and simulated effect for each SNP
CSV.write(ARGS_[8] * "betas.txt", df)