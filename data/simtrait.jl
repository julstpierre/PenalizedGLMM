# ========================================================================
# Code for simulating binary traits with environmental exposure from 1000G data
# ========================================================================
using CSV, DataFrames, SnpArrays, DataFramesMeta, StatsBase, LinearAlgebra, Distributions, CodecZlib

# ------------------------------------------------------------------------
# Initialize parameters
# ------------------------------------------------------------------------
# Assign default command-line arguments
const ARGS_ = isempty(ARGS) ? ["0.8", "0", "0.1", "10000", "0.005", "ALL", ""] : ARGS

# Fraction of variance due to polygenic additive effect (logit scale)
h2_g = parse(Float64, ARGS_[1])

# Fraction of residual variance due to unobserved shared environmental effect (logit scale)
h2_d = parse(Float64, ARGS_[2])

# Prevalence
pi0 = parse(Float64, ARGS_[3])	

# Number of snps to randomly select accros genome
p = parse(Int, ARGS_[4])

# Percentage of causal SNPs
c = parse(Float64, ARGS_[5])

# Number of snps to use for GRM estimation
p_kin = 50000

# ------------------------------------------------------------------------
# Load the covariate file
# ------------------------------------------------------------------------
# Read plink fam file
samples = @chain CSV.read("1000G/1000G.fam", DataFrame; header = false) begin  
    @select!(:FID = :Column1, :IID = :Column2)
end
n = size(samples, 1)

# Combine into a DataFrame
dat = @chain CSV.read("1000G/covars.csv", DataFrame) begin
    @transform!(:FID = 0, :IID = :ind, :SEX = 1 * (:gender .== "male"), :POP = :super_pop, :AGE = round.(rand(Normal(50, 5), n), digits = 0))
	rightjoin(samples, on = [:IID, :FID])
    @select!(:FID, :IID, :POP, :SEX, :AGE, :PC1, :PC2, :PC3, :PC4, :PC5, :PC6, :PC7, :PC8, :PC9, :PC10)
end	    	  

# Randomly sample subjects by POP for training and test sets
grpdat = groupby(dat, :POP)
train_ids = [sample(grpdat[i].IID, Int(ceil(nrow(grpdat[i]) * 0.80)); replace = false) for i in 1:length(grpdat)] |>
                x -> reduce(vcat, x)

# Add indicator variable for training subjects
dat.train = [dat.IID[i] in train_ids for i in 1:n]

#-------------------------------------------------------------------------
# Load genotype Data
#-------------------------------------------------------------------------
# Read plink bim file
_1000G = SnpArray("1000G/1000G.bed")

# Compute MAF in training set for each SNP and in each Population
_maf = DataFrame()
_maf.EUR = maf(@view(_1000G[(dat.POP .== "EUR") .& dat.train, :]))
_maf.EAS = maf(@view(_1000G[(dat.POP .== "EAS") .& dat.train, :]))
_maf.AMR = maf(@view(_1000G[(dat.POP .== "AMR") .& dat.train, :]))
_maf.SAS = maf(@view(_1000G[(dat.POP .== "SAS") .& dat.train, :]))
_maf.AFR = maf(@view(_1000G[(dat.POP .== "AFR") .& dat.train, :]))

# Remove snps with MAF = 0.5 in the training data
_maf.ALL = maf(@view(_1000G[dat.train, :]))
snps = setdiff(axes(_1000G, 2), findall(_maf.ALL .== 0.50))

# Compute range for MAFs among the 5 populations
_maf.range = vec(maximum([_maf.EUR _maf.EAS _maf.AMR _maf.SAS _maf.AFR], dims = 2) - minimum([_maf.EUR _maf.EAS _maf.AMR _maf.SAS _maf.AFR], dims = 2))

# Sample p candidate SNPs randomly accross genome, convert to additive model and impute
snp_inds = sample(snps,  weights(_maf.range / sum(_maf.range))[snps], p, replace = false, ordered = true)
G = convert(Matrix{Float64}, @view(_1000G[:, snp_inds]), impute = true)

# Save filtered plink file
rowmask, colmask = trues(n), [col in snp_inds for col in 1:size(_1000G, 2)]
SnpArrays.filter("1000G/1000G", rowmask, colmask, des = ARGS_[7] * "geno")

if ARGS_[6] == "ALL"
    # Causal SNPs are included in the GRM
    grm_inds = sample(setdiff(snps, snp_inds), p_kin - p, replace = false, ordered = true) |>
               x -> [x; snp_inds] |>
               sort
elseif ARGS_[6] == "NONE"
    # Causal SNPs are excluded from the GRM
    grm_inds = sample(setdiff(snps, snp_inds), p_kin, replace = false, ordered = true)
end

# Estimated GRM
GRM = 2 * grm(_1000G, cinds = grm_inds)

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
open(GzipCompressorStream, ARGS_[7] * "grm.txt.gz", "w") do stream
    CSV.write(stream, DataFrame(GRM, :auto))
end

# ------------------------------------------------------------------------
# Simulate phenotypes
# ------------------------------------------------------------------------
# Variance components
sigma2_e = pi^2 / 3 + log(1.3)^2 * var(dat.SEX) + log(1.05)^2 * var(dat.AGE / 10)
sigma2_g = 1/2 * h2_g / (1 - h2_g - h2_d) * sigma2_e
sigma2_d = h2_d / (1 - h2_g - h2_d) * sigma2_e

# Simulate fixed effects for randomly sampled causal snps
W = zeros(p)
s = sample(1:p, Integer(round(p*c)), replace = false)
W[s] .= sigma2_g/length(s)
beta = rand.([Normal(0, sqrt(W[i])) for i in 1:p])

# Simulate fixed effect for dichotomous environmental effect
Z = (dat.POP .== ["SAS" "AFR"]) |>
    x -> x ./ std(x, dims=1, corrected = false)
gamma = rand(Normal(0, sqrt(sigma2_d/2)), 2)

# Simulate random effects
b = rand(MvNormal(sigma2_g * GRM))

# Standardize G
mu = mean(G, dims = 1) 
s = std(G, dims = 1, corrected = false)
G = (G .- mu) ./ s

# Simulate binary traits
logit(x) = log(x / (1 - x))
expit(x) = exp(x) / (1 + exp(x))
final_dat = @chain dat begin
	@transform!(:logit_pi = logit(pi0) .- log(1.3) * :SEX + log(1.05) * (:AGE / 10) + G * beta + Z * gamma + b)
    @transform!(:pi = expit.(:logit_pi))
    @transform(:y = rand.([Binomial(1, :pi[i]) for i in 1:nrow(dat)]))
    select!(Not([:pi, :logit_pi]))
end

# Prevalence by Population
println("Prevalence is ", round(mean(final_dat.y), digits = 3))
println(combine(groupby(final_dat, :POP), :y => mean))

#----------------------
# Write csv files
#---------------------
# CSV file containing covariates
CSV.write(ARGS_[7] * "covariate.txt", final_dat)

# Convert simulated effect for each SNP on original genotype scale
df = SnpData("1000G/1000G").snp_info[snp_inds, [1,4]]
df.beta = [ beta[i] / s[i] for i in 1:p ]

# Save MAF for each SNP and in each Population
df.mafEUR = _maf.EUR[snp_inds]
df.mafEAS = _maf.EAS[snp_inds]
df.mafAMR = _maf.AMR[snp_inds]
df.mafSAS = _maf.SAS[snp_inds]
df.mafAFR = _maf.AFR[snp_inds]
df.mafrange = _maf.range[snp_inds]

# CSV file containing MAFs and simulated effect for each SNP
CSV.write(ARGS_[7] * "betas.txt", df)