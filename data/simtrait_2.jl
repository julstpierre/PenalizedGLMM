# ========================================================================
# Code for simulating binary traits from UKBB data with shared environment effects resulting from sampling design 
# ========================================================================
using CSV, DataFrames, SnpArrays, DataFramesMeta, StatsBase, LinearAlgebra, Distributions, CodecZlib

# ------------------------------------------------------------------------
# Initialize parameters
# ------------------------------------------------------------------------
# Assign default command-line arguments
const ARGS_ = isempty(ARGS) ? ["0.3", "0.3", "0.1", "0.1", "10000", "0.003", "20", ""] : ARGS

# Fraction of variance due to polygenic additive effect (logit scale)
h2_g = parse(Float64, ARGS_[1])

# Fraction of residual variance due to unobserved shared environmental effect (logit scale)
h2_d = parse(Float64, ARGS_[2])

# Prevalence for Non Caucasian
pi0 = parse(Float64, ARGS_[3])	

# Prevalence for Caucasian
pi1 = parse(Float64, ARGS_[4])	

# Number of snps to randomly select accros genome
p = parse(Int, ARGS_[5])

# Percentage of causal SNPs
c = parse(Float64, ARGS_[6])

# Number of sampling units
m = parse(Int64, ARGS_[7]) 

# ------------------------------------------------------------------------
# Load the covariate file
# ------------------------------------------------------------------------
# Read plink fam file
samples = @chain CSV.read("UKBB.fam", DataFrame; header = false) begin  
    @select!(:FID = :Column1, :IID = :Column2)
end

# Caucasian individuals
caucasians = @chain CSV.read("include_Caucasian.txt", DataFrame; header = false) begin
    @select!(:FID = :Column1, :IID = :Column2)
    @transform(:ETHNICITY = "Caucasian", :CAUCASIAN = 1)
end
    
# Non-caucasian individuals
non_caucasians = @chain CSV.read("include_notCaucasian.txt", DataFrame; header = false) begin
    @select!(:FID = :Column1, :IID = :Column2)
    @transform!(:ETHNICITY = "Non-Caucasian", :CAUCASIAN = 0)
end

# Combine into a DataFrame
dat = @chain CSV.read("covars_full.txt", DataFrame) begin
    @select!(:FID, :IID, :SEX, :AGE, :PCA1, :PCA2, :PCA3, :PCA4, :PCA5, :PCA6, :PCA7, :PCA8, :PCA9, :PCA10)
	rightjoin(samples, on = [:FID, :IID])
	leftjoin(vcat(caucasians, non_caucasians), on = [:FID, :IID])
    @transform!(:SEX = parse.(Int, :SEX))
end	    	  
n = size(dat, 1)

# ------------------------------------------------------------------------
# Split into m groups
# ------------------------------------------------------------------------
# Create grouping variable and shuffle randomly
dat.grp = repeat(1:m, Integer(nrow(dat)/m)) |> x -> sample(x, length(x), replace=false)

#-------------------------------------------------------------------------
# Load genotype Data
#-------------------------------------------------------------------------
# Read plink bim file
UKBB = SnpArray("UKBB.bed")

# Sample p candidate SNPs randomly accross genome, convert to additive model, scale and impute
snp_inds = sample(axes(UKBB, 2), p, replace = false, ordered = true)
G = convert(Matrix{Float64}, @view(UKBB[:, snp_inds]), center = true, scale = true, impute = true)

# Save filtered plink file
rowmask, colmask = trues(n), [col in snp_inds for col in 1:size(UKBB, 2)]
SnpArrays.filter("UKBB", rowmask, colmask, des = ARGS_[8] * "geno")

# ------------------------------------------------------------------------
# Simulate phenotypes
# ------------------------------------------------------------------------
# Variance components
sigma2_e = pi^2 / 3
sigma2_g = h2_g / (1 - h2_g - h2_d) * sigma2_e
sigma2_d = h2_d / (1 - h2_g - h2_d) * sigma2_e

# Simulate fixed effects for randomly sampled causal snps
W = zeros(p)
s = sample(1:p, Integer(round(p*c)), replace = false)
W[s] .= sigma2_g/length(s)
beta = rand.([Normal(0, sqrt(W[i])) for i in 1:p])

# Simulate random effects for each group
Z = dat.grp .== collect(1:m)'
gamma = rand.([Normal(0, sqrt(sigma2_d)) for i in 1:m])

# Simulate binary traits
logit(x) = log(x / (1 - x))
expit(x) = exp(x) / (1 + exp(x))
final_dat = @chain dat begin
	@transform!(:logit_pi = logit(pi0) .+ (logit(pi1) - logit(pi0)) * :CAUCASIAN - log(1.3) * :SEX + log(1.05) * ((:AGE .- 56) / 10) + G * beta + Z * gamma)
    @transform!(:pi = expit.(:logit_pi))
    @transform(:y = rand.([Binomial(1, :pi[i]) for i in 1:n]))
    select!(Not([:pi, :logit_pi, :ETHNICITY]))
end

# Create dummy variable for group variable
final_dat = Z[:, 2:end] |> x-> convert.(Int, x) |> x-> DataFrame(x, :auto) |> x-> [final_dat x] 

#-------------------------
# Write csv files
#-------------------------
# Covariates file
CSV.write(ARGS_[8] * "covariate.txt", final_dat)

# Genetic predictors file
df = SnpData("UKBB").snp_info[snp_inds, [1,2,4]]
df.beta = beta
CSV.write(ARGS_[8] * "betas.txt", df)