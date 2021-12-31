# ========================================================================
# Code for simulating binary traits with environmental exposure from UKBB data
# ========================================================================
using CSV, DataFrames, SnpArrays, DataFramesMeta, StatsBase, LinearAlgebra, Distributions, CodecZlib

# ------------------------------------------------------------------------
# Initialize parameters
# ------------------------------------------------------------------------
# Assign default command-line arguments
const ARGS_ = isempty(ARGS) ? ["0.3", "0", "0.1", "0.1", "10000", "0.003", "0.5", "data/"] : ARGS

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

# Fraction of Caucasian/Non-Caucasian in the first group
w = [parse(Float64, ARGS_[7]), 1 - parse(Float64, ARGS_[7])] 

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
end	    	  
n = size(dat, 1)

# ------------------------------------------------------------------------
# Split into two groups
# ------------------------------------------------------------------------
# Create function to randomly sample rows from DataFrame
function data_sample(data, frac::Float64)
    size = Int(round(frac * nrow(data)))
    rows = sample(1:nrow(data), size, replace = false)
    out = data[rows, :]
end

# Sample subjects from each population into two groups
pop = combine(groupby(dat, :ETHNICITY), nrow)[!, 1]
exp_dat = similar(dat, 0)
for i in 1:length(pop)
    global exp_dat = @chain data_sample(filter(:ETHNICITY => ==(pop[i]), dat), w[i]) begin
        append!(exp_dat)
    end
end

# Create dummy variable for environmental exposure
grp_dat = @chain exp_dat begin
    @transform!(:Exposed = 1)
    @select!(:FID, :IID, :Exposed)
    outerjoin(dat, on = [:FID, :IID])
    @transform!(:Exposed = -ismissing.(:Exposed) .+ 1, :SEX = parse.(Int, :SEX))
    @orderby(:FID)
end

# Environment relatedness matrix
K_D = Array{Float64}(undef, n, n)
for i in 1:n 
    for j in i:n
		K_D[i, j] = ifelse(grp_dat.Exposed[i] == grp_dat.Exposed[j], 1, 0)
    end
end
LowerTriangular(K_D) .= transpose(UpperTriangular(K_D))

#-------------------------------------------------------------------------
# Load genotype Data
#-------------------------------------------------------------------------
# Read plink bim file
UKBB = SnpArray("UKBB.bed")

# Sample 10,000 SNPs with infinitesimal polygenic effects
grm_inds = sample(axes(UKBB, 2), 10000, replace = false)
X = convert(Matrix{Float64}, @view(UKBB[:, grm_inds]), center = true, scale = true, impute = true)
K = X * X' / length(grm_inds)

# Ensure that K is positive definite
function posdef(K::Matrix{Float64}, xi::Float64 = 1e-4, n::Int64 = size(K, 1))
    while !isposdef(K)
        K = K + xi * Diagonal(ones(n));
        xi = 10*xi;
    end
    return(K = K)
end
K = posdef(K)

# Sample p candidate SNPs randomly accross genome, convert to additive model, scale and impute
snp_inds = sample(setdiff(axes(UKBB, 2), grm_inds), p, replace = false, ordered = true)
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

# Simulate random effects
b = c > 0 ? rand(MvNormal(0.5 * sigma2_g * K + sigma2_d * K_D)) : rand(MvNormal(sigma2_g * K + sigma2_d * K_D))

# Simulate fixed effects for randomly sampled causal snps
W = zeros(p)
s = sample(1:p, Integer(round(p*c)), replace = false)
W[s] .= sigma2_g/(2 * length(s))
beta = rand.([Normal(0, sqrt(W[i])) for i in 1:p])

# Simulate binary traits
logit(x) = log(x / (1 - x))
expit(x) = exp(x) / (1 + exp(x))
final_dat = @chain grp_dat begin
	@transform!(:logit_pi = logit(pi0) .+ (logit(pi1) - logit(pi0)) * :CAUCASIAN - log(1.3) * :SEX + log(1.05) * ((:AGE .- 56) / 10) + G * beta + b)
    @transform!(:pi = expit.(:logit_pi))
    @transform(:y = rand.([Binomial(1, :pi[i]) for i in 1:n]))
    select!(Not([:pi, :logit_pi, :ETHNICITY]))
end

# Write csv files
CSV.write(ARGS_[8] * "covariate.txt", final_dat)

df = SnpData("UKBB").snp_info[snp_inds, [1,2,4]]
df.beta = beta
CSV.write(ARGS_[8] * "betas.txt", df)