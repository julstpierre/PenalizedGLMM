# ========================================================================
# Code for simulating binary traits with environmental exposure from UKBB data
# ========================================================================
using CSV, DataFrames, SnpArrays, DataFramesMeta, StatsBase, LinearAlgebra, Distributions, CodecZlib

# ------------------------------------------------------------------------
# Initialize parameters
# ------------------------------------------------------------------------
# Assign default command-line arguments
const ARGS_ = isempty(ARGS) ? ["10000", "0.05", "data/"] : ARGS

# Fraction of Caucasian/Non-Caucasian in the first group
w = 0.5; w = [1, 1 - w] 

# Fraction of variance due to unobserved shared environmental effect (logit scale)
h2_d = 0

# Prevalence for Non Caucasian
pi0 = 0.1 	

# Prevalence for Caucasian
pi1 = 0.1 	

# Number of snps to randomly select accros genome
p = parse(Int, ARGS_[1])

# Percentage of causal SNPs
c = parse(Float64, ARGS_[2])

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

# Create GRM using 50000 randomly sampled SNPs
grm_inds = sample(axes(UKBB, 2), 50000, replace = false)
K = grm(UKBB, method=:GRM, cinds = grm_inds)

# Ensure that K is positive definite
xi = 1e-4;
while minimum(eigen(K).values) < 0
    global K = K + xi * Diagonal(ones(n));
    global xi = 10*xi;
end

# Write GRM to a compressed csv file
open(GzipCompressorStream, ARGS_[3] * "grm.txt.gz", "w") do stream
    CSV.write(stream, DataFrame(K), :auto))
end

# Sample p SNPs randomly accross genome, convert to additive model, scale and impute
snp_inds = sample(setdiff(axes(UKBB, 2), grm_inds), p, replace = false, ordered = true)
G = convert(Matrix{Float64}, @view(UKBB[:, snp_inds]), center = true, scale = true, impute = true)

# Save filtered plink file
rowmask, colmask = trues(n), [col in snp_inds for col in 1:size(UKBB, 2)]
SnpArrays.filter("UKBB", rowmask, colmask, des = ARGS_[3] * "geno")

# ------------------------------------------------------------------------
# Simulate phenotypes
# ------------------------------------------------------------------------
# Variance components
sigma2 = 2
sigma2_d = h2_d * sigma2
sigma2_g = (1 - h2_d) * sigma2

# Simulate fixed effects for randomly sampled causal snps
W = zeros(p)
s = sample(1:p, Integer(round(p*c)))
W[s] .= sigma2_g/length(s)
beta = rand.([Normal(0, sqrt(W[i])) for i in 1:p])

# Simulate random effects
b = rand(MvNormal(G * beta, sigma2_g * K + sigma2_d * K_D ))

# Simulate binary traits
logit(x) = log(x / (1 - x))
expit(x) = exp(x) / (1 + exp(x))
final_dat = @chain grp_dat begin
	@transform!(:logit_pi = logit(pi0) .+ (logit(pi1) - logit(pi0)) * :CAUCASIAN - log(1.3) * :SEX + log(1.05) * ((:AGE .- 56) / 10) + b)
    @transform!(:pi = expit.(:logit_pi))
    @transform(:y = rand.([Binomial(1, :pi[i]) for i in 1:n]))
    select!(Not([:pi, :logit_pi, :ETHNICITY]))
end

# Write csv files
CSV.write(ARGS_[3] * "covariate.txt", final_dat)

df = SnpData("UKBB").snp_info[snp_inds, [1,2,4]]
df.beta = beta
CSV.write(ARGS_[3] * "betas.txt", df)