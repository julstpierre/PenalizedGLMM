# ========================================================================
# Code for simulating binary traits with environmental exposure from HGDP+1KG data
# ========================================================================
using CSV, DataFrames, SnpArrays, DataFramesMeta, StatsBase, LinearAlgebra, Distributions, CodecZlib, SparseArrays, ADMIXTURE

# ------------------------------------------------------------------------
# Initialize parameters
# ------------------------------------------------------------------------
# Assign default command-line arguments
const ARGS_ = isempty(ARGS) ? ["0.2", "0.1", "0.4", "0.2", "0.1", "10000", "0.01", "0.1", "true", "NONE" , ""] : ARGS

# Fraction of variance due to fixed polygenic additive effect (logit scale)
h2_g = parse(Float64, ARGS_[1])

# Fraction of variance due to fixed GEI effect (logit scale)
h2_GEI = parse(Float64, ARGS_[2])

# Fraction of variance due to random polygenic additive effect (logit scale)
h2_b = parse(Float64, ARGS_[3])

# Fraction of variance due to environmental exposure (logit scale)
h2_d = parse(Float64, ARGS_[4])

# Prevalence
pi0 = parse(Float64, ARGS_[5])  

# Number of snps to randomly select accros genome
p = parse(Int, ARGS_[6])

# Fraction of causal SNPs
c = parse(Float64, ARGS_[7])

# Fraction of GEI effects among causal SNPs
c_ = parse(Float64, ARGS_[8])

# Hierarchical GEI effects
hier = parse(Bool, ARGS_[9])

# Overlap between kinship and causal SNPs
kin = ARGS_[10]

# Number of snps to use for GRM estimation
p_kin = 50000

# Directory where source data is located
datadir = ARGS_[11]

# ------------------------------------------------------------------------
# Load the covariate file
# ------------------------------------------------------------------------
# Read plink fam file
samples = @chain CSV.read(datadir * "HGDP+1KG/HGDP+1KG.fam", DataFrame; header = false) begin  
    @select!(:FID = :Column1, :IID = :Column2)
end

# Combine into a DataFrame
dat = @chain CSV.read(datadir * "HGDP+1KG/covars.csv", DataFrame) begin
    @transform!(:FID = 0, :IID = :ind, :SEX = 1 * (:gender .== "male"), :POP = :super_pop, :AGE = round.(rand(Normal(50, 5), length(:ind)), digits = 0))
    rightjoin(samples, on = [:IID, :FID])
    @select!(:FID, :IID, :POP, :SEX, :AGE, :related, :related_exclude, :PC1, :PC2, :PC3, :PC4, :PC5, :PC6, :PC7, :PC8, :PC9, :PC10)
end

# Randomly sample subjects by POP for training and test sets
grpdat = groupby(filter(:related => f-> f==false, dat), :POP)
prop = (0.80 * nrow(dat) - sum(dat.related)) / sum(dat.related .== false)
train_ids = [sample(grpdat[i].IID, Int(ceil(nrow(grpdat[i]) * prop)); replace = false) for i in 1:length(grpdat)] |>
                x -> reduce(vcat, x)

# Add indicator variable for training subjects
dat.train = [dat.IID[i] in train_ids || dat.related[i] for i in 1:size(dat, 1)]

#-------------------------------------------------------------------------
# Load genotype Data
#-------------------------------------------------------------------------
# Read plink bim file
_HGDP1KG = SnpArray(datadir * "HGDP+1KG/HGDP+1KG.bed")

# Compute MAF in training set for each SNP and in each Population
_maf = DataFrame()
_maf.EAS = maf(@view(_HGDP1KG[(dat.POP .== "EAS") .& dat.train, :]))
_maf.AMR = maf(@view(_HGDP1KG[(dat.POP .== "AMR") .& dat.train, :]))
_maf.CSA = maf(@view(_HGDP1KG[(dat.POP .== "CSA") .& dat.train, :]))
_maf.OCE = maf(@view(_HGDP1KG[(dat.POP .== "OCE") .& dat.train, :]))
_maf.EUR = maf(@view(_HGDP1KG[(dat.POP .== "EUR") .& dat.train, :]))
_maf.AFR = maf(@view(_HGDP1KG[(dat.POP .== "AFR") .& dat.train, :]))
_maf.MID = maf(@view(_HGDP1KG[(dat.POP .== "MID") .& dat.train, :]))

# Compute range for MAFs among the populations
inds = trues(size(dat, 1))
_maf.range = vec(maximum([_maf.EAS _maf.AMR _maf.CSA _maf.OCE _maf.EUR _maf.AFR _maf.MID], dims = 2) - minimum([_maf.EAS _maf.AMR _maf.CSA _maf.OCE _maf.EUR _maf.AFR _maf.MID], dims = 2))

# Remove SNPs with MAF = 0 or 0.5 either in the train set or in the train+test set
_maf.ALL = maf(@view(_HGDP1KG[inds, :]))
_maf.ALLtrain = maf(@view(_HGDP1KG[inds .& dat.train, :]))
snps = findall((_maf.ALL .!= 0) .& (_maf.ALL .!= 0.5) .& (_maf.ALLtrain .!= 0) .& (_maf.ALLtrain .!= 0.5))

# Sample p candidate SNPs randomly accross genome, convert to additive model and impute
snp_inds = sample(snps, p, replace = false, ordered = true)
G = convert(Matrix{Float64}, @view(_HGDP1KG[inds, snp_inds]), impute = true, center = true, scale = true)

# Function to return standard deviation and mean of each SNP
function standardizeG(s::AbstractSnpArray, model, scale::Bool, T = AbstractFloat)
    n, m = size(s)
    μ, σ = Array{T}(undef, m), Array{T}(undef, m)   
    @inbounds for j in 1:m
        μj, mj = zero(T), 0
        for i in 1:n
            vij = SnpArrays.convert(T, s[i, j], model)
            μj += isnan(vij) ? zero(T) : vij
            mj += isnan(vij) ? 0 : 1
        end
        μj /= mj
        μ[j] = μj
        σ[j] = model == ADDITIVE_MODEL ? sqrt(μj * (1 - μj / 2)) : sqrt(μj * (1 - μj))
    end
    
    # Return centre and scale parameters
    if scale 
       return μ, σ
    else 
       return μ, []
    end
end
muG, sG = standardizeG(@view(_HGDP1KG[inds, snp_inds]), ADDITIVE_MODEL, true)

# Save filtered plink file
rowmask, colmask = inds, [col in snp_inds for col in 1:size(_HGDP1KG, 2)]
SnpArrays.filter(datadir * "HGDP+1KG/HGDP+1KG", rowmask, colmask, des = "geno")

if kin == "ALL"
    # Causal SNPs are included in the GRM
    grm_inds = sample(setdiff(snps, snp_inds), p_kin - p, replace = false, ordered = true) |>
               x -> [x; snp_inds] |>
               sort
elseif kin == "NONE"
    # Causal SNPs are excluded from the GRM
    grm_inds = sample(setdiff(snps, snp_inds), p_kin, replace = false, ordered = true)
end

# Estimated GRM
GRM = 2 * grm(_HGDP1KG, cinds = grm_inds, method = :Robust)[inds, inds]

# # Create a plink data set using unrelated individuals only for ADMIXTURE
# SnpArrays.filter(datadir * "HGDP+1KG/HGDP+1KG", 
#                  findall(dat.related_exclude .== false), 
#                  grm_inds, 
#                  des = "admix"
#                 )

# # Create .pop file
# CSV.write("admix.pop", dat[findall(dat.related_exclude .== false),[:POP]], header = false)

# # Obtain estimated allele frequencies and estimated ancestry fractions using ADMIXTURE
# admixture("admix.bed", 7, supervised = true)
# run(`mv admix.7.P admix.7.P.in`)

# # Create a plink data with all individuals for ADMIXTURE
# SnpArrays.filter(datadir * "HGDP+1KG/HGDP+1KG", 
#                  inds, 
#                  grm_inds, 
#                  des = "admix"
#                 )

# # Project all individuals using ADMIXTURE
# run(`$ADMIXTURE_EXE -P admix.bed 7`)

# # Compute GRM
# # first read in the P and Q matrix output from ADMIXTURE and tranpose them
# Pt = CSV.read("admix.7.P", DataFrame, header = false) |> Matrix |> transpose
# Qt = CSV.read("admix.7.Q", DataFrame, header = false) |> Matrix |> transpose
# GRM = 2 * SnpArrays.grm_admixture(SnpArray("admix.bed"), Pt, Qt)

# Make sure GRM is posdef
function posdef(K, n = size(K, 1), xi = 1e-4)
    while !isposdef(K)
        println(xi)
        K = K + xi * Diagonal(ones(n))
        xi = 2 * xi
    end
    return(K)
end
GRM = posdef(GRM)
    
# Save GRM in compressed file
open(GzipCompressorStream, "grm.txt.gz", "w") do stream
    CSV.write(stream, DataFrame(GRM, :auto))
end

# GEI similarity matrix
D = dat.SEX
V_D = D * D'
for j in findall(x -> x == 0, D), i in findall(x -> x == 0, D)  
    V_D[i, j] = 1 
end
GRM_D = GRM .* V_D

# clean up
rm("admix.7.P", force = true)
rm("admix.7.P.in", force = true)
rm("admix.7.Q", force = true)
rm("admix.bim", force = true)
rm("admix.bed", force = true)
rm("admix.fam", force = true)
rm("admix.pop", force = true)

# ------------------------------------------------------------------------
# Simulate phenotypes
# ------------------------------------------------------------------------
# Variance components
sigma2_e = pi^2 / 3 + log(1.3)^2 * var(dat.SEX) + log(1.05)^2 * var(dat.AGE / 10)
sigma2_g = h2_g / (1 - h2_g - h2_b - h2_d - h2_GEI) * sigma2_e
sigma2_GEI = h2_GEI / (1 - h2_g - h2_b - h2_d - h2_GEI) * sigma2_e
sigma2_b = h2_b / (1 - h2_g - h2_b - h2_d - h2_GEI) * sigma2_e
sigma2_d = h2_d / (1 - h2_g - h2_b - h2_d - h2_GEI) * sigma2_e

# Simulate fixed effects for randomly sampled causal snps
W = zeros(p)
s = sample(1:p, Integer(round(p*c)), replace = false, ordered = true)
W[s] .= sigma2_g/length(s)
beta = rand.([Normal(0, sqrt(W[i])) for i in 1:p])

# Simulate fixed GEI effects for randomly sampled causal snps
W = zeros(p)
s_ = hier ? sample(s, Integer(round(p*c*c_)), replace = false, ordered = true) : sample(1:p, Integer(round(p*c*c_)), replace = false, ordered = true)
W[s_] .= sigma2_GEI/length(s_)
gamma = rand.([Normal(0, sqrt(W[i])) for i in 1:p])

# Simulate random effects
b = h2_b > 0 ? rand(MvNormal(sigma2_b * GRM)) : zeros(size(dat, 1))
b += h2_d > 0 ? rand(MvNormal(sigma2_d * GRM_D)) : zeros(size(dat, 1))

# Simulate binary traits
logit(x) = log(x / (1 - x))
expit(x) = exp(x) / (1 + exp(x))
final_dat = @chain dat begin
    @transform!(:logit_pi = logit(pi0) .- log(1.3) * :SEX + log(1.05) * (:AGE .- mean(:AGE))/ 10 + G * beta + (G .* :SEX) * gamma + b)
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
CSV.write("covariate.txt", final_dat)

# Convert simulated effect for each SNP on original genotype scale
df = SnpData(datadir * "HGDP+1KG/HGDP+1KG").snp_info[snp_inds, [1,4]]
df.beta = [beta[i] / sG[i] for i in 1:p]
df.gamma = [gamma[i] / sG[i] for i in 1:p]

# Save MAF for each SNP and in each Population
df.mafEUR = _maf.EUR[snp_inds]
df.mafAMR = _maf.AMR[snp_inds]
df.mafCSA = _maf.CSA[snp_inds]
df.mafEAS = _maf.EAS[snp_inds]
df.mafAFR = _maf.AFR[snp_inds]
df.mafOCE = _maf.OCE[snp_inds]
df.mafMID = _maf.MID[snp_inds]
df.mafrange = _maf.range[snp_inds]

# CSV file containing MAFs and simulated effect for each SNP
CSV.write("betas.txt", df)