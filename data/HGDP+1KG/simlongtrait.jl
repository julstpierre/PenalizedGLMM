# ========================================================================
# Code for simulating binary or continuous traits with environmental exposure from HGDP+1KG data
# ========================================================================
using CSV, DataFrames, SnpArrays, DataFramesMeta, StatsBase, LinearAlgebra, Distributions, CodecZlib, SparseArrays, RCall
# using ADMIXTURE

# ------------------------------------------------------------------------
# Initialize parameters
# ------------------------------------------------------------------------
# Assign default command-line arguments
const ARGS_ = isempty(ARGS) ? ["0.2", "0.1", "0.4", "0.2", "0.1", "10000", "0.01", "0.5", "true", "", "bin"] : ARGS

# Fraction of variance due to fixed polygenic additive effect (logit scale)
h2_g = parse(Float64, ARGS_[1])

# Fraction of variance due to fixed GEI effect (logit scale)
h2_GEI = parse(Float64, ARGS_[2])

# Fraction of variance due to random polygenic additive effect (logit scale)
h2_b = parse(Float64, ARGS_[3])

# Fraction of variance due to environmental exposure (logit scale)
h2_d = parse(Float64, ARGS_[4])

# Prevalence
prev = parse(Float64, ARGS_[5])  

# Number of snps to randomly select accros genome
p = parse(Int, ARGS_[6])

# Fraction of causal SNPs
c = parse(Float64, ARGS_[7])

# Fraction of GEI effects among causal SNPs
c_ = parse(Float64, ARGS_[8])

# Hierarchical GEI effects
hier = parse(Bool, ARGS_[9])

# Directory where source data is located
datadir = ARGS_[11]

# Simulate Logistic() or Normal() distribution
dist = ARGS_[12] == "bin" ? Logistic() : Normal()

# ------------------------------------------------------------------------
# Load the covariate file
# ------------------------------------------------------------------------
# Read plink fam file
samples = @chain CSV.read(datadir * "HGDP+1KG/HGDP+1KG.fam", DataFrame; header = false) begin  
    @select!(:FID = :Column1, :IID = :Column2)
end

# Combine into a DataFrame
dat = @chain CSV.read(datadir * "HGDP+1KG/covars.csv", DataFrame) begin
    @transform!(:FID = 0, :IID = :ind, :SEX = 1 * (:gender .== "male"), :POP = :super_pop, :AGE1 = round.(rand(Normal(50, 5), length(:ind)), digits = 0))
    @transform!(:AGE2 = :AGE1 .+ 1, :AGE3 = :AGE1 .+ 2, :AGE4 = :AGE1 .+ 3, :AGE5 = :AGE1 .+ 4)
    stack([:AGE1, :AGE2, :AGE3, :AGE4, :AGE5], value_name = :AGE)
    sort([:IID, :AGE])
    rightjoin(samples, on = [:IID, :FID], order = :right)
    @select!(:FID, :IID, :POP, :SEX, :AGE, :related, :related_exclude)
end

# Randomly sample subjects by POP for training and test sets
grpdat = groupby(filter(:related => f-> f==false, dat), :POP)
prop = (0.80 * nrow(dat) - sum(dat.related)) / sum(dat.related .== false)
train_ids = [sample(unique(grpdat[i].IID), Int(ceil(nrow(unique(grpdat[i], :IID)) * prop)); replace = false) for i in 1:length(grpdat)] |>
                x -> reduce(vcat, x)

# Add indicator variable for training subjects
dat.train = [dat.IID[i] in train_ids || dat.related[i] for i in 1:size(dat, 1)]

# Create dataset with unique IDs
uniquedat = unique(dat, :IID)
n, m = nrow(uniquedat), nrow(dat)

#-------------------------------------------------------------------------
# Load genotype Data
#-------------------------------------------------------------------------
# Read plink bim file
_HGDP1KG = SnpArray(datadir * "HGDP+1KG/HGDP+1KG.bed")

# Compute MAF in training set for each SNP and in each Population
_maf = DataFrame()
_maf.EAS = maf(@view(_HGDP1KG[(uniquedat.POP .== "EAS") .& uniquedat.train, :]))
_maf.AMR = maf(@view(_HGDP1KG[(uniquedat.POP .== "AMR") .& uniquedat.train, :]))
_maf.CSA = maf(@view(_HGDP1KG[(uniquedat.POP .== "CSA") .& uniquedat.train, :]))
_maf.OCE = maf(@view(_HGDP1KG[(uniquedat.POP .== "OCE") .& uniquedat.train, :]))
_maf.EUR = maf(@view(_HGDP1KG[(uniquedat.POP .== "EUR") .& uniquedat.train, :]))
_maf.AFR = maf(@view(_HGDP1KG[(uniquedat.POP .== "AFR") .& uniquedat.train, :]))
_maf.MID = maf(@view(_HGDP1KG[(uniquedat.POP .== "MID") .& uniquedat.train, :]))

# Compute range for MAFs among the populations
inds = trues(size(uniquedat, 1))
_maf.range = vec(maximum([_maf.EAS _maf.AMR _maf.CSA _maf.OCE _maf.EUR _maf.AFR _maf.MID], dims = 2) - minimum([_maf.EAS _maf.AMR _maf.CSA _maf.OCE _maf.EUR _maf.AFR _maf.MID], dims = 2))

# Remove SNPs with MAF = 0 or 0.5 either in the train set or in the train+test set
_maf.ALL = maf(@view(_HGDP1KG[inds, :]))
_maf.ALLtrain = maf(@view(_HGDP1KG[inds .& uniquedat.train, :]))
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

# Read GRM


# GEI similarity matrix
D = uniquedat.SEX
V_D = D * D'
for j in findall(x -> x == 0, D), i in findall(x -> x == 0, D)  
    V_D[i, j] = 1 
end
GRM_D = GRM .* V_D

# ------------------------------------------------------------------------
# Simulate phenotypes
# ------------------------------------------------------------------------
# Simulate population structure
POP = [dat.POP .== unique(dat.POP)[i] for i in 1:length(unique(dat.POP))] |> x-> mapreduce(permutedims, vcat, x)'
pi0 = POP * rand(Uniform(0.1, 0.9), size(POP, 2))

# Simulate Logistic or Normal error
e = rand(dist, size(dat, 1))

# Variance components
sigma2_e = var(e)
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
a = rand(MvNormal([0.25 0.3; 0.3 0.5]), n)'
b = h2_b > 0 ? rand(MvNormal(sigma2_b * GRM)) : zeros(size(dat, 1))
b += h2_d > 0 ? rand(MvNormal(sigma2_d * GRM_D)) : zeros(size(dat, 1))

# Create L matrix
L = [dat.IID .== unique(dat.IID)[i] for i in 1:n] |> x-> mapreduce(permutedims, vcat, x) |> x-> Float64.(x)
a_ = sum((L' * a) .* hcat(ones(m), dat.AGE), dims=2)

# Simulate outcome
logit(x) = log(x / (1 - x))
final_dat = @chain dat begin
    @transform!(:logit_pi = vec(logit.(pi0) .- log(1.3) * :SEX + log(1.05) * (:AGE .- mean(:AGE))/ 10 + L' * (G * beta) + (L' * (G * gamma)) .* :SEX + a_ + L' * b + e))
    @transform(:y = dist == Logistic() ? Int.(:logit_pi .> quantile(:logit_pi, 1 - prev)) : :logit_pi)
    select!(Not([:logit_pi]))
end

# Prevalence by Population
println("Mean of y is ", round(mean(final_dat.y), digits = 3))
println(combine(groupby(final_dat, :POP), :y => mean))
    
#----------------------
# Write csv files
#---------------------
# Add PCs to covariate file and write to CSV
final_dat.PC1 = L' * PCs[:, 1]; final_dat.PC2 = L' * PCs[:, 2]
final_dat.PC3 = L' * PCs[:, 3]; final_dat.PC4 = L' * PCs[:, 4]
final_dat.PC5 = L' * PCs[:, 5]; final_dat.PC6 = L' * PCs[:, 6]
final_dat.PC7 = L' * PCs[:, 7]; final_dat.PC8 = L' * PCs[:, 8]
final_dat.PC9 = L' * PCs[:, 9]; final_dat.PC10 = L' * PCs[:, 10]
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