# ========================================================================
# Code for simulating binary or continuous traits with environmental exposure from HGDP+1KG data
# ========================================================================
using CSV, DataFrames, SnpArrays, DataFramesMeta, StatsBase, LinearAlgebra, Distributions, CodecZlib, SparseArrays, RCall, RData, BlockDiagonals

# ------------------------------------------------------------------------
# Initialize parameters
# ------------------------------------------------------------------------
# Assign default command-line arguments
const ARGS_ = isempty(ARGS) ? ["0.02", "0", "0.1", "0", "0.2", "10000", "0.01", "0.5", "true", ""] : ARGS

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
datadir = ARGS_[10]

# ------------------------------------------------------------------------
# Load the covariate file
# ------------------------------------------------------------------------
# Read plink fam file
samples = @chain CSV.read(datadir * "HGDP+1KG/HGDP+1KG.fam", DataFrame; header = false) begin  
    @select!(:FID = :Column1, :IID = :Column2)
end

# Combine into a DataFrame
dat_ = @chain CSV.read(datadir * "HGDP+1KG/covars.csv", DataFrame) begin
    @transform!(:FID = 0, :IID = :ind, :SEX = 1 * (:gender .== "male"), :POP = :super_pop, :AGE1 = round.(rand(Normal(50, 5), length(:ind)), digits = 0))
    @transform!(:AGE2 = :AGE1 .+ 1, :AGE3 = :AGE1 .+ 2, :AGE4 = :AGE1 .+ 3, :AGE5 = :AGE1 .+ 4)
    stack([:AGE1, :AGE2, :AGE3, :AGE4, :AGE5], value_name = :AGE, variable_name = :AGEvar)
    @transform(:TIME = (:AGE .- mean(:AGE)) / 5, :EXP = rand(Normal(), length(:ind)))
    sort([:IID, :AGE])
    rightjoin(samples, on = [:IID, :FID], order = :right)
    @select!(:FID, :IID, :POP, :SEX, :AGE, :TIME, :EXP, :related, :related_exclude, :PC1, :PC2, :PC3, :PC4, :PC5, :PC6, :PC7, :PC8, :PC9, :PC10)
end

# Randomly sample from 1 to 5 rows from each individual
rows = [sample(1:5, sample(1:5), replace = false, ordered = true) for i in 1:length(groupby(dat_, :IID))]
dat = reduce(vcat, [groupby(dat_, :IID)[i][rows[i],:] for i in 1:length(rows)])

# Randomly sample subjects by POP for training and test sets
grpdat = groupby(filter(:related => f-> f==false, dat), :POP)
prop = (0.80 * nrow(dat) - sum(dat.related)) / sum(dat.related .== false)
train_ids = [sample(unique(grpdat[i].IID), Int(ceil(nrow(unique(grpdat[i], :IID)) * prop)); replace = false) for i in 1:length(grpdat)] |>
                x -> reduce(vcat, x)

# Add indicator variable for training subjects
dat.train = [dat.IID[i] in train_ids || dat.related[i] for i in 1:size(dat, 1)]

# Create dataset with unique IDs
uniquedat = unique(dat, :IID)
m, n= nrow(uniquedat), nrow(dat)

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

# Remove snps that have been used to calculate GRM
pruned = load(datadir * "HGDP+1KG/pruned_snps.rds")

# Sample p candidate SNPs randomly accross genome, convert to additive model and impute
snp_inds = sample(setdiff(snps, pruned), p, replace = false, ordered = true)
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

# Read full GRM
grm_ids = CSV.read(datadir * "HGDP+1KG/grm_ids.txt", DataFrame; header = false)[:, 1]
grminds = [findfirst(x -> x == uniquedat.IID[i], grm_ids) for i in 1:m]

GRM = open(GzipDecompressorStream, datadir * "HGDP+1KG/grm.txt.gz", "r") do stream
        Matrix(CSV.read(stream, DataFrame))[grminds, grminds]
end

# Make sure grm is posdef
function posdef(GRM::Matrix{T}) where T
    xi, n = 1e-4, size(GRM, 1)
    while !isposdef(GRM)
        GRM = GRM + xi * Diagonal(ones(n))
        xi = 2 * xi
    end
    
    return GRM
end
GRM = posdef(GRM)

# # Read sparse GRM 
# include(dirname(dirname(datadir)) * "/src/utils.jl")
# GRM, GRM_ids = read_sparse_grm(datadir * "HGDP+1KG/sparse_grm.rds", uniquedat.IID)
# GRM = Matrix(GRM)

# # Change order of covariate file to match sparse GRM
# covrowinds = reduce(vcat, [findall(dat.IID .== GRM_ids[i]) for i in 1:length(GRM_ids)])
# dat = dat[covrowinds, :]
# uniquedat = unique(dat, :IID)

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
# POP = [dat.POP .== unique(dat.POP)[i] for i in 1:length(unique(dat.POP))] |> x-> mapreduce(permutedims, vcat, x)'
# pi0 = POP * rand(Uniform(0.1, 0.9), size(POP, 2))

# Simulate no population structure
pi0 = 0.7315

# Simulate Normal error
e = rand(Normal(0, 1), size(dat, 1))

# Simulate random intercept and random slopes
L = [dat.IID .== unique(dat.IID)[i] for i in 1:m] |> x-> mapreduce(permutedims, vcat, x)' |> x-> Float64.(x)
a = rand(MvNormal([0.4 -0.2 0.1; -0.2 0.5 0.2; 0.1 0.2 0.3]), m)'
a_ = sum((L * a) .* hcat(one.(dat.TIME), dat.TIME, dat.EXP), dims=2)

# Variance components
sigma2_e = var(e) + var(a_)
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

# Simulate polygenic random effects
b = h2_b > 0 ? rand(MvNormal(sigma2_b * GRM)) : zeros(m)
b += h2_d > 0 ? rand(MvNormal(sigma2_d * GRM_D)) : zeros(m)

# Simulate outcome
logit(x) = log(x / (1 - x))
final_dat = @chain dat begin
    @transform!(:y = vec(logit(pi0) .- log(1.3) * :SEX + log(1.05) * :AGE + L * (G * beta) + (L * (G * gamma)) .* :SEX + a_ + L * b + e))
    @transform(:ybin = Int.(:y .> quantile(:y, 1 - prev)))
end

# Prevalence by Population
println("Mean of y is ", round(mean(final_dat.y), digits = 3))
println("Prevalence is ", round(mean(final_dat.ybin), digits = 3))
println(combine(groupby(final_dat, :POP), :ybin => mean))
    
#----------------------
# Write csv files
#---------------------
# Write covariate file to CSV
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