const datadir = "data"
const covfile = datadir * "/covariate.txt"
const plinkfile = datadir * "/geno"
const grmfile = datadir * "/grm.txt.gz"

# read covariate file
covdf = CSV.read(covfile, DataFrame)

# read PLINK files
geno = SnpArray(plinkfile * ".bed")

# Read grm file
K = open(GzipDecompressorStream, grmfile, "r") do stream
    Matrix(CSV.read(stream, DataFrame))
end

# Environment relatedness matrix
n = size(K, 1)
K_D = Array{Float64}(undef, n, n)
for i in 1:n 
    for j in i:n
		K_D[i, j] = ifelse(covdf.Exposed[i] == covdf.Exposed[j], 1, 0)
    end
end
LowerTriangular(K_D) .= transpose(UpperTriangular(K_D))

# Variance-Covariance Matrix with known variance components
Phi = 0.2 * 2 * (K + K_D)

# Sample p SNPs randomly, convert to additive model
p = 20
snp_inds = sample(1:size(geno, 2), p, replace = false, ordered = true)
G = convert(Matrix{Float64}, @view(geno[:, snp_inds]), center = true, scale = true, impute = true)

# Using GLM 
glm(G, covdf[:,:y], Binomial())

# Compare Newton's method with lower-bound and BFGS methods 
using BenchmarkTools
@btime glmm_fit(covdf[:,:y], G, binomial, Phi)
@btime glmm_fit(covdf[:,:y], G, binomial, method = "LB", Phi)
@btime glmm_fit(covdf[:,:y], G, binomial, method = "BFGS", Phi)

using GFlops
@count_ops glmm_fit(covdf[:,:y], G, binomial, Phi)
@count_ops glmm_fit(covdf[:,:y], G, binomial, method = "LB", Phi)
@count_ops glmm_fit(covdf[:,:y], G, binomial, method = "BFGS", Phi)