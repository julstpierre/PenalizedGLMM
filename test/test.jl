using LinearAlgebra, CSV, DataFrames

# Define directories where data is located
const datadir = "data"
const covfile = datadir * "/covariate.txt"
const plinkfile = datadir * "/UKBB"
const grmfile = datadir * "/grm.txt.gz"

# read covariate file
covdf = CSV.read(covfile, DataFrame)
n = size(covdf, 1)

# Environment relatedness matrix
K_D = Array{Float64}(undef, n, n)
for i in 1:n 
    for j in i:n
		K_D[i, j] = ifelse(covdf.Exposed[i] == covdf.Exposed[j], 1, 0)
    end
end
LowerTriangular(K_D) .= transpose(UpperTriangular(K_D))
M = push!(Any[], K_D)

pglmm(@formula(y ~ SEX + AGE), covfile, plinkfile, grmfile, M = M)
pglmm(@formula(y ~ SEX + AGE), covfile, plinkfile, grmfile, family = Normal(), link = IdentityLink(), M = M)