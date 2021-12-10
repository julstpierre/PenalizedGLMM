# Load packages
using Pkg; Pkg.activate("..")
using PenalizedGLMM
using GLM, GLMNet, SnpArrays, CSV, DataFrames

# Assign default command-line arguments
const ARGS_ = isempty(ARGS) ? ["../Simulations/data/data1/", "ALL"] : ARGS

# Define directories where data is located
const datadir = ARGS_[1]
const covfile = datadir * "covariate.txt"
const plinkfile = datadir * "geno"
const grmfile = datadir * "grm.txt.gz"

#------------------------------
# Model with one random effect 
#------------------------------
# Estimate covariate effects and variance components under the null
nullmodel = pglmm_null(@formula(y ~ SEX + AGE), covfile, grmfile)

# Fit a penalized logistic mixed model
modelfit = pglmm(nullmodel, plinkfile, irwls_maxiter = 50, verbose = true, GIC_crit = ARGS_[2])

# Genetic predictors effects at each λ   
pglmm_β = modelfit.betas[3:end,:]

# Find λ that gives minimum GIC
pglmmAIC_β = pglmm_β[:, argmin(modelfit.GIC["AIC",:])]
pglmmBIC_β = pglmm_β[:, argmin(modelfit.GIC["BIC",:])]
pglmmHDBIC_β = pglmm_β[:, argmin(modelfit.GIC["HDBIC",:])]

#----------------------------
# Compare with glmnet
#----------------------------
# convert PLINK genotype to matrix, convert to additive model (default), scale and impute
geno = SnpArray(plinkfile * ".bed")
G = convert(Matrix{Float64}, geno, model = ADDITIVE_MODEL, center = true, scale = true, impute = true)
p = size(G, 2)

# Combine non-genetic and genetic covariates, and convert y to a two-column matrix
covdf = CSV.read(covfile, DataFrame)
varlist = ["AGE", "SEX", "PCA1","PCA2","PCA3","PCA4","PCA5","PCA6","PCA7","PCA8","PCA9","PCA10"]
X = [Array(covdf[:, varlist]) G]

# y must be a matrix with one column per class
y = convert(Matrix{Float64}, [covdf.y .== 0 covdf.y .== 1])

# Fit a penalized logistic model using GLMNet
fit_glmnet = glmnet(X, y, Binomial(), penalty_factor = [zeros(length(varlist)); ones(p)])
glmnet_β = fit_glmnet.betas[(length(varlist) + 1):end,:]

# Select best penalized logistic model using GLMNet cross-validation
cv_glmnet = glmnetcv(X, y, Binomial(), penalty_factor = [zeros(length(varlist)); ones(p)])
glmnetcv_β = cv_glmnet.path.betas[(length(varlist) + 1):end, argmin(cv_glmnet.meanloss)]

#---------------------------------
# Compare results with real values
#---------------------------------
# Read file with real values
betas = CSV.read(datadir * "betas.txt", DataFrame)
rename!(betas, :beta => :true_beta)

# Save betas for pglmm with AIC, BIC and HDBIC, and glmnet_cv
betas.pglmmAIC_beta = pglmmAIC_β
betas.pglmmBIC_beta = pglmmBIC_β
betas.pglmmHDBIC_beta = pglmmHDBIC_β
betas.glmnetcv_beta = glmnetcv_β

# False positive rate (FPR) at 5% for pglmm and glmnet
betas.pglmmFPR5_beta = pglmm_β[:, findlast(mean((pglmm_β .> 0) .& (betas.true_beta .== 0), dims=1) .< 0.05)[2]]
betas.glmnetFPR5_beta = glmnet_β[:, findlast(mean((glmnet_β .> 0) .& (betas.true_beta .== 0), dims=1) .< 0.05)[2]]

# Estimated variance compoenent τ
betas.tau = repeat(nullmodel.τ, nrow(betas))

# Save results
CSV.write(datadir * "results.txt", select(betas, 
                                          :true_beta, 
                                          :pglmmAIC_beta,
                                          :pglmmBIC_beta,
                                          :pglmmHDBIC_beta, 
                                          :pglmmFPR5_beta,
                                          :glmnetcv_beta, 
                                          :glmnetFPR5_beta, 
                                          :tau
                                          )
)

# #------------------------------
# # Model with two random effects
# #------------------------------
# # read covariate file
# covdf = CSV.read(covfile, DataFrame)
# n = size(covdf, 1)

# # Environment relatedness matrix
# K_D = Array{Float64}(undef, n, n)
# for i in 1:n 
#     for j in i:n
# 		K_D[i, j] = ifelse(covdf.Exposed[i] == covdf.Exposed[j], 1, 0)
#     end
# end
# LowerTriangular(K_D) .= transpose(UpperTriangular(K_D))

# # Estimate covariate effects and variance components under the null
# nullmodel = pglmm_null(@formula(y ~ SEX + AGE), covfile, grmfile, M = push!(Any[], K_D))

# # Fit a penalized model
# modelfit = pglmm(nullmodel, plinkfile)