# Load packages
using Pkg; Pkg.activate("..")
using PenalizedGLMM
using GLM, GLMNet, SnpArrays, CSV, DataFrames, LinearAlgebra

# Assign default command-line arguments
const ARGS_ = isempty(ARGS) ? ["", "0.01"] : ARGS

# Define directories where data is located
const datadir = ARGS_[1]
const covfile = datadir * "covariate.txt"
const plinkfile = datadir * "geno"
const grmfile = datadir * "grm.txt.gz"

#-------------------------------------------------------------------
# PenalizedGLMM
#-------------------------------------------------------------------
# Fit null model with one random effect
nullmodel = pglmm_null(@formula(y ~ SEX + AGE), covfile, grmfile)

# Fit a penalized logistic mixed model
modelfit = pglmm(nullmodel, plinkfile, verbose = true, GIC_crit = "ALL")

# Genetic predictors effects at each λ   
pglmm_β = modelfit.betas[3:end,:]

# Find λ that gives minimum GIC
pglmmAIC_β = pglmm_β[:, argmin(modelfit.GIC["AIC",:])]
pglmmBIC_β = pglmm_β[:, argmin(modelfit.GIC["BIC",:])]

#----------------------------
# GLMNet
#----------------------------
# convert PLINK genotype to matrix, convert to additive model (default), scale and impute
geno = SnpArray(plinkfile * ".bed")
G = convert(Matrix{Float64}, geno, model = ADDITIVE_MODEL, impute = true)
p = size(G, 2)

# Read covariate file
covdf = CSV.read(covfile, DataFrame)

# y must be a matrix with one column per class
y = convert(Matrix{Float64}, [covdf.y .== 0 covdf.y .== 1])

#----------------------------
# Lasso with no PC
#----------------------------
# Combine covariate with genetic predictors
varlist = ["AGE", "SEX"]
X = [Array(covdf[:, varlist]) G]

# Fit a penalized logistic model using GLMNet with no PCs
fit_glmnet = glmnet(X, y, Binomial(), penalty_factor = [zeros(length(varlist)); ones(p)])
glmnet_β = fit_glmnet.betas[(length(varlist) + 1):end,:]

# Select best penalized logistic model using GLMNet cross-validation
cv_glmnet = glmnetcv(X, y, Binomial(), penalty_factor = [zeros(length(varlist)); ones(p)])
cv_glmnet_β = cv_glmnet.path.betas[(length(varlist) + 1):end, argmin(cv_glmnet.meanloss)]

#----------------------------
# Lasso with 10 PCs
#----------------------------
# Combine covariate with genetic predictors
varlistwithPC = ["AGE", "SEX", "PC1","PC2","PC3","PC4","PC5","PC6","PC7","PC8","PC9","PC10"]
XwithPC = [Array(covdf[:, varlistwithPC]) G]

# Fit a penalized logistic model using GLMNet with 10 PCs
fit_glmnetPC = glmnet(XwithPC, y, Binomial(), penalty_factor = [zeros(length(varlistwithPC)); ones(p)])
glmnetPC_β = fit_glmnetPC.betas[(length(varlistwithPC) + 1):end,:]

# Select best penalized logistic model using GLMNet cross-validation
cv_glmnetPC = glmnetcv(XwithPC, y, Binomial(), penalty_factor = [zeros(length(varlistwithPC)); ones(p)])
cv_glmnetPC_β = cv_glmnetPC.path.betas[(length(varlistwithPC) + 1):end, argmin(cv_glmnetPC.meanloss)]

#---------------------------------
# Compare results with real values
#---------------------------------
# Read file with real values
betas = CSV.read(datadir * "betas.txt", DataFrame)

# Return coefficients on original scale
s = std(G, dims = 1, corrected = false)
lmul!(inv(Diagonal(vec(s))), betas.beta)

# Save betas for pglmm with AIC, BIC and HDBIC, and glmnet_cv
betas.pglmmAIC = pglmmAIC_β
betas.pglmmBIC = pglmmBIC_β
betas.cv_glmnet = cv_glmnet_β
betas.cv_glmnetPC = cv_glmnetPC_β

#-----------------------------------------------------
# False positive rate (FPR) for pglmm and glmnet
#-----------------------------------------------------
# Create DataFrame for fitted values
yhat = DataFrame()
fpr = parse(Float64, ARGS_[2])

# pglmm
pglmmFPR_ind = findlast(sum((pglmm_β .!= 0) .& (betas.beta .== 0), dims = 1) / sum(betas.beta .== 0) .< fpr)[2]
betas.pglmmFPR = pglmm_β[:, pglmmFPR_ind]
yhat.pglmmFPR = PenalizedGLMM.predict(modelfit, X, outtype = :prob)[:,pglmmFPR_ind]

# glmnet with no PCs
glmnetFPR_ind = findlast(sum((glmnet_β .!= 0) .& (betas.beta .== 0), dims = 1) / sum(betas.beta .== 0) .< fpr)[2]
betas.glmnetFPR = glmnet_β[:, glmnetFPR_ind]
yhat.glmnetFPR = GLMNet.predict(fit_glmnet, X, outtype = :prob)[:,glmnetFPR_ind]

# glmnet with 10 PCs
glmnetPCFPR_ind = findlast(sum((glmnetPC_β .!= 0) .& (betas.beta .== 0), dims = 1) / sum(betas.beta .== 0) .< fpr)[2]
betas.glmnetPCFPR = glmnetPC_β[:, glmnetPCFPR_ind]
yhat.glmnetPCFPR = GLMNet.predict(fit_glmnetPC, XwithPC, outtype = :prob)[:,glmnetPCFPR_ind]

#-----------------------
# Save results
#-----------------------
CSV.write(datadir * "results.txt", select(betas, 
                                            :beta, 
                                            :pglmmAIC,
                                            :pglmmBIC, 
                                            :pglmmFPR,
                                            :cv_glmnet,
                                            :cv_glmnetPC, 
                                            :glmnetFPR,
                                            :glmnetPCFPR
                                        )
)
CSV.write(datadir * "fitted_values.txt", select(yhat, :pglmmFPR, :glmnetFPR, :glmnetPCFPR))
CSV.write(datadir * "pglmm_tau.txt", DataFrame(tau = nullmodel.τ, h2 = nullmodel.τ / sum([nullmodel.τ' pi^2/3])))