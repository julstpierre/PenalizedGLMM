# Load packages
using Pkg; Pkg.activate("..")
using PenalizedGLMM
using GLM, GLMNet, SnpArrays, CSV, DataFrames, LinearAlgebra

# Assign default command-line arguments
const ARGS_ = isempty(ARGS) ? ["", "1RE", "ALL"] : ARGS

# Define directories where data is located
const datadir = ARGS_[1]
const covfile = datadir * "covariate.txt"
const plinkfile = datadir * "geno"
const grmfile = datadir * "grm.txt.gz"

#-------------------------------------------------------------------
# PenalizedGLMM
#-------------------------------------------------------------------
# Fit null model with one random effect
nullmodel = pglmm_null(@formula(y ~ SEX + AGE + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15 + x16 + x17 + x18 + x19), covfile, grmfile)

# Fit a penalized logistic mixed model
modelfit = pglmm(nullmodel, plinkfile, verbose = true, GIC_crit = ARGS_[3])

# Genetic predictors effects at each λ   
pglmm_β = modelfit.betas[22:end,:]

# Find λ that gives minimum GIC
pglmmAIC_β = pglmm_β[:, argmin(modelfit.GIC["AIC",:])]
pglmmBIC_β = pglmm_β[:, argmin(modelfit.GIC["BIC",:])]

# Model with two random effects
if ARGS_[2] == "2REs"
    # Read covariate file
    covdf = CSV.read(covfile, DataFrame)
    n = nrow(covdf)

    # Environment relatedness matrix
    K_D = Array{Float64}(undef, n, n)
    for i in 1:n 
        for j in i:n
            K_D[i, j] = ifelse(covdf.grp[i] == covdf.grp[j], 1, 0)
        end
    end
    K_D = Symmetric(K_D)

    # Model with two random effects
    nullmodel2 = pglmm_null(@formula(y ~ SEX + AGE), covfile, grmfile, M = push!([], K_D))
    modelfit2 = pglmm(nullmodel2, plinkfile, verbose = true, GIC_crit = ARGS_[3])

    # Genetic predictors effects at each λ   
    pglmm2_β = modelfit2.betas[3:end,:]

    # Find λ that gives minimum GIC
    pglmm2AIC_β = pglmm2_β[:, argmin(modelfit2.GIC["AIC",:])]
    pglmm2BIC_β = pglmm2_β[:, argmin(modelfit2.GIC["BIC",:])]

end

#----------------------------
# Compare with glmnet
#----------------------------
# convert PLINK genotype to matrix, convert to additive model (default), scale and impute
geno = SnpArray(plinkfile * ".bed")
G = convert(Matrix{Float64}, geno, model = ADDITIVE_MODEL, center = true, scale = true, impute = true)
p = size(G, 2)

# Combine non-genetic and genetic covariates, and convert y to a two-column matrix
covdf = CSV.read(covfile, DataFrame)
varlist = ["AGE", "SEX", "PCA1","PCA2","PCA3","PCA4","PCA5","PCA6","PCA7","PCA8","PCA9","PCA10",
           "x1","x2","x3","x4","x5","x6","x7","x8","x9","x10","x11","x12","x13","x14","x15","x16","x17","x18","x19"]
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
betas.glmnetcv_beta = glmnetcv_β
if ARGS_[2] == "2REs"
    betas.pglmm2AIC_beta = pglmm2AIC_β
    betas.pglmm2BIC_beta = pglmm2BIC_β
end

#-----------------------------------------------------
# False positive rate (FPR) at 1% for pglmm and glmnet
#-----------------------------------------------------
# Create DataFrame for fitted values
yhat = DataFrame()

# pglmm
pglmmFPR_ind = findlast(sum((pglmm_β .!= 0) .& (betas.true_beta .== 0), dims = 1) / sum(betas.true_beta .== 0) .< 0.01)[2]
betas.pglmmFPR_beta = pglmm_β[:, pglmmFPR_ind]
yhat.pglmmFPR = modelfit.fitted_means[:, pglmmFPR_ind]

if ARGS_[2] == "2REs"
    pglmm2FPR_ind = findlast(sum((pglmm2_β .!= 0) .& (betas.true_beta .== 0), dims = 1) / sum(betas.true_beta .== 0) .< 0.01)[2]
    betas.pglmm2FPR_beta = pglmm2_β[:, pglmm2FPR_ind]
    yhat.pglmm2FPR = modelfit2.fitted_means[:, pglmm2FPR_ind]
end

# glmnet
glmnetFPR_ind = findlast(sum((glmnet_β .!= 0) .& (betas.true_beta .== 0), dims = 1) / sum(betas.true_beta .== 0) .< 0.01)[2]
betas.glmnetFPR_beta = glmnet_β[:, glmnetFPR_ind]
yhat.glmnetFPR = GLMNet.predict(fit_glmnet, X, outtype = :prob)[:,glmnetFPR_ind]

#-----------------------
# Save results
#-----------------------
if ARGS_[2] == "1RE"
    CSV.write(datadir * "results.txt", select(betas, 
                                            :true_beta, 
                                            :pglmmAIC_beta,
                                            :pglmmBIC_beta, 
                                            :pglmmFPR_beta,
                                            :glmnetcv_beta, 
                                            :glmnetFPR_beta
                                            )
    )
    CSV.write(datadir * "fitted_values.txt", select(yhat, :pglmmFPR, :glmnetFPR))
    CSV.write(datadir * "pglmm_tau.txt", DataFrame(tau = nullmodel.τ, h2 = nullmodel.τ / sum([nullmodel.τ' pi^2/3])))

elseif ARGS_[2] == "2REs"
    CSV.write(datadir * "results.txt", select(betas, 
                                            :true_beta, 
                                            :pglmmAIC_beta,
                                            :pglmmBIC_beta, 
                                            :pglmmFPR_beta,
                                            :pglmm2AIC_beta,
                                            :pglmm2BIC_beta, 
                                            :pglmm2FPR_beta,
                                            :glmnetcv_beta, 
                                            :glmnetFPR_beta
                                            )
    )
    CSV.write(datadir * "fitted_values.txt", select(yhat, :pglmmFPR, :pglmm2FPR, :glmnetFPR))
    CSV.write(datadir * "pglmm_tau.txt", DataFrame(tau = nullmodel2.τ, h2 = nullmodel2.τ / sum([nullmodel2.τ' pi^2/3])))
end

