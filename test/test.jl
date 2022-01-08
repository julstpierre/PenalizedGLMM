# Load packages
using Pkg; Pkg.activate("..")
using PenalizedGLMM
using GLM, GLMNet, CSV, DataFrames

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
nullmodel = pglmm_null(@formula(y ~ SEX + AGE), covfile, grmfile)

# Fit a penalized logistic mixed model
modelfit = pglmm(nullmodel, plinkfile, verbose = true, GIC_crit = ARGS_[3])

# Genetic predictors effects at each λ   
pglmm_β = modelfit.betas[3:end,:]

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
