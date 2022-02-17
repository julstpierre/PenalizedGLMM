# Load packages
using Pkg; Pkg.activate("..")
using PenalizedGLMM
using GLM, GLMNet, SnpArrays, CSV, DataFrames, LinearAlgebra

# Assign default command-line arguments
const ARGS_ = isempty(ARGS) ? [""] : ARGS

# Define directories where data is located
const datadir = ARGS_[1]
const covfile = datadir * "covariate.txt"
const plinkfile = datadir * "geno"
const snpfile = datadir * "snps.txt"
const grmfile = datadir * "grm.txt.gz"

#-------------------------------------------------------------------
# PenalizedGLMM
#-------------------------------------------------------------------
# Read covariate file
covdf = CSV.read(covfile, DataFrame)
trainrowinds = findall(covdf.train)
testrowinds = setdiff(1:nrow(covdf), trainrowinds)

# Fit null model with one random effect
nullmodel = pglmm_null(@formula(y ~ SEX + AGE), covfile, grmfile, covrowinds = trainrowinds, grminds = trainrowinds)

# Fit a penalized logistic mixed model
if isfile(plinkfile * ".bed")
        modelfit = pglmm(nullmodel, plinkfile, geneticrowinds = trainrowinds, verbose = true)
elseif isfile(snpfile)
        modelfit = pglmm(nullmodel, snpfile = snpfile, geneticrowinds = trainrowinds, verbose = true)
end

# Genetic predictors effects at each λ   
pglmm_β = modelfit.betas[3:end,:]

# Find λ that gives minimum GIC
pglmmAIC = PenalizedGLMM.GIC(modelfit, :AIC)
pglmmBIC = PenalizedGLMM.GIC(modelfit, :BIC)

pglmmAIC_β  = pglmm_β[:,pglmmAIC]
pglmmBIC_β  = pglmm_β[:,pglmmBIC]

#----------------------------
# GLMNet
#----------------------------
if isfile(plinkfile * ".bed")
	# convert PLINK genotype to matrix, convert to additive model (default) and impute
	geno = SnpArray(plinkfile * ".bed")
	G = convert(Matrix{Float64}, @view(geno[trainrowinds,:]), model = ADDITIVE_MODEL, impute = true)
elseif isfile(snpfile)
	# Read genotype from csv file and convert to matrix
	geno = CSV.read(snpfile, DataFrame)
	G = convert.(Float64, Matrix(geno[trainrowinds,:]))
end

p = size(G, 2)

# y must be a matrix with one column per class
y = convert(Matrix{Float64}, [covdf.y .== 0 covdf.y .== 1]) |> 
        x -> x[trainrowinds,:]

#----------------------------
# Lasso with no PC
#----------------------------
# Combine covariate with genetic predictors
varlist = ["SEX", "AGE"]
X = [Array(covdf[trainrowinds, varlist]) G]

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
varlistwithPC = ["SEX","AGE","PC1","PC2","PC3","PC4","PC5","PC6","PC7","PC8","PC9","PC10"]
XwithPC = [Array(covdf[trainrowinds, varlistwithPC]) G]

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

# Save betas for pglmm with AIC, BIC and HDBIC, and glmnet_cv
betas.pglmmAIC = pglmmAIC_β
betas.pglmmBIC = pglmmBIC_β
betas.cv_glmnet = cv_glmnet_β
betas.cv_glmnetPC = cv_glmnetPC_β

# Create DataFrame for predicted values
yhat = DataFrame()

# Create Arrays for test set
if isfile(plinkfile)
	Gnew = convert(Matrix{Float64}, @view(geno[testrowinds,:]), model = ADDITIVE_MODEL, impute = true)
elseif isfile(snpfile)
	Gnew = convert.(Float64, Matrix(geno[testrowinds,:]))
end

Xnew = [Array(covdf[testrowinds, varlist]) Gnew]
XwithPCnew = [Array(covdf[testrowinds, varlistwithPC]) Gnew]

# pglmm (AIC)
yhat.pglmmAIC = PenalizedGLMM.predict(modelfit, Xnew, grmfile, grmrowinds = testrowinds, grmcolinds = trainrowinds, s = [pglmmAIC], outtype = :prob) |> x-> vec(x)
yhat.pglmmBIC = PenalizedGLMM.predict(modelfit, Xnew, grmfile, grmrowinds = testrowinds, grmcolinds = trainrowinds, s = [pglmmBIC], outtype = :prob) |> x-> vec(x)
yhat.cv_glmnet = GLMNet.predict(fit_glmnet, Xnew, outtype = :prob)[:,argmin(cv_glmnet.meanloss)]
yhat.cv_glmnetPC = GLMNet.predict(fit_glmnetPC, XwithPCnew, outtype = :prob)[:,argmin(cv_glmnetPC.meanloss)]

#-----------------------
# Save results
#-----------------------
CSV.write(datadir * "results.txt", 
                select(betas, :beta, :pglmmAIC, :pglmmBIC, :cv_glmnet, :cv_glmnetPC)
)
CSV.write(datadir * "fitted_values.txt", 
                select(yhat, :pglmmAIC, :pglmmBIC, :cv_glmnet, :cv_glmnetPC)
)

#-----------------------------------------------------
# False positive rate (FPR) for pglmm and glmnet
#-----------------------------------------------------
for fpr in [0:0.001:0.01;]

        # pglmm
        pglmmFPR_ind = findlast(sum((pglmm_β .!= 0) .& (betas.beta .== 0), dims = 1) / sum(betas.beta .== 0) .<= fpr)[2]
        betas.pglmmFPR = pglmm_β[:, pglmmFPR_ind]
        yhat.pglmmFPR = PenalizedGLMM.predict(modelfit, Xnew, grmfile, grmrowinds = testrowinds, grmcolinds = trainrowinds, s = [pglmmFPR_ind], outtype = :prob) |> x-> vec(x)

        # glmnet with no PCs
        glmnetFPR_ind = findlast(sum((glmnet_β .!= 0) .& (betas.beta .== 0), dims = 1) / sum(betas.beta .== 0) .<= fpr)[2]
        betas.glmnetFPR = glmnet_β[:, glmnetFPR_ind]
        yhat.glmnetFPR = GLMNet.predict(fit_glmnet, Xnew, outtype = :prob)[:,glmnetFPR_ind]

        # glmnet with 10 PCs
        glmnetPCFPR_ind = findlast(sum((glmnetPC_β .!= 0) .& (betas.beta .== 0), dims = 1) / sum(betas.beta .== 0) .<= fpr)[2]
        betas.glmnetPCFPR = glmnetPC_β[:, glmnetPCFPR_ind]
        yhat.glmnetPCFPR = GLMNet.predict(fit_glmnetPC, XwithPCnew, outtype = :prob)[:,glmnetPCFPR_ind]

        #-----------------------
        # Save results
        #-----------------------
        CSV.write(datadir * "results_" * "fpr" * string(fpr) * ".txt", 
                  select(betas, :beta, :pglmmFPR, :glmnetFPR, :glmnetPCFPR)
        )
        CSV.write(datadir * "fitted_values_" * "fpr" * string(fpr) * ".txt", 
                  select(yhat, :pglmmFPR, :glmnetFPR, :glmnetPCFPR)
        )
end

CSV.write(datadir * "pglmm_tau.txt", DataFrame(tau = nullmodel.τ, h2 = nullmodel.τ / sum([nullmodel.τ' pi^2/3])))