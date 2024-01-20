# Load packages
using Pkg; Pkg.activate("..")
using PenalizedGLMM
using GLM, GLMNet, SnpArrays, CSV, DataFrames, LinearAlgebra, ROCAnalysis

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
trainrowinds = findall(covdf.set .== "train")
tunerowinds = findall(covdf.set .== "tune")
testrowinds = findall(covdf.set .== "test")

# Fit null model with one random effect
nullmodel = pglmm_null(@formula(y ~ SEX + AGE), covfile, grmfile, covrowinds = trainrowinds, grminds = trainrowinds)

# Fit a penalized logistic mixed model
if isfile(plinkfile * ".bed")
        modelfit = pglmm(nullmodel, plinkfile, geneticrowinds = trainrowinds, verbose = true)
elseif isfile(snpfile)
        modelfit = pglmm(nullmodel, snpfile = snpfile, geneticrowinds = trainrowinds, verbose = true)
end

# Genetic predictors effects at each λ 
varlist = ["SEX", "AGE"]  
pglmm_β = modelfit.betas[(length(varlist) + 1):end,:]

# Find λ that gives minimum GIC
pglmmAIC = PenalizedGLMM.GIC(modelfit, :AIC)
pglmmBIC = PenalizedGLMM.GIC(modelfit, :BIC)

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

# y must be a matrix with one column per class
y = convert(Matrix{Float64}, [covdf.y .== 0 covdf.y .== 1]) |> 
        x -> x[trainrowinds,:]

p = size(G, 2)

#----------------------------
# Lasso with no PC
#----------------------------
# Combine covariate with genetic predictors
X = [Array(covdf[trainrowinds, varlist]) G]

# Fit a penalized logistic model using GLMNet with no PCs
fit_glmnet = glmnet(X, y, Binomial(), penalty_factor = [zeros(length(varlist)); ones(p)])
glmnet_β = fit_glmnet.betas[(length(varlist) + 1):end,:]

# Select best penalized logistic model using GLMNet cross-validation
cv_glmnet = glmnetcv(X, y, Binomial(), penalty_factor = [zeros(length(varlist)); ones(p)])

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

#-----------------------------------------------------------------------
# Make prediction on combined tune+test set
#-----------------------------------------------------------------------
# Create DataFrame for predicted values
yhat = DataFrame()

# Create Arrays for tune+test set
# Genetic predictors
if isfile(plinkfile * ".bed")
	Gnew = convert(Matrix{Float64}, @view(geno[sort([testrowinds; tunerowinds]),:]), model = ADDITIVE_MODEL, impute = true)
elseif isfile(snpfile)
	Gnew = convert.(Float64, Matrix(geno[sort([testrowinds; tunerowinds]),:]))
end

# Covariates
Xnew = [Array(covdf[sort([testrowinds; tunerowinds]), varlist]) Gnew]
XwithPCnew = [Array(covdf[sort([testrowinds; tunerowinds]), varlistwithPC]) Gnew]

# Make prediction on combined tune+test set
yhat.pglmmAIC = PenalizedGLMM.predict(modelfit, Xnew, grmfile, grmrowinds = sort([testrowinds; tunerowinds]), grmcolinds = trainrowinds, s = [pglmmAIC], outtype = :prob) |> x-> vec(x)
yhat.pglmmBIC = PenalizedGLMM.predict(modelfit, Xnew, grmfile, grmrowinds = sort([testrowinds; tunerowinds]), grmcolinds = trainrowinds, s = [pglmmBIC], outtype = :prob) |> x-> vec(x)
yhat.cv_glmnet = GLMNet.predict(fit_glmnet, Xnew, outtype = :prob)[:,argmin(cv_glmnet.meanloss)]
yhat.cv_glmnetPC = GLMNet.predict(fit_glmnetPC, XwithPCnew, outtype = :prob)[:,argmin(cv_glmnetPC.meanloss)]

#-----------------------------------------------------------------------
# Find best model using tune set, and make prediction on test set only
#-----------------------------------------------------------------------
# Genetic predictors
# Tune set
if isfile(plinkfile * ".bed")
	Gtune = convert(Matrix{Float64}, @view(geno[tunerowinds,:]), model = ADDITIVE_MODEL, impute = true)
elseif isfile(snpfile)
	Gtune = convert.(Float64, Matrix(geno[tunerowinds,:]))
end

# Test set
if isfile(plinkfile * ".bed")
	Gtest = convert(Matrix{Float64}, @view(geno[testrowinds,:]), model = ADDITIVE_MODEL, impute = true)
elseif isfile(snpfile)
	Gtest = convert.(Float64, Matrix(geno[testrowinds,:]))
end

# Covariates
Xtune = [Array(covdf[tunerowinds, varlist]) Gtune]
XwithPCtune = [Array(covdf[tunerowinds, varlistwithPC]) Gtune]

Xtest = [Array(covdf[testrowinds, varlist]) Gtest]
XwithPCtest = [Array(covdf[testrowinds, varlistwithPC]) Gtest]

# Find best model based on AUC
# pglmm
pglmmtune = PenalizedGLMM.predict(modelfit, Xtune, grmfile, grmrowinds = tunerowinds, grmcolinds = trainrowinds, outtype = :prob)
pglmm_best_model = [ROCAnalysis.auc(roc(pglmmtune[covdf[tunerowinds,:y] .== 0, i], pglmmtune[covdf[tunerowinds,:y] .== 1, i])) for i in 1:size(pglmmtune, 2)] |>
        x->argmax(x)

# glmnet
glmnettune = GLMNet.predict(fit_glmnet, Xtune, outtype = :prob)
glmnet_best_model = [ROCAnalysis.auc(roc(glmnettune[covdf[tunerowinds,:y] .== 0, i], glmnettune[covdf[tunerowinds,:y] .== 1, i])) for i in 1:size(glmnettune, 2)] |>
        x->argmax(x)

# glmnetPC
glmnetPCtune = GLMNet.predict(fit_glmnetPC, XwithPCtune, outtype = :prob)
glmnetPC_best_model = [ROCAnalysis.auc(roc(glmnetPCtune[covdf[tunerowinds,:y] .== 0, i], glmnetPCtune[covdf[tunerowinds,:y] .== 1, i])) for i in 1:size(glmnetPCtune, 2)] |>
        x->argmax(x)

# Make prediction on test set
_yhat = DataFrame()
_yhat.pglmm = PenalizedGLMM.predict(modelfit, Xtest, grmfile, grmrowinds = testrowinds, grmcolinds = trainrowinds, s = [pglmm_best_model], outtype = :prob) |> x-> vec(x)
_yhat.glmnet = GLMNet.predict(fit_glmnet, Xtest, outtype = :prob)[:, glmnet_best_model ]
_yhat.glmnetPC = GLMNet.predict(fit_glmnetPC, XwithPCtest, outtype = :prob)[:, glmnetPC_best_model ]

#------------------------------------------
# Compare estimated betas with real values
#------------------------------------------
# Read file with real values
betas = CSV.read(datadir * "betas.txt", DataFrame)

# Estimated betas for pglmm
betas.pglmmAIC = pglmm_β[:, pglmmAIC]
betas.pglmmBIC = pglmm_β[:, pglmmBIC]
betas.pglmm = pglmm_β[:, pglmm_best_model]

# Estimated betas for glmnet
betas.glmnet = fit_glmnet.betas[(length(varlist) + 1):end, glmnet_best_model]
betas.cv_glmnet = cv_glmnet.path.betas[(length(varlist) + 1):end, argmin(cv_glmnet.meanloss)]

# Estimated betas for glmnetPC
betas.glmnetPC = fit_glmnetPC.betas[(length(varlistwithPC) + 1):end, glmnetPC_best_model]
betas.cv_glmnetPC = cv_glmnetPC.path.betas[(length(varlistwithPC) + 1):end, argmin(cv_glmnetPC.meanloss)]

#-----------------------
# Save results
#-----------------------
CSV.write(datadir * "results.txt", 
                select(betas, :beta, :pglmm, :pglmmAIC, :pglmmBIC, :glmnet, :cv_glmnet, :glmnetPC, :cv_glmnetPC)
)
CSV.write(datadir * "fitted_values_tune_test.txt", 
                select(yhat, :pglmmAIC, :pglmmBIC, :cv_glmnet, :cv_glmnetPC)
)
CSV.write(datadir * "fitted_values_test.txt", 
                select(_yhat, :pglmm, :glmnet, :glmnetPC)
)

#-----------------------------------------------------
# False positive rate (FPR) for pglmm and glmnet
#-----------------------------------------------------
for fpr in [0:0.001:0.01;]

        # pglmm
        pglmmFPR_ind = findlast(sum((pglmm_β .!= 0) .& (betas.beta .== 0), dims = 1) / sum(betas.beta .== 0) .<= fpr)[2]
        betas.pglmmFPR = pglmm_β[:, pglmmFPR_ind]
        yhat.pglmmFPR = PenalizedGLMM.predict(modelfit, Xnew, grmfile, grmrowinds = sort([testrowinds; tunerowinds]), grmcolinds = trainrowinds, s = [pglmmFPR_ind], outtype = :prob) |> x-> vec(x)

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

#-----------------------------------------------------
# Model size for pglmm and glmnet
#-----------------------------------------------------
for size_ in range(5, 50, length = 10)

        # pglmm
        pglmmsize_ind = findlast(sum(pglmm_β .!= 0, dims = 1) .<= size_)[2]
        betas.pglmmsize = pglmm_β[:, pglmmsize_ind]
        yhat.pglmmsize = PenalizedGLMM.predict(modelfit, Xnew, grmfile, grmrowinds = sort([testrowinds; tunerowinds]), grmcolinds = trainrowinds, s = [pglmmsize_ind], outtype = :prob) |> x-> vec(x)

        # glmnet with no PCs
        glmnetsize_ind = findlast(sum(glmnet_β .!= 0, dims = 1) .<= size_)[2]
        betas.glmnetsize = glmnet_β[:, glmnetsize_ind]
        yhat.glmnetsize = GLMNet.predict(fit_glmnet, Xnew, outtype = :prob)[:,glmnetsize_ind]

        # glmnet with 10 PCs
        glmnetPCsize_ind = findlast(sum(glmnetPC_β .!= 0, dims = 1) .<= size_)[2]
        betas.glmnetPCsize = glmnetPC_β[:, glmnetPCsize_ind]
        yhat.glmnetPCsize = GLMNet.predict(fit_glmnetPC, XwithPCnew, outtype = :prob)[:,glmnetPCsize_ind]

        #-----------------------
        # Save results
        #-----------------------
        CSV.write(datadir * "results_" * "size" * string(size_) * ".txt", 
                  select(betas, :beta, :pglmmsize, :glmnetsize, :glmnetPCsize)
        )

        CSV.write(datadir * "fitted_values_" * "size" * string(size_) * ".txt", 
                  select(yhat, :pglmmsize, :glmnetsize, :glmnetPCsize)
        )
end

CSV.write(datadir * "pglmm_tau.txt", DataFrame(tau = nullmodel.τ, h2 = nullmodel.τ / sum([nullmodel.τ' pi^2/3])))