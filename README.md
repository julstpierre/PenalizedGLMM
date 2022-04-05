# PenalizedGLMM

PenalizedGLMM is a Julia package that fits entire Lasso regularization paths for linear or logistic mixed models using block coordinate descent.

## Quick start


```julia
using PenalizedGLMM
```


```julia
using CSV, DataFrames, StatsBase, GLM
```


```julia
# Define directory where data is located
const datadir = "data/"
const covfile = datadir * "covariate.txt"
const plinkfile = datadir * "geno"
const grmfile = datadir * "grm.txt.gz";
```


```julia
# Read covariate file and split into train and test sets
covdf = CSV.read(covfile, DataFrame)
trainrowinds = sample(1:nrow(covdf), Int(floor(nrow(covdf) * 0.8)); replace = false)
testrowinds = setdiff(1:nrow(covdf), trainrowinds);
```


```julia
# Fit null model with one random effect on the training set
nullmodel = pglmm_null(@formula(y ~ SEX), covfile, grmfile, covrowinds = trainrowinds, grminds = trainrowinds)
```




    (φ = 1.0, τ = [0.6617376626899395], α = [-0.6211902843653399, -0.13209449127176048], η = [-1.0815442913774027, -1.4605425615282233, -0.7342121620949644, -1.132061492617605, -1.0752634130093959, -1.0434025442620023, -1.1513399760469247, -0.8227523934857359, -0.714502608267745, -0.8558075481019416  …  -1.3449024373068048, -1.2808631686945593, -0.48530160149918355, -1.2206807082144773, -0.1700411781149116, -1.048773637868374, 0.4082501359858177, -1.6555451027494212, -1.0485003831834194, -1.0286230207288944], converged = true, τV = [0.6614482972887938 0.07024137884028683 … 0.0814650935169359 0.07015346364542127; 0.07024137884028683 0.6402162504324866 … 0.08391943731881317 0.054522291769375814; … ; 0.0814650935169359 0.08391943731881317 … 0.6324191067970655 0.07315436125076488; 0.07015346364542127 0.054522291769375814 … 0.07315436125076488 0.6194557619369717], y = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1  …  0, 0, 0, 0, 1, 0, 1, 0, 0, 0], X = [1.0 0.0; 1.0 0.0; … ; 1.0 0.0; 1.0 0.0], family = Binomial{Float64}(n=1, p=0.5))




```julia
# The estimated variance components of the models are equal to
nullmodel.φ, nullmodel.τ
```




    (1.0, [0.6617376626899395])




```julia
# The estimated intercept and fixed effect for SEX are equal to
nullmodel.α
```




    2-element Vector{Float64}:
     -0.6211902843653399
     -0.13209449127176048




```julia
# We can check if the algorithm has converged
nullmodel.converged
```




    true




```julia
# Fit a penalized logistic mixed model
# modelfit = pglmm(nullmodel, plinkfile, geneticrowinds = trainrowinds)
```


```julia
# Find λ that gives minimum AIC or BIC
#pglmmAIC = PenalizedGLMM.GIC(modelfit, :AIC)
#pglmmBIC = PenalizedGLMM.GIC(modelfit, :BIC)
```
