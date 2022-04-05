# PenalizedGLMM

PenalizedGLMM is a Julia package that fits Lasso regularization paths for high-dimensional genetic data using block coordinate descent for linear or logistic mixed models.

## Installation

This package requires Julia v1.6.2 or later. The package is not yet registered and can be installed via


```julia
Pkg.add(url = "https://github.com/julstpierre/PenalizedGLMM.jl")
```

For this tutorial, we will be needing the following packages:


```julia
using PenalizedGLMM, CSV, DataFrames, StatsBase, GLM
```

## Example data sets

The data folder of the package contains genetic data for 2504 individuals from the 1000Genomes Project in PLINK format. The covariate.txt file contains SEX and phenotype info for all individuals. Finally, we also include a GRM in the form of a compressed .txt file that was calculated using the function `grm` from [SnpArrays.jl](https://openmendel.github.io/SnpArrays.jl/latest/). 


```julia
const datadir = "data/"
const covfile = datadir * "covariate.txt"
const plinkfile = datadir * "geno"
const grmfile = datadir * "grm.txt.gz";
```

## Basic usage

We read the example covariate file and split the subjects into train and test sets.


```julia
covdf = CSV.read(covfile, DataFrame)
trainrowinds = sample(1:nrow(covdf), Int(floor(nrow(covdf) * 0.8)); replace = false)
testrowinds = setdiff(1:nrow(covdf), trainrowinds);
```

We fit the null model on the training set, with SEX as fixed effect and one random effect with variance-covariance structure parametrized by the GRM.


```julia
nullmodel = pglmm_null(@formula(y ~ SEX), covfile, grmfile, covrowinds = trainrowinds, grminds = trainrowinds);
```

The estimated variance components of the models are equal to


```julia
nullmodel.φ, nullmodel.τ
```




    (1.0, [0.6347418361047935])



The estimated intercept and fixed effect for SEX are equal to


```julia
nullmodel.α
```




    2-element Vector{Float64}:
     -0.6211902843653399
     -0.13209449127176048



We can check that the AIREML algorithm has effectively converged


```julia
nullmodel.converged
```




    true



After obtaining the variance components estimates under the null, we fit a penalized logistic mixed model using a lasso regularization:


```julia
modelfit = pglmm(nullmodel, plinkfile, geneticrowinds = trainrowinds)
```


```julia
# Find λ that gives minimum AIC or BIC
#pglmmAIC = PenalizedGLMM.GIC(modelfit, :AIC)
#pglmmBIC = PenalizedGLMM.GIC(modelfit, :BIC)
```
