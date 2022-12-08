# PenalizedGLMM

PenalizedGLMM is a Julia package that fits Lasso regularization paths for high-dimensional genetic data using block coordinate descent for linear or logistic mixed models.

## Installation

This package requires Julia v1.6.2 or later. The package is not yet registered and can be installed via


```julia
using Pkg
Pkg.add(url = "https://github.com/julstpierre/PenalizedGLMM.jl")
```


```julia
using PenalizedGLMM
```

## Example data sets

The data folder of the package contains genetic data for 2504 individuals and 5000 SNPs from the [1000Genomes](https://www.internationalgenome.org/data/) Project in PLINK format. The covariate.txt file contains the population from which each individual belongs, the first 10 PCs, sex and training/test sets assignments. We also simulated a covariate AGE and a binary phenotype for all individuals. Finally, we also include a GRM in the form of a compressed .txt file that was calculated using the function `grm` from [SnpArrays.jl](https://openmendel.github.io/SnpArrays.jl/latest/).


```julia
const datadir = "data/"
const covfile = datadir * "covariate.txt"
const plinkfile = datadir * "geno"
const grmfile = datadir * "grm.txt.gz";
```

## 1. Estimation of variance components under the null

We read the example covariate file and find the corresponding rows for subjects in the train and test sets:


```julia
using CSV, DataFrames
covdf = CSV.read(covfile, DataFrame)
trainrowinds = findall(covdf.train)
testrowinds = setdiff(1:nrow(covdf), trainrowinds);
```

We fit the null logistic mixed model on the training set, with AGE, SEX as fixed effects and one random effect with variance-covariance structure parametrized by the GRM:


```julia
using GLM
nullmodel = pglmm_null(@formula(y ~ AGE + SEX) 
                      ,covfile
                      ,grmfile 
                      ,covrowinds = trainrowinds 
                      ,grminds = trainrowinds);
```

By default, the dispersion parameter φ for the binomial distribution is equal to 1. The estimated variance components are equal to


```julia
print([nullmodel.φ, nullmodel.τ[1]])
```

    [1.0, 0.6427207752071908]

The estimated intercept and fixed effects for AGE and SEX are equal to


```julia
print(nullmodel.α)
```

    [-0.7236231919887682, 0.0003074705440105167, 0.023197298500290876]

We can check that the AI-REML algorithm has effectively converged:


```julia
print(nullmodel.converged)
```

    true

## 2. Penalized logistic mixed model
After obtaining the variance components estimates under the null, we fit a penalized logistic mixed model using a lasso regularization term on the SNP effects in order to perform variable selection:


```julia
modelfit = pglmm(nullmodel, plinkfile, geneticrowinds = trainrowinds)
```

The coefficients for the genetic predictors at each value of λ are stored in `modelfit.betas`:


```julia
modelfit.betas
```




    5000×100 view(::Matrix{Float64}, :, 1:100) with eltype Float64:
     0.0  0.0  0.0  0.0  0.0  0.0  0.0  …   0.0         0.0         0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0      0.0         0.0         0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0      0.0         0.0         0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0      0.0         0.0         0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0      0.0         0.0         0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0  …   0.0         0.0         0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0      0.0693873   0.0705072   0.0716954
     0.0  0.0  0.0  0.0  0.0  0.0  0.0      0.0         0.0         0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0      0.0266505   0.0272976   0.0276218
     0.0  0.0  0.0  0.0  0.0  0.0  0.0      0.0337146   0.0327807   0.0316525
     0.0  0.0  0.0  0.0  0.0  0.0  0.0  …   0.0         0.0         0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0      0.0         0.0         0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0      0.0         0.0         0.0
     ⋮                        ⋮         ⋱                          
     0.0  0.0  0.0  0.0  0.0  0.0  0.0      0.0         0.0         0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0      0.0         0.0         0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0  …   0.0         0.0         0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0     -0.107755   -0.110852   -0.113882
     0.0  0.0  0.0  0.0  0.0  0.0  0.0      0.0         0.0         0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0      0.0         0.0         0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0      0.0         0.0         0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0  …   0.0         0.0         0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0     -0.0972381  -0.100543   -0.103532
     0.0  0.0  0.0  0.0  0.0  0.0  0.0      0.0         0.0         0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0      0.0         0.0         0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0      0.0         0.0         0.0



We can calculate the number of non-zero coefficients for the genetic predictors at each value of λ:


```julia
[length(findall(modelfit.betas[:, j] .!= 0)) for j in 1:size(modelfit.betas, 2)]'
```




    1×100 adjoint(::Vector{Int64}) with eltype Int64:
     0  2  2  5  5  5  5  5  5  6  6  7  9  …  1027  1032  1034  1036  1038  1042



To find the optimal λ, we can use AIC or BIC as model selection criteria:


```julia
pglmmAIC = PenalizedGLMM.GIC(modelfit, :AIC)
pglmmBIC = PenalizedGLMM.GIC(modelfit, :BIC);
print(modelfit.lambda[[pglmmAIC, pglmmBIC]])
```

    [30.28301918732795, 55.440282469160415]

The estimated values for the intercept and non-genetic covariates are stored in `modelfit.a0` and `modelfit.alphas` respectively:


```julia
[modelfit.a0'; modelfit.alphas]
```




    3×100 Matrix{Float64}:
      9.00291e-18   0.0338059    0.0141006   …  -23.9261      -24.2947
     -2.70731e-17   4.16088e-5   3.2427e-5       -0.00187031   -0.00185894
     -4.32384e-16  -0.00111175  -0.00326714      -0.122787     -0.124612



## 3. Polygenic Risk Score (PRS)

We can calculate a PRS for each individual in the test set using the predict function. By default, predictions are obtained on the full lasso path, but it is also possible to provide a vector containing indices for the values of the regulatization parameter λ. For example, we can predict a PRS for each individual in the test set using the estimated coefficients from the models obtained by using AIC and BIC respectively:


```julia
yhat = PenalizedGLMM.predict(modelfit
                            ,covfile
                            ,grmfile
                            ,plinkfile
                            ,covrowinds = testrowinds
                            ,covars = ["AGE", "SEX"]
                            ,geneticrowinds = testrowinds
                            ,grmrowinds = testrowinds
                            ,grmcolinds = trainrowinds
                            ,s = [pglmmAIC, pglmmBIC]
                            ,outtype = :prob
                            ) |>
        x-> DataFrame(x, [:AIC, :BIC])

print(first(yhat, 5))
```

    [1m5×2 DataFrame[0m
    [1m Row [0m│[1m AIC      [0m[1m BIC      [0m
    [1m     [0m│[90m Float64  [0m[90m Float64  [0m
    ─────┼────────────────────
       1 │ 0.334916  0.472156
       2 │ 0.254587  0.357044
       3 │ 0.613588  0.557021
       4 │ 0.682049  0.570351
       5 │ 0.28341   0.344761

We can determine which model provides the best prediction accuracy by comparing AUCs for the PRSs obtained via AIC and BIC. We use the [ROCAnalysis.jl](https://juliapackages.com/p/rocanalysis) package to calculate AUC for each model:


```julia
using ROCAnalysis
ctrls = (covdf[testrowinds,:y] .== 0)
cases = (covdf[testrowinds,:y] .== 1)

[ROCAnalysis.auc(roc(yhat[ctrls, i], yhat[cases, i])) for i in ("AIC", "BIC")]' |> 
    x-> DataFrame(Matrix(x), [:AIC, :BIC]) |> print
```

    [1m1×2 DataFrame[0m
    [1m Row [0m│[1m AIC      [0m[1m BIC     [0m
    [1m     [0m│[90m Float64  [0m[90m Float64 [0m
    ─────┼───────────────────
       1 │ 0.782088  0.76029

We see that both models result in comparable prediction accuracies, but the model using BIC has selected 12 times less predictors than the model based on AIC:


```julia
[length(findall(modelfit.betas[:,k] .!= 0)) for k in (pglmmAIC, pglmmBIC)]' |> 
    x-> DataFrame(Matrix(x), [:AIC, :BIC]) |> print
```

    [1m1×2 DataFrame[0m
    [1m Row [0m│[1m AIC   [0m[1m BIC   [0m
    [1m     [0m│[90m Int64 [0m[90m Int64 [0m
    ─────┼──────────────
       1 │   181     15

If we know which predictors are truly causal (for simulated data), then we can compare the true positive rate (TPR), false positive rate (FPR) and false discovery rate (FDR) of each model selection strategy:


```julia
betas = CSV.read(datadir * "betas.txt", DataFrame)
true_betas = findall(betas.beta .!= 0)

AIC_betas = findall(modelfit.betas[:,pglmmAIC] .!= 0)
BIC_betas = findall(modelfit.betas[:,pglmmBIC] .!= 0);

TPR = [length(intersect(true_betas, x)) / length(true_betas) for x in (AIC_betas, BIC_betas)]' 

FPR = [(length(x) - length(intersect(true_betas, x))) / (size(modelfit.betas, 1) - length(true_betas)) for x in (AIC_betas, BIC_betas)]'

FDR = [(length(x) - length(intersect(true_betas, x))) / length(x) for x in (AIC_betas, BIC_betas)]'

DataFrame(hcat(["TPR", "FPR", "FDR"], [TPR; FPR; FDR]), [:Metric, :AIC, :BIC]) |> print
```

    [1m3×3 DataFrame[0m
    [1m Row [0m│[1m Metric [0m[1m AIC       [0m[1m BIC         [0m
    [1m     [0m│[90m Any    [0m[90m Any       [0m[90m Any         [0m
    ─────┼────────────────────────────────
       1 │ TPR     0.58       0.22
       2 │ FPR     0.0307071  0.000808081
       3 │ FDR     0.839779   0.266667
