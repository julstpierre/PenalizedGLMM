# PenalizedGLMM

PenalizedGLMM is a Julia package that fits Lasso regularization paths for high-dimensional genetic data using block coordinate descent for linear or logistic mixed models.

## Installation

This package requires Julia v1.6.2 or later. The package is not yet registered and can be installed via


```julia
Pkg.add(url = "https://github.com/julstpierre/PenalizedGLMM.jl")
```

For this tutorial, we will be needing the following packages:


```julia
using PenalizedGLMM, CSV, DataFrames, StatsBase, GLM, SnpArrays, ROCAnalysis
```

## Example data sets

The data folder of the package contains genetic data for 2504 individuals from the 1000Genomes Project in PLINK format. The covariate.txt file contains SEX and binary phenotype info for all individuals. Finally, we also include a GRM in the form of a compressed .txt file that was calculated using the function `grm` from [SnpArrays.jl](https://openmendel.github.io/SnpArrays.jl/latest/). 


```julia
const datadir = "data/"
const covfile = datadir * "covariate.txt"
const plinkfile = datadir * "geno"
const grmfile = datadir * "grm.txt.gz";
```

## 1. Estimate the variance components under the null

We read the example covariate file and split the subjects into train and test sets.


```julia
covdf = CSV.read(covfile, DataFrame)
trainrowinds = sample(1:nrow(covdf), Int(floor(nrow(covdf) * 0.8)); replace = false)
testrowinds = setdiff(1:nrow(covdf), trainrowinds);
```

We fit the null logistic mixed model on the training set, with SEX as fixed effect and one random effect with variance-covariance structure parametrized by the GRM.


```julia
nullmodel = pglmm_null(@formula(y ~ SEX) 
                      ,covfile
                      ,grmfile 
                      ,covrowinds = trainrowinds 
                      ,grminds = trainrowinds);
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



## 2. Fit a penalized logistic mixed model

After obtaining the variance components estimates under the null, we fit a penalized logistic mixed model using a lasso regularization term on the SNP effects in order to perform variable selection.


```julia
modelfit = pglmm(nullmodel, plinkfile, geneticrowinds = trainrowinds);
```

The coefficients for each value of λ are stored in `modelfit.betas`


```julia
modelfit.betas
```




    5001×100 view(::Matrix{Float64}, 2:5002, 1:100) with eltype Float64:
     -0.125515  -0.126354  -0.127168  …  -0.664979   -0.672408   -0.679633
      0.0        0.0        0.0           0.0         0.0         0.0
      0.0        0.0        0.0           0.0         0.0         0.0
      0.0        0.0        0.0           0.0         0.0         0.0
      0.0        0.0        0.0          -0.0252843  -0.0238376  -0.0227283
      0.0        0.0        0.0       …   0.0         0.0         0.0
      0.0        0.0        0.0           0.0         0.0         0.0
      0.0        0.0        0.0           0.205792    0.202325    0.198666
      0.0        0.0        0.0          -0.0188797  -0.0194645  -0.0195953
      0.0        0.0        0.0           0.0397702   0.04399     0.0476746
      0.0        0.0        0.0       …   0.0         0.0         0.0
      0.0        0.0        0.0          -0.107175   -0.108742   -0.109795
      0.0        0.0        0.0           0.0         0.0         0.0
      ⋮                               ⋱                          
      0.0        0.0        0.0           0.0         0.0         0.0
      0.0        0.0        0.0       …   0.329599    0.334526    0.339608
      0.0        0.0        0.0           0.0         0.0         0.0
      0.0        0.0        0.0           0.0774474   0.0784331   0.0793867
      0.0        0.0        0.0           0.0         0.0         0.0
      0.0        0.0        0.0           0.0         0.0         0.0
      0.0        0.0        0.0       …   0.0         0.0         0.0
      0.0        0.0        0.0           0.0         0.0         0.0
      0.0        0.0        0.0           0.0         0.0         0.0
      0.0        0.0        0.0           0.0         0.0         0.0
      0.0        0.0        0.0           0.0         0.0         0.0
      0.0        0.0        0.0       …   0.0         0.0         0.0



 We can calculate the number of non-zero coefficients for each value of λ


```julia
[length(findall(x -> x != 0, view(modelfit.betas, :,k))) for k in 1:size(modelfit.betas, 2)]'
```




    1×100 adjoint(::Vector{Int64}) with eltype Int64:
     2  2  2  2  2  3  3  3  3  4  6  7  9  …  1049  1058  1063  1065  1073  1079



To find the optimal λ, we can use AIC or BIC


```julia
pglmmAIC = PenalizedGLMM.GIC(modelfit, :AIC);
pglmmBIC = PenalizedGLMM.GIC(modelfit, :BIC);
```

## 3. Calculate Polygenic Risk Score (PRS) on test individuals

To make predictions on the test set, we convert PLINK genotype to matrix, using the package [SnpArrays.jl](https://openmendel.github.io/SnpArrays.jl/latest/). We convert to additive model (default) and impute missing values.


```julia
geno = SnpArray(plinkfile * ".bed")
Gtest = convert(Matrix{Float64}, @view(geno[testrowinds,:]), model = ADDITIVE_MODEL, impute = true);
```

We combine genotype with the covariate(s) into an array.


```julia
Xtest = [covdf[testrowinds, "SEX"] Gtest];
```

Finally, we can make prediction using the `PenalizedGLMM.predict` function. By default, predictions for the full lasso path are calculated. We can also obtain predictions for optimal λ only, for example by comparing predictions obtained using AIC and BIC.


```julia
yhat = PenalizedGLMM.predict(modelfit
                            ,Xtest
                            ,grmfile
                            ,grmrowinds = testrowinds
                            ,grmcolinds = trainrowinds
                            ,s = [pglmmAIC, pglmmBIC]
                            ,outtype = :prob
                            ) |>
        x-> DataFrame(x, [:AIC, :BIC])

first(yhat, 5)
```




<div class="data-frame"><p>5 rows × 2 columns</p><table class="data-frame"><thead><tr><th></th><th>AIC</th><th>BIC</th></tr><tr><th></th><th title="Float64">Float64</th><th title="Float64">Float64</th></tr></thead><tbody><tr><th>1</th><td>0.244937</td><td>0.232804</td></tr><tr><th>2</th><td>0.239528</td><td>0.243225</td></tr><tr><th>3</th><td>0.190408</td><td>0.18723</td></tr><tr><th>4</th><td>0.333759</td><td>0.296043</td></tr><tr><th>5</th><td>0.171658</td><td>0.160262</td></tr></tbody></table></div>



We can determine which model provides best prediction accuracy by comparing AUCs for the PRSs obtained via AIC and BIC. We use the [ROCAnalysis.jl](https://juliapackages.com/p/rocanalysis) package to calculate AUC for each model.


```julia
ctrls = (covdf[testrowinds,:y] .== 0)
cases = (covdf[testrowinds,:y] .== 1)

[ROCAnalysis.auc(roc(yhat[ctrls, i], yhat[cases, i])) for i in ("AIC", "BIC")]' |> 
    x-> DataFrame(Matrix(x), [:AIC, :BIC])
```




<div class="data-frame"><p>1 rows × 2 columns</p><table class="data-frame"><thead><tr><th></th><th>AIC</th><th>BIC</th></tr><tr><th></th><th title="Float64">Float64</th><th title="Float64">Float64</th></tr></thead><tbody><tr><th>1</th><td>0.8045</td><td>0.799065</td></tr></tbody></table></div>



We see that both models result in comparable prediction accuracies, but the model using BIC has selected almost 4 times less predictors than the model based on AIC:


```julia
[length(findall(x -> x != 0, view(modelfit.betas, :,k))) for k in (pglmmAIC, pglmmBIC)]' |> 
    x-> DataFrame(Matrix(x), [:AIC, :BIC])
```




<div class="data-frame"><p>1 rows × 2 columns</p><table class="data-frame"><thead><tr><th></th><th>AIC</th><th>BIC</th></tr><tr><th></th><th title="Int64">Int64</th><th title="Int64">Int64</th></tr></thead><tbody><tr><th>1</th><td>88</td><td>23</td></tr></tbody></table></div>


