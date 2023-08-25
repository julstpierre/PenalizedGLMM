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

    [1.0, 0.718662801592849]

The estimated intercept and fixed effects for AGE and SEX are equal to


```julia
print(nullmodel.α)
```

    [-0.03317721475728723, -0.009395389811651098, 0.028378808111444494]

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
Matrix(modelfit.betas)
```




    5000×100 Matrix{Float64}:
     0.0  0.0  0.0  0.0  0.0  0.0  0.0  …   0.0         0.0         0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0      0.0         0.0         0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0      0.0         0.0         0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0     -0.021803   -0.0240124  -0.0265078
     0.0  0.0  0.0  0.0  0.0  0.0  0.0      0.0         0.0         0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0  …   0.0         0.0         0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0      0.0         0.0         0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0      0.0         0.0         0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0      0.0         0.0         0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0      0.0         0.0         0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0  …   0.0         0.0         0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0      0.0         0.0         0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0      0.0         0.0         0.0
     ⋮                        ⋮         ⋱                          
     0.0  0.0  0.0  0.0  0.0  0.0  0.0      0.0         0.0         0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0      0.0         0.0         0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0  …   0.0         0.0         0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0      0.425332    0.432151    0.438802
     0.0  0.0  0.0  0.0  0.0  0.0  0.0      0.0         0.0         0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0      0.0         0.0         0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0      0.0         0.0         0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0  …   0.0         0.0         0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0     -0.164794   -0.167476   -0.170247
     0.0  0.0  0.0  0.0  0.0  0.0  0.0      0.0708796   0.0721145   0.0731258
     0.0  0.0  0.0  0.0  0.0  0.0  0.0      0.0         0.0         0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0      0.0         0.0         0.0



We can calculate the number of non-zero coefficients for the genetic predictors at each value of λ:


```julia
[length(findall(modelfit.betas[:, j] .!= 0)) for j in 1:size(modelfit.betas, 2)]'
```




    1×100 adjoint(::Vector{Int64}) with eltype Int64:
     0  1  1  2  2  3  4  6  7  7  8  9  …  1142  1145  1145  1148  1150  1154



To find the optimal λ, we can use AIC or BIC as model selection criteria:


```julia
pglmmAIC = PenalizedGLMM.GIC(modelfit, :AIC)
pglmmBIC = PenalizedGLMM.GIC(modelfit, :BIC);
print(modelfit.lambda[[pglmmAIC, pglmmBIC]])
```

    [38.45282781207581, 61.227776080476154]

The estimated values for the intercept and non-genetic covariates are stored in `modelfit.a0` and `modelfit.alphas` respectively:


```julia
Matrix([modelfit.a0'; modelfit.alphas])
```




    3×100 Matrix{Float64}:
     -1.14681e-13   0.0341377   …  22.6047     22.8143     23.011
     -5.50032e-15   2.79311e-5     -0.0299055  -0.0304224  -0.0309717
     -2.3118e-14   -8.56448e-5     -0.162442   -0.163483   -0.164323



## 3. Polygenic Risk Score (PRS)

We can calculate a PRS for each individual in the test set using the predict function. By default, predictions are obtained on the full lasso path, but it is also possible to provide a vector containing indices for the values of the regularization parameter λ. For example, we can predict a PRS for each individual in the test set using the estimated coefficients from the models obtained by using AIC and BIC respectively:


```julia
using Latexify
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
                            );
```

The predicted probabilities for the first 5 subjects are equal to:


```julia
DataFrame(hcat(covdf.IID[testrowinds], yhat)[1:5, :], [:IID, :AIC, :BIC]) |> 
    x-> latexify(x, fmt = "%5.4f", latex = false)
```




|     IID |    AIC |    BIC |
| -------:| ------:| ------:|
| HG00110 | 0.3479 | 0.3332 |
| HG00113 | 0.3687 | 0.3455 |
| HG00121 | 0.3261 | 0.3154 |
| HG00130 | 0.4837 | 0.3726 |
| HG00132 | 0.1856 | 0.2237 |




We can determine which model provides the best prediction accuracy by comparing AUCs for the PRSs obtained via AIC and BIC. We use the [ROCAnalysis.jl](https://juliapackages.com/p/rocanalysis) package to calculate AUC for each model:


```julia
using ROCAnalysis
ctrls = (covdf[testrowinds,:y] .== 0)
cases = (covdf[testrowinds,:y] .== 1)

[ROCAnalysis.auc(roc(yhat[ctrls, i], yhat[cases, i])) for i in 1:2]' |> 
    x-> DataFrame(hcat("AUC", x), [:~, :AIC, :BIC]) |> x-> latexify(x, fmt = "%5.4f", latex = false)
```




|   ~ |    AIC |    BIC |
| ---:| ------:| ------:|
| AUC | 0.7416 | 0.7414 |




We see that the models based on AIC and BIC resulted in the same higher prediction accuracy, but the model based on BIC has selected 10 times less predictors than the model based on AIC:


```julia
[length(findall(modelfit.betas[:,k] .!= 0)) for k in (pglmmAIC, pglmmBIC)]' |> 
    x-> DataFrame(hcat("Number of predictors", x), [:~, :AIC, :BIC]) |> x-> latexify(x, fmt = "%5.0f", latex = false)
```




|                    ~ | AIC | BIC |
| --------------------:| ---:| ---:|
| Number of predictors | 101 |   9 |




If we know which predictors are truly causal (for simulated data), then we can compare the true positive rate (TPR), false positive rate (FPR) and false discovery rate (FDR) of each model selection strategy:


```julia
betas = CSV.read(datadir * "betas.txt", DataFrame)
true_betas = findall(betas.beta .!= 0)

AIC_betas = findall(modelfit.betas[:,pglmmAIC] .!= 0)
BIC_betas = findall(modelfit.betas[:,pglmmBIC] .!= 0);

TPR = [length(intersect(true_betas, x)) / length(true_betas) for x in (AIC_betas, BIC_betas)]' 

FPR = [(length(x) - length(intersect(true_betas, x))) / (size(modelfit.betas, 1) - length(true_betas)) for x in (AIC_betas, BIC_betas)]'

FDR = [(length(x) - length(intersect(true_betas, x))) / length(x) for x in (AIC_betas, BIC_betas)]'

DataFrame(hcat(["AIC", "BIC"], [TPR; FPR; FDR]'), [:Model, :TPR, :FPR, :FDR]) |> x-> latexify(x, fmt = "%5.4f", latex = false)
```




| Model |    TPR |    FPR |    FDR |
| -----:| ------:| ------:| ------:|
|   AIC | 0.3000 | 0.0174 | 0.8515 |
|   BIC | 0.1400 | 0.0004 | 0.2222 |




## 4. Joint selection of main genetic effects and gene by environment (GEI) effects

To perform joint selection of main genetic and GEI effects in sparse regularized logistic mixed models, one simply need to define a binary exposure variable for which it is believed there might important interaction effects with genetic predictors. 

Instead of fitting the null and sparse group lasso models separately, we use the `pglmm_cv` function which allows to perform cross-validation. We use sex as a dummy for environmental exposure, and include a second kinship matrix that accounts for the similarity of individuals due to random polygenic GEI effects. The `rho` parameter controls the relative sparsity of the GEI effects for each SNP, and we suggest testing a range of values from 0 to 0.9.


```julia
modelfit = pglmm_cv(@formula(y ~ AGE + SEX), 
                    covfile, 
                    grmfile, 
                    plinkfile, 
                    covrowinds = trainrowinds, 
                    grminds = trainrowinds, 
                    geneticrowinds = trainrowinds, 
                    GEIvar = "SEX", 
                    GEIkin = true, 
                    rho = [0, 0.5, 0.9], 
                    nlambda = 20,
                    nfolds = 4
);
```

We obtain a solution path for each value of `rho`:


```julia
modelfit.path
```




    3-element Vector{PenalizedGLMM.pglmmPath{Binomial{Float64}, SparseArrays.SparseVector{Float64, Int64}, SparseArrays.SparseMatrixCSC{Float64, Int64}, Float64, Vector{Float64}, Matrix{Float64}}}:
     Logistic Solution Path (20 solutions for 10002 predictors):
    ──────────────────────────────────
           df   pct_dev         λ    ρ
    ──────────────────────────────────
     [1]    2  0.188183  102.939   0.0
     [2]    4  0.18935    98.2606  0.0
     [3]    4  0.190446   93.7945  0.0
     [4]    6  0.191791   89.5314  0.0
     [5]    6  0.193525   85.4621  0.0
     [6]    8  0.195076   81.5777  0.0
     [7]   12  0.201035   77.8699  0.0
     [8]   12  0.204125   74.3306  0.0
     [9]   16  0.207401   70.9521  0.0
    [10]   18  0.212282   67.7272  0.0
    [11]   18  0.216017   64.6489  0.0
    [12]   20  0.21952    61.7105  0.0
    [13]   26  0.224584   58.9057  0.0
    [14]   28  0.228649   56.2283  0.0
    [15]   40  0.234235   53.6727  0.0
    [16]   56  0.242755   51.2332  0.0
    [17]   76  0.252626   48.9045  0.0
    [18]   94  0.262022   46.6818  0.0
    [19]  124  0.275344   44.56    0.0
    [20]  138  0.287114   42.5347  0.0
    ──────────────────────────────────
     Logistic Solution Path (20 solutions for 10002 predictors):
    ─────────────────────────────────
          df   pct_dev         λ    ρ
    ─────────────────────────────────
     [1]   2  0.188183  205.879   0.5
     [2]   3  0.189288  196.521   0.5
     [3]   3  0.19037   187.589   0.5
     [4]   4  0.191383  179.063   0.5
     [5]   4  0.193078  170.924   0.5
     [6]   4  0.194616  163.155   0.5
     [7]   7  0.196886  155.74    0.5
     [8]   8  0.200135  148.661   0.5
     [9]   9  0.203714  141.904   0.5
    [10]   9  0.207429  135.454   0.5
    [11]  10  0.211254  129.298   0.5
    [12]  11  0.214844  123.421   0.5
    [13]  14  0.218793  117.811   0.5
    [14]  14  0.222961  112.457   0.5
    [15]  16  0.226801  107.345   0.5
    [16]  32  0.233004  102.466   0.5
    [17]  39  0.240634   97.8091  0.5
    [18]  47  0.249146   93.3635  0.5
    [19]  58  0.25873    89.12    0.5
    [20]  74  0.269658   85.0694  0.5
    ─────────────────────────────────
     Logistic Solution Path (20 solutions for 10002 predictors):
    ─────────────────────────────────
          df   pct_dev         λ    ρ
    ─────────────────────────────────
     [1]   2  0.188183  1029.39   0.9
     [2]   3  0.189288   982.606  0.9
     [3]   3  0.19037    937.945  0.9
     [4]   4  0.191383   895.314  0.9
     [5]   4  0.193078   854.621  0.9
     [6]   4  0.194616   815.777  0.9
     [7]   7  0.196886   778.699  0.9
     [8]   8  0.200135   743.306  0.9
     [9]   9  0.203714   709.521  0.9
    [10]   9  0.207429   677.272  0.9
    [11]  10  0.211254   646.489  0.9
    [12]  11  0.214844   617.105  0.9
    [13]  14  0.218793   589.057  0.9
    [14]  14  0.222961   562.283  0.9
    [15]  16  0.226801   536.727  0.9
    [16]  32  0.233004   512.332  0.9
    [17]  38  0.240634   489.045  0.9
    [18]  46  0.249106   466.818  0.9
    [19]  57  0.258614   445.6    0.9
    [20]  72  0.269464   425.347  0.9
    ─────────────────────────────────




```julia
print([modelfit.rho; modelfit.lambda])
```

    PenalizedGLMM.TuningParms[value = 0.0, index = 1, value = 48.90454152938238, index = 17]

The best model based on cross-validation is found for $\rho=0$ and $\lambda=48.9045$.


```julia
print([length(findall(modelfit.path[1].betas[:, 17] .!= 0)); 
       length(findall(modelfit.path[1].gammas[:, 17] .!= 0))]')
```

    [37 37]

The number of selected main effects and GEI effects for the best model are equal, since when $\rho=0$ the sparse group lasso is equivalent to a group lasso. Finally, we can obtain predictions on the test set for the full lasso path using the `predict` function:


```julia
yhat_path = PenalizedGLMM.predict(modelfit.path
                                ,covfile
                                ,grmfile
                                ,plinkfile
                                ,covrowinds = testrowinds
                                ,covrowtraininds = trainrowinds
                                ,covars = ["AGE", "SEX"]
                                ,GEIvar = "SEX"
                                ,GEIkin = true
                                ,geneticrowinds = testrowinds
                                ,grmrowinds = testrowinds
                                ,grmcolinds = trainrowinds
                                ,outtype = :prob
                                );
```
