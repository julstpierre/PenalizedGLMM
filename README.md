```julia
using PenalizedGLMM
```


```julia
using CSV, StatsBase
```


```julia
# Define directory where data is located
const datadir = "data/"
const covfile = datadir * "covariate.txt"
const plinkfile = datadir * "geno"
const grmfile = datadir * "grm.txt.gz"
;
```


```julia
# Read covariate file and split into train and test sets
covdf = CSV.read(covfile, DataFrame)
trainrowinds = sample(1:nrow(covdf), Int(floor(nrow(covdf) * 0.8)); replace = false)
testrowinds = setdiff(1:nrow(covdf), trainrowinds)
;
```


```julia
# Fit null model with one random effect on the training set
nullmodel = pglmm_null(@formula(y ~ SEX), covfile, grmfile, covrowinds = trainrowinds, grminds = trainrowinds)
```




    (φ = 1.0, τ = [0.5785829161713041], α = [-0.6914574895991064, -0.05699758175634297], η = [-1.1238507215440368, -1.1540271336033499, -1.3805020338750404, -1.1640028386300991, -1.5309284838239745, -0.32585110275742446, -1.084412547369445, -1.1645193056005803, 0.16025168814587776, -1.4945599804452891  …  -1.142444822496975, 0.4471429528836868, -1.1906327803089385, 0.4829717752925058, -1.0929216289268027, -0.3480207641760802, -0.8685702830303061, -1.1388686441997067, -0.7970635238533132, -0.3291115648156864], converged = true, τV = [0.824930229383528 -0.0713704617454726 … -0.05700464383352525 0.20770976930425203; -0.0713704617454726 0.5585168862776144 … 0.009898941808310227 -0.0660808421309462; … ; -0.05700464383352525 0.009898941808310227 … 0.5748638399977538 -0.05562354278598693; 0.20770976930425203 -0.0660808421309462 … -0.05562354278598693 0.7866657564833134], y = [0, 0, 0, 0, 0, 1, 0, 0, 1, 0  …  0, 1, 0, 1, 0, 1, 0, 0, 0, 1], X = [1.0 0.0; 1.0 0.0; … ; 1.0 0.0; 1.0 0.0], family = Binomial{Float64}(n=1, p=0.5))




```julia
# The estimated variance components of the models are equal to
nullmodel.φ, nullmodel.τ
```




    (1.0, [0.5785829161713041])




```julia
# The estimated intercept and fixed effect for SEX are equal to
nullmodel.α
```




    2-element Vector{Float64}:
     -0.6914574895991064
     -0.05699758175634297




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


```julia

```
