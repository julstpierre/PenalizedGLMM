module PenalizedGLMM

using GLM
using LinearAlgebra
using CSV, CodecZlib, DataFrames, Distributions
using SnpArrays

export pglmm
export glmm_fit 

include("src/pglmm.jl")
include("src/pglmm_fit.jl")
include("src/utils.jl")

end