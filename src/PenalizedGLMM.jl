module PenalizedGLMM

using GLM
using LinearAlgebra, SparseArrays
using CSV, CodecZlib, DataFrames, Distributions
using SnpArrays
using GLMNet

export pglmm_null
export pglmm

include("src/pglmm_null.jl")
include("src/pglmm.jl")

end