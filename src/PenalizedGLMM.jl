__precompile__()

module PenalizedGLMM

using GLM
using CSV, CodecZlib, DataFrames, Distributions
using SnpArrays

export pglmm

include("src/pglmm.jl")

end