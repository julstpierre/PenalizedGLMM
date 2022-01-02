# ========================================================================
# Code for estimating ancestry using ADMIXTURE.jl
# ========================================================================
# load package
using ADMIXTURE, CSV, DataFrames, SnpArrays, CodecZlib

# Estimate ancestry from K=3 populations
P, Q = admixture("UKBB.bed", 4)

# The output files of ADMIXTURE are available in the working directory.
Pt = CSV.read("UKBB.4.P", DataFrame, header = false) |> Matrix |> transpose
Qt = CSV.read("UKBB.4.Q", DataFrame, header = false) |> Matrix |> transpose

# Compute estimated kinship
UKBB = SnpArray("UKBB.bed")
GRM = SnpArrays.grm_admixture(UKBB, Pt, Qt) |> x -> 2*x


# Save GRM in compressed file
open(GzipCompressorStream, "grm.txt.gz", "w") do stream
    CSV.write(stream, DataFrame(GRM, :auto))
end