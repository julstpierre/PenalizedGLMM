"""
    pglmm(nullformula, covfile, geneticfile; kwargs...)
    pglmm(nullformula, df, geneticfile; kwargs...)
# Positional arguments 
- `nullformula::FormulaTerm`: formula for the null model.
- `covfile::AbstractString`: covariate file (csv) with one header line, including the phenotype.  
- `df::DataFrame`: DataFrame containing response and regressors for null model.
- `plinkfile::AbstractString`: PLINK file name containing genetic information,
    without the .bed, .fam, or .bim extensions. Moreover, bed, bim, and fam file with 
    the same `geneticfile` prefix need to exist.
- `grmfile::AbstractString`: GRM file name.
# Keyword arguments
- `covrowinds::Union{Nothing,AbstractVector{<:Integer}}`: sample indices for covariate file.   
- `link::GLM.Link`: `LogitLink()` (default).
- `snpmodel`: `ADDITIVE_MODEL` (default), `DOMINANT_MODEL`, or `RECESSIVE_MODEL`.
- `snpinds::Union{Nothing,AbstractVector{<:Integer}}`: SNP indices for bed/vcf file.
- `geneticrowinds::Union{Nothing,AbstractVector{<:Integer}}`: sample indices for bed/vcf file.
"""
function pglmm(
    # positional arguments
    nullformula::FormulaTerm,
    covfile::AbstractString,
    plinkfile::AbstractString,
    grmfile::AbstractString;
    # keyword arguments
    family::UnivariateDistribution = Binomial(),
    link::GLM.Link = LogitLink(),
    kwargs...
    )

    # read covariate file
    covdf = CSV.read(covfile, DataFrame)

    # read PLINK files
    geno = SnpArray(plinkfile * ".bed")

    # read grm file
    grm = open(GzipDecompressorStream, grmfile, "r") do stream
        CSV.read(stream, DataFrame)
    end

    # fit null GLM
    nullfit = glm(nullformula, covdf, family, link)
end