"""
    pglmm_cv(nullmode, plinkfile; kwargs...)
# Positional arguments 
- `nullmodel`: null model obtained by fitting pglmm_null.
- `plinkfile::AbstractString`: PLINK file name containing genetic information,
    without the .bed, .fam, or .bim extensions. Moreover, bed, bim, and fam file with 
    the same `geneticfile` prefix need to exist.
# Keyword arguments
- `snpfile::Union{Nothing, AbstractString}`: TXT file name containing genetic data if not in PLINK format.
- `snpmodel`: `ADDITIVE_MODEL` (default), `DOMINANT_MODEL`, or `RECESSIVE_MODEL`.
- `snpinds::Union{Nothing,AbstractVector{<:Integer}}`: SNP indices for bed/vcf file.
- `geneticrowinds::Union{Nothing,AbstractVector{<:Integer}}`: sample indices for bed/vcf file.
- `irls_tol::Float64` = 1e-7 (default)`: tolerance for the irls loop.
- `irls_maxiter::Integer = 500 (default)`: maximum number of iterations for the irls loop.
- `K_::Union{Nothing, Integer} = nothing (default)`: stop the full lasso path search after K_th value of λ.
- `verbose::Bool = false (default)`: print number of irls iterations at each value of λ.
- `standardize_X::Bool = true (default)`: standardize non-genetic covariates. Coefficients are returned on original scale.
- `standardize_G::Bool = true (default)`: standardize genetic predictors. Coefficients are returned on original scale.
- `criterion`: criterion for coordinate descent convergence. Can be equal to `:coef` (default) or `:obj`.
- `earlystop::Bool = true (default)`: should full lasso path search stop earlier if deviance change is smaller than MIN_DEV_FRAC_DIFF or higher than MAX_DEV_FRAC ? 
- `method = cd (default)`: which method to use to estimate random effects vector. Can be equal to `:cd` (default) for coordinate descent or `:conjgrad` for conjuguate gradient descent.
- `nfolds = 10 (default)`: number of cross-validation folds - default is 10.
- `foldid::Union{Nothing, AbstractVector{<:Integer}}`: an optional vector of values between 1 and nfold identifying what fold each observation is in. If supplied, nfolds can be missing.
- `type_measure`: loss to use for cross-validation. Can be equal to `:deviance` (default), which uses deviance for logistic regression, or `:auc` for AUC.
"""
function pglmm_cv(
    # positional arguments
    nullformula::FormulaTerm,
    covfile::AbstractString,
    grmfile::AbstractString,
    plinkfile::Union{Nothing, AbstractString} = nothing;
    # keyword arguments
    snpfile::Union{Nothing, AbstractString} = nothing,
    snpmodel = ADDITIVE_MODEL,
    snpinds::Union{Nothing,AbstractVector{<:Integer}} = nothing,
    covrowinds::Union{Nothing,AbstractVector{<:Integer}} = nothing,
    grminds::Union{Nothing,AbstractVector{<:Integer}} = nothing,
    geneticrowinds::Union{Nothing,AbstractVector{<:Integer}} = nothing,
    family::UnivariateDistribution = Binomial(),
    link::GLM.Link = LogitLink(),
    GEIvar::Union{Nothing,AbstractString} = nothing,
    GEIkin::Bool = true,
    M::Union{Nothing, Vector{Any}} = nothing,
    tol::T = 1e-5,
    maxiter::Integer = 500,
    irls_tol::T = 1e-7,
    irls_maxiter::Integer = 500,
    K::Integer = 100,
    rho::Union{Real, AbstractVector{<:Real}} = 0.5,
    verbose::Bool = false,
    standardize_X::Bool = true,
    standardize_G::Bool = true,
    criterion = :coef,
    earlystop::Bool = false,
    method = :cd,
    nfolds::Integer = 5,
    foldid::Union{Nothing, AbstractVector{<:Integer}} = nothing,
    type_measure = :auc,
    kwargs...
    ) where T

    # Split into nfolds
    foldid = isnothing(foldid) ? shuffle!((1:size(nullmodel.X, 1)) .% nfolds) .+ 1 : foldid

    # Fit null model separately for each fold
    nullmodel = [pglmm_null(
        nullformula,
        covfile,
        grmfile,
        covrowinds = covrowinds[foldid .!= i],
        grminds = grminds[foldid .!= i],
        family = family,
        link = link,
        GEIvar = GEIvar,
        GEIkin = GEIkin,
        M = M,
        tol = tol,
        maxiter = maxiter
        )  for i in 1:nfolds]

    # Fit model separately for each fold
    modelfit = [pglmm(
        nullmodel[i], 
        plinkfile,
        snpfile = snpfile,
        snpmodel = snpmodel,
        snpinds = snpinds,
        geneticrowinds = geneticrowinds[foldid .!= i],
        irls_tol = irls_tol,
        irls_maxiter = irls_maxiter,
        K = K,
        rho = rho,
        verbose = verbose,
        standardize_X = standardize_X,
        standardize_G = standardize_G,
        criterion = criterion,
        earlystop = earlystop,
        method = method
        ) for i in 1:nfolds]

    # Make predictions for each fold
    yhat = [PenalizedGLMM.predict(
        modelfit[i],
        covfile,
        grmfile,
        plinkfile,
        snpfile = snpfile,
        snpmodel = snpmodel,
        snpinds = snpinds,
        covrowinds = covrowinds[foldid .== i],
        covrowtraininds = covrowinds[foldid .!= i],
        covars = coefnames(apply_schema(nullformula, schema(nullformula, covdf)).rhs), 
        geneticrowinds = geneticrowinds[foldid .== i],
        grmrowinds = grminds[foldid .== i],
        grmcolinds = grminds[foldid .!= i],
        M = M,
        GEIvar = GEIvar,
        GEIkin = GEIkin,
        outtype = :prob
        ) for i in 1:nfolds]

    # Compute AUC for each fold
    covdf = !isnothing(covrowinds) ? CSV.read(covfile, DataFrame)[covrowinds, :] : CSV.read(covfile, DataFrame)
    ctrls = [(covdf[foldid .== i, coefnames(apply_schema(nullformula, schema(nullformula, covdf)).lhs)] .== 0) for i in 1:nfolds]
    cases = [(covdf[foldid .== i, coefnames(apply_schema(nullformula, schema(nullformula, covdf)).lhs)] .== 1) for i in 1:nfolds]

    auc_means = [ROCAnalysis.auc(roc(yhat[i][ctrls[i], j], yhat[i][cases[i], j])) for i in 1:nfolds, j in 1:size(yhat[i], 2)] |> 
                    x->mean(x, dims = 1) |>
                    x->argmax(x)
end