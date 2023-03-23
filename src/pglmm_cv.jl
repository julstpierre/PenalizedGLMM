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
- `upper_bound::Bool = false (default)`: For logistic regression, should an upper-bound be used on the Hessian ?
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
    nlambda::Integer = 100,
    rho::Union{Real, AbstractVector{<:Real}} = 0.5,
    verbose::Bool = false,
    standardize_X::Bool = true,
    standardize_G::Bool = true,
    criterion = :coef,
    earlystop::Bool = false,
    method = :cd,
    upper_bound::Bool = false,
    tau::Union{Nothing, Vector{T}} = nothing,
    nfolds::Integer = 5,
    foldid::Union{Nothing, AbstractVector{<:Integer}} = nothing,
    type_measure = :deviance,
    nthreads = Threads.nthreads(),
    kwargs...
    ) where T
    
    # Fit null model using all observations
    nullmodel_full = pglmm_null(
        nullformula,
        covfile,
        grmfile,
        covrowinds = covrowinds,
        grminds = grminds,
        family = family,
        link = link,
        GEIvar = GEIvar,
        GEIkin = GEIkin,
        M = M,
        tol = tol,
        maxiter = maxiter
        )

    # Fit lasso model using all observations
    modelfit_full = pglmm(
        nullmodel_full, 
        plinkfile,
        snpfile = snpfile,
        snpmodel = snpmodel,
        snpinds = snpinds,
        geneticrowinds = geneticrowinds,
        irls_tol = irls_tol,
        irls_maxiter = irls_maxiter,
        nlambda = nlambda,
        rho = rho,
        verbose = verbose,
        standardize_X = standardize_X,
        standardize_G = standardize_G,
        criterion = criterion,
        earlystop = earlystop,
        method = method,
        upper_bound = upper_bound,
        tau = tau
        )

    # Read covariate file
    covdf = CSV.read(covfile, DataFrame)

    # Check if covrowinds is missing
    if isnothing(covrowinds) covrowinds = 1:nrow(covdf) end
    covdf = covdf[covrowinds, :]

    # Check if grminds is missing
    if isnothing(grminds) grminds = 1:nrow(covdf) end

    # Check if geneticrowinds is missing
    if isnothing(geneticrowinds) geneticrowinds = 1:nrow(covdf) end

    # Split observations into nfolds with the same case:control ratio in each fold if applicable
    if isnothing(foldid)
        if family != Binomial()
            foldid = shuffle!((1:nrow(covdf)) .% nfolds) .+ 1    
        else
            foldid = Vector{Int}(undef, nrow(covdf))
            ctrls = findall(GLM.response(nullformula, covdf) .== 0)
            cases = findall(GLM.response(nullformula, covdf) .== 1)
            for x in (ctrls, cases) foldid[x] = shuffle!((1:length(x)) .% nfolds) .+ 1 end
        end
    end

    # Fit null model separately for each fold
    if nthreads == 1
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
    else
        nullmodel = Vector{}(undef, nfolds)
        Threads.@threads for i = 1:nfolds
            nullmodel[i] = pglmm_null(
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
            )
       end
    end

    # Fit model separately for each fold
    if nthreads == 1
        modelfit = [pglmm(
                nullmodel[i], 
                plinkfile,
                snpfile = snpfile,
                snpmodel = snpmodel,
                snpinds = snpinds,
                geneticrowinds = geneticrowinds[foldid .!= i],
                irls_tol = irls_tol,
                irls_maxiter = irls_maxiter,
                nlambda = nlambda,
                lambda = [modelfit_full[i].lambda for i in 1:length(rho)],
                rho = rho,
                verbose = verbose,
                standardize_X = standardize_X,
                standardize_G = standardize_G,
                criterion = criterion,
                earlystop = earlystop,
                method = method,
                upper_bound = upper_bound,
                tau = tau
                ) for i in 1:nfolds]
    else
        modelfit = Vector{}(undef, nfolds)
        Threads.@threads for i = 1:nfolds
            modelfit[i] = pglmm(
                nullmodel[i], 
                plinkfile,
                snpfile = snpfile,
                snpmodel = snpmodel,
                snpinds = snpinds,
                geneticrowinds = geneticrowinds[foldid .!= i],
                irls_tol = irls_tol,
                irls_maxiter = irls_maxiter,
                nlambda = nlambda,
                lambda = [modelfit_full[i].lambda for i in 1:length(rho)],
                rho = rho,
                verbose = verbose,
                standardize_X = standardize_X,
                standardize_G = standardize_G,
                criterion = criterion,
                earlystop = earlystop,
                method = method,
                upper_bound = upper_bound,
                tau = tau
            )
       end
    end

    # Make predictions for each fold
    if nthreads == 1
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
    else 
        yhat = Vector{}(undef, nfolds)
        Threads.@threads for i = 1:nfolds
            yhat[i] = PenalizedGLMM.predict(
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
            )
        end
    end

    if type_measure == :deviance
        # Read GRM for test subjects
        GRM = [open(GzipDecompressorStream, grmfile, "r") do stream
            Symmetric(Matrix(CSV.read(stream, DataFrame)))[grminds[foldid .== i], grminds[foldid .== i]]
        end for i in 1:nfolds]

        # Create list of similarity matrices
        V = [push!(Any[], GRM[i]) for i in 1:nfolds]

        # Add GEI similarity matrix
        if !isnothing(GEIvar)
            D = [covdf[foldid .== i, GEIvar] for i in 1:nfolds]
            if GEIkin
                V_D = [D[i] * D[i]' for i in 1:nfolds]
                for k in 1:nfolds, j in findall(x -> x == 0, D[k]), i in findall(x -> x == 0, D[k])  
                        V_D[k][i, j] = 1 
                end
                [push!(V[i], sparse(GRM[i] .* V_D[i])) for i in 1:nfolds]
            end
        end

        # Covariance matrix
        τV = [sum(nullmodel[i].τ .* V[i]) for i in 1:nfolds]

        # Predict random effects for each fold
        b = [PenalizedGLMM.predict(
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
            outtype = :random
            ) for i in 1:nfolds]

        # Compute deviance for each fold
        meanloss = [model_dev(family, b[i][:, j], τV[i], covdf[foldid .== i, coefnames(apply_schema(nullformula, schema(nullformula, covdf)).lhs)], yhat[i][:, j]) for i in 1:nfolds, j in 1:size(yhat[1], 2)] |> 
                        x -> vec(mean(x, dims = 1))

        j = ceil(Int,  argmin(meanloss) / nlambda)
        jj = argmin(meanloss[((j-1)*nlambda+1):(j*nlambda)])
    elseif type_measure == :auc
        # Compute AUC for each fold
        ctrls = [(covdf[foldid .== i, coefnames(apply_schema(nullformula, schema(nullformula, covdf)).lhs)] .== 0) for i in 1:nfolds]
        cases = [(covdf[foldid .== i, coefnames(apply_schema(nullformula, schema(nullformula, covdf)).lhs)] .== 1) for i in 1:nfolds]

        meanloss = [ROCAnalysis.auc(roc(yhat[i][ctrls[i], j], yhat[i][cases[i], j])) for i in 1:nfolds, j in 1:size(yhat[1], 2)] |> 
                        x  -> vec(mean(x, dims = 1))

        j = ceil(Int,  argmax(meanloss)/ nlambda)
        jj = argmax(meanloss[((j-1)*nlambda+1):(j*nlambda)])
    end

    # Return lasso path and optimal values of rho and lambda
    return(path = modelfit_full, rho = TuningParms(rho[j], j), lambda = TuningParms(modelfit_full[j].lambda[jj], jj), meanloss = meanloss)

end