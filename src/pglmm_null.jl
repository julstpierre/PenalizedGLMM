"""
    pglmm_null(nullformula, covfile, grmfile; kwargs...)
# Positional arguments 
- `nullformula::FormulaTerm`: formula for the null model.
- `covfile::AbstractString`: covariate file (csv) with one header line, including the phenotype.
- `grmfile::AbstractString`: GRM file name.
# Keyword arguments
- `covrowinds::Union{Nothing,AbstractVector{<:Integer}}`: sample indices for covariate file.
- `grminds::Union{Nothing,AbstractVector{<:Integer}}`: sample indices for GRM file.
- `family::UnivariateDistribution:` `Binomial()` (default)   
- `link::GLM.Link`: `LogitLink()` (default).
- `M::Union{Nothing, Vector{Any}}`: vector containing other similarity matrices if >1 random effect is included.
- `tol::Float64 = 1e-5 (default)`: tolerance for convergence of PQL estimates.
- `maxiter::Integer = 500 (default)`: maximum number of iterations for AI-REML algorithm.
"""
function pglmm_null(
    # positional arguments
    nullformula::FormulaTerm,
    covfile::AbstractString,
    grmfile::AbstractString;
    # keyword arguments
    covrowinds::Union{Nothing,AbstractVector{<:Integer}} = nothing,
    grminds::Union{Nothing,AbstractVector{<:Integer}} = nothing,
    family::UnivariateDistribution = Binomial(),
    link::GLM.Link = LogitLink(),
    M::Union{Nothing, Vector{Any}} = nothing,
    tol::Float64 = 1e-5,
    maxiter::Integer = 500,
    kwargs...
    )

    #--------------------------------------------------------------
    # Read input files
    #--------------------------------------------------------------
    # read covariate file
    covdf = CSV.read(covfile, DataFrame)

    if !isnothing(covrowinds)
        covdf = covdf[covrowinds,:]
    end 

    # read grm file
    GRM = open(GzipDecompressorStream, grmfile, "r") do stream
        Symmetric(Matrix(CSV.read(stream, DataFrame)))
    end

    if !isnothing(grminds)
        GRM = GRM[grminds, grminds]
    end

    # Initialize number of subjects and genetic predictors
    n = size(GRM, 1)

    # Make sure grm is posdef
    xi = 1e-4
    while !isposdef(GRM)
        GRM = GRM + xi * Diagonal(ones(n))
        xi = 10 * xi
    end

    #--------------------------------------------------------------
    # Define link and variance functions
    #--------------------------------------------------------------
    # Define the derivative of link function g at the mean value ??
    if link == LogitLink()
        dg = function(??::Array{Float64}) 1 ./ (?? .* (1 .- ??)) end
    elseif link == IdentityLink()
        dg = function(??::Array{Float64}) 1 end
    end

     # Define the weights for link function g, variance function v(??) at the mean value ??
    if family == Binomial() && link == LogitLink()
        W = function(??::Array{Float64}, ??::Float64) Diagonal(?? .* (1 .- ??)) end
    elseif family == Normal() && link == IdentityLink()
        W = function(??::Array{Float64}, ??::Float64) 1/?? * Diagonal(ones(n)) end
    end

    #--------------------------------------------------------------
    # Estimation of variance components under H0
    #--------------------------------------------------------------
    # fit null GLM
    nullfit = glm(nullformula, covdf, family, link)
    
    # Define the design matrix
    X = modelmatrix(nullfit)

    # Obtain initial values for ??
    ??_0 = GLM.coef(nullfit)

    # Obtain initial values for Ytilde
    y = GLM.response(nullformula, covdf)
    ?? = GLM.predict(nullfit)
    ?? = GLM.linkfun.(link, ??)
    Ytilde = ?? + dg(??) .* (y - ??)

    # Number of variance components in the model
    if isnothing(M)
        K = 1
        V = push!(Any[], GRM)
    else
        K = 1 + size(M, 1)
        V = reverse(push!(M, GRM))
    end

    # For Normal family, dispersion parameter needs to be estimated
    if family == Normal() 
        K += 1
        V = reverse(push!(reverse(V), Diagonal(ones(n))))
    end 

    # Obtain initial values for variance components
    theta0 = fill(var(Ytilde) / K, K)

    # Initialize number of steps
    nsteps = 1

    # Iterate until convergence
    while true
        # Update variance components estimates
        fit0 = glmmfit_ai(family, theta0, V, X, Ytilde, W(??, first(theta0)), K)
        if nsteps == 1
            theta = theta0 + n^-1 * theta0.^2 .* fit0.S
        else
            theta = max.(theta0 + fit0.AI \ fit0.S, 10^-6 * var(Ytilde))
        end

        # Update working response
        fit = glmmfit_ai(family, theta, V, X, Ytilde, W(??, first(theta)), K, fit_only = true)
        ??, ?? = fit.??, fit.??
        ?? = GLM.linkinv.(link, ??)
        Ytilde = ?? + dg(??) .* (y - ??)

        # Check termination conditions
        if  2 * maximum(vcat(abs.(?? - ??_0) ./ (abs.(??) + abs.(??_0) .+ tol), abs.(theta - theta0) ./ (abs.(theta) + abs.(theta0) .+ tol))) < tol || nsteps >= maxiter 
            # Check if maximum number of iterations was reached
            converged = ifelse(nsteps < maxiter, true, false)

            # For binomial, set ?? = 1. Else, return first element of theta as ??
            if family == Binomial()
                ??, ?? = 1.0, theta
                ??V = sum(?? .* V)
            elseif family == Normal()
                ??, ?? = first(theta), theta[2:end]
                ??V = sum(?? .* V[2:end])
            end

            return(?? = ??, 
                   ?? = ??, 
                   ?? = ??, 
                   ?? = ??,
                   converged = converged, 
                   ??V = ??V,
                   y = y,
                   X = X,
                   family = family)
            break
        else
            theta0 = theta
            ??_0 = ??
            nsteps += 1
        end
    end

end

function glmmfit_ai(
    family::UnivariateDistribution,
    theta::Vector{Float64}, 
    V::Vector{Any},
    X::Matrix{Float64},
    Ytilde::Vector{Float64},
    W::Diagonal{Float64, Vector{Float64}},
    K::Integer;
    fit_only::Bool = false
    )

    # Define inverse of ??
    ??_inv = family == Normal() ? Symmetric(inv(cholesky(W^-1 + sum(theta[2:end] .* V[2:end])))) : Symmetric(inv(cholesky(W^-1 + sum(theta .* V))))
    X??_inv = X' * ??_inv
    X??_invX = Symmetric(X??_inv * X)
    cov = inv(cholesky(X??_invX))
    covX??_inv = cov * X??_inv

    ?? = covX??_inv * Ytilde
    PY = ??_inv * Ytilde - X??_inv' * ??
    ?? = Ytilde - W^-1 * PY

    if fit_only
        return(?? = ??, ?? = ??)
    else
        # Define the score of the restricted quasi-likelihood with respect to variance components
        VPY = [V[k] * PY for k in 1:K]
        PVPY = [??_inv * VPY[k] - X??_inv' * covX??_inv * VPY[k] for k in 1:K]
        S = [PY' * VPY[k] - sum(??_inv .* V[k]) - sum(X??_inv .* (covX??_inv * V[k])) for k in 1:K]

        # Define the average information matrix AI
        AI = Array{Float64}(undef, K, K)
        for k in 1:K
            for l in k:K
                AI[k, l] = VPY[k]' * PVPY[l]
            end
        end
        AI = Symmetric(AI)

        return(AI = AI, S = S, ?? = ??, ?? = ??)
    end
end