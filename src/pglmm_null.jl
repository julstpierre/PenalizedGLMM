"""
    pglmm_null(nullformula, covfile, geneticfile; kwargs...)
# Positional arguments 
- `nullformula::FormulaTerm`: formula for the null model.
- `covfile::AbstractString`: covariate file (csv) with one header line, including the phenotype.
- `grmfile::AbstractString`: GRM file name.
# Keyword arguments
- `covrowinds::Union{Nothing,AbstractVector{<:Integer}}`: sample indices for covariate file.
- `family::UnivariateDistribution:` `Binomial()` (default)   
- `link::GLM.Link`: `LogitLink()` (default).
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
    # Define the derivative of link function g at the mean value μ
    if link == LogitLink()
        dg = function(μ::Array{Float64}) 1 ./ (μ .* (1 .- μ)) end
    elseif link == IdentityLink()
        dg = function(μ::Array{Float64}) 1 end
    end

     # Define the weights for link function g, variance function v(μ) at the mean value μ
    if family == Binomial() && link == LogitLink()
        W = function(μ::Array{Float64}, φ::Float64) Diagonal(μ .* (1 .- μ)) end
    elseif family == Normal() && link == IdentityLink()
        W = function(μ::Array{Float64}, φ::Float64) 1/φ * Diagonal(ones(n)) end
    end

    #--------------------------------------------------------------
    # Estimation of variance components under H0
    #--------------------------------------------------------------
    # fit null GLM
    nullfit = glm(nullformula, covdf, family, link)
    
    # Define the design matrix
    X = modelmatrix(nullfit)

    # Obtain initial values for α
    α_0 = GLM.coef(nullfit)

    # Obtain initial values for Ytilde
    y = GLM.response(nullformula, covdf)
    μ = GLM.predict(nullfit)
    η = GLM.linkfun.(link, μ)
    Ytilde = η + dg(μ) .* (y - μ)

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
        K = K + 1
        V = reverse(push!(reverse(V), Diagonal(ones(n))))
    end

    # Obtain initial values for variance components
    theta0 = fill(var(Ytilde) / K, K)

    # Initialize number of steps
    nsteps = 1

    # Iterate until convergence
    while true
        # Update variance components estimates
        fit0 = glmmfit_ai(theta0, V, X, Ytilde, W(μ, first(theta0)), K)
        if nsteps == 1
            theta = theta0 + n^-1 * theta0.^2 .* fit0.S
        else
            theta = max.(theta0 + fit0.AI \ fit0.S, 10^-6 * var(Ytilde))
        end

        # Update working response
        fit = glmmfit_ai(theta, V, X, Ytilde, W(μ, first(theta)), K, fit_only = "TRUE")
        α, η = fit.α, fit.η
        μ = GLM.linkinv.(link, η)
        Ytilde = η + dg(μ) .* (y - μ)

        # Check termination conditions
        if  2 * maximum(vcat(abs.(α - α_0) ./ (abs.(α) + abs.(α_0) .+ tol), abs.(theta - theta0) ./ (abs.(theta) + abs.(theta0) .+ tol))) < tol || nsteps >= maxiter 
            # Check if maximum number of iterations was reached
            converged = ifelse(nsteps < maxiter, true, false)

            # For binomial, set φ = 1. Else, return first element of theta as φ
            if family == Binomial()
                φ, τ = 1.0, theta
                τV = sum(τ .* V)
            elseif family == Normal()
                φ, τ = first(theta), deleteat!(theta, 1)
                τV = sum(τ .* deleteat!(V, 1))
            end

            return(φ = φ, 
                   τ = τ, 
                   α = α, 
                   η = η,
                   converged = converged, 
                   τV = τV,
                   y = y,
                   X = X,
                   family = family)
            break
        else
            theta0 = theta
            α_0 = α
            nsteps += 1
        end
    end

end

function glmmfit_ai(
    theta::Vector{Float64}, 
    V::Vector{Any},
    X::Matrix{Float64},
    Ytilde::Vector{Float64},
    W::Diagonal{Float64, Vector{Float64}},
    K::Integer;
    fit_only = "FALSE"
    )
    # Define inverse of Σ
    Σ_inv = Symmetric(inv(cholesky(W^-1 + sum(theta .* V))))
    XΣ_inv = X' * Σ_inv
    XΣ_invX = Symmetric(XΣ_inv * X)
    cov = inv(cholesky(XΣ_invX))
    covXΣ_inv = cov * XΣ_inv

    α = covXΣ_inv * Ytilde
    PY = Σ_inv * Ytilde - XΣ_inv' * α
    η = Ytilde - W^-1 * PY

    if fit_only == "TRUE"
        return(α = α, η = η)
    else
        # Define the score of the restricted quasi-likelihood with respect to variance components
        VPY = [V[k] * PY for k in 1:K]
        PVPY = [Σ_inv * VPY[k] - XΣ_inv' * covXΣ_inv * VPY[k] for k in 1:K]
        S = [PY' * VPY[k] - sum(Σ_inv .* V[k]) - sum(XΣ_inv .* (covXΣ_inv * V[k])) for k in 1:K]

        # Define the average information matrix AI
        AI = Array{Float64}(undef, K, K)
        for k in 1:K
            for l in k:K
                AI[k, l] = VPY[k]' * PVPY[l]
            end
        end
        AI = Symmetric(AI)

        return(AI = AI, S = S, α = α, η = η)
    end
end