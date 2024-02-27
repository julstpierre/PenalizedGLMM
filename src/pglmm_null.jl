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
- `tau::Union{Nothing, Vector{T}} = nothing (default)`: Fix the value(s) for variance component(s).
- `method = :AIREML (default)`: Method to estimate variance components. Alternatively, one can use :AIML for ML estimation.
"""
function pglmm_null(
    # positional arguments
    nullformula::FormulaTerm,
    covfile::AbstractString;
    # keyword arguments
    grmfile::Union{Nothing, AbstractString} = nothing,
    GRM::Union{Nothing, Matrix{T}, BlockDiagonal{T, Matrix{T}}} = nothing,
    covrowinds::Union{Nothing,AbstractVector{<:Integer}} = nothing,
    grminds::Union{Nothing,AbstractVector{<:Integer}} = nothing,
    family::UnivariateDistribution = Binomial(),
    link::GLM.Link = LogitLink(),
    GEIvar::Union{Nothing,AbstractString} = nothing,
    GEIkin::Bool = true,
    M::Union{Nothing, Vector{Any}} = nothing,
    tol::T = 1e-5,
    maxiter::Integer = 1000,
    tau::Union{Nothing, Vector{T}} = nothing,
    method = :AIREML,
    idvar::Union{Nothing, Symbol, String} = nothing,
    reformula::Union{Nothing, FormulaTerm} = nothing,
    verbose::Bool = false,
    standardize_Z::Bool = false,
    standardize_X::Bool = false,
    kwargs...
    ) where T

    #--------------------------------------------------------------
    # Read input files
    #--------------------------------------------------------------
    # read covariate file
    covdf = !isnothing(covrowinds) ? CSV.read(covfile, DataFrame)[covrowinds,:] : CSV.read(covfile, DataFrame)
    n = size(covdf, 1)

    # read grm file
    if !isnothing(grmfile)
        GRM = open(GzipDecompressorStream, grmfile, "r") do stream
            Symmetric(Matrix(CSV.read(stream, DataFrame)))
        end
    end

    # Reorder GRM
    if !isnothing(grminds)
        GRM = GRM[grminds, grminds] |> x->Symmetric(x)
    end

    # Make sure grm is posdef
    xi = 1e-4
    while !isposdef(Matrix(GRM))
        GRM = GRM + xi * Diagonal(ones(size(GRM, 1)))
        xi = 2 * xi
    end
    m = size(GRM, 1)

    #--------------------------------------------------------------
    # Define link and variance functions
    #--------------------------------------------------------------
    # Define the derivative of link function g at the mean value μ
    if link == LogitLink()
        dg = function(μ::Array{Float64}) 
            [1 / (μ[i] * (1 - μ[i])) for i in 1:length(μ)] 
        end
    elseif link == IdentityLink()
        dg = function(μ::Array{Float64}) 1 end
    end

    # Function to update linear predictor and mean at each iteration
    PMIN = 1e-5
    PMAX = 1-1e-5
    function updateμ(::Binomial, η::Vector{T}, link::GLM.Link) where T
        μ = GLM.linkinv.(link, η)
        μ = [μ[i] < PMIN ? PMIN : μ[i] > PMAX ? PMAX : μ[i] for i in 1:length(μ)]
        return(μ)
    end

    function updateμ(::Normal, η::Vector{T}, link::GLM.Link) where T
        μ = GLM.linkinv.(link, η)
        return(μ)
    end

    function compute_weights(::Binomial, μ::Vector{T}, kwargs...) where T
        Diagonal(μ .* (1 .- μ))
    end

    function compute_weights(::Normal, μ::Vector{T}, φ::Float64) where T
        1/φ * Diagonal(ones(length(μ)))
    end

    #--------------------------------------------------------------
    # Estimation of variance components under H0
    #--------------------------------------------------------------
    # fit null GLM
    nullfit = glm(nullformula, covdf, family, link)
    
    # Define the design matrix
    X = modelmatrix(nullfit)
    if standardize_X
            X = all(X[:, 1] .== 1) ? hcat(ones(n), X[:, 2:end] ./ sqrt.(diag(cov(X[:, 2:end])))') : X ./ sqrt.(diag(cov(X)))'
    end
    X_nodup = !isnothing(idvar) ? Matrix(unique(DataFrame([covdf[:, idvar] X], :auto), 1)[:, 2:end]) : X
    
    # Obtain initial values for α
    α_0 = GLM.coef(nullfit)

    # Obtain initial values for Ytilde
    y = GLM.response(nullformula, covdf)
    η = GLM.linkfun.(link, GLM.predict(nullfit))
    μ = updateμ(family, η, link)
    Ytilde = η + dg(μ) .* (y - μ)

    # Create list of similarity matrices
    V = push!(Any[], GRM)

    # Add GEI similarity matrix
    ind_E = nothing
    if !isnothing(GEIvar)
        ind_E = findall(coefnames(nullfit) .== GEIvar)
        if GEIkin
            E = vec(X_nodup[:, ind_E])
            V_E = E * E'
            for j in findall(x -> x == 0, E), i in findall(x -> x == 0, E)  
                    V_E[i, j] = 1 
            end
            GRM_E = sparse(Matrix(GRM) .* V_E)
            
            if GRM isa BlockDiagonal
                # Convert GRM_D into BD matrix, keeping zeros within each block
                GRM_E[(V_E .== 0) .&& (Matrix(GRM) .!= 0)] .= 1
                GRM_E[(V_E .== 0) .&& (Matrix(GRM) .!= 0)] .= 0
                push!(V, BlockDiagonal(GRM_E))
            else
                push!(V, GRM_E)
            end
    
        end
    end

    # Add variance components in the model
    if !isnothing(M) 
        [push!(V, M[i]) for i in 1:length(M)] 
    end

    # For Normal family, dispersion parameter needs to be estimated
    if family == Normal() pushfirst!(V, Diagonal(ones(n))) end 

    # Obtain initial values for variance components
    K = length(V)
    if !isnothing(tau)
        @assert K == length(tau) "The number of variance components in tau must be equal to the number of kinship matrices."
        theta0 = theta = tau
    else
        theta0 = fill(var(Ytilde) / K, K)
    end
    W = compute_weights(family, μ, first(theta0))
    
    #--------------------------------------------------------------
    # Longitudinal data
    #--------------------------------------------------------------
    L = nothing
    if !isnothing(idvar)
        # Create L matrix assuming covdf is sorted by repeated ids
        L = [ones(sum(covdf[:, idvar] .== unique(covdf[:, idvar])[i]), 1) for i in 1:m] |> x-> BlockDiagonal(x)
    end
    
    Z, D = nothing, nothing
    if !isnothing(reformula)
        z = modelmatrix(glm(reformula, covdf, family, link))

        # Initialize longitudinal variance components
        D = Matrix(Diagonal(ones(size(z, 2))))
        Γ = cholesky(D).L

        # Standardize z
        if standardize_Z
            z = all(z[:, 1] .== 1) ? hcat(ones(n), z[:, 2:end] ./ sqrt.(diag(cov(z[:, 2:end])))') : z ./ sqrt.(diag(cov(z)))'
        end
        Z = [z[covdf[:, idvar] .== unique(covdf[:, idvar])[i], :] for i in 1:m]
    end

    #--------------------------------------------------------------
    # Estimation of variance components
    #--------------------------------------------------------------
    # Initialize number of steps
    nsteps = 1
    prev_obj = Inf

    if isnothing(tau)
            fit = glmmfit_ai(family, theta0, V, Z, L, D, X, Ytilde, W, K)
            theta0 = theta0 + n^-1 * theta0.^2 .* fit.S
            theta0 = max.(theta0 + fit.AI \ fit.S, 10^-6 * var(Ytilde))
    end

    # Iterate until convergence
    while true

        # Update variance components estimates
        if isnothing(tau)
            fit = glmmfit_ai(family, theta0, V, Z, L, D, X, Ytilde, W, K)
            theta = max.(theta0 + fit.AI \ fit.S, 10^-6 * var(Ytilde))
        end

        # Update D
        D_new = !isnothing(D) ? glmmfit_ai(family, theta, V, Z, L, D, X, Ytilde, W, K, update_D = true) : nothing
        psi, psi0 = !isnothing(D) ? [vec(D_new), vec(D)] : [0,0]

        # Update working response
        fit = glmmfit_ai(family, theta, V, Z, L, D, X, Ytilde, W, K, fit_only = true)
        α, η = fit.α, fit.η
        μ = updateμ(family, η, link)
        W = compute_weights(family, μ, first(theta))
        Ytilde = η + dg(μ) .* (y - μ)
        
        # Check termination conditions
        Δ = maximum(vcat(abs.(α - α_0) ./ (abs.(α) + abs.(α_0) .+ tol), abs.(theta - theta0) ./ (abs.(theta) + abs.(theta0) .+ tol), abs.(psi - psi0) ./ (abs.(psi) + abs.(psi0) .+ tol)))
        if  2 * Δ < tol || nsteps >= maxiter
            
            # Check if maximum number of iterations was reached
            converged = ifelse(nsteps < maxiter, true, false)

            # For binomial, set φ = 1. Else, return first element of theta as φ
            if family == Binomial()
                φ, τ = 1.0, theta
            elseif family == Normal()
                φ, τ = first(theta), theta[2:end]
            end

            return(φ = φ, 
                   τ = τ, 
                   α = α, 
                   η = η,
                   converged = converged,
                   V = V,
                   y = y,
                   X = X,
                   ind_E = ind_E,
                   family = family,
                   D = D_new)
            break
        else
            theta0 = theta
            α_0 = α
            D = D_new
            nsteps += 1
            verbose && family == Normal() && println("nsteps = $nsteps; D = $D; φ = $(first(theta)); τ = $(theta[2:end]); Δ = $Δ")
            verbose && family == Binomial() && println("nsteps = $nsteps; D = $D; τ = $theta; Δ = $Δ")
        
        end
    end

end

# AI-REML algorithm
function glmmfit_ai(
    family::UnivariateDistribution,
    theta::Vector{T}, 
    V::Vector{Any},
    Z::Nothing,
    L::Nothing,
    D::Nothing,
    X::Matrix{T},
    Ytilde::Vector{T},
    W::Diagonal{T, Vector{T}},
    K::Integer;
    fit_only::Bool = false
    ) where T

    # Define inverse of Σ
    Σ = family == Normal() ? W^-1 + sum(theta[2:end] .* V[2:end]) : W^-1 + sum(theta .* V)
    if !(Σ isa BlockDiagonal) Σ = cholesky(Σ) end
    XΣ_inv = X' / Σ
    XΣ_invX = Symmetric(XΣ_inv * X) |> x-> cholesky(x)
    covXΣ_inv = XΣ_invX \ XΣ_inv

    α = covXΣ_inv * Ytilde
    PY = Σ \ Ytilde - XΣ_inv' * α
    η = Ytilde - W^-1 * PY

    if fit_only
        return(α = α, η = η)
    else
        # Define the score of the restricted quasi-likelihood with respect to variance components
        VPY = [V[k] * PY for k in 1:K]
        PVPY = [Σ \ VPY[k] - XΣ_inv' * covXΣ_inv * VPY[k] for k in 1:K]
        S = [PY' * VPY[k] - sum(Matrix(inv(Σ)) .* Matrix(V[k])) - sum(XΣ_inv .* (covXΣ_inv * V[k])) for k in 1:K]

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

# AI-REML for repeated measurements with random slopes
function glmmfit_ai(
    family::UnivariateDistribution,
    theta::Vector{T}, 
    V::Vector{Any},
    Z::Vector{Matrix{T}},
    L::BlockDiagonal{T, Matrix{T}},
    D::Matrix{T},
    X::Matrix{T},
    Ytilde::Vector{T},
    W::Diagonal{T, Vector{T}},
    K::Integer;
    fit_only::Bool = false,
    update_D::Bool = false
    ) where T

    # Calculate inverse of R and usefull quantities
    m = length(Z)
    R_inv = [Z[i] * D * Z[i]' for i in 1:m] |> x->BlockDiagonal(x) + W^-1 |> x-> inv(x)
    LR_inv = [sum(blocks(R_inv)[i], dims = 1) for i in 1:m] |> x-> BlockDiagonal(x)
    LR_invL = [sum(blocks(LR_inv)[i]) for i in 1:m] |> x-> Diagonal(x)
    LR_invX = LR_inv * X
    XR_inv = X' * R_inv 

    # Define inverse of Σ using matrix inversion lemma
    τV_inv = family == Normal() ? inv(sum(theta[2:end] .* V[2:end])) : inv(sum(theta .* V))
    Σ_L_inv = inv(LR_invL + τV_inv)
    XΣ_invX = XR_inv * X - LR_invX' * (Σ_L_inv * LR_invX)
    XΣ_invYtilde = XR_inv * Ytilde - LR_invX' * (Σ_L_inv * (LR_inv * Ytilde))
    
    # Estimate α
    α = XΣ_invX \ XΣ_invYtilde
    r = Ytilde - X * α
    PY = R_inv * r - LR_inv' * (Σ_L_inv * (LR_inv * r))
    η = Ytilde - W^-1 * PY

    if fit_only
        return(α = α, η = η)
    elseif update_D
        # Update the covariance matrix D
        DZ = [D * Z[i]' for i in 1:m] |> x->BlockDiagonal(x)
        a = DZ * PY
        a_ = [a[((i-1)*size(Z[1], 2)+1):i*size(Z[1], 2)] for i in 1:m]
        aat = [a_[i] * a_[i]' for i in 1:m] |> x-> sum(x)

        ZR_invZ = [Z[i]' * blocks(R_inv)[i] * Z[i] for i in 1:m] |> x-> sum(x)
        LR_invZ = [blocks(LR_inv)[i] * Z[i] for i in 1:m] |> x-> reduce(vcat, x)

        M = ZR_invZ - LR_invZ' * Σ_L_inv * LR_invZ |> x-> cholesky(Symmetric(x))
        D = inv(M.U) * sqrt(M.U * aat * M.L) * inv(M.L)

        return(D = D)
    else
        # Update variance components using AI-(RE)ML algorithm
        LΣ_invL = LR_invL - LR_invL * Σ_L_inv * LR_invL
        LΣ_invX = LR_invX - LR_invL * Σ_L_inv * LR_invX
        LPL = LΣ_invL - LΣ_invX * (XΣ_invX \ LΣ_invX')
        LPY = L'PY
        VLPY = family == Normal() ? pushfirst!([V[k] * LPY for k in 2:K], PY) : [V[k] * LPY for k in 1:K]
        
        # Define the score of the restricted quasi-likelihood with respect to variance components
        if family == Normal()

            R_inv2 = R_inv*R_inv
            LR_inv2 = [sum(blocks(R_inv2)[i], dims = 1) for i in 1:m] |> x-> BlockDiagonal(x)
            LR_inv2L = [sum(blocks(LR_inv2)[i]) for i in 1:m] |> x-> Diagonal(x)

            S = [LPY' * V[k] * LPY - tr(LΣ_invL * V[k]) for k in 2:K]
            pushfirst!(S, PY'PY - tr(R_inv) + tr(LR_inv2L * Σ_L_inv))

            # Compute Σ_inv and P
            # Σ_inv = R_inv - LR_inv' * Σ_L_inv * LR_inv
            # XΣ_inv = X'Σ_inv
            # P = Σ_inv - XΣ_inv' * (XΣ_invX \ XΣ_inv)
            # PL = P * L

            # Approximate P by R_inv
            P = R_inv
            PL = BlockDiagonal([sum(blocks(P)[i], dims=2) for i in 1:m])
            LPL_ = pushfirst!(repeat([PL], K-1), P)
        else
            S = [LPY' * V[k] * LPY - tr(LΣ_invL * V[k]) for k in 1:K]
            LPL_ = repeat([LPL], K)
        end

        # Define the average information matrix AI
        AI = Array{T}(undef, K, K)
        for k in 1:K
            for l in k:K
                if k == 1
                    AI[k, l] = VLPY[k]' * LPL_[l] * VLPY[l]
                else
                    AI[k, l] = VLPY[k]' * LPL * VLPY[l]
                end
            end
        end
        AI = Symmetric(AI)
        
        return(AI = AI, S = S, α = α, η = η)
    end
end