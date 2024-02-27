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
function pglm_null(
    # positional arguments
    nullformula::FormulaTerm,
    covdf::DataFrame;
    # keyword arguments
    covrowinds::Union{Nothing,AbstractVector{<:Integer}} = nothing,
    family::UnivariateDistribution = Binomial(),
    link::GLM.Link = LogitLink(),
    tol::T = 1e-5,
    maxiter::Integer = 500,
    idvar::Union{Nothing, Symbol, String} = nothing,
    reformula::Union{Nothing, FormulaTerm} = nothing,
    verbose::Bool = false,
    standardize_Z::Bool = false,
    standardize_X::Bool = true,
    kwargs...
    ) where T

    #--------------------------------------------------------------
    # Read input files
    #--------------------------------------------------------------
    # read covariate file
    n, m = size(covdf, 1), size(unique(covdf, idvar), 1)

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

    # Obtain initial values for α
    α_0 = GLM.coef(nullfit)

    # Obtain initial values for Ytilde
    y = GLM.response(nullformula, covdf)
    η = GLM.linkfun.(link, GLM.predict(nullfit))
    μ = updateμ(family, η, link)
    δ = dg(μ) .* (y - μ)
    Ytilde = η + δ

    # Obtain initial values for variance components
    theta0 = var(Ytilde)
    W = compute_weights(family, μ, theta0)
    
    #--------------------------------------------------------------
    # Longitudinal data
    #--------------------------------------------------------------
    Z, D = nothing, nothing
    if !isnothing(idvar) && !isnothing(reformula)
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

    # Iterate until convergence
    while true

        # Estimate variance components and update working response
        # while true
            α, η, obj, theta, Γ_new = glmmfit_ai(family, theta0, Z, Γ, X, Ytilde, W, nsteps, α_0, prev_obj, η, δ)
            μ = updateμ(family, η, link)
            W = compute_weights(family, μ, first(theta))
            δ = dg(μ) .* (y - μ)
            Ytilde = η + δ
            converged = abs(obj - prev_obj) < 1e-7 * obj
            converged && break
            α_0 = α
            theta0 = theta
            Γ = Γ_new
            prev_obj = obj
            println("obj = $obj")
        # end

        psi, psi0 =[vech(Γ_new), vech(Γ)]
        
        # Check termination conditions
        Δ = maximum(vcat(abs.(α - α_0) ./ (abs.(α) + abs.(α_0) .+ tol), abs.(theta - theta0) ./ (abs.(theta) + abs.(theta0) .+ tol), abs.(psi - psi0) ./ (abs.(psi) + abs.(psi0) .+ tol)))
        if  2 * Δ < tol || nsteps >= maxiter
            
            # Check if maximum number of iterations was reached
            converged = ifelse(nsteps < maxiter, true, false)

            # For binomial, set φ = 1. Else, return first element of theta as φ
            if family == Binomial()
                φ = 1.0
            elseif family == Normal()
                φ = theta
            end

            return(φ = φ, 
                   α = α, 
                   η = η,
                   converged = converged,
                   y = y,
                   X = X,
                   family = family,
                   Γ = Γ_new)
            break
        else
            theta0 = theta
            α_0 = α
            Γ = Γ_new
            prev_obj = obj
            nsteps += 1
            println("nsteps = $nsteps; D = $(Γ * Γ''); θ = $(theta); Δ = $Δ; obj = $obj")
        end
    end

end

# AI-REML algorithm
function glmmfit_ai(
    family::UnivariateDistribution,
    theta::T, 
    Z::Vector{Matrix{T}},
    Γ::LowerTriangular{T, Matrix{T}},
    X::Matrix{T},
    Ytilde::Vector{T},
    W::Diagonal{T, Vector{T}},
    nsteps::Integer,
    Ytilde_0::Vector{T},
    prev_obj::T,
    η::Vector{T},
    δ::Vector{T};
    fit_only::Bool = false,
    tol = 1e-5
    ) where T

    # Define inverse of Σ
    m = length(Z)
    ZΓ = [Z[i] * Γ for i in 1:m]
    Σ = [ZΓ[i] * ZΓ[i]' for i in 1:m] |> x->BlockDiagonal(x) + W^-1
    XΣ_inv = X' / Σ
    XΣ_invX = Symmetric(XΣ_inv * X) |> x-> cholesky(x)
    covXΣ_inv = XΣ_invX \ XΣ_inv

    α = covXΣ_inv * Ytilde
    PY = Σ \ Ytilde - XΣ_inv' * α
    obj = logL(Γ, Z, W, PY, Ytilde)

    # If loss function did not decrease, take a half step to ensure convergence
    if obj > prev_obj
        s = 1.0
        # d = α - α_0
        while obj > prev_obj
            s /= 1.1
            s < 1e-12 && break
            # α = α_0 + s * d
            Ytilde = η + s * δ
            α = covXΣ_inv * Ytilde
            PY = Σ \ Ytilde - XΣ_inv' * α
            obj = logL(Γ, Z, W, PY, Ytilde)
        end
    end 

    # Compute eta
    η = Ytilde - W^-1 * PY
    μ = updateμ(family, η, link)
    W = compute_weights(family, μ, first(theta))

    if fit_only
        return(α = α, η = η)
    elseif family == Normal()
        # Define the score of the restricted quasi-likelihood with respect to variance components
        Σ_inv = [Z[i] * D * Z[i]' for i in 1:m] |> x->BlockDiagonal(x) + theta * Diagonal(ones(length(Ytilde))) |> x-> inv(x)
        VPY = PY
        PVPY = Σ_inv * VPY - XΣ_inv' * covXΣ_inv * VPY
        S = PY' * VPY - tr(Σ_inv) - sum(XΣ_inv .* covXΣ_inv)

        # Update variance components estimates
        if nsteps == 1
            n = length(Ytilde)
            theta = theta + n^-1 * theta^2 * S
        end
            
        # Define the average information matrix AI
        AI = VPY' * PVPY
        theta = max.(theta + AI \ S, 10^-6 * var(Ytilde))

        # Update inverse of Σ
        Σ_inv = [Z[i] * D * Z[i]' for i in 1:m] |> x->BlockDiagonal(x) + theta * Diagonal(ones(length(Ytilde))) |> x-> inv(x)
    end

    # # # Update D matrix (method 1)
    # ZΣ_inv = [Z[i]' / blocks(Σ)[i] for i in 1:m]
    # ZΣ_invZ = [ZΣ_inv[i] * Z[i] for i in 1:m] |> x-> sum(x)
    # DZ = [D * Z[i]' for i in 1:m]
    # a = BlockDiagonal(DZ) * PY
    # a_ = [a[((i-1)*size(Z[1], 2)+1):i*size(Z[1], 2)] for i in 1:m]
    # aat = [a_[i] * a_[i]' for i in 1:m] |> x-> sum(x)

    # M = cholesky(Symmetric(ZΣ_invZ))
    # D_new = inv(M.U) * sqrt(M.U * aat * M.L) * inv(M.L)
    # Γ_new = cholesky(Symmetric(D_new)).L

    # # Update D matrix (method 2)
    # DZ = [D * Z[i]' for i in 1:m]
    # a = BlockDiagonal(DZ) * PY
    # a_ = [a[((i-1)*size(Z[1], 2)+1):i*size(Z[1], 2)] for i in 1:m]
    # aat = [a_[i] * a_[i]' for i in 1:m]
    # ZW = BlockDiagonal(Z)'W
    # D_new = mean([inv(blocks(ZW)[i] * Z[i] + inv(D)) + aat[i] for i in 1:m])
    # Γ_new_ = cholesky(Symmetric(D_new)).L

    # Update D matrix (method 3 gradient descent)
    # Γ_new = Γ - ivech(inv(hessian(Γ, Z, W, PY)) * vech(grad(Γ, Z, W, PY)))
    Γ_new = Γ - grad(Γ, Z, W, PY)

    # If loss function did not decrease, take a half step to ensure convergence
    prev_obj = obj
    obj = logL(Γ_new, Z, W, PY, Ytilde)
    if obj > prev_obj
        s = 1.0
        d = Γ_new - Γ
        while obj > prev_obj
            s /= 2
            s < 1e-12 && break
            Γ_new = Γ + s * d
            obj = logL(Γ_new, Z, W, PY, Ytilde)
        end
    end

    return(α = α, η = η, obj = obj, theta = theta, Γ_new = Γ_new)
end

# Write function to return half-vectorization operator
function vech(A::LowerTriangular{T, Matrix{T}}) where T
    A[tril(trues(size(A)))] 
end

# Write function to return matrix based on inverse of half-vectorization operator
function ivech(b::Vector{T}) where T
    r = Int((-1 + sqrt(1 + 8 * length(b))) / 2)
    A = zeros(r, r)
    A[tril(trues(size(A)))] = b
    LowerTriangular(A)
end

# # Write function to calculate gradient of negative log-likelihood wr to Γ
# function grad(Γ::LowerTriangular{T, Matrix{T}}, Z::Vector{Matrix{T}}, W::Diagonal{T, Vector{T}}, PY::Vector{T}) where T
#     m, r = length(Z), size(Z[1], 2)
#     ZΓ = [Z[i] * Γ for i in 1:m]
#     Σ = [ZΓ[i] * ZΓ[i]' for i in 1:m] |> x->BlockDiagonal(x) + W^-1
#     ZΣ_inv = [Z[i]' / blocks(Σ)[i] for i in 1:m]
#     ZΣ_invZ = [ZΣ_inv[i] * Z[i] for i in 1:m] |> x-> Symmetric(sum(x))
#     ZPY = BlockDiagonal(Z)'PY
#     ZPY_ = [ZPY[((i-1)*r+1):i*r] for i in 1:m]
#     ZPYZPYt = [ZPY_[i] * ZPY_[i]' for i in 1:m] |> x-> sum(x) |> x-> Symmetric(x)

#     ∇Γ = zero(Γ)
#     for k in findall(tril(trues(size(Γ))))
#         A = zeros(r, r)
#         A[k] = 1
#         ∇Γ[k] = -0.5 * tr((ZPYZPYt - ZΣ_invZ) * (A * Γ' + Γ * A'))
#     end
#     return ∇Γ
# end

# Write function to calculate gradient of -2 negative log-likelihood wr to Γ
function grad(Γ::LowerTriangular{T, Matrix{T}}, Z::Vector{Matrix{T}}, W::Diagonal{T, Vector{T}}, PY::Vector{T}) where T
    m, r = length(Z), size(Z[1], 2)
    ZΓ = [Z[i] * Γ for i in 1:m]
    Σ = [ZΓ[i] * ZΓ[i]' for i in 1:m] |> x->BlockDiagonal(x) + W^-1
    Σ_inv = inv(Σ)

    ∇Γ = zero(Γ)
    for k in findall(tril(trues(size(Γ))))
        dΓ = zeros(r, r)
        dΓ[k] = 1
        dΣ = BlockDiagonal([Z[i] * (dΓ * Γ' + Γ * dΓ') * Z[i]' for i in 1:m])
        ∇Γ[k] = tr(Σ_inv * dΣ) - PY'dΣ * PY
    end
    return ∇Γ
end

# Write function to calculate hessian of -2 negative log-likelihood wr to Γ
function hessian(Γ::LowerTriangular{T, Matrix{T}}, Z::Vector{Matrix{T}}, W::Diagonal{T, Vector{T}}, PY::Vector{T}) where T
    m, r = length(Z), size(Z[1], 2)
    ZΓ = [Z[i] * Γ for i in 1:m]
    Σ = [ZΓ[i] * ZΓ[i]' for i in 1:m] |> x->BlockDiagonal(x) + W^-1
    Σ_inv = inv(Σ)

    idxs = findall(tril(trues(size(Γ))))
    H = zeros(length(idxs), length(idxs))
    for i in 1:length(idxs)
        s1,r1 = Tuple(idxs[i])
        dΓs1r1 = zeros(r, r)
        dΓs1r1[idxs[i]] = 1
        dΣs1r1 = BlockDiagonal([Z[i] * (dΓs1r1 * Γ' + Γ * dΓs1r1') * Z[i]' for i in 1:m])
        Σ_invdΣs1r1 = Σ_inv * dΣs1r1

        for j in i:length(idxs)
            s2,r2 = Tuple(idxs[j])
            dΓs2r2  = zeros(r, r)
            dΓs2r2[idxs[j]]  = 1
            if i == j
                dΣs2r2 = dΣs1r1
                Σ_invdΣs2r2 = Σ_invdΣs1r1
            else
                dΣs2r2 = BlockDiagonal([Z[i] * (dΓs2r2 * Γ' + Γ * dΓs2r2') * Z[i]' for i in 1:m])
                Σ_invdΣs2r2 = Σ_inv * dΣs2r2
            end

            H[i,j] = -tr(Σ_invdΣs1r1 * Σ_invdΣs2r2) + 2 * PY'dΣs1r1 * Σ_invdΣs2r2 * PY

            if r1 == r2
                d2Σ = BlockDiagonal([Z[i] * (dΓs1r1 * dΓs2r2' + dΓs2r2 * dΓs1r1') * Z[i]' for i in 1:m])   
                H[i,j] += tr(Σ_inv * d2Σ) - PY'd2Σ * PY
            end
        end
    end
    return Symmetric(H)
end

# Write function for -2 negative log-likelihood
function logL(Γ::LowerTriangular{T, Matrix{T}}, Z::Vector{Matrix{T}}, W::Diagonal{T, Vector{T}}, PY::Vector{T}, Ytilde::Vector{T}) where T
    m = length(Z)
    ZΓ = BlockDiagonal([Z[i] * Γ for i in 1:m])

    if det(Σ) < 0
        Inf
    else
        logdet(ZΓ'W * ZΓ + Diagonal(ones(size(ZΓ, 2)))) + PY'Ytilde
    end 
end
