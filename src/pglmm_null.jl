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
- `tol::Float64 = 1e-5 (default)`: tolerance for convergence of PQL estimates.
- `maxiter::Integer = 500 (default)`: maximum number of iterations for AI-REML algorithm.
- `method = :AIREML (default)`: Method to estimate variance components. Alternatively, one can use :AIML for ML estimation.
"""
function pglmm_null(
    # positional arguments
    nullformula::FormulaTerm;
    # keyword arguments
    covfile::Union{DataFrame, AbstractString} = nothing,
    covrowinds::Union{Nothing,AbstractVector{<:Integer}} = nothing,
    grmfile::Union{Nothing, AbstractString} = nothing,
    GRM::Union{Nothing, Matrix{T}, BlockDiagonal{T, Matrix{T}}} = nothing,
    grminds::Union{Nothing,AbstractVector{<:Integer}} = nothing,
    family::UnivariateDistribution = Binomial(),
    link::GLM.Link = LogitLink(),
    GEIvar::Union{Nothing,AbstractString} = nothing,
    GEIkin::Bool = false,
    tol::T = 1e-5,
    maxiter::Integer = 50,
    method::Symbol  = :REML,
    idvar::Union{Nothing, Symbol, String} = nothing,
    reformula::Union{Nothing, FormulaTerm} = nothing,
    verbose::Bool = false,
    standardize_Z::Bool = false,
    kwargs...
    ) where T

    #--------------------------------------------------------------
    # Read input files
    #--------------------------------------------------------------
    # read covariate file
    covdf = isa(covfile, AbstractString) ? CSV.read(covfile, DataFrame) : isa(covfile, DataFrame) ? covfile : error("covfile is not a DataFrame of AbstractString")
    covdf = !isnothing(covrowinds) ? covdf[covrowinds,:] : covdf
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

    @assert n == m || !isnothing(idvar) "The number of individuals in the GRM and covariate file must be equal. You must either use the covrowinds or grminds keyword arguments. In case where individuals have repeated observations, you must use the idvar keyword argument."

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
                # Convert GRM_E into BD matrix, keeping zeros within each block
                GRM_E[(V_E .== 0) .&& (Matrix(GRM) .!= 0)] .= 1
                GRM_E[(V_E .== 0) .&& (Matrix(GRM) .!= 0)] .= 0
                push!(V, BlockDiagonal(GRM_E))
            else
                push!(V, GRM_E)
            end
    
        end
    end

    # For Normal family, dispersion parameter needs to be estimated
    IsNormal = family == Normal()
    if IsNormal pushfirst!(V, Diagonal(ones(n))) end 
    
    #--------------------------------------------------------------
    # Longitudinal data
    #--------------------------------------------------------------hg
    if !isnothing(idvar)
        # Create L matrix assuming covdf is sorted by repeated ids
        L = [ones(sum(covdf[:, idvar] .== unique(covdf[:, idvar])[i]), 1) for i in 1:m] |> x-> BlockDiagonal(x)
    else
        L = Diagonal(ones(m))
    end
    
    r, Z, diagidx, H = 0, nothing, [], L
    if !isnothing(reformula)

        @assert !isnothing(idvar) "idvar is missing"

        # Random effect formula
        z = modelmatrix(glm(reformula, covdf, family, link))
        if standardize_Z
            z = all(z[:, 1] .== 1) ? hcat(ones(n), z[:, 2:end] ./ sqrt.(diag(cov(z[:, 2:end])))') : z ./ sqrt.(diag(cov(z)))'
        end

        # Create BlockDiagonal matrix for each random effect
        r, Z = size(z,2), Any[]
        diagidx = [1, r+1, 2*r, 3*r-2, 4*r-5, 5*r-9][1:r]
        for j in 1:r
            push!(Z, [reshape(z[covdf[:, idvar] .== unique(covdf[:, idvar])[i], j], :, 1) for i in 1:m] |> x->BlockDiagonal(x))
        end
        H = sparse(hcat(Matrix(L), reduce(hcat, Matrix.(Z))))

        # Create relatedness matrices
        for j in 1:r
            for k in j:r
                if j == k
                    push!(V, BlockDiagonal(blocks(Z[j]) .* blocks(Z[j]')))
                else
                    push!(V, BlockDiagonal(blocks(Z[j]) .* blocks(Z[k]') + blocks(Z[k]) .* blocks(Z[j]')))
                end
            end
        end
    end

    #--------------------------------------------------------------
    # Estimation of variance components
    #--------------------------------------------------------------
    # Create indices for variance and covariance parameters
    K, q = length(V), Int(length(V) - r * (r + 1) / 2)
    idxtau = reduce(vcat, [1:q, diagidx .+ q])
    idxcovtau = setdiff(1:K, idxtau)
    covariance_idx = zeros(Int, length(idxcovtau) , 3)
    covariance_idx[:,1] = idxcovtau
    ii = 1
    for i in 1:length(idxtau[q+1:end])-1
        for j in (i+1):length(idxtau[q+1:end])
            covariance_idx[ii,:] = [idxcovtau[ii], idxtau[q+1:end][i], idxtau[q+1:end][j]] 
            ii += 1
        end
    end

    # Obtain initial values for variance components
    theta0 = fill(var(Ytilde) / K, K)
    theta0[idxcovtau] .= 0
    theta0 = theta0 ./ mean.(diag.(V))
    W = compute_weights(family, μ, first(theta0))

    # Initialize covariance matrix
    D = Matrix(Diagonal(theta0[idxtau[q+1:end]]))
    ii = 1
    for i in 1:r-1
        for j in i+1:r
            D[i,j] = theta0[idxcovtau[ii]]
            ii += 1
        end
    end
    D = Symmetric(D)

    # Initialize number of steps and perform EM step to estimate theta
    nsteps = 1
    fit = glmmfit_ai(family, theta0, V, D, Z, L, H, X, Ytilde, W, K, method = method)
    theta0 = max.(theta0 + n^-1 * theta0.^2 .* fit.S, 0)

    # Iterate until convergence
    while true

        # Update variance components estimates
        fit = glmmfit_ai(family, theta0, V, D, Z, L, H, X, Ytilde, W, K, method = method)
        dtheta = fit.AI \ fit.S
        theta = theta0 + dtheta

        # Make sure variance parameters are positive
        if isempty(idxcovtau)
            theta[theta .< tol .&& theta0 .< tol] .= 0
            while any(theta .< 0)
                dtheta .= dtheta / 2
                theta .= theta0 .+ dtheta
                theta[theta .< tol .&& theta0 .< tol] .= 0
            end
            theta[theta .< tol] .= 0
        else
            fixrho_idx0 = abs.(theta0[covariance_idx[:, 1]]) .> (1 - 1.01 * tol) * sqrt.(theta0[covariance_idx[:, 2]] .* theta0[covariance_idx[:, 3]])
            theta[idxtau[theta[idxtau] .< tol .&& theta0[idxtau] .< tol]] .= 0

            fixrho_idx = @fastmath abs.(theta[covariance_idx[:, 1]]) .> (1 - 1.01 * tol) * sqrt.(theta[covariance_idx[:, 2]] .* theta[covariance_idx[:, 3]])
            theta[covariance_idx[fixrho_idx .&& fixrho_idx0, 1]] .= sign.(theta[covariance_idx[fixrho_idx .&& fixrho_idx0, 1]]) .* sqrt.(theta[covariance_idx[fixrho_idx .&& fixrho_idx0, 2]] .* theta[covariance_idx[fixrho_idx .&& fixrho_idx0, 3]])

            while any(theta[idxtau] .< 0) || any(@fastmath abs.(theta[covariance_idx[:, 1]]) .> sqrt.(theta[covariance_idx[:, 2]] .* theta[covariance_idx[:, 3]]))
                dtheta = dtheta / 2
                theta = theta0 + dtheta
                theta[idxtau[theta[idxtau] .< tol .&& theta0[idxtau] .< tol]] .= 0

                fixrho_idx = @fastmath [abs(theta[covariance_idx[i, 1]]) > (1 - 1.01 * tol) * sqrt(theta[covariance_idx[i, 2]] * theta[covariance_idx[i, 3]]) for i in 1:length(idxcovtau)]
                theta[covariance_idx[fixrho_idx .&& fixrho_idx0, 1]] .= sign.(theta[covariance_idx[fixrho_idx .&& fixrho_idx0, 1]]) .* sqrt.(theta[covariance_idx[fixrho_idx .&& fixrho_idx0, 2]] .* theta[covariance_idx[fixrho_idx .&& fixrho_idx0, 3]])
            end

            theta[idxtau[theta[idxtau] .< tol]] .= 0
            theta[covariance_idx[fixrho_idx, 1]] = sign.(theta[covariance_idx[fixrho_idx, 1]]) .* sqrt.(theta[covariance_idx[fixrho_idx, 2]] .* theta[covariance_idx[fixrho_idx, 3]])
        end

        # Update working response
        fit = glmmfit_ai(family, theta, V, D, Z, L, H, X, Ytilde, W, K, fit_only = true, method = method)
        α, η = fit.α, fit.η
        μ = updateμ(family, η, link)
        W = compute_weights(family, μ, first(theta))
        Ytilde = η + dg(μ) .* (y - μ)

        # Update covariance matrix
        D = Matrix(Diagonal(theta[idxtau[q+1:end]]))
        ii = 1
        for i in 1:r-1
            for j in i+1:r
                D[i,j] = theta[idxcovtau[ii]]
                ii += 1
            end
        end
        D = Symmetric(D)

        # Check termination conditions
        Δ = maximum(vcat(abs.(α - α_0) ./ (abs.(α) + abs.(α_0) .+ tol), abs.(theta - theta0) ./ (abs.(theta) + abs.(theta0) .+ tol)))
        if  2 * Δ < tol || nsteps >= maxiter
            
            # Check if maximum number of iterations was reached
            converged = ifelse(nsteps < maxiter, true, false)

            # For binomial, set φ = 1. Else, return first element of theta as φ
            if family == Binomial()
                φ, τ = 1.0, theta[1:q]
            elseif family == Normal()
                φ, τ = first(theta), theta[2:q]
            end

            # Define sum(tau_k * V_k)
            τV = sum(τ .* V[1+IsNormal:q])

            # Make sure τV and D are positive definite matrices
            xi = 10e-6
            while !isposdef(Matrix(τV))
                τV = τV + xi * Diagonal(ones(m))
                xi = 10 * xi
            end

            xi = 10e-6
            while !isposdef(D)
                D = D + xi * Diagonal(ones(r))
                xi = 10 * xi
            end

            # Return output
            return(φ = φ, 
                   τ = τ, 
                   α = α, 
                   η = η,
                   b = fit.b,
                   converged = converged,
                   τV = τV,
                   y = y,
                   X = X,
                   L = L,
                   H = H,
                   ind_E = ind_E,
                   D = D,
                   family = family,
                   nsteps = nsteps)
            break
        else
            theta0 = theta
            α_0 = α
            nsteps += 1
            verbose && family == Normal() && println("nsteps = $nsteps; φ = $(first(theta)); τ = $(theta[2:q]); D = $D; Δ = $Δ")
            verbose && family == Binomial() && println("nsteps = $nsteps; τ = $(theta[1:q]); D = $D; Δ = $Δ")
        
        end
    end

end

# AI-REML algorithm
function glmmfit_ai(
    family::UnivariateDistribution,
    theta::Vector{T}, 
    V::Vector{Any},
    D::Symmetric{T, Matrix{T}},
    Z::Nothing,
    L::Diagonal{T, Vector{T}},
    H::Diagonal{T, Vector{T}},
    X::Matrix{T},
    Ytilde::Vector{T},
    W::Diagonal{T, Vector{T}},
    K::Integer;
    fit_only::Bool = false,
    method::Symbol
    ) where T

    # Define inverse of Σ
    τV = family == Normal() ? sum(theta[2:end] .* V[2:end]) : sum(theta .* V)
    Σ = W^-1 + τV
    if !(Σ isa BlockDiagonal) Σ = cholesky(Σ) end
    XΣ_inv = X' / Σ
    XΣ_invX = Symmetric(XΣ_inv * X) |> x-> cholesky(x)
    covXΣ_inv = XΣ_invX \ XΣ_inv

    # Estimate α and b
    α = covXΣ_inv * Ytilde
    PY = Σ \ Ytilde - XΣ_inv' * α
    η = Ytilde - W^-1 * PY
    b = Matrix(τV) * PY

    if fit_only
        return(α = α, η = η, b = b)
    else
        # Define the score of the restricted quasi-likelihood with respect to variance components
        VPY = [V[k] * PY for k in 1:K]
        PVPY = [Σ \ VPY[k] - XΣ_inv' * covXΣ_inv * VPY[k] for k in 1:K]
        S = [PY' * VPY[k] - sum(Matrix(inv(Σ)) .* Matrix(V[k])) - sum(XΣ_inv .* (covXΣ_inv * V[k])) for k in 1:K]

        # Define the average information matrix AI
        AI = Array{T}(undef, K, K)
        for k in 1:K
            for l in k:K
                AI[k, l] = VPY[k]' * PVPY[l]
            end
        end
        AI = Symmetric(AI)

        return(AI = AI, S = S, α = α, η = η, b = b)
    end
end

# AI-REML algorithm for repeated measurements
function glmmfit_ai(
    family::UnivariateDistribution,
    theta::Vector{T}, 
    V::Vector{Any},
    D::Symmetric{T, Matrix{T}},
    Z::Union{Vector{Any}, Nothing},
    L::BlockDiagonal{T, Matrix{T}},
    H::AbstractMatrix,
    X::Matrix{T},
    Ytilde::Vector{T},
    W::Diagonal{T, Vector{T}},
    K::Integer;
    fit_only::Bool = false,
    method::Symbol
    ) where T

    # Calculate inverse of R and useful quantities
    m, r = size(L, 2), isnothing(Z) ? 0 : length(Z)
    q = Int(K - r * (r + 1) / 2)
    R_inv = !isnothing(Z) ? sum(theta[(q+1):end] .* V[(q+1):end]) + W^-1 |> x-> inv(x) : W
    LR_inv = !isnothing(Z) ? [sum(blocks(R_inv)[i], dims = 1) for i in 1:m] |> x-> BlockDiagonal(x) : L'R_inv
    LR_invL = [sum(blocks(LR_inv)[i]) for i in 1:m] |> x-> Diagonal(x)
    LR_invX = LR_inv * X
    LR_invYtilde = LR_inv * Ytilde
    XR_inv = X' * R_inv 

    # Define inverse of Σ using matrix inversion lemma
    IsNormal = family == Normal()
    τV = sum(theta[(1+IsNormal):q] .* V[(1+IsNormal):q])
    τV_inv = inv(τV)
    Σ_L = LR_invL + τV_inv
    XΣ_invX = XR_inv * X - LR_invX' * (Σ_L \ LR_invX)
    XΣ_invYtilde = XR_inv * Ytilde - LR_invX' * (Σ_L \ LR_invYtilde)
    
    # Estimate α
    α = XΣ_invX \ XΣ_invYtilde
    e = Ytilde - X * α
    PY = R_inv * e - LR_inv' * (Σ_L \ (LR_inv * e))
    η = Ytilde - W^-1 * PY

    # Estimate b
    HPY = H'PY
    b = BlockDiagonal([Matrix(τV), kron(D, Diagonal(ones(m)))]) * HPY

    if fit_only
        return(α = α, η = η, b = b)
    else
        # Define the derivative of V times PY
        dVPY = IsNormal ? [PY] : []
        for i in (1+IsNormal):q
            push!(dVPY, L * (V[i] * L'PY))
        end
        for i in (q+1):K
            push!(dVPY, V[i] * PY)
        end

        # Compute useful quantities to calculate trace of P for dispersion parameter only
        if IsNormal
            R_inv2 = R_inv^2
            LR_inv2 = !isnothing(Z) ? [sum(blocks(R_inv2)[i], dims = 1) for i in 1:m] |> x-> BlockDiagonal(x) : L'R_inv2
            LR_inv2L = [sum(blocks(LR_inv2)[i]) for i in 1:m] |> x-> Diagonal(x)
        end

        # Compute useful quantities to calculate the trace of Σ_inv * dV
        LΣ_invL = LR_invL - LR_invL * (Σ_L \ Matrix(LR_invL))
        if !isnothing(Z)
            ZR_inv = [BlockDiagonal(blocks(Z[i]') .* blocks(R_inv)) for i in 1:r]
            ZR_invZ = [j >= i ? BlockDiagonal(blocks(ZR_inv[i]) .* blocks(Z[j])) : 0 for i in 1:r, j in 1:r]
            LR_invZ = [Diagonal(sum.(blocks(ZR_inv[i]))) for i in 1:r]
            ZΣ_invZ = ZR_invZ - [j >= i ? LR_invZ[i] * (Σ_L \ Matrix(LR_invZ[j])) : 0 for i in 1:r, j in 1:r]
        end

        # Define the trace of Σ_inv * dV (for ML) or P * dV (for REML)
        trΣ_invdV = IsNormal ? [tr(R_inv) - tr(Matrix(LR_inv2L) / Σ_L)] : []

        if method == :ML
            for k in (1+IsNormal):q
                isa(V[k], BlockDiagonal) && push!(trΣ_invdV, tr(LΣ_invL * V[k]))
                isa(V[k], Matrix) && push!(trΣ_invdV, sum(LΣ_invL .* V[k]))
            end
            if !isnothing(Z)
                for i in 1:r, j in i:r
                    i == j ? push!(trΣ_invdV, tr(ZΣ_invZ[i, j])) : push!(trΣ_invdV, 2*tr(ZΣ_invZ[i, j]))
                end
            end
            # Define the score
            S = [PY'dVPY[i] - trΣ_invdV[i] for i in 1:K]

        elseif method == :REML
            # Compute useful quantities to calculate the trace of P * dV
            XΣ_inv = XR_inv - LR_invX' * (Σ_L \ Matrix(LR_inv))
            XΣ_invL = XΣ_inv * L 
            LPL = LΣ_invL - XΣ_invL' * (XΣ_invX \ XΣ_invL)
            if !isnothing(Z)
                ZR_invX = [ZR_inv[i] * X for i in 1:r]
                ZΣ_invX = ZR_invX - [LR_invZ[i]' * (Σ_L \ LR_invX) for i in 1:r]
                ZPZ = ZΣ_invZ - [j >= i ? ZΣ_invX[i] * (XΣ_invX \ ZΣ_invX[j]') : 0 for i in 1:r, j in 1:r]
            end

            trPdV = IsNormal ? [trΣ_invdV[1] - sum(XΣ_inv .* (XΣ_invX \ XΣ_inv))] : []
            for k in (1+IsNormal):q
                isa(V[k], BlockDiagonal) && push!(trPdV, tr(LPL * V[k]))
                isa(V[k], Matrix) && push!(trPdV, sum(LPL .* V[k]))
            end
            if !isnothing(Z)
                for i in 1:r, j in i:r
                    i == j ? push!(trPdV, tr(ZPZ[i, j])) : push!(trPdV, 2*tr(ZPZ[i, j]))
                end
            end
            # Define the score
            S = [PY'dVPY[i] - trPdV[i] for i in 1:K]
        end

        # Calculate the average information matrix
        dVPYR_invdVPY = [j >= i ? dVPY[i]' * R_inv * dVPY[j] : 0 for i in 1:K, j in 1:K]
        LR_invdVPY = [LR_inv * dVPY[i] for i in 1:K]
        dVPYΣ_invdVPY = dVPYR_invdVPY - [j >= i ? LR_invdVPY[i]' * (Σ_L \ LR_invdVPY[j]) : 0 for i in 1:K, j in 1:K]

        dVPYR_invX = [(XR_inv * dVPY[i])' for i in 1:K]
        dVPYΣ_invX = [dVPYR_invX[i] - LR_invdVPY[i]' * (Σ_L \ LR_invX) for i in 1:K]
        dVPYPdVPY = dVPYΣ_invdVPY - [j >= i ? dVPYΣ_invX[i] * (XΣ_invX \ dVPYΣ_invX[j]') : 0 for i in 1:K, j in 1:K]

        AI = Symmetric(dVPYPdVPY)
        
        return(AI = AI, S = S, α = α, η = η, b = b)
    end
end
