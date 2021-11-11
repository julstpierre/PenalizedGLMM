"""
    pglmm(plinkfile; kwargs...)
# Positional arguments 
- `nullmodel`: Null model obtained by fitting pglmm_null.
- `plinkfile::AbstractString`: PLINK file name containing genetic information,
    without the .bed, .fam, or .bim extensions. Moreover, bed, bim, and fam file with 
    the same `geneticfile` prefix need to exist.
# Keyword arguments
- `snpmodel`: `ADDITIVE_MODEL` (default), `DOMINANT_MODEL`, or `RECESSIVE_MODEL`.
- `snpinds::Union{Nothing,AbstractVector{<:Integer}}`: SNP indices for bed/vcf file.
- `geneticrowinds::Union{Nothing,AbstractVector{<:Integer}}`: sample indices for bed/vcf file.
- `irwls_tol::Float64` = 1e-7 (default)`: tolerance for the IRWLS loop.
- `irwls_maxiter::Integer = 20 (default)`: maximum number of Newton iterations for the IRWLS loop.
"""
function pglmm(
    # positional arguments
    nullmodel,
    plinkfile::AbstractString;
    # keyword arguments
    snpmodel = ADDITIVE_MODEL,
    snpinds = nothing,
    geneticrowinds = nothing,
    irwls_tol::Float64 = 1e-7,
    irwls_maxiter::Integer = 20,
    kwargs...
    )

    # read PLINK files
    geno = SnpArray(plinkfile * ".bed")
    
    # Convert genotype file to matrix, convert to additive model (default), scale and impute
    if isnothing(snpinds) 
        snpinds = 1:size(geno, 2) 
    end
    if isnothing(geneticrowinds) 
        geneticrowinds = 1:size(geno, 1) 
    end
    G = convert(Matrix{Float64}, @view(geno[geneticrowinds, snpinds]), model = snpmodel, center = true, scale = true, impute = true)

    # Initialize number of subjects and genetic predictors
    n, p = size(G)
    @assert n == length(nullmodel.y) "Genotype matrix and y must have same number of rows"

    # Initialize number of non-genetic covariates (including intercept)
    k = size(nullmodel.X, 2)
    
    # Center and scale covariates
    X = (nullmodel.X .- mean(nullmodel.X, dims = 1)) ./ std(nullmodel.X, dims = 1)
    X = isnan(X[1, 1]) ? [ones(n) X[:, 2:end]] : X

    # Penalty factor
    p_f = [zeros(k); ones(p)]
    
    # Spectral decomposition of sum(τ * V)
    eigvals, U = eigen(nullmodel.τV)

    # Define (normalized) weights for each observation
    if nullmodel.family == Binomial()
        μ = GLM.linkinv.(LogitLink(), nullmodel.η)
        Ytilde = nullmodel.η + 4 * (nullmodel.y - μ)
        β = [nullmodel.α; zeros(p)]
        c = 4
    elseif nullmodel.family == Gaussian()
        Ytilde = nullmodel.y
        c = nullmodel.φ
    end
    w = (1 ./(c .+ eigvals))

    # Transform X and Y
    Xstar = U' * [X  G]
    Ystar = U' * Ytilde

    # Penalized model
    # For linear mixed model, we can call glmnet directly
    if nullmodel.family == Gaussian()
        path = glmnet(Xstar, Ystar; weights = w, penalty_factor = p_f, intercept = false)
    # For binomial logistic mixed model, we need to perform IRWLS
    elseif nullmodel.family == Binomial()
        path = pglmm_lasso(Xstar, Ystar, nullmodel.y, U; β = β, w = w, p_f = p_f, K_ = 20) 
    end 
end

function pglmm_lasso(
    Xstar::AbstractMatrix{Float64},
    Ystar::AbstractVector{Float64}, 
    y::AbstractVector{Int64},
    U::AbstractMatrix{Float64}; 
    β::AbstractVector{Float64},
    w::AbstractVector{Float64},
    p_f::AbstractVector{Float64},
    irwls_tol::Float64 = 1e-7,
    irwls_maxiter::Integer = 20,
    δ::Float64 = 0.001,
    K_::Union{Nothing, Integer} = nothing
    )

    # Define inverse of Σ_tilde
    UD_inv = U * Diagonal(w)
    UD_invUX = UD_inv * Xstar

    # Initialize null deviance
    μ = mean(y)
    μ_star = U' * (mean(U * Ystar) * ones(n))
    nulldev = -2 * sum(y .* log.(μ ./ (1 .- μ)) .+ log.(1 .- μ)) + sum(w .* (1 .- 4 * w) .* (Ystar - μ_star).^2)

    # Sequence of λ
    w_n = w / sum(w)
    λ_seq, K = lambda_seq(Ystar, Xstar; weights = w_n, penalty_factor = p_f)
    K = isnothing(K_) ? K : K_

    # Define weighted sum of squares
    wXstar = w_n .* Xstar
    Swxx = vec(sum(wXstar .* Xstar, dims = 1))

    # Initialize indices for the non-null β's         
    inds = findall(x -> x != 0, vec(β))

    # Initialize array to store output for each λ
    betas = sparse(Array{Float64}(undef, size(Xstar, 2), K))
    converged = Array{Bool}(undef, K)
    pct_dev = Array{Float64}(undef, K)

    # Loop through sequence of λ
    for i = 1:K

        # Current value of λ
        λ = λ_seq[i]

        # Initialize objective function
        loss = Inf

        # Penalized iterative weighted least squares (IWLS)
        for _ in 1:irwls_maxiter
   
            # Run coordinate descent inner loop to obtain β
            β_0 = β 
            β = glmnet(Xstar, Ystar; weights = w_n, penalty_factor = p_f, lambda = [λ], intercept = false).betas
            #r = Ystar - Xstar[:, inds] * β_0[inds]
            #β, inds = cd_lasso(r, Xstar, wXstar, Swxx; β_0 = β_0, p_f = p_f, λ = λ)

            # Update working response
            Ystar0 = Ystar
            Ystar, μ = wrkresp()  

            # Update deviance and loss function
            loss0 = loss
            dev = -2 * sum(y .* log.(μ ./ (1 .- μ)) .+ log.(1 .- μ)) + sum(w .* (1 .- 4 * w) .* (Ystar - Xstar[:,inds] * β[inds]).^2)
            loss = dev/2 + λ * sum(p_f[inds] .* abs.(β[inds]))
            
            # If loss function did not decrease, take a half step to ensure convergence
            s = 1.0
            d = β - β_0
            ∇f = -Xstar[:,inds]' * (w .* (Ystar0 - Xstar[:,inds] * β_0[inds]))
            Δ = ∇f' * d[inds] + λ * (sum(p_f[inds] .* abs.(β[inds])) - sum(p_f[inds] .* abs.(β_0[inds])))    
            while loss > loss0 + δ * s * Δ
                s /= 2
                β = β_0 + s * d
                Ystar, μ = wrkresp() 
                dev = -2 * sum(y .* log.(μ ./ (1 .- μ)) .+ log.(1 .- μ)) + sum(w .* (1 .- 4 * w) .* (Ystar - Xstar[:,inds] * β[inds]).^2)
                loss = dev/2 + λ * sum(p_f[inds] .* abs.(β[inds]))
            end

            # Check termination conditions
            converged[i] = abs(loss - loss0) < irwls_tol * loss
            converged[i] && break
        end

        # Store ouput from IRWLS loop
        betas[:, i] = β 
        pct_dev[i] = 1 - dev/nulldev
        @assert converged[i] "IRWLS failed to converge in $irwls_maxiter iterations at λ = $λ"
    end

    # Display summary output
    df = [length(findall(x -> x != 0, vec(betas[:,k]))) for k in 1:K]
    show(stdout, "text/plain", ["" "df" "pct_dev" "λ" "converged"; collect(1:K) df pct_dev[1:K] λ_seq[1:K] converged[1:K]])
    return(betas = betas)
end

# Function to compute sequence of values for λ
function lambda_seq(
    y::AbstractVector{Float64}, 
    X::AbstractMatrix{Float64}; 
    weights::AbstractVector{Float64},
    penalty_factor::AbstractVector{Float64},
    K::Integer = 100
    )

    # Define weighted outcome
    wY = weights .* y

    # Define matrix of penalised predictors
    X = (X .* penalty_factor')

    λ_min_ratio = (length(y) < size(X, 2) ? 1e-2 : 1e-4)
    λ_max = maximum(abs.((X' * wY)))
    λ_min = λ_max * λ_min_ratio
    λ_step = log(λ_min_ratio)/(K - 1)
    λ_seq = exp.(collect(log(λ_max):λ_step:log(λ_min)))
    
    return(λ_seq, length(λ_seq))
end

# Define Softtreshold function
function  softtreshold(z::AbstractFloat{Float64}, γ::AbstractFloat{Float64}) :: AbstractFloat{Float64}
    if z > γ
        z - γ
    elseif z < -γ
        z + γ
    else
        0
    end
end

# Function to update working response at each iteration
function wrkresp()
    η = U * Ystar0 - 4 * UD_inv * Ystar0 + 4 * UD_invUX[:,inds] * β[inds]
    μ = GLM.linkinv.(LogitLink(), η)
    Ystar = U' * vec(η + 4 * (y - μ)) 
    return(Ystar, μ)
end

# Function to perform coordinate descent with a lasso penalty
function cd_lasso(
    # positional arguments
    r::AbstractVector{Float64},
    X::AbstractMatrix{Float64},
    wX::AbstractMatrix{Float64}, 
    Swxx::AbstractVector{Float64};
    #keywords arguments
    β_0::CompressedPredictorMatrix,
    p_f::AbstractVector{Float64} = ones(size(X, 2)), 
    λ::AbstractFloat,
    cd_maxiter::Integer = 100000,
    cd_tol::Real=1e-7
    )

    converged = false
    maxΔ = zero(Float64)

    for cd_iter in 1:cd_maxiter
        # At firs iteration, perform one coordinate cycle and 
        # record active set of coefficients that are nonzero
        if cd_iter == 1 || converged
            inds = Int64[]

            for j in 1:size(X, 2)
                v = dot(wX[:, j], r)
                λj = λ * p_f[j]
                
                if β_0[j] != 0
                    v += β_0[j] * Swxx[j]
                    append!(inds, j)
                else
                    # Adding a new variable to the model
                    abs(v) < λj && continue
                    append!(inds, j)
                end
                β[j] = softtreshold(v, λj) / Swxx[j]
                r += X[:, j] * (β_0[j] - β[j])

                maxΔ = max(maxΔ, Swxx[j] * (β_0[j] - β[j])^2)
            end

            # Check termination condition at last iteration
            converged && maxΔ < cd_tol && break
        end

        # Cycle over coefficients in active set only until convergence
        sort!(inds)
        for j in inds
            β_0[j] = β[j]
            β[j] == 0 && continue
            
            v = dot(wX[:, j], r) + β_0[j] * Swxx[j]
            β[j] = softtreshold(v, λ * p_f[j]) / Swxx[j]
            r += X[:, j] * (β_0[j] - β[j])

            maxΔ = max(maxΔ, Swxx[j] * (β_0[j] - β[j])^2)
        end

        # Check termination condition before last iteration
        converged = maxΔ < cd_tol
    end

    # Assess convergence of coordinate descent
    @assert converged "Coordinate descent failed to converge in $cd_maxiter iterations at λ = $λ"

    return(β, inds)

end