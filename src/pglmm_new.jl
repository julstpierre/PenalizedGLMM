"""
    pglmm(nullmode, plinkfile; kwargs...)
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
"""
function pglmm(
    # positional arguments
    nullmodel,
    plinkfile::Union{Nothing, AbstractString} = nothing;
    # keyword arguments
    snpfile::Union{Nothing, AbstractString} = nothing,
    snpmodel = ADDITIVE_MODEL,
    snpinds::Union{Nothing,AbstractVector{<:Integer}} = nothing,
    geneticrowinds::Union{Nothing,AbstractVector{<:Integer}} = nothing,
    irls_tol::Float64 = 1e-7,
    irls_maxiter::Integer = 500,
    K::Integer = 100,
    verbose::Bool = false,
    standardize_X::Bool = true,
    standardize_G::Bool = true,
    criterion = :coef,
    earlystop::Bool = true,
    method = :cd,
    kwargs...
    )

    ## keyword arguments
    # snpmodel = ADDITIVE_MODEL
    # snpinds = nothing
    # geneticrowinds = trainrowinds
    # irls_tol = 1e-7
    # irls_maxiter = 500
    # K = 100
    # verbose = true
    # standardize_X = true
    # standardize_G = true
    # criterion = :coef
    # earlystop= true
    # method = :cd 

    # Read genotype file
    if !isnothing(plinkfile)

	    # read PLINK files
	    geno = SnpArray(plinkfile * ".bed")
        snpinds_ = isnothing(snpinds) ? (1:size(geno, 2)) : snpinds 
        geneticrowinds_ = isnothing(geneticrowinds) ? (1:size(geno, 1)) : geneticrowinds
        
        # Read genotype and calculate mean and standard deviation
	    G = SnpLinAlg{Float64}(geno, model = snpmodel, impute = true, center = true, scale = standardize_G) |> x-> @view(x[geneticrowinds_, snpinds_])
        muG, sG = standardizeG(@view(geno[geneticrowinds_, snpinds_]), snpmodel, standardize_G)

    elseif !isnothing(snpfile)

        # read CSV file
        geno = CSV.read(snpfile, DataFrame)
        
        # Convert genotype file to matrix, convert to additive model (default) and impute
        snpinds_ = isnothing(snpinds) ? (1:size(geno, 2)) : snpinds 
        geneticrowinds_ = isnothing(geneticrowinds) ? (1:size(geno, 1)) : geneticrowinds
        G = convert.(Float64, Matrix(geno[geneticrowinds_, snpinds_]))

	    # standardize genetic predictors
    	G, muG, sG = standardizeX(G, standardize_G)
    end

    # Initialize number of subjects and predictors (including intercept)
    (n, p), k = size(G), size(nullmodel.X, 2)
    @assert n == length(nullmodel.y) "Genotype matrix and y must have same number of rows"

    # Spectral decomposition of sum(τ * V)
    eigvals, U = eigen(nullmodel.τV)
    eigvals .= 1 ./ eigvals
    UD_invUt = U * Diagonal(eigvals) * U'
   
    # Rotate random effects vector 
    δ = Array{Float64}(undef, n)
    b = nullmodel.η - nullmodel.X * nullmodel.α
    mul!(δ, U', b)

    # Initialize working variable
    y = nullmodel.y
    if nullmodel.family == Binomial()
        μ, ybar = GLM.linkinv.(LogitLink(), nullmodel.η), mean(y)
        w = μ .* (1 .- μ)
        Ytilde = nullmodel.η + (y - μ) ./ w
        nulldev = -2 * sum(y * log(ybar / (1 - ybar)) .+ log(1 - ybar))
    elseif nullmodel.family == Normal()
        Ytilde, μ = y, nullmodel.η
        w = 1 / nullmodel.φ
        nulldev = w * (y .- mean(y)).^2
    end

    # Initialize residuals and null deviance
    r = Ytilde - nullmodel.η

    # standardize non-genetic covariates
    intercept = all(nullmodel.X[:,1] .== 1)
    X, muX, sX = standardizeX(nullmodel.X, standardize_X, intercept)
    ind_D = !isnothing(nullmodel.ind_D) ? nullmodel.ind_D .- intercept : nothing
    D, muD, sD = !isnothing(ind_D) ? (vec(X[:, nullmodel.ind_D]), muX[ind_D], sX[ind_D]) : repeat([nothing], 3)

    # Initialize β, γ and penalty factors
    α, β, γ = sparse(zeros(k)), sparse(zeros(p)), sparse(zeros(p))
    p_fX = zeros(k); p_fG = ones(p)

    # Sequence of λ
    λ_seq = lambda_seq(y - μ, X, G, D; p_fX = p_fX, p_fG = p_fG)
    
    # Fit penalized model
    path = pglmm_fit(nullmodel.family, Ytilde, y, X, G, U, D, nulldev, r, μ, α, β, γ, δ, p_fX, p_fG, λ_seq, K, w, eigvals, verbose, criterion, earlystop, irls_tol, irls_maxiter, method)

    # Separate intercept from coefficients
    a0, alphas = intercept ? (path.alphas[1,:], path.alphas[2:end,:]) : (nothing, path.alphas)
    betas = path.betas
    gammas = !isnothing(D) ? path.gammas : nothing

    # Return coefficients on original scale
    if isnothing(gammas)
        if !isempty(sX) & !isempty(sG)
            lmul!(inv(Diagonal(sX)), alphas), lmul!(inv(Diagonal(sG)), betas)
        elseif !isempty(sX)
            lmul!(inv(Diagonal(sX)), alphas)
        elseif !isempty(sG)
            lmul!(inv(Diagonal(sG)), betas)
        end

        a0 .-= vec(muX' * alphas + muG' * betas)
    else
        if !isempty(sX) & !isempty(sG)
            lmul!(inv(Diagonal(sX)), alphas), lmul!(inv(Diagonal(sG)), betas), lmul!(inv(Diagonal(sD .* sG)), gammas)
        elseif !isempty(sX)
            lmul!(inv(Diagonal(sX)), alphas), lmul!(inv(Diagonal(sD .* ones(p))), gammas)
        elseif !isempty(sG)
            lmul!(inv(Diagonal(sG)), betas), lmul!(inv(Diagonal(sG)), gammas)
        end

        a0 .-= vec(muX' * alphas + muG' * betas - (muD .* muG)' * gammas)
        alphas[ind_D, :] -= muG' * gammas; betas -= muD' .* gammas
    end

    # Return lasso path
    pglmmPath(nullmodel.family, a0, alphas, betas, gammas, nulldev, path.pct_dev, path.λ, 0, path.fitted_values, y, UD_invUt, nullmodel.τ, intercept)
end

# Controls early stopping criteria with automatic λ
const MIN_DEV_FRAC_DIFF = 1e-5
const MAX_DEV_FRAC = 0.999

# Function to fit a penalized mixed model
function pglmm_fit(
    ::Binomial,
    Ytilde::Vector{T},
    y::Vector{Int64},
    X::Matrix{T},
    G::SubArray{T, 2, SnpLinAlg{T}, Tuple{Vector{Int64}, UnitRange{Int64}}},
    U::Matrix{T},
    D::Nothing,
    nulldev::T,
    r::Vector{T},
    μ::Vector{T},
    α::SparseVector{T},
    β::SparseVector{T},
    γ::SparseVector{T},
    δ::Vector{T},
    p_fX::Vector{T},
    p_fG::Vector{T},
    λ_seq::Vector{T},
    K::Int64,
    w::Vector{T},
    eigvals::Vector{T},
    verbose::Bool,
    criterion,
    earlystop::Bool,
    irls_tol::T,
    irls_maxiter::Int64,
    method
) where T

    # Initialize array to store output for each λ
    alphas = zeros(length(α), K)
    betas = zeros(length(β), K)
    pct_dev = zeros(T, K)
    dev_ratio = convert(T, NaN)
    fitted_means = zeros(length(y), K)

    # Loop through sequence of λ
    i = 0
    for _ = 1:K
        # Next iterate
        i += 1
        converged = false
        
        # Current value of λ
        λ = λ_seq[i]
        dλ = 2 * λ_seq[i] - λ_seq[max(1, i-1)]

        # Check strong rule
        compute_strongrule(dλ, p_fX, p_fG, α = α, β = β, X = X, G = G, y = y, μ = μ)

        # Initialize objective function and save previous deviance ratio
        dev = loss = convert(T, Inf)
        last_dev_ratio = dev_ratio

        # Iterative weighted least squares (IRLS)
        for irls in 1:irls_maxiter

            # Update random effects vector δ
            update_δ(Val(method); U = U, λ = λ, family = Binomial(), Ytilde = Ytilde, y = y, w = w, r = r, δ = δ, eigvals = eigvals, criterion = criterion, μ = μ)

            # Run coordinate descent inner loop to update β
            Swxx, Swgg = cd_lasso(X, G, λ; family = Binomial(), Ytilde = Ytilde, y = y, w = w, r = r, α = α, β = β, δ = δ, p_fX = p_fX, p_fG = p_fG, eigvals = eigvals, criterion = criterion)

            # Update μ and w
            μ, w = updateμ(r, Ytilde)
            
            # Update deviance and loss function
            prev_loss = loss
            dev = LogisticDeviance(δ, eigvals, y, μ)
            loss = dev/2 + last(λ) * P(α, β, p_fX, p_fG)
            
            # If loss function did not decrease, take a half step to ensure convergence
            if loss > prev_loss + length(μ)*eps(prev_loss)
                verbose && println("step-halving because loss=$loss > $prev_loss + $(length(μ)*eps(prev_loss)) = length(μ)*eps(prev_loss)")
                #= s = 1.0
                d = β - β_last
                while loss > prev_loss
                    s /= 2
                    β = β_last + s * d
                    μ, w = updateμ(r, Ytilde) 
                    dev = LogisticDeviance(δ, eigvals, y, μ)
                    loss = dev/2 + last(λ) * P(α, β, p_fX, p_fG)
                end =#
            end 

            # Update working response and residuals
            Ytilde, r = wrkresp(y, μ, w)

            # Check termination conditions
            converged = abs(loss - prev_loss) < irls_tol * loss
            
            # Check KKT conditions at last iteration
            if converged
                maxΔ, converged = cycle(X, G, λ, Val(true), r = r, α = α, β = β, Swxx = Swxx, Swgg = Swgg, w = w, p_fX = p_fX, p_fG = p_fG)
            end
            converged && verbose && println("Number of irls iterations = $irls at $i th value of λ.")
            converged && break  
        end
        @assert converged "IRLS failed to converge in $irls_maxiter iterations at λ = $λ"

        # Store ouput from irls loop
        alphas[:, i] = convert(Vector{Float64}, α)
        betas[:, i] = convert(Vector{Float64}, β)
        dev_ratio = dev/nulldev
        pct_dev[i] = 1 - dev_ratio
        fitted_means[:, i] = μ

        # Test whether we should continue
        earlystop && ((last_dev_ratio - dev_ratio < MIN_DEV_FRAC_DIFF) || pct_dev[i] > MAX_DEV_FRAC) && break
    end

    return(alphas = view(alphas, :, 1:i), betas = view(betas, :, 1:i), pct_dev = pct_dev[1:i], λ = λ_seq[1:i], fitted_values = view(fitted_means, :, 1:i))
end

# Function to fit a penalized mixed model
function pglmm_fit(
    ::Binomial,
    Ytilde::Vector{T},
    y::Vector{Int64},
    X::Matrix{T},
    G::SubArray{T, 2, SnpLinAlg{T}, Tuple{Vector{Int64}, UnitRange{Int64}}},
    U::Matrix{T},
    D::Vector{T},
    nulldev::T,
    r::Vector{T},
    μ::Vector{T},
    α::SparseVector{T},
    β::SparseVector{T},
    γ::SparseVector{T},
    δ::Vector{T},
    p_fX::Vector{T},
    p_fG::Vector{T},
    λ_seq::Vector{T},
    K::Int64,
    w::Vector{T},
    eigvals::Vector{T},
    verbose::Bool,
    criterion,
    earlystop::Bool,
    irls_tol::T,
    irls_maxiter::Int64,
    method
) where T

    # Initialize array to store output for each λ
    alphas = zeros(length(α), K)
    betas = zeros(length(β), K)
    gammas = zeros(length(γ), K)
    pct_dev = zeros(T, K)
    dev_ratio = convert(T, NaN)
    fitted_means = zeros(length(y), K)

    # Loop through sequence of λ
    i = 0
    for _ = 1:K
        # Next iterate
        i += 1
        converged = false
        
        # Current value of λ
        λ = λ_seq[i]
        dλ = 2 * λ_seq[i] - λ_seq[max(1, i-1)]

        # Check strong rule
        compute_strongrule(dλ, λ_seq[max(1, i-1)], p_fX, p_fG, D, α = α, β = β, γ = γ, X = X, G = G, y = y, μ = μ)

        # Initialize objective function and save previous deviance ratio
        dev = loss = convert(T, Inf)
        last_dev_ratio = dev_ratio

        # Iterative weighted least squares (IRLS)
        for irls in 1:irls_maxiter

            # Update random effects vector δ
            update_δ(Val(method); U = U, λ = λ, family = Binomial(), Ytilde = Ytilde, y = y, w = w, r = r, δ = δ, eigvals = eigvals, criterion = criterion, μ = μ)

            # Run coordinate descent inner loop to update β
            Swxx, Swgg, Swdg = cd_lasso(D, X, G, λ; family = Binomial(), Ytilde = Ytilde, y = y, w = w, r = r, α = α, β = β, δ = δ, γ = γ, p_fX = p_fX, p_fG = p_fG, eigvals = eigvals, criterion = criterion)

            # Update μ and w
            μ, w = updateμ(r, Ytilde)
            
            # Update deviance and loss function
            prev_loss = loss
            dev = LogisticDeviance(δ, eigvals, y, μ)
            loss = dev/2 + last(λ) * P(α, β, γ, p_fX, p_fG)
            
            # If loss function did not decrease, take a half step to ensure convergence
            if loss > prev_loss + length(μ)*eps(prev_loss)
                verbose && println("step-halving because loss=$loss > $prev_loss + $(length(μ)*eps(prev_loss)) = length(μ)*eps(prev_loss)")
                #= s = 1.0
                d = β - β_last
                while loss > prev_loss
                    s /= 2
                    β = β_last + s * d
                    μ, w = updateμ(r, Ytilde) 
                    dev = LogisticDeviance(δ, eigvals, y, μ)
                    loss = dev/2 + last(λ) * P(α, β, γ, p_fX, p_fG)
                end =#
            end 

            # Update working response and residuals
            Ytilde, r = wrkresp(y, μ, w)

            # Check termination conditions
            converged = abs(loss - prev_loss) < irls_tol * loss
            
            # Check KKT conditions at last iteration
            if converged
                maxΔ, converged = cycle(D, X, G, λ, Val(true), r = r, α = α, β = β, γ = γ, Swxx = Swxx, Swgg = Swgg, Swdg = Swdg, w = w, p_fX = p_fX, p_fG = p_fG)
            end
            converged && verbose && println("Number of irls iterations = $irls at $i th value of λ.")
            converged && break  
        end
        @assert converged "IRLS failed to converge in $irls_maxiter iterations at λ = $λ"

        # Store ouput from irls loop
        alphas[:, i] = convert(Vector{Float64}, α)
        betas[:, i] = convert(Vector{Float64}, β)
        gammas[:, i] = convert(Vector{Float64}, γ)
        dev_ratio = dev/nulldev
        pct_dev[i] = 1 - dev_ratio
        fitted_means[:, i] = μ

        # Test whether we should continue
        earlystop && ((last_dev_ratio - dev_ratio < MIN_DEV_FRAC_DIFF) || pct_dev[i] > MAX_DEV_FRAC) && break
    end

    return(alphas = view(alphas, :, 1:i), betas = view(betas, :, 1:i), gammas = view(gammas, :, 1:i), pct_dev = pct_dev[1:i], λ = λ_seq[1:i], fitted_values = view(fitted_means, :, 1:i))
end

# Function to perform coordinate descent with a lasso penalty
function cd_lasso(
    # positional arguments
    X::Matrix{T},
    G::SubArray{T, 2, SnpLinAlg{T}, Tuple{Vector{Int64}, UnitRange{Int64}}},
    λ::T;
    #keywords arguments
    family::UnivariateDistribution,
    r::Vector{T},
    α::SparseVector{T},
    β::SparseVector{T},
    δ::Vector{T},
    Ytilde::Vector{T},
    y::Vector{Int},
    w::Vector{T}, 
    eigvals::Vector{T}, 
    p_fX::Vector{T},
    p_fG::Vector{T},
    cd_maxiter::Integer = 10000,
    cd_tol::Real=1e-7,
    criterion,
    kwargs...
    ) where T

    converged = true
    loss = Inf

    # Compute sum of squares
    Swxx, Swgg = zero(α), zero(β)

    # Non-genetic effects
    for j in α.nzind
        @inbounds Swxx[j] = compute_Swxx(X, w, j)
    end

    # Genetic effects
    for j in β.nzind
        @inbounds Swgg[j] = compute_Swxx(G, w, j)
    end

    # Coordinate descent algorithm
    for cd_iter in 1:cd_maxiter

        # Perform one coordinate descent cycle
        maxΔ = cycle(X, G, λ, Val(false), r = r, α = α, β = β, Swxx = Swxx, Swgg = Swgg, w = w, p_fX = p_fX, p_fG = p_fG)

        # Check termination condition before last iteration
        if criterion == :obj
            # Update μ
            μ, = updateμ(r, Ytilde)

            # Update deviance and loss function
            prev_loss = loss
            dev = model_dev(family, δ, w, r, eigvals, y, μ)
            loss = dev/2 + λ * P(α, β, p_fX, p_fG)

            # Check termination condition
            converged && abs(loss - prev_loss) < cd_tol * loss && break
            converged = abs(loss - prev_loss) < cd_tol * loss 

        elseif criterion == :coef
            converged && maxΔ < cd_tol && break
            converged = maxΔ < cd_tol
        end
    end

    # Assess convergence of coordinate descent
    @assert converged "Coordinate descent failed to converge in $cd_maxiter iterations at λ = $λ"

    return(Swxx, Swgg)
end

function cd_lasso(
    # positional arguments
    D::Vector{T},
    X::Matrix{T},
    G::SubArray{T, 2, SnpLinAlg{T}, Tuple{Vector{Int64}, UnitRange{Int64}}},
    λ::T;
    #keywords arguments
    family::UnivariateDistribution,
    r::Vector{T},
    α::SparseVector{T},
    β::SparseVector{T},
    δ::Vector{T},
    γ::SparseVector{T},
    Ytilde::Vector{T},
    y::Vector{Int},
    w::Vector{T}, 
    eigvals::Vector{T}, 
    p_fX::Vector{T},
    p_fG::Vector{T},
    cd_maxiter::Integer = 10000,
    cd_tol::Real=1e-7,
    criterion,
    ) where T

    converged = true
    loss = Inf

    # Compute sum of squares
    Swxx, Swgg, Swdg = zero(α), zero(β), zero(γ)

    # Non-genetic effects
    for j in α.nzind
        @inbounds Swxx[j] = compute_Swxx(X, w, j)
    end

    # Genetic effects
    for j in β.nzind
        @inbounds Swgg[j] = compute_Swxx(G, w, j)
    end

    # GEI effects
    for j in γ.nzind
        @inbounds Swdg[j] = compute_Swxx(D, G, w, j)
    end


    # Coordinate descent algorithm
    for cd_iter in 1:cd_maxiter

        # Perform one coordinate descent cycle
        maxΔ, = cycle(D, X, G, λ, Val(false), r = r, α = α, β = β, γ = γ, Swxx = Swxx, Swgg = Swgg, Swdg = Swdg, w = w, p_fX = p_fX, p_fG = p_fG)

        # Check termination condition before last iteration
        if criterion == :obj
            # Update μ
            μ, = updateμ(r, Ytilde)

            # Update deviance and loss function
            prev_loss = loss
            dev = model_dev(family, δ, w, r, eigvals, y, μ)
            loss = dev/2 + λ * P(α, β, γ, p_fX, p_fG)

            # Check termination condition
            converged && abs(loss - prev_loss) < cd_tol * loss && break
            converged = abs(loss - prev_loss) < cd_tol * loss 

        elseif criterion == :coef
            converged && maxΔ < cd_tol && break
            converged = maxΔ < cd_tol
        end
    end

    # Assess convergence of coordinate descent
    @assert converged "Coordinate descent failed to converge in $cd_maxiter iterations at λ = $λ"

    return(Swxx, Swgg, Swdg)
end

function cd_lasso(
    # positional arguments
    U::Matrix{T},
    λ::T;
    #keywords arguments
    family::UnivariateDistribution,
    r::Vector{T},
    δ::Vector{T},
    Ytilde::Vector{T},
    y::Vector{Int},
    w::Vector{T},
    eigvals::Vector{T}, 
    cd_maxiter::Integer = 10000,
    cd_tol::Real=1e-7,
    criterion
    ) where T

    converged = true
    loss = Inf

    # Compute sum of squares
    Swuu = zero(δ)
    for j in 1:length(δ)
        @inbounds Swuu[j] = compute_Swxx(U, w, j)
    end

    # Coordinate descent algorithm
    for cd_iter in 1:cd_maxiter

        # Perform one coordinate descent cycle
        maxΔ = cycle(U, λ, r = r, δ = δ, Swuu = Swuu, w = w, eigvals = eigvals)

        # Check termination condition before last iteration
        if criterion == :obj
            # Update μ
            μ, = updateμ(r, Ytilde)

            # Update deviance and loss function
            prev_loss = loss
            dev = model_dev(family, δ, w, r, eigvals, y, μ)
            loss = dev/2 + λ * P(α, β, γ, p_fX, p_fG)

            # Check termination condition
            converged && abs(loss - prev_loss) < cd_tol * loss && break
            converged = abs(loss - prev_loss) < cd_tol * loss 

        elseif criterion == :coef
            converged && maxΔ < cd_tol && break
            converged = maxΔ < cd_tol
        end
    end

    # Assess convergence of coordinate descent
    @assert converged "Coordinate descent failed to converge in $cd_maxiter iterations at λ = $λ"

end

function cycle(
    # positional arguments
    X::Matrix{T},
    G::SubArray{T, 2, SnpLinAlg{T}, Tuple{Vector{Int64}, UnitRange{Int64}}},
    λ::T,
    all_pred::Val{false};
    #keywords arguments
    r::Vector{T},
    α::SparseVector{T},
    β::SparseVector{T},
    Swxx::SparseVector{T},
    Swgg::SparseVector{T},
    w::Vector{T}, 
    p_fX::Vector{T},
    p_fG::Vector{T}
    ) where T

    maxΔ = zero(T)

    # Cycle over coefficients in active set only until convergence
    # Non-genetic covariates
    for j in α.nzind
        last_α = α[j]
        v = compute_grad(X, w, r, j) + last_α * Swxx[j]
        new_α = softtreshold(v, λ * p_fX[j]) / Swxx[j]
        r = update_r(X, r, last_α - new_α, j)

        maxΔ = max(maxΔ, Swxx[j] * (last_α - new_α)^2)
        α[j] = new_α
    end

    # Genetic predictors
    for j in β.nzind
        last_β = β[j]
        v = compute_grad(G, w, r, j) + last_β * Swgg[j]
        new_β = softtreshold(v, λ * p_fG[j]) / Swgg[j]
        r = update_r(G, r, last_β - new_β, j)

        maxΔ = max(maxΔ, Swgg[j] * (last_β - new_β)^2)
        β[j] = new_β
    end

    maxΔ
end

function cycle(
    # positional arguments
    D::Vector{T},
    X::Matrix{T},
    G::SubArray{T, 2, SnpLinAlg{T}, Tuple{Vector{Int64}, UnitRange{Int64}}},
    λ::T,
    all_pred::Val{false};
    #keywords arguments
    r::Vector{T},
    α::SparseVector{T},
    β::SparseVector{T},
    γ::SparseVector{T},
    Swxx::SparseVector{T},
    Swgg::SparseVector{T},
    Swdg::SparseVector{T},
    w::Vector{T}, 
    p_fX::Vector{T},
    p_fG::Vector{T}
    ) where T

    maxΔ = zero(T)

    # Cycle over coefficients in active set only until convergence
    # Non-genetic covariates
    for j in α.nzind
        last_α = α[j]
        v = compute_grad(X, w, r, j) + last_α * Swxx[j]
        new_α = softtreshold(v, λ * p_fX[j]) / Swxx[j]
        r = update_r(X, r, last_α - new_α, j)

        maxΔ = max(maxΔ, Swxx[j] * (last_α - new_α)^2)
        α[j] = new_α
    end

    # GEI and genetic effects
    for j in γ.nzind
        λj = λ * p_fG[j]

        # Update GEI effect
        last_γ = γ[j]
        v = compute_grad(D, G, w, r, j) + last_γ * Swdg[j]
        if abs(v) > λj
            new_γ = softtreshold(v, λj) / (Swdg[j] + λj / norm((γ[j], β[j])))
            r = update_r(D, G, r, last_γ - new_γ, j)

            maxΔ = max(maxΔ, Swdg[j] * (last_γ - new_γ)^2)
            γ[j] = new_γ
        end

        # Update genetic effect
        last_β = β[j]
        v = compute_grad(G, w, r, j) + last_β * Swgg[j]
        new_β = γ[j] != 0 ? v / (Swgg[j] + λj / norm((γ[j], β[j]))) : softtreshold(v, λj) / Swgg[j]
        r = update_r(G, r, last_β - new_β, j)

        maxΔ = max(maxΔ, Swgg[j] * (last_β - new_β)^2)
        β[j] = new_β

    end

    maxΔ
end

function cycle(
    # positional arguments
    X::Matrix{T},
    G::SubArray{T, 2, SnpLinAlg{T}, Tuple{Vector{Int64}, UnitRange{Int64}}},
    λ::T,
    all_pred::Val{true};
    #keywords arguments
    r::Vector{T},
    α::SparseVector{T},
    β::SparseVector{T},
    Swxx::SparseVector{T},
    Swgg::SparseVector{T},
    w::Vector{T}, 
    p_fX::Vector{T},
    p_fG::Vector{T}
    ) where T

    maxΔ = zero(T)
    kkt_check = true

    # At first and last iterations, cycle through all predictors
    # Non-genetic covariates
    for j in 1:length(α)
        v = compute_grad(X, w, r, j)
        λj = λ * p_fX[j]
        
        if j in α.nzind
            last_α = α[j]
            v += last_α * Swxx[j]
        else
            # Adding a new variable to the model
            abs(v) <= λj && continue
            kkt_check = false
            last_α = 0
            Swxx[j] = compute_Swxx(X, w, j)
        end
        new_α = softtreshold(v, λj) / Swxx[j]
        r = update_r(X, r, last_α - new_α, j)

        maxΔ = max(maxΔ, Swxx[j] * (last_α - new_α)^2)
        α[j] = new_α
    end

    # Genetic covariates
    for j in 1:length(β)
        v = compute_grad(G, w, r, j)
        λj = λ * p_fG[j]

        if j in β.nzind
            last_β = β[j]
            v += last_β * Swgg[j]
        else
            # Adding a new variable to the model
            abs(v) <= λj && continue
            kkt_check = false
            last_β = 0
            Swgg[j] = compute_Swxx(G, w, j)
        end
        new_β = softtreshold(v, λj) / Swgg[j]
        r = update_r(G, r, last_β - new_β, j)

        maxΔ = max(maxΔ, Swgg[j] * (last_β - new_β)^2)
        β[j] = new_β
    end

    return(maxΔ, kkt_check)
end

function cycle(
    # positional arguments
    D::Vector{T},
    X::Matrix{T},
    G::SubArray{T, 2, SnpLinAlg{T}, Tuple{Vector{Int64}, UnitRange{Int64}}},
    λ::T,
    all_pred::Val{true};
    #keywords arguments
    r::Vector{T},
    α::SparseVector{T},
    β::SparseVector{T},
    γ::SparseVector{T},
    Swxx::SparseVector{T},
    Swgg::SparseVector{T},
    Swdg::SparseVector{T},
    w::Vector{T}, 
    p_fX::Vector{T},
    p_fG::Vector{T}
    ) where T

    maxΔ = zero(T)
    kkt_check = true

    # At first and last iterations, cycle through all predictors
    # Non-genetic covariates
    for j in 1:length(α)
        v = compute_grad(X, w, r, j)
        λj = λ * p_fX[j]
        
        if j in α.nzind
            last_α = α[j]
            v += last_α * Swxx[j]
        else
            # Adding a new variable to the model
            abs(v) <= λj && continue
            kkt_check = false
            last_α = 0
            Swxx[j] = compute_Swxx(X, w, j)
        end
        new_α = softtreshold(v, λj) / Swxx[j]
        r = update_r(X, r, last_α - new_α, j)

        maxΔ = max(maxΔ, Swxx[j] * (last_α - new_α)^2)
        α[j] = new_α
    end

    # GEI and genetic effects
    for j in 1:length(γ)
        v = compute_grad(D, G, w, r, j)
        λj = λ * p_fG[j]

        if j in γ.nzind || abs(v) > λj
            # Update GEI effect
            if j in γ.nzind 
                last_γ = γ[j]
                v += last_γ * Swdg[j]
            else
                kkt_check = false
                last_γ, γ[j] = 0, 1
                Swdg[j] = compute_Swxx(D, G, w, j)
            end

            new_γ = softtreshold(v, λj) / (Swdg[j] + λj / norm((last_γ, β[j])))
            r = update_r(D, G, r, last_γ - new_γ, j)

            # Update genetic effect
            v = compute_grad(G, w, r, j)

            if j in β.nzind 
                last_β = β[j]
                v += last_β * Swgg[j]
            else
                kkt_check = false
                last_β, β[j] = 0, 1
                Swgg[j] = compute_Swxx(G, w, j)
            end

            new_β = new_γ != 0 ? v / (Swgg[j] + λj / norm((last_γ, last_β))) : softtreshold(v, λj) / Swgg[j]
            r = update_r(G, r, last_β - new_β, j)

            maxΔ = max(maxΔ, Swgg[j] * (last_β - new_β)^2, Swdg[j] * (last_γ - new_γ)^2)
            β[j], γ[j] = new_β, new_γ

            continue
        end

        # Genetic effects only
        v = compute_grad(G, w, r, j)

        if j in β.nzind
            last_β = β[j]
            v += last_β * Swgg[j]
        else
            # Adding a new variable to the model
            abs(v) <= λj && continue
            kkt_check = false
            last_β = 0
            Swgg[j] = compute_Swxx(G, w, j)
        end
        new_β = softtreshold(v, λj) / Swgg[j]
        r = update_r(G, r, last_β - new_β, j)

        maxΔ = max(maxΔ, Swgg[j] * (last_β - new_β)^2)
        β[j] = new_β

    end

    return(maxΔ, kkt_check)
end

function cycle(
    # positional arguments
    U::Matrix{T},
    λ::T;
    #keywords arguments
    r::Vector{T},
    δ::Vector{T},
    Swuu::Vector{T},
    w::Vector{T}, 
    eigvals::Vector{T}
    ) where T

    maxΔ = zero(T)

    # Cycle through all predictors
    for j in 1:size(U, 2)
        last_δ = δ[j]
        v = compute_grad(U, w, r, j) + last_δ * Swuu[j]
        new_δ = v / (Swuu[j] + eigvals[j])
        r = update_r(U, r, last_δ - new_δ, j)

        maxΔ = max(maxΔ, Swuu[j] * (last_δ - new_δ)^2)
        δ[j] = new_δ
    end

    maxΔ
end

# Function to update random effects vector
function update_δ(
    # positional arguments
    ::Val{:cd};
    #keywords arguments
    U::Matrix{T}, 
    λ::T, 
    family::UnivariateDistribution, 
    Ytilde::Vector{T}, 
    y::Vector{Int64}, 
    w::Vector{T}, 
    r::Vector{T}, 
    δ::Vector{T}, 
    eigvals::Vector{T}, 
    criterion = criterion,
    kwargs...
    ) where T

    cd_lasso(U, λ; family = family, Ytilde = Ytilde, y = y, w = w, r = r, δ = δ, eigvals = eigvals, criterion = criterion)
end

function update_δ(
    # positional arguments
    ::Val{:conjgrad};
    #keywords arguments
    U::Matrix{T},
    y::Vector{Int64}, 
    w::Vector{T}, 
    r::Vector{T}, 
    δ::Vector{T}, 
    eigvals::Vector{T}, 
    μ::Vector{T},
    kwargs...
    ) where T

    delta_δ = conjgrad(δ = δ, eigvals = eigvals, U = U, y = y, μ = μ, w = w)
    r += U * delta_δ
end

# Conjuguate gradient descent to update random effects vector
function conjgrad(
    ;
    #keywords arguments
    δ::Vector{T},
    eigvals::Vector{T},
    U::Matrix{T},
    y::Vector{Int64},
    μ::Vector{T},
    w::Vector{T},
    tol::T = 1e-7
    ) where T
    
    # Initialization
    converged = false 
    A = U' * Diagonal(w) * U + Diagonal(eigvals)
    r = eigvals .* δ - U' * (y - μ)
    p = -r
    k, delta_δ = 0, zero(δ)

    for _ in 1:size(U, 1)
        # Check convergence
        converged = norm(r) < tol
        # converged && println("Conjuguate gradient has converged in $k steps.") 
        converged && break

        # Next iteration
        k += 1
        alpha = dot(r, r) / dot(p, A, p) 
        new_δ = δ + alpha * p
        delta_δ += δ - new_δ
        δ = new_δ
        new_r = r + alpha * A * p
        beta = dot(new_r, new_r) / dot(r, r)
        r = new_r
        p = -r + beta * p
    end

    @assert converged "Conjuguate gradient descent failed to converge."
    delta_δ
end

modeltype(::Normal) = "Least Squares GLMNet"
modeltype(::Binomial) = "Logistic"

struct pglmmPath{F<:Distribution, A<:AbstractArray, B<:AbstractArray, T<:AbstractFloat, C<:SubArray{T, 2, Matrix{T}, Tuple{Base.Slice{Base.OneTo{Int}}, UnitRange{Int}}, true}}
    family::F
    a0::A                                       # intercept values for each solution
    alphas::B                                   # coefficient values for each solution
    betas::Union{B, C}                                 
    gammas::Union{Nothing, C}
    null_dev::T                                 # Null deviance of the model
    pct_dev::A                                  # R^2 values for each solution
    lambda::A                                   # lamda values corresponding to each solution
    npasses::Int                                # actual number of passes over the
                                                # data for all lamda values
    fitted_values                               # fitted_values
    y::Union{Vector{Int}, A}                    # eigenvalues vector
    UD_invUt::B                                 # eigenvectors matrix times diagonal weights matrix
    τ::A                                        # estimated variance components
    intercept::Bool                             # boolean for intercept
end

function show(io::IO, g::pglmmPath)
    if isnothing(g.gammas)
        df = [length(findall(x -> x != 0, vec(view([g.alphas; g.betas], :,k)))) for k in 1:size(g.betas, 2)]
        println(io, "$(modeltype(g.family)) Solution Path ($(size(g.betas, 2)) solutions for $(size([g.alphas; g.betas], 1)) predictors):") #in $(g.npasses) passes):"
    else
        df = [length(findall(x -> x != 0, vec(view([g.alphas; g.betas; g.gammas], :,k)))) for k in 1:size(g.betas, 2)]
        println(io, "$(modeltype(g.family)) Solution Path ($(size(g.betas, 2)) solutions for $(size([g.alphas; g.betas; g.gammas], 1)) predictors):") #in $(g.npasses) passes):"
    end    
    print(io, CoefTable(Union{Vector{Int},Vector{Float64}}[df, g.pct_dev, g.lambda], ["df", "pct_dev", "λ"], []))
end

# Function to compute sequence of values for λ
function lambda_seq(
    r::Vector{T}, 
    X::Matrix{T},
    G::SubArray{T, 2, SnpLinAlg{T}, Tuple{Vector{Int64}, UnitRange{Int64}}},
    D::Union{Vector{T}, Nothing}; 
    p_fX::Vector{T},
    p_fG::Vector{T},
    K::Integer = 100
    ) where T

    λ_min_ratio = (length(r) < size(G, 2) ? 1e-2 : 1e-4)
    λ_max = lambda_max(nothing, X, r, p_fX)
    λ_max = lambda_max(D, G, r, p_fG, λ_max)
    λ_min = λ_max * λ_min_ratio
    λ_step = log(λ_min_ratio)/(K - 1)
    λ_seq = exp.(collect(log(λ_max+100*eps(λ_max)):λ_step:log(λ_min)))

    λ_seq
end

# Function to compute λ_max for the lasso
function lambda_max(D::Nothing, X::AbstractMatrix{T}, r::AbstractVector{T}, p_f::AbstractVector{T}, λ_max::T = zero(T)) where T
    seq = findall(!iszero, p_f)
    for j in seq
        x = abs(compute_grad(X, r, j))
        if x > λ_max
            λ_max = x
        end
    end
    return(λ_max)
end

# Function to compute λ_max for the group lasso
function lambda_max(D::AbstractVector{T}, X::AbstractMatrix{T}, r::AbstractVector{T}, p_f::AbstractVector{T}, λ_max::T = zero(T)) where T

    seq = findall(!iszero, p_f)
    for j in seq
        x = compute_max(D, X, r, j)
        if x > λ_max
            λ_max = x
        end
    end
    return(λ_max)
end

# Define softtreshold function
function softtreshold(z::T, γ::T) :: T where T
    if z > γ
        z - γ
    elseif z < -γ
        z + γ
    else
        0
    end
end

# Function to update working response and residual
function wrkresp(
    y::Vector{Int64},
    μ::Vector{T},
    w::Vector{T}
) where T
    η = GLM.linkfun.(LogitLink(), μ)
    Ytilde = [η[i] + (y[i] - μ[i]) / w[i] for i in 1:length(y)]
    r = Ytilde - η
    return(Ytilde, r)
end

# Function to update linear predictor and mean at each iteration
const PMIN = 1e-5
const PMAX = 1-1e-5
function updateμ(r::Vector{T}, Ytilde::Vector{T}) where T
    η = Ytilde - r
    μ = GLM.linkinv.(LogitLink(), η)
    μ = [μ[i] < PMIN ? PMIN : μ[i] > PMAX ? PMAX : μ[i] for i in 1:length(μ)]
    w = μ .* (1 .- μ)
    return(μ, w)
end

# Functions to calculate deviance
model_dev(::Binomial, δ::Vector{T}, w::Vector{T}, r::Vector{T}, eigvals::Vector{T}, y::Vector{Int64}, μ::Vector{Float64}) where T = LogisticDeviance(δ, eigvals, y, μ)
model_dev(::Normal, δ::Vector{T}, w::T, r::Vector{T}, eigvals::Vector{T}, kargs...) where T = NormalDeviance(δ, w, r, eigvals)

function LogisticDeviance(δ::Vector{T}, eigvals::Vector{T}, y::Vector{Int64}, μ::Vector{T}) where T
    -2 * sum(y .* log.(μ ./ (1 .- μ)) .+ log.(1 .- μ)) + dot(δ, Diagonal(eigvals), δ)
end

function NormalDeviance(δ::Vector{T}, w::T, r::Vector{T}, eigvals::Vector{T}) where T
    w * dot(r, r) + dot(δ, Diagonal(eigvals), δ)
end

# Predict phenotype
function predict(path::pglmmPath, 
                  X::AbstractMatrix{T}, 
                  grmfile::AbstractString; 
                  grmrowinds::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                  grmcolinds::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                  M::Union{Nothing, Vector{Any}} = nothing,
                  s::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                  fixed_effects_only::Bool = false,
                  outtype = :response
                 ) where T

    # read file containing the m x (N-m) kinship matrix between m test and (N-m) training subjects
    GRM = open(GzipDecompressorStream, grmfile, "r") do stream
        Symmetric(Matrix(CSV.read(stream, DataFrame)))
    end

    if !isnothing(grmrowinds)
        GRM = GRM[grmrowinds, :]
    end

    if !isnothing(grmcolinds)
        GRM = GRM[:, grmcolinds]
    end

    # Covariance matrix between test and training subjects
    V = isnothing(M) ? push!(Any[], GRM) : reverse(push!(M, GRM))
    Σ_12 = sum(path.τ .* V)

    # Number of predictions to compute. User can provide index s for which to provide predictions, 
    # rather than computing predictions for the whole path.
    s = isnothing(s) ? (1:size(path.betas, 2)) : s

    # Linear predictor
    η = !isnothing(path.gammas) ? path.a0[s]' .+ X * [path.alphas; path.betas; path.gammas][:,s] : path.a0[s]' .+ X * [path.alphas; path.betas][:,s]

    if fixed_effects_only == false
        if path.family == Binomial()
            η += Σ_12 * (path.y .- path.fitted_values[:,s])
        elseif path.family == Normal()
            η += Σ_12 * path.UD_invUt * path.fitted_values[:,s]
        end
    end

    # Return linear predictor (default) or fitted probs
    if outtype == :response
        return(η)
    elseif outtype == :prob
        return(GLM.linkinv.(LogitLink(), η))
    end
end 

function predict(path::pglmmPath, 
                  covfile::AbstractString,
                  grmfile::AbstractString,
                  plinkfile::Union{Nothing, AbstractString} = nothing;
                  # keyword arguments
                  snpfile::Union{Nothing, AbstractString} = nothing,
                  snpmodel = ADDITIVE_MODEL,
                  snpinds::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                  covrowinds::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                  covars::Union{Nothing,AbstractVector{<:String}} = nothing, 
                  geneticrowinds::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                  grmrowinds::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                  grmcolinds::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                  M::Union{Nothing, Vector{Any}} = nothing,
                  s::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                  fixed_effects_only::Bool = false,
                  GEIvar::Union{Nothing,AbstractString} = nothing,
                  outtype = :response
                 ) where T
    
    #--------------------------------------------------------------
    # Read covariate file
    #--------------------------------------------------------------
    covdf = CSV.read(covfile, DataFrame)

    if !isnothing(covrowinds)
        covdf = covdf[covrowinds,:]
    end 

    if !isnothing(covars)
        covdf = covdf[:, covars]
    end 

    X = Matrix(covdf)
    nX, k =size(X)

    #--------------------------------------------------------------
    # Read file containing the m x (N-m) kinship matrix between m test and (N-m) training subjects
    #--------------------------------------------------------------
    GRM = open(GzipDecompressorStream, grmfile, "r") do stream
        Symmetric(Matrix(CSV.read(stream, DataFrame)))
    end

    if !isnothing(grmrowinds)
        GRM = GRM[grmrowinds, :]
    end

    if !isnothing(grmcolinds)
        GRM = GRM[:, grmcolinds]
    end

    @assert nX == size(GRM, 1) "GRM and covariates matrix must have same number of rows."
    
    #--------------------------------------------------------------
    # Read genotype file
    #--------------------------------------------------------------
    if !isnothing(plinkfile)

        # read PLINK files
        geno = SnpArray(plinkfile * ".bed")
    
        # Convert genotype file to matrix, convert to additive model (default) and impute
        snpinds_ = isnothing(snpinds) ? (1:size(geno, 2)) : snpinds 
        geneticrowinds_ = isnothing(geneticrowinds) ? (1:size(geno, 1)) : geneticrowinds
        geno = isnothing(snpinds) && isnothing(geneticrowinds) ? geno : SnpArrays.filter(plinkfile, geneticrowinds_, snpinds_)
        
        # Read genotype
        G = SnpLinAlg{Float64}(geno, model = snpmodel, impute = true, center = false, scale = false)

    elseif !isnothing(snpfile)

        # read CSV file
        geno = CSV.read(snpfile, DataFrame)
        
        # Convert genotype file to matrix, convert to additive model (default) and impute
        snpinds_ = isnothing(snpinds) ? (1:size(geno, 2)) : snpinds 
        geneticrowinds_ = isnothing(geneticrowinds) ? (1:size(geno, 1)) : geneticrowinds
        G = convert.(Float64, Matrix(geno[geneticrowinds_, snpinds_]))

    end

    # Initialize number of subjects and predictors (including intercept)
    nG, p = size(G)
    @assert nG == size(GRM, 1) "GRM and genotype matrix must have same number of rows."
    
    #--------------------------------------------------------------
    # Compute predictions
    #--------------------------------------------------------------
    # Covariance matrix between test and training subjects
    V = isnothing(M) ? push!(Any[], GRM) : reverse(push!(M, GRM))
    Σ_12 = sum(path.τ .* V)

    # Number of predictions to compute. User can provide index s for which to provide predictions, 
    # rather than computing predictions for the whole path.
    s = isnothing(s) ? (1:size(path.betas, 2)) : s

    # Linear predictor
    η = path.a0[s]' .+ X * path.alphas[:,s] .+ G * path.betas[:,s]

    if !isnothing(GEIvar)
        D = covdf[:, GEIvar]
        η += (D .* G) * path.gammas[:,s]
    end

    if fixed_effects_only == false
        if path.family == Binomial()
            η += Σ_12 * (path.y .- path.fitted_values[:,s])
        elseif path.family == Normal()
            η += Σ_12 * path.UD_invUt * path.fitted_values[:,s]
        end
    end

    # Return linear predictor (default) or fitted probs
    if outtype == :response
        return(η)
    elseif outtype == :prob
        return(GLM.linkinv.(LogitLink(), η))
    end
end 

# GIC penalty parameter
function GIC(path::pglmmPath, criterion)
    
    # Obtain number of rows (n), predictors (p) and λ values (K)
    n = size(path.y, 1)
    m, (p, K) = size(path.alphas, 1), size(path.betas)
    df = path.intercept .+ [length(findall(x -> x != 0, vec(view([path.alphas; path.betas], :, k)))) for k in 1:K] .+ length(path.τ)
    df += !isnothing(path.gammas) ? [length(findall(x -> x != 0, vec(view(path.gammas, :, k)))) for k in 1:K] : zero(df)

    # Define GIC criterion
    if criterion == :BIC
        a_n = log(n)
    elseif criterion == :AIC
        a_n = 2
    elseif criterion == :HDBIC
        a_n = !isnothing(path.gammas) ? log(log(n)) * log(m + 2 * p) : log(log(n)) * log(m + p)
    end

    # Compute deviance for each value of λ
    dev = (1 .- path.pct_dev) * path.null_dev
    GIC = dev .+ a_n * df

    # Return betas with lowest GIC value
    return(argmin(GIC))
end

# Standardize predictors for lasso
function standardizeX(X::AbstractMatrix{T}, standardize::Bool, intercept::Bool = false) where T
    mu = intercept ? vec([0 mean(X[:,2:end], dims = 1)]) : vec(mean(X, dims = 1))
    if standardize
        s = intercept ? vec([1 std(X[:,2:end], dims = 1, corrected = false)]) : vec(std(X, dims = 1, corrected = false)) 
        if any(s .== zero(T))
            @warn("One predictor is a constant, hence it can't been standardized!")
            s[s .== 0] .= 1 
        end
        for j in 1:size(X,2), i in 1:size(X, 1) 
            @inbounds X[i,j] = (X[i,j] .- mu[j]) / s[j]
        end
    else
        for j in 1:size(X,2), i in 1:size(X, 1) 
            @inbounds X[i,j] = X[i,j] .- mu[j]
        end
        s = []
    end

    # Remove first term if intercept
    if intercept 
         popfirst!(mu); popfirst!(s)
    end

    X, mu, s
end

# Calculate mean and scale for genotype data
function standardizeG(s::AbstractSnpArray, model, scale::Bool, T = AbstractFloat)
    n, m = size(s)
    μ, σ = Array{T}(undef, m), Array{T}(undef, m)	
    @inbounds for j in 1:m
        μj, mj = zero(T), 0
        for i in 1:n
            vij = SnpArrays.convert(T, s[i, j], model)
            μj += isnan(vij) ? zero(T) : vij
            mj += isnan(vij) ? 0 : 1
        end
        μj /= mj
        μ[j] = μj
        σ[j] = model == ADDITIVE_MODEL ? sqrt(μj * (1 - μj / 2)) : sqrt(μj * (1 - μj))
    end
    
    # Return centre and scale parameters
    if scale 
	   return μ, σ
    else 
	   return μ, []
    end
end

function compute_grad(X::AbstractMatrix{T}, w::AbstractVector{T}, r::AbstractVector{T}, whichcol::Int) where T
    v = zero(T)
    for i = 1:size(X, 1)
        @inbounds v += X[i, whichcol] * r[i] * w[i]
    end
    v
end

function compute_grad(D::AbstractVector{T}, X::AbstractMatrix{T}, w::AbstractVector{T}, r::AbstractVector{T}, whichcol::Int) where T
    v = zero(T)
    for i = 1:size(X, 1)
        @inbounds v += D[i] * X[i, whichcol] * r[i] * w[i]
    end
    v
end

function compute_grad(X::AbstractMatrix{T}, r::AbstractVector{T}, whichcol::Int) where T
    v = zero(T)
    for i = 1:size(X, 1)
        @inbounds v += X[i, whichcol] * r[i]
    end
    v
end

function compute_max(D::AbstractVector{T}, X::AbstractMatrix{T}, r::AbstractVector{T}, whichcol::Int) where T
    v = zeros(2)
    for i = 1:size(X, 1)
        @inbounds v[1] += X[i, whichcol] * r[i]
        @inbounds v[2] += D[i] * X[i, whichcol] * r[i]
    end
    maximum(abs.(v))
end

function compute_prod(X::AbstractMatrix{T}, y::AbstractVector{Int}, p::AbstractVector{T}, whichcol::Int) where T
    v = zero(T)
    for i = 1:size(X, 1)
        @inbounds v += X[i, whichcol] * (y[i] - p[i])
    end
    v
end

function compute_prod(D::AbstractVector{T}, X::AbstractMatrix{T}, y::AbstractVector{Int}, p::AbstractVector{T}, whichcol::Int) where T
    v = zero(T)
    for i = 1:size(X, 1)
        @inbounds v += D[i] * X[i, whichcol] * (y[i] - p[i])
    end
    v
end

function compute_Swxx(X::AbstractMatrix{T}, w::AbstractVector{T}, whichcol::Int) where T
    s = zero(T)
    for i = 1:size(X, 1)
        @inbounds s += X[i, whichcol]^2 * w[i]
    end
    s
end

function compute_Swxx(D::AbstractVector{T}, X::AbstractMatrix{T}, w::AbstractVector{T}, whichcol::Int) where T
    s = zero(T)
    for i = 1:size(X, 1)
        @inbounds s += (D[i] * X[i, whichcol])^2 * w[i]
    end
    s
end

function update_r(X::AbstractMatrix{T}, r::AbstractVector{T}, deltaβ::T, whichcol::Int) where T
    for i = 1:size(X, 1)
        @inbounds r[i] += X[i, whichcol] * deltaβ
    end
    r
end

function update_r(D::AbstractVector{T}, X::AbstractMatrix{T}, r::AbstractVector{T}, deltaβ::T, whichcol::Int) where T
    for i = 1:size(X, 1)
        @inbounds r[i] += D[i] * X[i, whichcol] * deltaβ
    end
    r
end

function P(α::SparseVector{T}, β::SparseVector{T}, p_fX::Vector{T}, p_fG::Vector{T}) where T
    x = zero(T)
    @inbounds @simd for i in α.nzind
            x += p_fX[i] * abs(α[i])
    end
    @inbounds @simd for i in β.nzind
            x += p_fG[i] * abs(β[i])
    end
    x
end

function P(α::SparseVector{T}, β::SparseVector{T}, γ::SparseVector{T}, p_fX::Vector{T}, p_fG::Vector{T}) where T
    x = zero(T)
    @inbounds @simd for i in α.nzind
            x += p_fX[i] * abs(α[i])
    end
    @inbounds @simd for i in β.nzind
            x += p_fG[i] * (norm((β[i], γ[i])) + abs(γ[i]))
    end
    x
end

# Compute strongrule for the lasso
function compute_strongrule(dλ::T, p_fX::Vector{T}, p_fG::Vector{T}; α::SparseVector{T}, β::SparseVector{T}, X::Matrix{T}, G::SubArray{T, 2, SnpLinAlg{T}, Tuple{Vector{Int64}, UnitRange{Int64}}}, y::Vector{Int}, μ::Vector{T}) where T
    for j in 1:length(α)
        j in α.nzind && continue
        c = compute_prod(X, y, μ, j)
        abs(c) <= dλ * p_fX[j] && continue
        
        # Force a new variable to the model
        α[j] = 1; α[j] = 0
    end
    
    for j in 1:length(β)
        j in β.nzind && continue
        c = compute_prod(G, y, μ, j)
        abs(c) <= dλ * p_fG[j] && continue
        
        # Force a new variable to the model
        β[j] = 1; β[j] = 0
    end
end

# Compute strongrule for the group lasso + lasso (CAP)
function compute_strongrule(dλ::T, λ::T, p_fX::Vector{T}, p_fG::Vector{T}, D::Vector{T}; α::SparseVector{T}, β::SparseVector{T}, γ::SparseVector{T}, X::Matrix{T}, G::SubArray{T, 2, SnpLinAlg{T}, Tuple{Vector{Int64}, UnitRange{Int64}}}, y::Vector{Int}, μ::Vector{T}) where T
    for j in 1:length(α)
        j in α.nzind && continue
        c = compute_prod(X, y, μ, j)
        abs(c) <= dλ * p_fX[j] && continue
        
        # Force a new variable to the model
        α[j] = 1; α[j] = 0
    end
    
    for j in 1:length(γ)
        j in γ.nzind && continue
        c1 = compute_prod(G, y, μ, j)
        c2 = softtreshold(compute_prod(D, G, y, μ, j), λ * p_fG[j])
        norm([c1, c2]) <= dλ * p_fG[j] && continue
        
        # Force a new group to the model
        γ[j] = 1; γ[j] = 0
        j in β.nzind && continue
        β[j] = 1; β[j] = 0
    end
end