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
- `method`: which method to use to estimate random effects vector. Can be equal to `:cd` (default) for coordinate descent or `:conjgrad` for conjuguate gradient descent.
- `upper_bound::Bool = false (default)`: For logistic regression, should an upper-bound be used on the Hessian ?
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
    irls_tol::T = 1e-7,
    irls_maxiter::Integer = 500,
    nlambda::Integer = 100,
    lambda::Union{Nothing, Vector{T}, Vector{Vector{T}}} = nothing,
    rho::Union{Real, AbstractVector{<:Real}} = 0.5,
    verbose::Bool = false,
    standardize_X::Bool = true,
    standardize_G::Bool = true,
    criterion = :coef,
    earlystop::Bool = false,
    method = :cd,
    upper_bound::Bool = false,
    kwargs...
    ) where T

    # # keyword arguments
    # snpmodel = ADDITIVE_MODEL
    # snpinds = nothing
    # geneticrowinds = trainrowinds
    # irls_tol = 1e-7
    # irls_maxiter = 500
    # nlambda = 100
    # rho = collect(0:0.1:0.5)
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
    eigvals, U = eigen(sum(nullmodel.τ .* nullmodel.V))
    eigvals .= 1 ./ eigvals
    UD_invUt = U * Diagonal(eigvals) * U'
   
    # Initialize random effects vector 
    b = nullmodel.η - nullmodel.X * nullmodel.α

    # Initialize working variable
    y = nullmodel.y
    if nullmodel.family == Binomial()
        μ, ybar = GLM.linkinv.(LogitLink(), nullmodel.η), mean(y)
        w = upper_bound ? repeat([0.25], length(y)) : μ .* (1 .- μ)
        Ytilde = nullmodel.η + (y - μ) ./ w
        nulldev = -2 * sum(y * log(ybar / (1 - ybar)) .+ log(1 - ybar))
    elseif nullmodel.family == Normal()
        Ytilde, μ = y, nullmodel.η
        w = 1 / nullmodel.φ
        nulldev = w * (y .- mean(y)).^2
    end

    # standardize non-genetic covariates
    intercept = all(nullmodel.X[:,1] .== 1)
    X, muX, sX = standardizeX(nullmodel.X, standardize_X, intercept)
    ind_D = !isnothing(nullmodel.ind_D) ? nullmodel.ind_D .- intercept : nothing
    D, muD, sD = !isnothing(ind_D) ? (vec(X[:, nullmodel.ind_D]), muX[ind_D], sX[ind_D]) : repeat([nothing], 3)

    # Penalty factors
    p_fX = zeros(k); p_fG = ones(p)

    # Sequence of λ
    rho = !isnothing(ind_D) ? rho : 0
    @assert all(0 .<= rho .< 1) "rho parameter must be in the range (0, 1]."
    x = length(rho)
    λ_seq = !isnothing(lambda) ? lambda : [lambda_seq(y - μ, X, G, D; p_fX = p_fX, p_fG = p_fG, rho = rho[j]) for j in 1:x]
   
    # Fit penalized model for each value of rho
    # λ_seq, path = Vector{typeof(μ)}(undef, x), Array{NamedTuple}(undef, x)
    # Threads.@threads for j in 1:x
    #        λ_seq[j] = lambda_seq(y - μ, X, G, D; p_fX = p_fX, p_fG = p_fG, rho = rho[j])
    #        path[j] = pglmm_fit(nullmodel.family, Ytilde, y, X, G, U, D, nulldev, r = Ytilde - nullmodel.η, μ, α = sparse(zeros(k)), β = sparse(zeros(p)), γ = sparse(zeros(p)), δ = U' * b, p_fX, p_fG, λ_seq[j], rho[j], nlambda, w, eigvals, verbose, criterion, earlystop, irls_tol, irls_maxiter, method)
    # end

    # Fit penalized model for each value of rho
    path = [pglmm_fit(nullmodel.family, Ytilde, y, X, G, U, D, nulldev, r = Ytilde - nullmodel.η, μ, α = sparse(zeros(k)), β = sparse(zeros(p)), γ = sparse(zeros(p)), δ = U' * b, p_fX, p_fG, λ_seq[j], rho[j], nlambda, w, eigvals, verbose, criterion, earlystop, irls_tol, irls_maxiter, method, upper_bound) for j in 1:x]

    # Separate intercept from coefficients
    a0, alphas = intercept ? ([path[j].alphas[1,:] for j in 1:x], [path[j].alphas[2:end,:] for j in 1:x]) : ([nothing for j in 1:x], [path[j].alphas for j in 1:x])
    betas = [path[j].betas for j in 1:x]
    gammas = !isnothing(D) ? [path[j].gammas for j in 1:x] : [nothing for j in 1:x]

    # Return coefficients on original scale
    if isnothing(gammas[1])
        if !isempty(sX) & !isempty(sG)
            [lmul!(inv(Diagonal(sX)), alphas[j]) for j in 1:x], [lmul!(inv(Diagonal(sG)), betas[j]) for j in 1:x]
        elseif !isempty(sX)
            [lmul!(inv(Diagonal(sX)), alphas[j]) for j in 1:x]
        elseif !isempty(sG)
            [lmul!(inv(Diagonal(sG)), betas[j]) for j in 1:x]
        end

        [a0[j] .-= spmul(muX, alphas[j]) + spmul(muG, betas[j]) for j in 1:x]
    else
        if !isempty(sX) & !isempty(sG)
            [lmul!(inv(Diagonal(sX)), alphas[j]) for j in 1:x], [lmul!(inv(Diagonal(sG)), betas[j]) for j in 1:x], [lmul!(inv(Diagonal(sD .* sG)), gammas[j]) for j in 1:x]
        elseif !isempty(sX)
            [lmul!(inv(Diagonal(sX)), alphas[j]) for j in 1:x], [lmul!(inv(Diagonal(sD .* ones(p))), gammas[j]) for j in 1:x]
        elseif !isempty(sG)
            [lmul!(inv(Diagonal(sG)), betas[j]) for j in 1:x], [lmul!(inv(Diagonal(sG)), gammas[j]) for j in 1:x]
        end

        [a0[j] .-= spmul(muX, alphas[j]) + spmul(muG, betas[j]) - spmul(muD .* muG, gammas[j]) for j in 1:x]
        [alphas[j][ind_D, :] -= spmul(muG, gammas[j])' for j in 1:x]; [betas[j] .-= muD' .* gammas[j] for j in 1:x]
    end

    # Return lasso path
    if !isnothing(ind_D)
        if length(rho) == 1
            pglmmPath(nullmodel.family, a0[1], alphas[1], betas[1], gammas[1], nulldev, path[1].pct_dev, path[1].λ, 0, path[1].fitted_values, y, UD_invUt, nullmodel.τ, intercept, rho[1])
        else
            [pglmmPath(nullmodel.family, a0[j], alphas[j], betas[j], gammas[j], nulldev, path[j].pct_dev, path[j].λ, 0, path[j].fitted_values, y, UD_invUt, nullmodel.τ, intercept, rho[j]) for j in 1:x]
        end
    else
        pglmmPath(nullmodel.family, a0[1], alphas[1], betas[1], gammas[1], nulldev, path[1].pct_dev, path[1].λ, 0, path[1].fitted_values, y, UD_invUt, nullmodel.τ, intercept, nothing)
    end
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
    G::Union{Matrix{T}, SubArray{T, 2, SnpLinAlg{T}, Tuple{Vector{Int64}, UnitRange{Int64}}}},
    U::Matrix{T},
    D::Nothing,
    nulldev::T,
    μ::Vector{T},
    p_fX::Vector{T},
    p_fG::Vector{T},
    λ_seq::Vector{T},
    rho::Real,
    nlambda::Int64,
    w::Vector{T},
    eigvals::Vector{T},
    verbose::Bool,
    criterion,
    earlystop::Bool,
    irls_tol::T,
    irls_maxiter::Int64,
    method,
    upper_bound::Bool;
    α::SparseVector{T},
    β::SparseVector{T},
    γ::SparseVector{T},
    δ::Vector{T},
    r::Vector{T}
) where T

    # Initialize array to store output for each λ
    alphas = spzeros(length(α), nlambda)
    betas = spzeros(length(β), nlambda)
    pct_dev = zeros(T, nlambda)
    dev_ratio = convert(T, NaN)
    fitted_means = zeros(length(y), nlambda)

    # Loop through sequence of λ
    i = 0
    for _ = 1:nlambda
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
            update_δ(Val(method); U = U, family = Binomial(), Ytilde = Ytilde, y = y, w = w, r = r, δ = δ, eigvals = eigvals, criterion = :coef, μ = μ)

            # Run coordinate descent inner loop to update β
            β_last = β
            Swxx, Swgg = cd_lasso(X, G, λ; family = Binomial(), Ytilde = Ytilde, y = y, w = w, r = r, α = α, β = β, δ = δ, p_fX = p_fX, p_fG = p_fG, eigvals = eigvals, criterion = criterion)

            # Update μ and w
            μ, w = updateμ(r, Ytilde)
            w = upper_bound ? repeat([0.25], length(μ)) : w

            # Update deviance and loss function
            prev_loss = loss
            dev = LogisticDeviance(δ, eigvals, y, μ)
            loss = dev/2 + last(λ) * P(α, β, p_fX, p_fG)
            
            # If loss function did not decrease, take a half step to ensure convergence
            if loss > prev_loss + length(μ)*eps(prev_loss)
                println("step-halving because loss=$loss > $prev_loss + $(length(μ)*eps(prev_loss)) = length(μ)*eps(prev_loss)")
                s = 1.0
                d = β - β_last
                while loss > prev_loss
                    s /= 2
                    β = β_last + s * d
                    μ, w = updateμ(r, Ytilde)
                    w = upper_bound ? repeat([0.25], length(μ)) : w 
                    dev = LogisticDeviance(δ, eigvals, y, μ)
                    loss = dev/2 + last(λ) * P(α, β, p_fX, p_fG)
                end 
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
        alphas[:, i] = α
        betas[:, i] = β
        dev_ratio = dev/nulldev
        pct_dev[i] = 1 - dev_ratio
        fitted_means[:, i] = μ

        # Test whether we should continue
        earlystop && ((last_dev_ratio - dev_ratio < MIN_DEV_FRAC_DIFF) || pct_dev[i] > MAX_DEV_FRAC) && break
    end

    return(alphas = alphas[:, 1:i], betas = betas[:, 1:i], pct_dev = pct_dev[1:i], λ = λ_seq[1:i], fitted_values = view(fitted_means, :, 1:i))
end

# Function to fit a penalized mixed model
function pglmm_fit(
    ::Binomial,
    Ytilde::Vector{T},
    y::Vector{Int64},
    X::Matrix{T},
    G::Union{Matrix{T}, SubArray{T, 2, SnpLinAlg{T}, Tuple{Vector{Int64}, UnitRange{Int64}}}},
    U::Matrix{T},
    D::Vector{T},
    nulldev::T,
    μ::Vector{T},
    p_fX::Vector{T},
    p_fG::Vector{T},
    λ_seq::Vector{T},
    rho::Real,
    nlambda::Int64,
    w::Vector{T},
    eigvals::Vector{T},
    verbose::Bool,
    criterion,
    earlystop::Bool,
    irls_tol::T,
    irls_maxiter::Int64,
    method,
    upper_bound::Bool;
    α::SparseVector{T},
    β::SparseVector{T},
    γ::SparseVector{T},
    δ::Vector{T},
    r::Vector{T}
) where T

    # Initialize array to store output for each λ
    alphas = spzeros(length(α), nlambda)
    betas = spzeros(length(β), nlambda)
    gammas = spzeros(length(β), nlambda)
    pct_dev = zeros(T, nlambda)
    dev_ratio = convert(T, NaN)
    fitted_means = zeros(length(y), nlambda)

    # Loop through sequence of λ
    i = 0
    for _ = 1:nlambda
        # Next iterate
        i += 1
        converged = false
        
        # Current value of λ
        λ = λ_seq[i]
        dλ = 2 * λ_seq[i] - λ_seq[max(1, i-1)]

        # Check strong rule
        compute_strongrule(dλ, λ_seq[max(1, i-1)], rho, p_fX, p_fG, D, α = α, β = β, γ = γ, X = X, G = G, y = y, μ = μ)

        # Initialize objective function and save previous deviance ratio
        dev = loss = convert(T, Inf)
        last_dev_ratio = dev_ratio

        # Iterative weighted least squares (IRLS)
        for irls in 1:irls_maxiter

            # Update random effects vector δ
            update_δ(Val(method); U = U, family = Binomial(), Ytilde = Ytilde, y = y, w = w, r = r, δ = δ, eigvals = eigvals, criterion = :coef, μ = μ)

            # Run coordinate descent inner loop to update β
            β_last = β
            Swxx, Swgg, Swdg = cd_lasso(D, X, G, λ, rho; family = Binomial(), Ytilde = Ytilde, y = y, w = w, r = r, α = α, β = β, δ = δ, γ = γ, p_fX = p_fX, p_fG = p_fG, eigvals = eigvals, criterion = criterion)

            # Update μ and w
            μ, w = updateμ(r, Ytilde)
            w = upper_bound ? repeat([0.25], length(μ)) : w

            # Update deviance and loss function
            prev_loss = loss
            dev = LogisticDeviance(δ, eigvals, y, μ)
            loss = dev/2 + last(λ) * P(α, β, γ, p_fX, p_fG, rho)
            
            # If loss function did not decrease, take a half step to ensure convergence
            if loss > prev_loss + length(μ)*eps(prev_loss)
                println("step-halving because loss=$loss > $prev_loss + $(length(μ)*eps(prev_loss)) = length(μ)*eps(prev_loss)")
                s = 1.0
                d = β - β_last
                while loss > prev_loss
                    s /= 2
                    β = β_last + s * d
                    μ, w = updateμ(r, Ytilde)
                    w = upper_bound ? repeat([0.25], length(μ)) : w 
                    dev = LogisticDeviance(δ, eigvals, y, μ)
                    loss = dev/2 + last(λ) * P(α, β, γ, p_fX, p_fG, rho)
                end
            end 

            # Update working response and residuals
            Ytilde, r = wrkresp(y, μ, w)

            # Check termination conditions
            converged = abs(loss - prev_loss) < irls_tol * loss
            
            # Check KKT conditions at last iteration
            if converged
                maxΔ, converged = cycle(D, X, G, λ, rho, Val(true), r = r, α = α, β = β, γ = γ, Swxx = Swxx, Swgg = Swgg, Swdg = Swdg, w = w, p_fX = p_fX, p_fG = p_fG)
            end
            converged && verbose && println("Number of irls iterations = $irls at $i th value of λ.")
            converged && break  
        end
        @assert converged "IRLS failed to converge in $irls_maxiter iterations at λ = $λ"

        # Store ouput from irls loop
        alphas[:, i] = α
        betas[:, i] = β
        gammas[:, i] = γ
        dev_ratio = dev/nulldev
        pct_dev[i] = 1 - dev_ratio
        fitted_means[:, i] = μ

        # Test whether we should continue
        earlystop && ((last_dev_ratio - dev_ratio < MIN_DEV_FRAC_DIFF) || pct_dev[i] > MAX_DEV_FRAC) && break
    end

    return(alphas = alphas[:, 1:i], betas = betas[:, 1:i], gammas = gammas[:, 1:i], pct_dev = pct_dev[1:i], λ = λ_seq[1:i], fitted_values = view(fitted_means, :, 1:i))
end

# Function to perform coordinate descent with a lasso penalty
function cd_lasso(
    # positional arguments
    X::Matrix{T},
    G::Union{Matrix{T}, SubArray{T, 2, SnpLinAlg{T}, Tuple{Vector{Int64}, UnitRange{Int64}}}},
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
    G::Union{Matrix{T}, SubArray{T, 2, SnpLinAlg{T}, Tuple{Vector{Int64}, UnitRange{Int64}}}},
    λ::T,
    rho::Real;
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
        maxΔ, = cycle(D, X, G, λ, rho, Val(false), r = r, α = α, β = β, γ = γ, Swxx = Swxx, Swgg = Swgg, Swdg = Swdg, w = w, p_fX = p_fX, p_fG = p_fG)

        # Check termination condition before last iteration
        if criterion == :obj
            # Update μ
            μ, = updateμ(r, Ytilde)

            # Update deviance and loss function
            prev_loss = loss
            dev = model_dev(family, δ, w, r, eigvals, y, μ)
            loss = dev/2 + λ * P(α, β, γ, p_fX, p_fG, rho)

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
    U::Matrix{T};
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
        maxΔ = cycle(U, r = r, δ = δ, Swuu = Swuu, w = w, eigvals = eigvals)

        # Check termination condition before last iteration
        if criterion == :coef
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
    G::Union{Matrix{T}, SubArray{T, 2, SnpLinAlg{T}, Tuple{Vector{Int64}, UnitRange{Int64}}}},
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
    G::Union{Matrix{T}, SubArray{T, 2, SnpLinAlg{T}, Tuple{Vector{Int64}, UnitRange{Int64}}}},
    λ::T,
    rho::Real,
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
    for j in β.nzind
        λj = λ * p_fG[j]

        # Update GEI effect
        last_γ, last_β = γ[j], β[j]
        v = compute_grad(D, G, w, r, j) + last_γ * Swdg[j]
        if abs(v) > rho * λj
            new_γ = softtreshold(v, rho * λj) / (Swdg[j] + (1 - rho) * λj / norm((last_γ, last_β)))
            r = update_r(D, G, r, last_γ - new_γ, j)

            maxΔ = max(maxΔ, Swdg[j] * (last_γ - new_γ)^2)
            γ[j] = new_γ
        end

        # Update genetic effect
        v = compute_grad(G, w, r, j) + last_β * Swgg[j]
        new_β = γ[j] != 0 ? v / (Swgg[j] + (1 - rho) * λj / norm((last_γ, last_β))) : softtreshold(v, (1 - rho) * λj) / Swgg[j]
        r = update_r(G, r, last_β - new_β, j)

        maxΔ = max(maxΔ, Swgg[j] * (last_β - new_β)^2)
        β[j] = new_β

    end

    maxΔ
end

function cycle(
    # positional arguments
    X::Matrix{T},
    G::Union{Matrix{T}, SubArray{T, 2, SnpLinAlg{T}, Tuple{Vector{Int64}, UnitRange{Int64}}}},
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
    G::Union{Matrix{T}, SubArray{T, 2, SnpLinAlg{T}, Tuple{Vector{Int64}, UnitRange{Int64}}}},
    λ::T,
    rho::Real,
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
    for j in 1:length(β)
        λj = λ * p_fG[j]
        v1 = compute_grad(G, w, r, j)
        v2 = compute_grad(D, G, w, r, j)

        if j in β.nzind
            last_β = β[j]
            v1 += last_β * Swgg[j]
        else
            # Adding a new variable to the model
            norm([v1, softtreshold(v2, rho * λj)]) <= (1 - rho) * λj && continue
            kkt_check = false
            last_β, β[j] = 0, 1
            Swgg[j] = compute_Swxx(G, w, j)
        end

        if j in γ.nzind
            last_γ = γ[j]
            v2 += last_γ * Swdg[j]
        else
            # Adding a new variable to the model
            if v2 != zero(v2)
                kkt_check = false
                last_γ, γ[j] = 0, 1
                Swdg[j] = compute_Swxx(D, G, w, j)
            else
                # Update β only
                new_β = softtreshold(v1, (1 - rho) * λj) / Swgg[j]
                r = update_r(G, r, last_β - new_β, j)

                maxΔ = max(maxΔ, Swgg[j] * (last_β - new_β)^2)
                β[j] = new_β
                continue
            end
        end

        # Update β and γ
        new_γ = softtreshold(v2, rho * λj) / (Swdg[j] + (1 - rho) * λj / norm((last_γ, last_β)))
        r = update_r(D, G, r, last_γ - new_γ, j)

        v = compute_grad(G, w, r, j) + last_β * Swgg[j]
        new_β = v / (Swgg[j] + (1 - rho) * λj / norm((last_γ, last_β)))
        r = update_r(G, r, last_β - new_β, j)

        maxΔ = max(maxΔ, Swgg[j] * (last_β - new_β)^2, Swdg[j] * (last_γ - new_γ)^2)
        β[j], γ[j] = new_β, new_γ

    end

    return(maxΔ, kkt_check)
end

function cycle(
    # positional arguments
    U::Matrix{T};
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

    cd_lasso(U; family = family, Ytilde = Ytilde, y = y, w = w, r = r, δ = δ, eigvals = eigvals, criterion = criterion)
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

mutable struct pglmmPath{F<:Distribution, A<:AbstractArray, B<:AbstractArray, T<:AbstractFloat, D<:AbstractArray, E<:AbstractArray}
    family::F
    a0::A                                       # intercept values for each solution
    alphas::B                                   # coefficient values for each solution
    betas::B                                
    gammas::Union{Nothing, B}
    null_dev::T                                 # Null deviance of the model
    pct_dev::D                                 # R^2 values for each solution
    lambda::D                                   # lamda values corresponding to each solution
    npasses::Int                                # actual number of passes over the data for all lamda values
    fitted_values                               # fitted_values
    y::Union{Vector{Int}, D}                    # eigenvalues vector
    UD_invUt::E                                # eigenvectors matrix times diagonal weights matrix
    τ::D                                        # estimated variance components
    intercept::Bool                             # boolean for intercept
    rho::Union{Nothing, Real}                   # rho tuninng parameter
end

function show(io::IO, g::pglmmPath)
    if isnothing(g.gammas)
        df = [length(findall(x -> x != 0, vec(view([g.alphas; g.betas], :,k)))) for k in 1:size(g.betas, 2)]
        println(io, "$(modeltype(g.family)) Solution Path ($(size(g.betas, 2)) solutions for $(size([g.alphas; g.betas], 1)) predictors):") #in $(g.npasses) passes):"
    else
        df = [length(findall(x -> x != 0, vec(view([g.alphas; g.betas; g.gammas], :,k)))) for k in 1:size(g.betas, 2)]
        println(io, "$(modeltype(g.family)) Solution Path ($(size(g.betas, 2)) solutions for $(size([g.alphas; g.betas; g.gammas], 1)) predictors):") #in $(g.npasses) passes):"
    end

    if !isnothing(g.rho) 
        print(io, CoefTable(Union{Vector{Int},Vector{Float64}}[df, g.pct_dev, g.lambda, g.rho * ones(length(g.lambda))], ["df", "pct_dev", "λ", "ρ"], []))
    else 
        print(io, CoefTable(Union{Vector{Int},Vector{Float64}}[df, g.pct_dev, g.lambda], ["df", "pct_dev", "λ"], []))
    end

end

# Function to compute sequence of values for λ
function lambda_seq(
    r::Vector{T}, 
    X::Matrix{T},
    G::Union{Matrix{T}, SubArray{T, 2, SnpLinAlg{T}, Tuple{Vector{Int64}, UnitRange{Int64}}}},
    D::Union{Vector{T}, Nothing}; 
    p_fX::Vector{T},
    p_fG::Vector{T},
    rho::Real,
    nlambda::Integer = 100
    ) where T

    λ_min_ratio = (length(r) < size(G, 2) ? 1e-2 : 1e-4)
    λ_max = lambda_max(nothing, X, r, p_fX)
    λ_max = lambda_max(D, G, r, p_fG, λ_max, rho = rho)
    λ_min = λ_max * λ_min_ratio
    λ_step = log(λ_min_ratio)/(nlambda - 1)
    λ_seq = exp.(collect(log(λ_max+100*eps(λ_max)):λ_step:log(λ_min)))

    λ_seq
end

# Function to compute λ_max for the lasso
function lambda_max(D::Nothing, X::AbstractMatrix{T}, r::AbstractVector{T}, p_f::AbstractVector{T}, λ_max::T = zero(T); kwargs...) where T
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
function lambda_max(D::AbstractVector{T}, X::AbstractMatrix{T}, r::AbstractVector{T}, p_f::AbstractVector{T}, λ_max::T = zero(T); rho::Real) where T

    seq = findall(!iszero, p_f)
    for j in seq
        x = compute_max(D, X, r, j, rho)
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
model_dev(::Binomial, b::Vector{T}, τV::Matrix{T}, y::Vector{Int64}, μ::Vector{Float64}) where T = LogisticDeviance(b, τV, y, μ)
model_dev(::Normal, δ::Vector{T}, w::T, r::Vector{T}, eigvals::Vector{T}, kargs...) where T = NormalDeviance(δ, w, r, eigvals)

function LogisticDeviance(δ::Vector{T}, eigvals::Vector{T}, y::Vector{Int64}, μ::Vector{T}) where T
    -2 * sum(y .* log.(μ ./ (1 .- μ)) .+ log.(1 .- μ)) + dot(δ, Diagonal(eigvals), δ)
end

function LogisticDeviance(b::Vector{T}, τV::Matrix{T}, y::Vector{Int64}, μ::Vector{T}) where T
    -2 * sum(y .* log.(μ ./ (1 .- μ)) .+ log.(1 .- μ)) + dot(b, inv(τV), b)
end

function NormalDeviance(δ::Vector{T}, w::T, r::Vector{T}, eigvals::Vector{T}) where T
    w * dot(r, r) + dot(δ, Diagonal(eigvals), δ)
end

# Predict phenotype
function predict(path, 
                  covfile::AbstractString,
                  grmfile::AbstractString,
                  plinkfile::Union{Nothing, AbstractString} = nothing;
                  # keyword arguments
                  snpfile::Union{Nothing, AbstractString} = nothing,
                  snpmodel = ADDITIVE_MODEL,
                  snpinds::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                  covrowinds::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                  covrowtraininds::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                  covars::Union{Nothing,AbstractVector{<:String}} = nothing, 
                  geneticrowinds::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                  grmrowinds::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                  grmcolinds::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                  M::Union{Nothing, Vector{Any}} = nothing,
                  s::Union{T, Vector{T}, Nothing} = nothing,
                  fixed_effects_only::Bool = false,
                  GEIvar::Union{Nothing,AbstractString} = nothing,
                  GEIkin::Bool = true,
                  outtype = :response
                 ) where T

    if isnothing(s)
        [predict(path[j], 
                  covfile,
                  grmfile,
                  plinkfile;
                  snpfile = snpfile,
                  snpmodel = snpmodel,
                  snpinds = snpinds,
                  covrowinds = covrowinds,
                  covrowtraininds = covrowtraininds,
                  covars = covars, 
                  geneticrowinds = geneticrowinds,
                  grmrowinds = grmrowinds,
                  grmcolinds = grmcolinds,
                  M = M,
                  s = 1:size(path[j].betas, 2),
                  fixed_effects_only = fixed_effects_only,
                  GEIvar = GEIvar,
                  GEIkin = GEIkin,
                  outtype = outtype
                 ) for j in 1:length(path)] |> x-> reduce(hcat,x)
    else
        [predict(path[s[j].rho.index], 
                  covfile,
                  grmfile,
                  plinkfile;
                  snpfile = snpfile,
                  snpmodel = snpmodel,
                  snpinds = snpinds,
                  covrowinds = covrowinds,
                  covrowtraininds = covrowtraininds,
                  covars = covars, 
                  geneticrowinds = geneticrowinds,
                  grmrowinds = grmrowinds,
                  grmcolinds = grmcolinds,
                  M = M,
                  s = s[j].lambda.index,
                  fixed_effects_only = fixed_effects_only,
                  GEIvar = GEIvar,
                  GEIkin = GEIkin,
                  outtype = outtype
                 ) for j in 1:length(s)] |> x-> reduce(hcat,x)
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
                  covrowtraininds::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                  covars::Union{Nothing,AbstractVector{<:String}} = nothing, 
                  geneticrowinds::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                  grmrowinds::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                  grmcolinds::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                  M::Union{Nothing, Vector{Any}} = nothing,
                  s::Union{Nothing,<:Integer,AbstractVector{<:Integer}} = nothing,
                  fixed_effects_only::Bool = false,
                  GEIvar::Union{Nothing,AbstractString} = nothing,
                  GEIkin::Bool = true,
                  outtype = :response
                 )
    
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
        
        # Read genotype
        G = SnpLinAlg{Float64}(geno, model = snpmodel, impute = true, center = false, scale = false) |> x-> @view(x[geneticrowinds_, snpinds_])

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
    # Create list of similarity matrices
    V = push!(Any[], GRM)

    # Add GEI similarity matrix
    if !isnothing(GEIvar)
        D = covdf[:, GEIvar]
        if GEIkin
            @assert length(path.τ) >= 2 "Only one variance component has been estimated under the null model."
        Dtrain = CSV.read(covfile, DataFrame)[covrowtraininds, GEIvar]
            V_D = D * Dtrain'
            for j in findall(x -> x == 0, Dtrain), i in findall(x -> x == 0, D)  
                    V_D[i, j] = 1 
            end
            push!(V, sparse(GRM .* V_D))
        end
    end

    # Add variance components in the model
    if !isnothing(M) 
        [push!(V, M[i]) for i in 1:length(M)] 
    end

    # Covariance matrix between test and training subjects
    Σ_12 = sum(path.τ .* V)

    # Number of predictions to compute. User can provide index s for which to provide predictions, 
    # rather than computing predictions for the whole path.
    s = isnothing(s) ? (1:size(path.betas, 2)) : s

    # Linear predictor
    η = path.a0[s]' .+ X * path.alphas[:,s] .+ G * path.betas[:,s]

    if !isnothing(GEIvar)
        η += (D .* G) * path.gammas[:,s]
    end

    if fixed_effects_only == false
        if path.family == Binomial()
            b = Σ_12 * (path.y .- path.fitted_values[:,s])
        elseif path.family == Normal()
            b = Σ_12 * path.UD_invUt * path.fitted_values[:,s]
        end
        η += b
    end

    # Return linear predictor (default), fitted probs or random effects
    if outtype == :response
        return(η)
    elseif outtype == :prob
        return(GLM.linkinv.(LogitLink(), η))
    elseif outtype == :random
        return(b)
    end
end 

# Define a structure for the tuning parameters tuple
struct TuningParms
  val::Real
  index::Int
end

Base.show(io::IO, g::TuningParms) = print(io, "value = $(g.val), index = $(g.index)")

# Define a structure for the GIC output tuple
struct GICTuple{T<:AbstractFloat}
  rho::TuningParms
  lambda::TuningParms
  GIC::T
end

Base.show(io::IO, g::GICTuple) = print(io, "rho = $(g.rho.val), lambda = $(g.lambda.val), GIC = $(g.GIC)")

# GIC penalty parameter
function GIC(path, criterion)
    a = [GIC(path[j], criterion, return_val = true) for j in 1:length(path)]
    j = argmin(getproperty.(a, :GIC))
    rho = path[j].rho
    jj = a[j].index
    lambda = path[j].lambda[jj]

    GICTuple(TuningParms(rho, j), TuningParms(lambda, jj), a[j].GIC)
end

function GIC(path::pglmmPath, criterion; return_val = false)
    
    # Obtain number of rows (n), predictors (p) and λ values (nlambda)
    n = size(path.y, 1)
    m, (p, nlambda) = size(path.alphas, 1), size(path.betas)
    df = path.intercept .+ [length(findall(x -> x != 0, vec(view([path.alphas; path.betas], :, k)))) for k in 1:nlambda] .+ length(path.τ)
    df += !isnothing(path.gammas) ? [length(findall(x -> x != 0, vec(view(path.gammas, :, k)))) for k in 1:nlambda] : zero(df)

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
    if return_val
        return(index = argmin(GIC), GIC = GIC[argmin(GIC)], rho = path.rho)
    else
        argmin(GIC)
    end
end

# Standardize predictors for lasso
function standardizeX(X::AbstractMatrix{T}, standardize::Bool, intercept::Bool = false) where T
    mu = intercept ? vec([0 mean(X[:,2:end], dims = 1)]) : vec(mean(X, dims = 1))
    Xs = zero(X)
    if standardize
        s = intercept ? vec([1 std(X[:,2:end], dims = 1, corrected = false)]) : vec(std(X, dims = 1, corrected = false)) 
        if any(s .== zero(T))
            @warn("One predictor is a constant, hence it can't been standardized!")
            s[s .== 0] .= 1 
        end
        for j in 1:size(X,2), i in 1:size(X, 1) 
            @inbounds Xs[i,j] = (X[i,j] .- mu[j]) / s[j]
        end
    else
        for j in 1:size(X,2), i in 1:size(X, 1) 
            @inbounds Xs[i,j] = X[i,j] .- mu[j]
        end
        s = []
    end

    # Remove first term if intercept
    if intercept 
         popfirst!(mu); popfirst!(s)
    end

    Xs, mu, s
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

function compute_max(D::AbstractVector{T}, X::AbstractMatrix{T}, r::AbstractVector{T}, whichcol::Int, rho::Real) where T
    v = zeros(2)
    for i = 1:size(X, 1)
        @inbounds v[1] += X[i, whichcol] * r[i]
        @inbounds v[2] += D[i] * X[i, whichcol] * r[i]
    end

    if rho == 1
        abs(v[2])
    else
        norm(v) / (1 - rho)
    end
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

function P(α::SparseVector{T}, β::SparseVector{T}, γ::SparseVector{T}, p_fX::Vector{T}, p_fG::Vector{T}, rho::Real) where T
    x = zero(T)
    @inbounds @simd for i in α.nzind
            x += p_fX[i] * abs(α[i])
    end
    @inbounds @simd for i in β.nzind
            x += p_fG[i] * ((1 - rho) * norm((β[i], γ[i])) + rho * abs(γ[i]))
    end
    x
end

# Compute strongrule for the lasso
function compute_strongrule(dλ::T, p_fX::Vector{T}, p_fG::Vector{T}; α::SparseVector{T}, β::SparseVector{T}, X::Matrix{T}, G::Union{Matrix{T},SubArray{T, 2, SnpLinAlg{T}, Tuple{Vector{Int64}, UnitRange{Int64}}}}, y::Vector{Int}, μ::Vector{T}) where T
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
function compute_strongrule(dλ::T, λ::T, rho::Real, p_fX::Vector{T}, p_fG::Vector{T}, D::Vector{T}; α::SparseVector{T}, β::SparseVector{T}, γ::SparseVector{T}, X::Matrix{T}, G::Union{Matrix{T},SubArray{T, 2, SnpLinAlg{T}, Tuple{Vector{Int64}, UnitRange{Int64}}}}, y::Vector{Int}, μ::Vector{T}) where T
    for j in 1:length(α)
        j in α.nzind && continue
        c = compute_prod(X, y, μ, j)
        abs(c) <= dλ * p_fX[j] && continue
        
        # Force a new variable to the model
        α[j] = 1; α[j] = 0
    end
    
    for j in 1:length(β)
        j in β.nzind && continue
        c1 = compute_prod(G, y, μ, j)
        c2 = softtreshold(compute_prod(D, G, y, μ, j), rho * λ * p_fG[j])
        norm([c1, c2]) <= (1 - rho) * dλ * p_fG[j] && continue
        
        # Force a new group to the model
        β[j] = 1; β[j] = 0
    end
end

# Function to compute product of sparse Matrix A with vector b => x = b' * A
function spmul(b::Vector{T}, A::SparseMatrixCSC) where T
    rows, cols, vals = findnz(A)
    x = zeros(size(A, 2))

    for j = 1:size(A, 2)
        v = zero(T)
        for i in findall(cols .== j)
            @inbounds v += b[rows[i]] * vals[i]
        end
        x[j] = v
    end
    x
end