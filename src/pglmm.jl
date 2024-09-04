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
    upper_bound::Bool = false,
    penalty_factor_X::Union{Nothing, Vector{T}} = nothing,
    penalty_factor_G::Union{Nothing, Vector{T}} = nothing,
    kwargs...
    ) where T

    # # keyword arguments
    # snpmodel = ADDITIVE_MODEL
    # # snpinds = nothing
    # irls_tol = 1e-7
    # irls_maxiter = 500
    # nlambda = 100
    # rho = collect(0:0.1:0.5)
    # verbose = true
    # standardize_X = true
    # standardize_G = true
    # criterion = :coef
    # earlystop= true
    # GEIvar = nothing
    # upper_bound = false
    # lambda = nothing
    # T = Float64

    # Read genotype file
    if !isnothing(plinkfile)

        # read PLINK files
        geno = SnpArray(plinkfile * ".bed")
        snpinds_ = isnothing(snpinds) ? (1:size(geno, 2)) : snpinds 
        geneticrowinds_ = isnothing(geneticrowinds) ? (1:size(geno, 1)) : geneticrowinds

        # Read genotype and calculate mean and standard deviation
        G = SnpLinAlg{Float64}(geno, model = snpmodel, impute = true, center = true, scale = standardize_G) |> x-> @view(x[Int.(nullmodel.L * geneticrowinds_), snpinds_])
        muG, sG = standardizeG(@view(geno[geneticrowinds_, snpinds_]), snpmodel, standardize_G)

    elseif !isnothing(snpfile)

        # read CSV file
        geno = CSV.read(snpfile, DataFrame)
        
        # Convert genotype file to matrix, convert to additive model (default) and impute
        snpinds_ = isnothing(snpinds) ? (1:size(geno, 2)) : snpinds 
        geneticrowinds_ = isnothing(geneticrowinds) ? (1:size(geno, 1)) : geneticrowinds

        G = convert.(Float64, Matrix(geno[Int.(nullmodel.L * geneticrowinds_), snpinds_]))

        # standardize genetic predictors
        G, muG, sG = standardizeX(G, standardize_G)
    end

    # Initialize number of subjects and predictors (including intercept)
    (n, p), k = size(G), size(nullmodel.X, 2)
    @assert n == length(nullmodel.y) "Genotype matrix and y must have same number of rows"

    # Spectral decomposition of τV
    eigvals, U = eigen(nullmodel.τV)

    if length(nullmodel.D) > 0
        # Compute eigenvalues and eigenvectors and add them to existing ones
        append!(eigvals, eigenkron(nullmodel.D, size(nullmodel.τV, 1)).values)
        U = BlockDiagonal([U, eigenkron(nullmodel.D, size(nullmodel.τV, 1)).vectors])

        #Sort by ascending eigenvalues
        ascorder = sortperm(eigvals)
        eigvals = eigvals[ascorder]
        U = Matrix(U)[:, ascorder]
    end

    U = sparse(U)
    eigvals .= nullmodel.φ ./ eigvals

    # Initialize random effects vector and rotate design matrix H
    b = nullmodel.b
    UH = nullmodel.H * U

    # Initialize working variable
    y = nullmodel.y
    if nullmodel.family == Binomial()
        μ, ybar = GLM.linkinv.(LogitLink(), nullmodel.η), mean(y)
        w = upper_bound ? repeat([0.25], length(y)) : μ .* (1 .- μ)
        Ytilde = nullmodel.η + (y - μ) ./ w
        nulldev = -2 * sum(y * log(ybar / (1 - ybar)) .+ log(1 - ybar))
    elseif nullmodel.family == Normal()
        Ytilde, μ = y, nullmodel.η
        w = one.(Ytilde)
        nulldev = sum((y .- mean(y)).^2) / nullmodel.φ
    end

    # standardize non-genetic covariates
    intercept = all(nullmodel.X[:,1] .== 1)
    X, muX, sX, α = standardizeX(nullmodel.X, standardize_X, nullmodel.α, intercept)
    ind_E = !isnothing(nullmodel.ind_E) ? nullmodel.ind_E .- intercept : nothing
    E, muE, sE = !isnothing(ind_E) ? (vec(X[:, nullmodel.ind_E]), muX[ind_E], sX[ind_E]) : repeat([nothing], 3)

    # Penalty factors
    if isnothing(penalty_factor_X)
        p_fX = zeros(k)
    else
        p_fX = penalty_factor_X
    end 

    if isnothing(penalty_factor_G)
        p_fG = ones(p)
    else
        p_fG = penalty_factor_G
    end 

    # Sequence of λ
    rho = !isnothing(ind_E) ? rho : 0
    @assert all(0 .<= rho .< 1) "rho parameter must be in the range (0, 1]."
    x = length(rho)
    λ_seq = !isnothing(lambda) ? lambda : [lambda_seq(y - μ, X, G, E; p_fX = p_fX, p_fG = p_fG, rho = rho[j]) for j in 1:x]

    # Fit penalized model for each value of rho
    # λ_seq, path = Vector{typeof(μ)}(undef, x), Array{NamedTuple}(undef, x)
    # Threads.@threads for j in 1:x
    #        λ_seq[j] = lambda_seq(y - μ, X, G, E; p_fX = p_fX, p_fG = p_fG, rho = rho[j])
    #        path[j] = pglmm_fit(nullmodel.family, Ytilde, y, X, G, U, E, nulldev, r = Ytilde - nullmodel.η, μ, α = sparse(zeros(k)), β = sparse(zeros(p)), γ = sparse(zeros(p)), δ = U' * b, p_fX, p_fG, λ_seq[j], rho[j], nlambda, w, eigvals, verbose, criterion, earlystop, irls_tol, irls_maxiter)
    # end

    # !!!!!!! To erase !!!!!!
    # r = Ytilde - nullmodel.η; α = sparse(α); β = sparse(zeros(p)); γ = sparse(zeros(p)); δ = U'b; U = UH; λ_seq = λ_seq[1]; rho = rho[1]; phi = nullmodel.φ

    # Fit penalized model for each value of rho
    path = [pglmm_fit(nullmodel.family, Ytilde, y, X, G, UH, E, nulldev, r = Ytilde - nullmodel.η, μ, α = sparse(α), β = sparse(zeros(p)), γ = sparse(zeros(p)), δ = U'b, nullmodel.φ, p_fX, p_fG, λ_seq[j], rho[j], nlambda, w, eigvals, verbose, criterion, earlystop, irls_tol, irls_maxiter, upper_bound) for j in 1:x]

    # Separate intercept from coefficients
    a0, alphas = intercept ? ([path[j].alphas[1,:] for j in 1:x], [path[j].alphas[2:end,:] for j in 1:x]) : ([nothing for j in 1:x], [path[j].alphas for j in 1:x])
    betas = [path[j].betas for j in 1:x]
    gammas = !isnothing(E) ? [path[j].gammas for j in 1:x] : [nothing for j in 1:x]

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
        [alphas[j][ind_E, :] -= spmul(muG, gammas[j])' for j in 1:x]; [betas[j] .-= muD' .* gammas[j] for j in 1:x]
    end

    # Return lasso path
    if !isnothing(ind_E)
        if length(rho) == 1
            pglmmPath(nullmodel.family, a0[1], alphas[1], betas[1], gammas[1], nulldev, path[1].pct_dev, path[1].λ, 0, path[1].fitted_values, y, nullmodel.φ, nullmodel.τ, intercept, rho[1], nullmodel.D)
        else
            [pglmmPath(nullmodel.family, a0[j], alphas[j], betas[j], gammas[j], nulldev, path[j].pct_dev, path[j].λ, 0, path[j].fitted_values, y, nullmodel.φ, nullmodel.τ, intercept, rho[j], nullmodel.D) for j in 1:x]
        end
    else
        pglmmPath(nullmodel.family, a0[1], alphas[1], betas[1], gammas[1], nulldev, path[1].pct_dev, path[1].λ, 0, path[1].fitted_values, y, nullmodel.φ, nullmodel.τ, intercept, nothing, nullmodel.D)
    end
end

# Controls early stopping criteria with automatic λ
const MIN_DEV_FRAC_DIFF = 1e-5
const MAX_DEV_FRAC = 0.999

# Function to fit a lasso penalized mixed model for a binary trait
function pglmm_fit(
    ::Binomial,
    Ytilde::Vector{T},
    y::Vector{Int},
    X::Matrix{T},
    G::AbstractMatrix{T},
    U::AbstractMatrix{T},
    E::Nothing,
    nulldev::T,
    μ::Vector{T},
    phi::T,
    p_fX::Vector{T},
    p_fG::Vector{T},
    λ_seq::Vector{T},
    rho::Real,
    nlambda::Int,
    w::Vector{T},
    eigvals::Vector{T},
    verbose::Bool,
    criterion,
    earlystop::Bool,
    irls_tol::T,
    irls_maxiter::Int,
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
        nzαind, nzβind = compute_strongrule(dλ, p_fX, p_fG, α = α, β = β, X = X, G = G, y = y, μ = μ)

        # Initialize objective function and save previous deviance ratio
        dev = loss = convert(T, Inf)
        last_dev_ratio = dev_ratio

        # Iterative weighted least squares (IRLS)
        for irls in 1:irls_maxiter

            # Update random effects vector δ
            update_δ(Binomial(), U = U, Ytilde = Ytilde, y = y, w = w, r = r, δ = δ, eigvals = eigvals, criterion = :coef, μ = μ)

            # Run coordinate descent inner loop to update β
            β_last = β
            Swxx, Swgg = cd_lasso(Binomial(), X, G, λ; Ytilde = Ytilde, y = y, w = w, r = r, α = α, β = β, δ = δ, p_fX = p_fX, p_fG = p_fG, eigvals = eigvals, criterion = criterion, phi = phi)

            # Update μ and w
            μ, w = updateμ(Binomial(), r, Ytilde)
            w = upper_bound ? repeat([0.25], length(μ)) : w

            # Update deviance and loss function
            prev_loss = loss
            dev = LogisticDeviance(δ, eigvals, y, μ)
            loss = dev/2 + last(λ) * P(α, β, p_fX, p_fG)
            
            # # If loss function did not decrease, take a half step to ensure convergence
            # if loss > prev_loss + length(μ)*eps(prev_loss)
            #     println("step-halving because loss=$loss > $prev_loss + $(length(μ)*eps(prev_loss)) = length(μ)*eps(prev_loss)")
            #     s = 1.0
            #     d = β - β_last
            #     while loss > prev_loss + length(μ)*eps(prev_loss)
            #         s /= 2
            #         β = β_last + s * d
            #         r = update_r(G, r, β_last - β)
            #         μ, w = updateμ(Binomial(), r, Ytilde)
            #         w = upper_bound ? repeat([0.25], length(μ)) : w 
            #         dev = LogisticDeviance(δ, eigvals, y, μ)
            #         loss = dev/2 + last(λ) * P(α, β, p_fX, p_fG)
            #     end 
            # end 

            # Update working response and residuals
            Ytilde, r = wrkresp(y, μ, w)

            # Check termination conditions
            converged = abs(loss - prev_loss) < irls_tol * loss
            
            # At last iteration, check KKT conditions on the strong set 
            if converged
                verbose && println("Checking KKT conditions on the strong set.")
                converged = cycle(X, G, λ, Val(true), r = r, α = α, β = β, Swxx = Swxx, Swgg = Swgg, w = w, p_fX = p_fX, p_fG = p_fG, nzαind = nzαind, nzβind = nzβind)
                !converged && verbose && println("KKT conditions not met, refitting the model.")
            end

            # Then, check KKT conditions on all predictors
            if converged
                verbose && println("Checking KKT conditions on all predictors.")
                converged = cycle(X, G, λ, Val(true), r = r, α = α, β = β, Swxx = Swxx, Swgg = Swgg, w = w, p_fX = p_fX, p_fG = p_fG)

                if !converged 
                    # Recalculate strong rule
                    verbose && println("KKT conditions not met, updating strong set and refitting the model.")
                    nzαind, nzβind = compute_strongrule(dλ, p_fX, p_fG, α = α, β = β, X = X, G = G, y = y, μ = μ)
                end
            end

            converged && verbose && println("Number of irls iterations = $irls at $i th value of λ.")
            converged && verbose && println("The number of active predictors is equal to $(sum(α .!= 0) + sum(β .!= 0) - 1).")
            converged && verbose && println("---------------------------------------------------")
            converged && break  
        end
        @assert converged "IRLS failed to converge in $irls_maxiter iterations at λ = $λ"

        # Store ouput from irls loop
        copyto!(alphas, 1:length(α), i:i, α, 1:length(α), 1:1)
        copyto!(betas, 1:length(β), i:i, β, 1:length(β), 1:1)
        dev_ratio = dev/nulldev
        copyto!(pct_dev, i, 1 - dev_ratio)
        copyto!(fitted_means, 1:length(μ), i:i, μ, 1:length(μ), 1:1)

        # Test whether we should continue
        earlystop && ((last_dev_ratio - dev_ratio < MIN_DEV_FRAC_DIFF) || pct_dev[i] > MAX_DEV_FRAC) && break
    end

    return(alphas = alphas[:, 1:i], betas = betas[:, 1:i], pct_dev = pct_dev[1:i], λ = λ_seq[1:i], fitted_values = view(fitted_means, :, 1:i))
end

# Function to fit a lasso penalized mixed model for a continous trait
function pglmm_fit(
    ::Normal,
    Ytilde::Vector{T},
    y::Vector{T},
    X::Matrix{T},
    G::AbstractMatrix{T},
    U::AbstractMatrix{T},
    E::Nothing,
    nulldev::T,
    μ::Vector{T},
    phi::T,
    p_fX::Vector{T},
    p_fG::Vector{T},
    λ_seq::Vector{T},
    rho::Real,
    nlambda::Int,
    w::Vector{T},
    eigvals::Vector{T},
    verbose::Bool,
    criterion,
    earlystop::Bool,
    irls_tol::T,
    irls_maxiter::Int,
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

    # Initilize sum of squares
    Swuu, Swxx, Swgg = zero(δ), zero(α), zero(β)
    for j in 1:length(δ)
        @inbounds Swuu[j] = compute_Swxx(U, w, j)
    end
    for j in 1:length(α)
        @inbounds Swxx[j] = compute_Swxx(X, w, j)
    end

    # Loop through sequence of λ
    i = 0
    for _ = 1:nlambda
        # Next iterate
        i += 1
        
        # Current value of λ
        λ = λ_seq[i]
        dλ = 2 * λ_seq[i] - λ_seq[max(1, i-1)]

        # Check strong rule
        nzαind, nzβind = compute_strongrule(dλ, p_fX, p_fG, α = α, β = β, X = X, G = G, y = y, μ = μ)
        dev = convert(T, Inf)
        last_dev_ratio = dev_ratio

        # Run coordinate descent outer loop
        while true

            # Update random effects vector δ
            update_δ(Normal(), U = U, Swuu = Swuu, Ytilde = Ytilde, y = y, w = w, r = r, δ = δ, eigvals = eigvals, criterion = :coef, μ = μ)

            # Run coordinate descent inner loop to update β
            cd_lasso(Normal(), X, G, λ; Ytilde = Ytilde, Swxx = Swxx, Swgg = Swgg, y = y, w = w, r = r, α = α, β = β, δ = δ, p_fX = p_fX, p_fG = p_fG, eigvals = eigvals, criterion = criterion, phi = phi)

            # Update μ
            μ = updateμ(Normal(), r, Ytilde)

            # Update deviance
            dev = 1 / phi * NormalDeviance(δ, w, r, eigvals)
            
            # Check KKT conditions on the strong set 
            verbose && println("Checking KKT conditions on the strong set.")
            converged = cycle(X, G, λ, Val(true), r = r, α = α, β = β, Swxx = Swxx, Swgg = Swgg, w = w, p_fX = p_fX, p_fG = p_fG, nzαind = nzαind, nzβind = nzβind)
            !converged && verbose && println("KKT conditions not met, refitting the model.")

            # Then, check KKT conditions on all predictors
            if converged
                verbose && println("Checking KKT conditions on all predictors.")
                converged = cycle(X, G, λ, Val(true), r = r, α = α, β = β, Swxx = Swxx, Swgg = Swgg, w = w, p_fX = p_fX, p_fG = p_fG)

                if !converged 
                    # Recalculate strong rule
                    verbose && println("KKT conditions not met, updating strong set and refitting the model.")
                    nzαind, nzβind = compute_strongrule(dλ, p_fX, p_fG, α = α, β = β, X = X, G = G, y = y, μ = μ)
                end
            end

            converged && verbose && println("The number of active predictors is equal to $(sum(α .!= 0) + sum(β .!= 0) - 1).")
            converged && verbose && println("---------------------------------------------------")
            converged && break
        end

        # Store ouput from irls loop
        copyto!(alphas, 1:length(α), i:i, α, 1:length(α), 1:1)
        copyto!(betas, 1:length(β), i:i, β, 1:length(β), 1:1)
        dev_ratio = dev/nulldev
        copyto!(pct_dev, i, 1 - dev_ratio)
        copyto!(fitted_means, 1:length(μ), i:i, μ, 1:length(μ), 1:1)

        # Test whether we should continue
        earlystop && ((last_dev_ratio - dev_ratio < MIN_DEV_FRAC_DIFF) || pct_dev[i] > MAX_DEV_FRAC) && break
    end

    return(alphas = alphas[:, 1:i], betas = betas[:, 1:i], pct_dev = pct_dev[1:i], λ = λ_seq[1:i], fitted_values = view(fitted_means, :, 1:i))
end

# Function to fit a sparse group lasso penalized mixed model for binary traits
function pglmm_fit(
    ::Binomial,
    Ytilde::Vector{T},
    y::Vector{Int},
    X::Matrix{T},
    G::AbstractMatrix{T},
    U::AbstractMatrix{T},
    E::Vector{T},
    nulldev::T,
    μ::Vector{T},
    phi::T,
    p_fX::Vector{T},
    p_fG::Vector{T},
    λ_seq::Vector{T},
    rho::Real,
    nlambda::Int,
    w::Vector{T},
    eigvals::Vector{T},
    verbose::Bool,
    criterion,
    earlystop::Bool,
    irls_tol::T,
    irls_maxiter::Int,
    upper_bound::Bool;
    α::SparseVector{T},
    β::SparseVector{T},
    γ::SparseVector{T},
    δ::Vector{T},
    r::Vector{T}
) where T
    println("Ytilde = $(Ytilde[1])")
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
        nzαind, nzβind = compute_strongrule(dλ, λ_seq[max(1, i-1)], rho, p_fX, p_fG, E, α = α, β = β, γ = γ, X = X, G = G, y = y, μ = μ)

        # Initialize objective function and save previous deviance ratio
        dev = loss = convert(T, Inf)
        last_dev_ratio = dev_ratio

        # Iterative weighted least squares (IRLS)
        for irls in 1:irls_maxiter

            # Update random effects vector δ
            update_δ(Binomial(), U = U, Ytilde = Ytilde, y = y, w = w, r = r, δ = δ, eigvals = eigvals, criterion = :coef, μ = μ)

            # Run coordinate descent inner loop to update β
            β_last = β
            Swxx, Swgg, Swdg = cd_lasso(Binomial(), E, X, G, λ, rho; Ytilde = Ytilde, y = y, w = w, r = r, α = α, β = β, δ = δ, γ = γ, p_fX = p_fX, p_fG = p_fG, eigvals = eigvals, criterion = criterion, phi = phi)

            # Update μ and w
            μ, w = updateμ(Binomial(), r, Ytilde)
            w = upper_bound ? repeat([0.25], length(μ)) : w

            # Update deviance and loss function
            prev_loss = loss
            dev = LogisticDeviance(δ, eigvals, y, μ)
            loss = dev/2 + last(λ) * P(α, β, γ, p_fX, p_fG, rho)
            
            # If loss function did not decrease, take a half step to ensure convergence
            if loss > prev_loss + length(μ)*eps(prev_loss) && !all(β.nzval .== 0)
                println("β = $β"); println("γ = $γ")
                println("step-halving because loss=$loss > $prev_loss + $(length(μ)*eps(prev_loss)) = length(μ)*eps(prev_loss)")
                s = 1.0
                d = β - β_last
                while loss > prev_loss
                    s /= 2
                    β = β_last + s * d
                    μ, w = updateμ(Binomial(), r, Ytilde)
                    w = upper_bound ? repeat([0.25], length(μ)) : w 
                    dev = LogisticDeviance(δ, eigvals, y, μ)
                    loss = dev/2 + last(λ) * P(α, β, γ, p_fX, p_fG, rho)
                end
            end 

            # Update working response and residuals
            Ytilde, r = wrkresp(y, μ, w)

            # Check termination conditions
            converged = abs(loss - prev_loss) < irls_tol * loss
            
            # Check KKT conditions on the strong set at last iteration
            if converged
                verbose && println("Checking KKT conditions on the strong set.")
                converged = cycle(E, X, G, λ, rho, Val(true), r = r, α = α, β = β, γ = γ, Swxx = Swxx, Swgg = Swgg, Swdg = Swdg, w = w, p_fX = p_fX, p_fG = p_fG, nzαind = nzαind, nzβind = nzβind)
                !converged && verbose && println("KKT conditions not met, refitting the model.")
            end

            if converged
                verbose && println("Checking KKT conditions on all predictors.")
                converged = cycle(E, X, G, λ, rho, Val(true), r = r, α = α, β = β, γ = γ, Swxx = Swxx, Swgg = Swgg, Swdg = Swdg, w = w, p_fX = p_fX, p_fG = p_fG)

                if !converged
                    # Recalculate strong rule
                    verbose && println("KKT conditions not met, updating strong set and refitting the model.")
                    nzαind, nzβind = compute_strongrule(dλ, λ_seq[max(1, i-1)], rho, p_fX, p_fG, E, α = α, β = β, γ = γ, X = X, G = G, y = y, μ = μ)
                end
            end

            converged && verbose && println("Number of irls iterations = $irls at $i th value of λ.")
            converged && verbose && println("The number of active predictors is equal to $(length(α.nzind) + length(β.nzind) + length(γ.nzind) - 1).")
            converged && verbose && println("---------------------------------------------------")
            converged && break    
        end
        @assert converged "IRLS failed to converge in $irls_maxiter iterations at λ = $λ"

        # Store ouput from irls loop
        copyto!(alphas, 1:length(α), i:i, α, 1:length(α), 1:1)
        copyto!(betas, 1:length(β), i:i, β, 1:length(β), 1:1)
        copyto!(gammas, 1:length(γ), i:i, γ, 1:length(γ), 1:1)
        dev_ratio = dev/nulldev
        copyto!(pct_dev, i, 1 - dev_ratio)
        copyto!(fitted_means, 1:length(μ), i:i, μ, 1:length(μ), 1:1)

        # Test whether we should continue
        earlystop && ((last_dev_ratio - dev_ratio < MIN_DEV_FRAC_DIFF) || pct_dev[i] > MAX_DEV_FRAC) && break
    end

    return(alphas = alphas[:, 1:i], betas = betas[:, 1:i], gammas = gammas[:, 1:i], pct_dev = pct_dev[1:i], λ = λ_seq[1:i], fitted_values = view(fitted_means, :, 1:i))
end

# Function to fit a sparse group lasso penalized mixed model for continous traits
function pglmm_fit(
    ::Normal,
    Ytilde::Vector{T},
    y::Vector{T},
    X::Matrix{T},
    G::AbstractMatrix{T},
    U::AbstractMatrix,
    E::Vector{T},
    nulldev::T,
    μ::Vector{T},
    phi::T,
    p_fX::Vector{T},
    p_fG::Vector{T},
    λ_seq::Vector{T},
    rho::Real,
    nlambda::Int,
    w::Vector{T},
    eigvals::Vector{T},
    verbose::Bool,
    criterion,
    earlystop::Bool,
    irls_tol::T,
    irls_maxiter::Int,
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

    # Initilize sum of squares
    Swuu, Swxx, Swgg, Swdg = zero(δ), zero(α), zero(β), zero(γ)
    for j in 1:length(δ)
        @inbounds Swuu[j] = compute_Swxx(U, w, j)
    end
    for j in 1:length(α)
        @inbounds Swxx[j] = compute_Swxx(X, w, j)
    end

    # Loop through sequence of λ
    i = 0
    for _ = 1:nlambda
        # Next iterate
        i += 1
        
        # Current value of λ
        λ = λ_seq[i]
        dλ = 2 * λ_seq[i] - λ_seq[max(1, i-1)]

        # Check strong rule
        delete_coeffs!(α, β, γ)
        nzαind, nzβind = compute_strongrule(dλ, λ_seq[max(1, i-1)], rho, p_fX, p_fG, E, α = α, β = β, γ = γ, X = X, G = G, y = y, μ = μ)
        dev = convert(T, Inf)
        last_dev_ratio = dev_ratio

        # Run coordinate descent outer loop
        while true

            # Update random effects vector δ
            update_δ(Normal(), U = U, Ytilde = Ytilde, y = y, w = w, r = r, δ = δ, Swuu = Swuu, eigvals = eigvals, criterion = :coef, μ = μ)

            # Run coordinate descent inner loop to update β
            cd_lasso(Normal(), E, X, G, λ, rho; Ytilde = Ytilde, Swxx = Swxx, Swgg = Swgg, Swdg =  Swdg, y = y, w = w, r = r, α = α, β = β, δ = δ, γ = γ, p_fX = p_fX, p_fG = p_fG, eigvals = eigvals, criterion = criterion, phi = phi)

            # Update μ
            μ = updateμ(Normal(), r, Ytilde)

            # Update deviance
            dev = 1 / phi * NormalDeviance(δ, w, r, eigvals)
            
            # Check KKT conditions on the strong set at last iteration
            verbose && println("Checking KKT conditions on the strong set.")
            converged = cycle(E, X, G, λ, rho, Val(true), r = r, α = α, β = β, γ = γ, Swxx = Swxx, Swgg = Swgg, Swdg = Swdg, w = w, p_fX = p_fX, p_fG = p_fG, nzαind = nzαind, nzβind = nzβind)
            !converged && verbose && println("KKT conditions not met, refitting the model.")

            if converged
                verbose && println("Checking KKT conditions on all predictors.")
                converged = cycle(E, X, G, λ, rho, Val(true), r = r, α = α, β = β, γ = γ, Swxx = Swxx, Swgg = Swgg, Swdg = Swdg, w = w, p_fX = p_fX, p_fG = p_fG)

                if !converged
                    # Recalculate strong rule
                    verbose && println("KKT conditions not met, updating strong set and refitting the model.")
                    nzαind, nzβind = compute_strongrule(dλ, λ_seq[max(1, i-1)], rho, p_fX, p_fG, E, α = α, β = β, γ = γ, X = X, G = G, y = y, μ = μ)
                end
            end

            converged && verbose && println("The number of active predictors is equal to $(length(α.nzind) + length(β.nzind) + length(γ.nzind) - 1).")
            converged && verbose && println("---------------------------------------------------")
            converged && break
        end

        # Store ouput from irls loop
        copyto!(alphas, 1:length(α), i:i, α, 1:length(α), 1:1)
        copyto!(betas, 1:length(β), i:i, β, 1:length(β), 1:1)
        copyto!(gammas, 1:length(γ), i:i, γ, 1:length(γ), 1:1)
        dev_ratio = dev/nulldev
        copyto!(pct_dev, i, 1 - dev_ratio)
        copyto!(fitted_means, 1:length(μ), i:i, μ, 1:length(μ), 1:1)

        # Test whether we should continue
        earlystop && ((last_dev_ratio - dev_ratio < MIN_DEV_FRAC_DIFF) || pct_dev[i] > MAX_DEV_FRAC) && break
    end

    return(alphas = alphas[:, 1:i], betas = betas[:, 1:i], gammas = gammas[:, 1:i], pct_dev = pct_dev[1:i], λ = λ_seq[1:i], fitted_values = view(fitted_means, :, 1:i))
end

# Function to perform coordinate descent with a lasso penalty
function cd_lasso(
    # positional arguments
    family::Normal,
    X::Matrix{T},
    G::AbstractMatrix{T},
    λ::T;
    #keywords arguments
    r::Vector{T},
    α::SparseVector{T},
    β::SparseVector{T},
    δ::Vector{T},
    Ytilde::Vector{T},
    y::Union{Vector{Int}, Vector{T}},
    w::Vector{T}, 
    eigvals::Vector{T}, 
    p_fX::Vector{T},
    p_fG::Vector{T},
    cd_maxiter::Integer = 10000,
    cd_tol::Real=1e-7,
    criterion,
    phi::T,
    Swxx::SparseVector{T},
    Swgg::SparseVector{T},
    kwargs...
    ) where T

    converged = true
    loss = Inf

    # Coordinate descent algorithm
    for cd_iter in 1:cd_maxiter

        # Perform one coordinate descent cycle
        maxΔ = cycle(X, G, λ, Val(false), r = r, α = α, β = β, Swxx = Swxx, Swgg = Swgg, w = w, p_fX = p_fX, p_fG = p_fG)

        # Check termination condition before last iteration
        if criterion == :obj
            # Update μ
            μ, = updateμ(family, r, Ytilde)

            # Update deviance and loss function
            prev_loss = loss
            dev = 1 / phi * model_dev(family, δ, w, r, eigvals, y, μ)
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

end

# Function to perform coordinate descent with a lasso penalty
function cd_lasso(
    # positional arguments
    family::Binomial,
    X::Matrix{T},
    G::AbstractMatrix{T},
    λ::T;
    #keywords arguments
    r::Vector{T},
    α::SparseVector{T},
    β::SparseVector{T},
    δ::Vector{T},
    Ytilde::Vector{T},
    y::Union{Vector{Int}, Vector{T}},
    w::Vector{T}, 
    eigvals::Vector{T}, 
    p_fX::Vector{T},
    p_fG::Vector{T},
    cd_maxiter::Integer = 10000,
    cd_tol::Real=1e-7,
    criterion,
    phi::T,
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
            μ, = updateμ(family, r, Ytilde)

            # Update deviance and loss function
            prev_loss = loss
            dev = 1 / phi * model_dev(family, δ, w, r, eigvals, y, μ)
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
    family::Normal,
    E::Vector{T},
    X::Matrix{T},
    G::AbstractMatrix{T},
    λ::T,
    rho::Real;
    #keywords arguments
    r::Vector{T},
    α::SparseVector{T},
    β::SparseVector{T},
    δ::Vector{T},
    γ::SparseVector{T},
    Ytilde::Vector{T},
    y::Union{Vector{Int}, Vector{T}},
    w::Vector{T}, 
    eigvals::Vector{T}, 
    p_fX::Vector{T},
    p_fG::Vector{T},
    cd_maxiter::Integer = 10000,
    cd_tol::Real=1e-7,
    criterion,
    phi::T,
    Swxx::SparseVector{T},
    Swgg::SparseVector{T},
    Swdg::SparseVector{T},
    ) where T

    converged = true
    loss = Inf

    # Coordinate descent algorithm
    for cd_iter in 1:cd_maxiter

        # Perform one coordinate descent cycle
        maxΔ, = cycle(E, X, G, λ, rho, Val(false), r = r, α = α, β = β, γ = γ, Swxx = Swxx, Swgg = Swgg, Swdg = Swdg, w = w, p_fX = p_fX, p_fG = p_fG)

        # Check termination condition before last iteration
        if criterion == :obj
            # Update μ
            μ, = updateμ(family, r, Ytilde)

            # Update deviance and loss function
            prev_loss = loss
            dev = 1 / phi * model_dev(family, δ, w, r, eigvals, y, μ)
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
end

function cd_lasso(
    # positional arguments
    family::Binomial,
    E::Vector{T},
    X::Matrix{T},
    G::AbstractMatrix{T},
    λ::T,
    rho::Real;
    #keywords arguments
    r::Vector{T},
    α::SparseVector{T},
    β::SparseVector{T},
    δ::Vector{T},
    γ::SparseVector{T},
    Ytilde::Vector{T},
    y::Union{Vector{Int}, Vector{T}},
    w::Vector{T}, 
    eigvals::Vector{T}, 
    p_fX::Vector{T},
    p_fG::Vector{T},
    cd_maxiter::Integer = 10000,
    cd_tol::Real=1e-7,
    criterion,
    phi::T
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
        @inbounds Swdg[j] = compute_Swxx(E, G, w, j)
    end


    # Coordinate descent algorithm
    for cd_iter in 1:cd_maxiter

        # Perform one coordinate descent cycle
        maxΔ, = cycle(E, X, G, λ, rho, Val(false), r = r, α = α, β = β, γ = γ, Swxx = Swxx, Swgg = Swgg, Swdg = Swdg, w = w, p_fX = p_fX, p_fG = p_fG)

        # Check termination condition before last iteration
        if criterion == :obj
            # Update μ
            μ, = updateμ(family, r, Ytilde)

            # Update deviance and loss function
            prev_loss = loss
            dev = 1 / phi * model_dev(family, δ, w, r, eigvals, y, μ)
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
    U::AbstractMatrix,
    ::Binomial;
    #keywords arguments
    r::Vector{T},
    δ::Vector{T},
    Ytilde::Vector{T},
    y::AbstractArray,
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

function cd_lasso(
    # positional arguments
    U::AbstractMatrix,
    ::Normal;
    #keywords arguments
    r::Vector{T},
    δ::Vector{T},
    Ytilde::Vector{T},
    y::AbstractArray,
    w::Vector{T},
    Swuu::Vector{T},
    eigvals::Vector{T}, 
    cd_maxiter::Integer = 10000,
    cd_tol::Real=1e-7,
    criterion
    ) where T

    converged = true
    loss = Inf

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
    G::AbstractMatrix{T},
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
        last_α, Swxxj = α[j], Swxx[j]
        v = compute_grad(X, w, r, j) + last_α * Swxxj
        new_α = softtreshold(v, λ * p_fX[j]) / Swxxj
        r = update_r(X, r, last_α - new_α, j)

        maxΔ = max(maxΔ, Swxxj * (last_α - new_α)^2)
        copyto!(α, j, new_α)
    end

    # Genetic predictors
    for j in β.nzind
        last_β, Swggj = β[j], Swgg[j]
        v = compute_grad(G, w, r, j) + last_β * Swggj
        new_β = softtreshold(v, λ * p_fG[j]) / Swggj
        r = update_r(G, r, last_β - new_β, j)

        maxΔ = max(maxΔ, Swggj * (last_β - new_β)^2)
        copyto!(β, j, new_β)
    end

    maxΔ
end

function cycle(
    # positional arguments
    E::Vector{T},
    X::Matrix{T},
    G::AbstractMatrix{T},
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
        last_α, Swxxj = α[j], Swxx[j]
        v = compute_grad(X, w, r, j) + last_α * Swxxj
        new_α = softtreshold(v, λ * p_fX[j]) / Swxxj
        r = update_r(X, r, last_α - new_α, j)

        maxΔ = max(maxΔ, Swxxj * (last_α - new_α)^2)
        copyto!(α, j, new_α)
    end

    # GEI and genetic effects
    for j in β.nzind
        λj = λ * p_fG[j]

        # Update GEI effect
        last_γ, last_β = γ[j], β[j]
        Swdgj = Swdg[j]
        v = compute_grad(E, G, w, r, j) + last_γ * Swdgj
        if abs(v) > rho * λj
            new_γ = softtreshold(v, rho * λj) / (Swdgj + sqrt(2) * (1 - rho) * λj / norm((last_γ, last_β)))
            r = update_r(E, G, r, last_γ - new_γ, j)

            maxΔ = max(maxΔ, Swdgj * (last_γ - new_γ)^2)
            copyto!(γ, j, new_γ)
        end

        # Update genetic effect
        Swggj = Swgg[j] 
        v = compute_grad(G, w, r, j) + last_β * Swggj
        new_β = γ[j] != 0 ? v / (Swggj + sqrt(2) * (1 - rho) * λj / norm((last_γ, last_β))) : softtreshold(v, sqrt(2) * (1 - rho) * λj) / Swggj
        r = update_r(G, r, last_β - new_β, j)

        maxΔ = max(maxΔ, Swggj * (last_β - new_β)^2)
        copyto!(β, j, new_β)
    end

    maxΔ
end

function cycle(
    # positional arguments
    X::Matrix{T},
    G::AbstractMatrix{T},
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
    p_fG::Vector{T},
    nzαind::Union{Nothing, Vector{Int}} = nothing, 
    nzβind::Union{Nothing, Vector{Int}} = nothing
    ) where T

    kkt_check = true

    # At first and last iterations, cycle through all predictors
    # Non-genetic covariates
    rangeα = !isnothing(nzαind) ? nzαind : 1:length(α)
    for j in rangeα
        λj = λ * p_fX[j]
        if j ∉ α.nzind
            # Adding a new variable to the model
            v = compute_grad(X, w, r, j)
            abs(v) <= λj && continue
            kkt_check = false
            copyto!(α, j, 1); copyto!(α, j, 0)
            copyto!(Swxx, j, compute_Swxx(X, w, j))
        end
    end

    # Genetic covariates
    rangeβ = !isnothing(nzβind) ? nzβind : 1:length(β)
    for j in rangeβ
        λj = λ * p_fG[j]
        if j ∉ β.nzind
            # Adding a new variable to the model
            v = compute_grad(G, w, r, j)
            abs(v) <= λj && continue
            kkt_check = false
            copyto!(β, j, 1); copyto!(β, j, 0)
            copyto!(Swgg, j, compute_Swxx(G, w, j))
        end
    end

    return(kkt_check)
end

function cycle(
    # positional arguments
    E::Vector{T},
    X::Matrix{T},
    G::AbstractMatrix{T},
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
    p_fG::Vector{T},
    nzαind::Union{Nothing, Vector{Int}} = nothing, 
    nzβind::Union{Nothing, Vector{Int}} = nothing
    ) where T

    kkt_check = true

    # At first and last iterations, cycle through all predictors
    # Non-genetic covariates
    rangeα = !isnothing(nzαind) ? nzαind : 1:length(α)
    for j in rangeα
        λj = λ * p_fX[j]
        if j ∉ α.nzind
            # Adding a new variable to the model
            v = compute_grad(X, w, r, j)
            abs(v) <= λj && continue
            kkt_check = false
            copyto!(α, j, 1); copyto!(α, j, 0)
            copyto!(Swxx, j, compute_Swxx(X, w, j))
        end
    end

    # GEI and genetic effects
    rangeβ = !isnothing(nzβind) ? nzβind : 1:length(β)
    for j in rangeβ
        λj = λ * p_fG[j]
        v1 = compute_grad(G, w, r, j)
        v2 = compute_grad(E, G, w, r, j)

        if j in β.nzind && j ∉ γ.nzind
            # Adding a new GEI to the model
            abs(v2) <= rho * λj && continue
            kkt_check = false
            copyto!(γ, j, 1); copyto!(γ, j, 0)
            copyto!(Swdg, j, compute_Swxx(E, G, w, j))
        elseif j ∉ β.nzind
            # Adding a new main effect to the model
            norm([v1, softtreshold(v2, rho * λj)]) <= sqrt(2) * (1 - rho) * λj && continue
            kkt_check = false
            copyto!(β, j, 1); copyto!(β, j, 0)
            copyto!(Swgg, j, compute_Swxx(G, w, j))

            # Adding a new GEI to the model
            abs(v2) <= rho * λj && continue
            copyto!(γ, j, 1); copyto!(γ, j, 0)
            copyto!(Swdg, j, compute_Swxx(E, G, w, j))
        end
    end

    return(kkt_check)
end

function cycle(
    # positional arguments
    U::AbstractMatrix;
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
    ::Normal;
    #keywords arguments
    U::AbstractMatrix,
    Swuu::Vector{T},
    Ytilde::Vector{T}, 
    y::AbstractVector, 
    w::Vector{T}, 
    r::Vector{T}, 
    δ::Vector{T}, 
    eigvals::Vector{T}, 
    criterion = criterion,
    kwargs...
    ) where T
    
    cd_lasso(U, Normal(); Ytilde = Ytilde, y = y, w = w, Swuu = Swuu, r = r, δ = δ, eigvals = eigvals, criterion = criterion)
end

function update_δ(
    # positional arguments
    ::Binomial;
    #keywords arguments
    U::AbstractMatrix,
    Ytilde::Vector{T}, 
    y::AbstractVector, 
    w::Vector{T}, 
    r::Vector{T}, 
    δ::Vector{T}, 
    eigvals::Vector{T}, 
    criterion = criterion,
    kwargs...
    ) where T
    
    cd_lasso(U, Binomial(); Ytilde = Ytilde, y = y, w = w, r = r, δ = δ, eigvals = eigvals, criterion = criterion)
end

modeltype(::Normal) = "Least Squares"
modeltype(::Binomial) = "Logistic"

mutable struct pglmmPath{F<:Distribution, A<:AbstractArray, B<:AbstractArray, T<:AbstractFloat, C<:AbstractArray, E<:AbstractArray}
    family::F
    a0::A                                       # intercept values for each solution
    alphas::B                                   # coefficient values for each solution
    betas::B                                
    gammas::Union{Nothing, B}
    null_dev::T                                 # Null deviance of the model
    pct_dev::C                                 # R^2 values for each solution
    lambda::C                                   # lambda values corresponding to each solution
    npasses::Int                                # actual number of passes over the data for all lamda values
    fitted_values                               # fitted_values
    y::Union{Vector{Int}, C}                    # outcome vector
    φ::T                                        # dispersion parameters
    τ::C                                        # estimated variance components
    intercept::Bool                             # boolean for intercept
    rho::Union{Nothing, Real}                   # rho tuninng parameter
    D::Union{Nothing, E}
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
    G::AbstractMatrix{T},
    E::Union{Vector{T}, Nothing}; 
    p_fX::Vector{T},
    p_fG::Vector{T},
    rho::Real,
    nlambda::Integer = 100
    ) where T

    λ_min_ratio = (length(r) < size(G, 2) ? 1e-2 : 1e-4)
    λ_max = lambda_max(nothing, X, r, p_fX)
    λ_max = lambda_max(E, G, r, p_fG, λ_max, rho = rho)
    λ_min = λ_max * λ_min_ratio
    λ_step = log(λ_min_ratio)/(nlambda - 1)
    λ_seq = exp.(collect(log(λ_max+100*eps(λ_max)):λ_step:log(λ_min)))

    λ_seq
end

# Function to compute λ_max for the lasso
function lambda_max(E::Nothing, X::AbstractMatrix{T}, r::AbstractVector{T}, p_f::AbstractVector{T}, λ_max::T = zero(T); kwargs...) where T
    seq = findall(x -> !iszero(x) && !isinf(x), p_f)
    for j in seq
        x = abs(compute_grad(X, r, j)) / p_f[j]
        if x > λ_max
            λ_max = x
        end
    end
    return(λ_max)
end

# Function to compute λ_max for the group lasso
function lambda_max(E::AbstractVector{T}, X::AbstractMatrix{T}, r::AbstractVector{T}, p_f::AbstractVector{T}, λ_max::T = zero(T); rho::Real) where T

    seq = findall(x -> !iszero(x) && !isinf(x), p_f)
    for j in seq
        x = compute_max(E, X, r, j, rho) / p_f[j]
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
    y::Vector{Int},
    μ::Vector{T},
    w::Vector{T}
) where T
    η = GLM.linkfun.(LogitLink(), μ)
    Ytilde = [η[i] + (y[i] - μ[i]) / w[i] for i in 1:length(y)]
    r = Ytilde - η
    return(Ytilde, r)
end

# Function to update linear predictor and mean at each iteration
function updateμ(::Normal, r::Vector{T}, Ytilde::Vector{T}) where T
    μ = Ytilde - r
    return(μ)
end

const PMIN = 1e-5
const PMAX = 1-1e-5
function updateμ(::Binomial, r::Vector{T}, Ytilde::Vector{T}) where T
    η = Ytilde - r
    μ = GLM.linkinv.(LogitLink(), η)
    μ = [μ[i] < PMIN ? PMIN : μ[i] > PMAX ? PMAX : μ[i] for i in 1:length(μ)]
    w = μ .* (1 .- μ)
    return(μ, w)
end

# Functions to calculate deviance
model_dev(::Binomial, δ::Vector{T}, w::Vector{T}, r::Vector{T}, eigvals::Vector{T}, y::Vector{Int}, μ::Vector{Float64}) where T = LogisticDeviance(δ, eigvals, y, μ)
model_dev(::Binomial, b::Vector{T}, τV::Matrix{T}, y::Vector{Int}, μ::Vector{T}) where T = LogisticDeviance(b, τV, y, μ)
model_dev(::Normal, δ::Vector{T}, w::Vector{T}, r::Vector{T}, eigvals::Vector{T}, kargs...) where T = NormalDeviance(δ, w, r, eigvals)

function LogisticDeviance(δ::Vector{T}, eigvals::Vector{T}, y::Vector{Int}, μ::Vector{T}) where T
    -2 * sum(y .* log.(μ ./ (1 .- μ)) .+ log.(1 .- μ)) + dot(δ, Diagonal(eigvals), δ)
end

function LogisticDeviance(b::Vector{T}, τV::Matrix{T}, y::Vector{Int}, μ::Vector{T}) where T
    -2 * sum(y .* log.(μ ./ (1 .- μ)) .+ log.(1 .- μ)) + dot(b, inv(τV), b)
end

function NormalDeviance(δ::Vector{T}, w::Vector{T}, r::Vector{T}, eigvals::Vector{T}) where T
    dot(sqrt.(w) .* r, sqrt.(w) .* r) + dot(δ, Diagonal(eigvals), δ)
end

# Predict phenotype
function predict(path, 
                  formula::FormulaTerm,
                  covfile::Union{DataFrame, AbstractString},
                  plinkfile::Union{Nothing, AbstractString} = nothing;
                  # keyword arguments
                  grmfile::Union{Nothing, AbstractString} = nothing,
                  snpfile::Union{Nothing, AbstractString} = nothing,
                  snpmodel = ADDITIVE_MODEL,
                  snpinds::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                  testrowinds::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                  trainrowinds::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                  idvar::Union{Nothing, Symbol, String} = nothing,
                  geneticrowinds::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                  grmcolinds::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                  M::Union{Nothing, Vector{Any}} = nothing,
                  s::Union{T, Vector{T}, Nothing} = nothing,
                  fixed_effects_only::Bool = false,
                  GEIvar::Union{Nothing,AbstractString} = nothing,
                  GEIkin::Bool = true,
                  GRM::Union{Nothing, Matrix{T}, BlockDiagonal{T, Matrix{T}}} = nothing,
                  reformula::Union{Nothing, FormulaTerm} = nothing,
                  standardize_Z::Bool = false,
                  outtype = :response
                 ) where T

    if isnothing(s)
        [predict(path[j],
                  formula, 
                  covfile,
                  plinkfile;
                  snpfile = snpfile,
                  grmfile = grmfile,
                  snpmodel = snpmodel,
                  snpinds = snpinds,
                  testrowinds = testrowinds,
                  trainrowinds = trainrowinds,
                  idvar = idvar,
                  geneticrowinds = geneticrowinds,
                  grmcolinds = grmcolinds,
                  M = M,
                  s = 1:size(path[j].betas, 2),
                  fixed_effects_only = fixed_effects_only,
                  GEIvar = GEIvar,
                  GEIkin = GEIkin,
                  GRM = GRM,
                  reformula = reformula,
                  standardize_Z = standardize_Z,
                  outtype = outtype
                 ) for j in 1:length(path)] |> x-> reduce(hcat,x)
    else
        [predict(path[s[j].rho.index],
                  formula, 
                  covfile,
                  plinkfile;
                  snpfile = snpfile,
                  grmfile = grmfile,
                  snpmodel = snpmodel,
                  snpinds = snpinds,
                  testrowinds = testrowinds,
                  trainrowinds = trainrowinds,
                  idvar = idvar,
                  geneticrowinds = geneticrowinds,
                  grmcolinds = grmcolinds,
                  M = M,
                  s = s[j].lambda.index,
                  fixed_effects_only = fixed_effects_only,
                  GEIvar = GEIvar,
                  GEIkin = GEIkin,
                  GRM = GRM,
                  reformula = reformula,
                  standardize_Z = standardize_Z,
                  outtype = outtype
                 ) for j in 1:length(s)] |> x-> reduce(hcat,x)
    end
end

function predict(path::pglmmPath,
                  formula::FormulaTerm, 
                  covfile::Union{DataFrame, AbstractString},
                  plinkfile::Union{Nothing, AbstractString} = nothing;
                  # keyword arguments
                  grmfile::Union{Nothing, AbstractString} = nothing,
                  snpfile::Union{Nothing, AbstractString} = nothing,
                  snpmodel = ADDITIVE_MODEL,
                  snpinds::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                  testrowinds::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                  trainrowinds::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                  idvar::Union{Nothing, Symbol, String} = nothing,
                  geneticrowinds::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                  grmcolinds::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                  M::Union{Nothing, Vector{Any}} = nothing,
                  s::Union{Nothing,<:Integer,AbstractVector{<:Integer}} = nothing,
                  fixed_effects_only::Bool = false,
                  GEIvar::Union{Nothing,AbstractString} = nothing,
                  GEIkin::Bool = true,
                  GRM::Union{Nothing, Matrix{T}, BlockDiagonal{T, Matrix{T}}} = nothing,
                  reformula::Union{Nothing, FormulaTerm} = nothing,
                  standardize_Z::Bool = false,
                  outtype = :response
                 ) where T
    
    #--------------------------------------------------------------
    # Read covariate and grm file
    #--------------------------------------------------------------
    covdf = isa(covfile, AbstractString) ? CSV.read(covfile, DataFrame) : isa(covfile, DataFrame) ? covfile : error("covfile is not a DataFrame of AbstractString")
    testrowinds = isnothing(testrowinds) ? (1:nrow(covdf)) : testrowinds

    X = modelmatrix(glm(formula, covdf[testrowinds,:], path.family))
    X = all(X[:,1] .== 1)  ? X[:, 2:end] : X
    nX, k = size(X)

    if !isnothing(grmfile)
        GRM = open(GzipDecompressorStream, grmfile, "r") do stream
            Symmetric(Matrix(CSV.read(stream, DataFrame)))
        end
    end

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
    @assert nG == nX "covariate and genotype matrices must have same number of rows."
    
    #--------------------------------------------------------------
    # Compute predictions
    #--------------------------------------------------------------
    # Create list of similarity matrices
    if !isnothing(idvar)
        m = length(unique(covdf[:, idvar]))
        L = [ones(sum(covdf[:, idvar] .== unique(covdf[:, idvar])[i]), 1) for i in 1:m] |> x-> BlockDiagonal(x)
    else
        L = Diagonal(ones(size(GRM, 1)))
    end
    V = push!(Any[], Matrix(L)[testrowinds, :] * GRM * Matrix(L)[trainrowinds, :]')

    # Add GEI similarity matrix
    if !isnothing(GEIvar)
        E = covdf[testrowinds, GEIvar]
        if GEIkin
            @assert length(path.τ) >= 2 "Only one variance component has been estimated under the null model."
        Etrain = covdf[trainrowinds, GEIvar]
            V_E = E * Etrain'
            for j in findall(x -> x == 0, Etrain), i in findall(x -> x == 0, E)  
                    V_E[i, j] = 1 
            end
            push!(V, sparse(V[1] .* V_E))
        end
    end

    # Covariance matrix between test and training subjects
    Σ = sum(path.τ .* V)

    # Number of predictions to compute. User can provide index s for which to provide predictions, 
    # rather than computing predictions for the whole path.
    s = isnothing(s) ? (1:size(path.betas, 2)) : s

    #--------------------------------
    # Linear predictor fixed effects
    #--------------------------------
    # Main effects
    Gbeta = reduce(hcat, [G[:, path.betas[:, i].nzind] * path.betas[:, i].nzval for i in s])
    η = path.a0[s]' .+ X * path.alphas[:,s] .+ Gbeta

    # GEI effects
    if !isnothing(GEIvar)
        η += reduce(hcat, [(E .* G[:, path.gammas[:, i].nzind]) * path.gammas[:, i].nzval for i in s])
    end

    #--------------------------------------------------------------
    # Longitudinal data
    #--------------------------------------------------------------
    if !isnothing(reformula)

        @assert !isnothing(idvar) "idvar is missing"

        # Random effect formula
        z = modelmatrix(glm(reformula, covdf, path.family))
        if standardize_Z
            z = all(z[:, 1] .== 1) ? hcat(ones(size(z, 1)), z[:, 2:end] ./ sqrt.(diag(cov(z[:, 2:end])))') : z ./ sqrt.(diag(cov(z)))'
        end

        # Create BlockDiagonal matrix for each random effect
        r, Z = size(z,2), Any[]
        for j in 1:r
            push!(Z, [reshape(z[covdf[:, idvar] .== unique(covdf[:, idvar])[i], j], :, 1) for i in 1:m] |> x->BlockDiagonal(x))
        end

        # Create relatedness matrices
        ZDZt, idx, Dvec, idx = [], 0, vech(path.D), 0
        for j in 1:r
            for k in j:r
                if j == k
                    idx += 1
                    push!(ZDZt, BlockDiagonal(blocks(Z[j]) .* blocks(Z[j]')) * Dvec[idx])
                else
                    idx += 1
                    push!(ZDZt, BlockDiagonal(blocks(Z[j]) .* blocks(Z[k]') + blocks(Z[k]) .* blocks(Z[j]')) * Dvec[idx])
                end
            end
        end

        Σ += Matrix(sum(ZDZt))[testrowinds, trainrowinds]

    end

    if fixed_effects_only == false
        if path.family == Binomial()
            b = Σ * (path.y .- path.fitted_values[:,s])
        elseif path.family == Normal()
            b = Σ * path.φ * (path.y .- path.fitted_values[:,s])
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
function standardizeX(X::AbstractMatrix{T}, standardize::Bool, alpha::AbstractVector{T}, intercept::Bool = false) where T
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

        α = alpha .* s 
    else
        for j in 1:size(X,2), i in 1:size(X, 1) 
            @inbounds Xs[i,j] = X[i,j] .- mu[j]
        end
        s = []
        α = alpha 
    end

    # Remove first term if intercept
    if intercept 
        popfirst!(mu); popfirst!(s)
        α[1] += alpha[2:end]'mu
    end

    Xs, mu, s, α
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

function compute_grad(E::AbstractVector{T}, X::AbstractMatrix{T}, w::AbstractVector{T}, r::AbstractVector{T}, whichcol::Int) where T
    v = zero(T)
    for i = 1:size(X, 1)
        @inbounds v += E[i] * X[i, whichcol] * r[i] * w[i]
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

function compute_max(E::AbstractVector{T}, X::AbstractMatrix{T}, r::AbstractVector{T}, whichcol::Int, rho::Real) where T
    v = zeros(2)
    for i = 1:size(X, 1)
        @inbounds v[1] += X[i, whichcol] * r[i]
        @inbounds v[2] += E[i] * X[i, whichcol] * r[i]
    end

    if rho == 1
        abs(v[2])
    else
        norm(v) / (sqrt(2) * (1 - rho))
    end
end

function compute_prod(X::AbstractMatrix{T}, y::Union{AbstractVector{Int}, AbstractVector{T}}, p::AbstractVector{T}, whichcol::Int) where T
    v = zero(T)
    for i = 1:size(X, 1)
        @inbounds v += X[i, whichcol] * (y[i] - p[i])
    end
    v
end

function compute_prod(E::AbstractVector{T}, X::AbstractMatrix{T}, y::Union{AbstractVector{Int}, AbstractVector{T}}, p::AbstractVector{T}, whichcol::Int) where T
    v = zero(T)
    for i = 1:size(X, 1)
        @inbounds v += E[i] * X[i, whichcol] * (y[i] - p[i])
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

function compute_Swxx(E::AbstractVector{T}, X::AbstractMatrix{T}, w::AbstractVector{T}, whichcol::Int) where T
    s = zero(T)
    for i = 1:size(X, 1)
        @inbounds s += (E[i] * X[i, whichcol])^2 * w[i]
    end
    s
end

function update_r(X::AbstractMatrix{T}, r::AbstractVector{T}, deltaβ::T, whichcol::Int) where T
    for i = 1:size(X, 1)
        @inbounds r[i] += X[i, whichcol] * deltaβ
    end
    r
end

function update_r(X::AbstractMatrix{T}, r::AbstractVector{T}, deltaβ::SparseVector{T}) where T
    for i = 1:size(X, 1)
        @inbounds r[i] += X[i, deltaβ.nzind]'deltaβ[deltaβ.nzind]
    end
    r
end

function update_r(E::AbstractVector{T}, X::AbstractMatrix{T}, r::AbstractVector{T}, deltaβ::T, whichcol::Int) where T
    for i = 1:size(X, 1)
        @inbounds r[i] += E[i] * X[i, whichcol] * deltaβ
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
            x += p_fG[i] * (sqrt(2) * (1 - rho) * norm((β[i], γ[i])) + rho * abs(γ[i]))
    end
    x
end

# Compute strongrule for the lasso
function compute_strongrule(dλ::T, p_fX::Vector{T}, p_fG::Vector{T}; α::SparseVector{T}, β::SparseVector{T}, X::Matrix{T}, G::AbstractMatrix{T}, y::AbstractArray, μ::Vector{T}) where T
    nzαind = copy(α.nzind)
    for j in 1:length(α)
        j in α.nzind && continue
        c = compute_prod(X, y, μ, j)
        abs(c) <= dλ * p_fX[j] && continue
        
        # Add a new variable to the strong set
        sort!(push!(nzαind, j))
    end
    
    nzβind = copy(β.nzind)
    for j in 1:length(β)
        j in β.nzind && continue
        c = compute_prod(G, y, μ, j)
        abs(c) <= dλ * p_fG[j] && continue
        
        # Add a new variable to the strong set
        sort!(push!(nzβind, j))
    end

    return nzαind, nzβind
end

# Compute strongrule for the sparse group lasso
function compute_strongrule(dλ::T, λ::T, rho::Real, p_fX::Vector{T}, p_fG::Vector{T}, E::Vector{T}; α::SparseVector{T}, β::SparseVector{T}, γ::SparseVector{T}, X::Matrix{T}, G::AbstractMatrix{T}, y::AbstractArray, μ::Vector{T}) where T
    nzαind = copy(α.nzind)
    for j in 1:length(α)
        j in α.nzind && continue
        c = compute_prod(X, y, μ, j)
        abs(c) <= dλ * p_fX[j] && continue
        
        # Add a new variable to the strong set
        sort!(push!(nzαind, j))
    end
    
    nzβind = copy(β.nzind)
    for j in 1:length(β)
        j in β.nzind && continue
        c1 = compute_prod(G, y, μ, j)
        c2 = softtreshold(compute_prod(E, G, y, μ, j), rho * λ * p_fG[j])
        norm([c1, c2]) <= sqrt(2) * (1 - rho) * dλ * p_fG[j] && continue
        
        # Add a new group to the strong set
        sort!(push!(nzβind, j))
    end

    return nzαind, nzβind
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

# Function to compute eigenvalues and eigenvectors of the kronecker product of a matrix with an identity matrix.
function eigenkron(A::AbstractMatrix{T}, n::Int) where T
    d, v = eigen(A)
    D = repeat(d, inner=n)
    V = kron(v, Diagonal(ones(n)))

    return(values=D, vectors=V)
end

# Function to delete zero coefficients from the active set
function delete_coeffs!(α, β)
    αzinds = findall(α[α.nzind] .== 0); deleteat!(α.nzval, αzinds); deleteat!(α.nzind, αzinds)
    βzinds = findall(β[β.nzind] .== 0); deleteat!(β.nzval, βzinds); deleteat!(β.nzind, βzinds)
end

function delete_coeffs!(α, β, γ)
    αzinds = findall(α[α.nzind] .== 0); deleteat!(α.nzval, αzinds); deleteat!(α.nzind, αzinds)
    βzinds = findall(β[β.nzind] .== 0); deleteat!(β.nzval, βzinds); deleteat!(β.nzind, βzinds)
    γzinds = findall(γ[γ.nzind] .== 0); deleteat!(γ.nzval, γzinds); deleteat!(γ.nzind, γzinds)
end