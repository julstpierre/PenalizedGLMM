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
    kwargs...
    )

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
    D_inv = eigenweights(nullmodel.family, eigvals, nullmodel.φ)
    UD_invUt = U * D_inv * U'
    eigvals = 1 ./ eigvals

    # Initialize GLMM weights
    if nullmodel.family == Binomial()
        w = 0.25
    elseif nullmodel.family == Normal()
        w = 1 / nullmodel.φ
    end
	
    # Initialize working variable
    y = nullmodel.y
    if nullmodel.family == Binomial()
        μ = GLM.linkinv.(LogitLink(), nullmodel.η)
        Ytilde = nullmodel.η + 4 * (y - μ)
        r = Ytilde - nullmodel.η
        b = nullmodel.η - nullmodel.X * nullmodel.α
        nulldev = LogisticDeviance(b, U, eigvals, y, μ)

    elseif nullmodel.family == Normal()
        Ytilde = y
        r = Ytilde - nullmodel.η
        b = nullmodel.η - nullmodel.X * nullmodel.α
        nulldev = model_dev(Normal(), b, U, w, r, eigvals)
    end

    # standardize non-genetic covariates
    intercept = all(nullmodel.X[:,1] .== 1)
    X, muX, sX = standardizeX(nullmodel.X, standardize_X, intercept)
    
    # Transform intercept because of centering
    if intercept 
        α = [dot([1; muX], nullmodel.α); nullmodel.α[2:end]]
    else 
        α = [dot(muX, nullmodel.α); nullmodel.α]
        X = hcat(ones(n), X)
        k += 1
    end

    # Initialize β, Swxx and penalty factors
    β = standardize_X ? sparse([α .* [1; sX]; zeros(p)]) : sparse([α; zeros(p)])
    Swxx = sparse([X' .^2 * w; zeros(p)])
    p_fX = zeros(k); p_fG = ones(p); p_f = [p_fX; p_fG]

    # Sequence of λ
    λ_seq = lambda_seq(r, X, G, w, p_fX, p_fG)
    
    # Fit penalized model
    path = pglmm_fit(nullmodel.family, Ytilde, y, X, G, nulldev, r, β, Swxx, b, p_f, λ_seq, K, UD_invUt, U, w, eigvals, verbose, criterion, earlystop, irls_tol, irls_maxiter)

    # Separate intercept from betas
    a0 = view(path.betas, 1, :)
    betas = view(path.betas, 2:(p + k), :)
    k = k - 1

    # Return coefficients on original scale
    if !isempty(sX) & !isempty(sG)
        lmul!(inv(Diagonal(vec([sX; sG]))), betas)
        a0 .-= vec([muX; muG]' * betas)
    elseif !isempty(sX)
        betas[1:k,:] = lmul!(inv(Diagonal(vec(sX))), betas[1:k,:])
        a0 .-=  vec(muX' * betas[1:k,:])
    elseif !isempty(sG)
        betas[(k+1):end,:] = lmul!(inv(Diagonal(vec(sG))), betas[(k+1):end,:])
        a0 .-=  vec(muG' * betas[(k+1):end,:])
    end

    # Return lasso path
    pglmmPath(nullmodel.family, a0, betas, nulldev, path.pct_dev, path.λ, 0, path.fitted_values, y, UD_invUt, nullmodel.τ, intercept)
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
    nulldev::T,
    r::Vector{T},
    β::SparseVector{T},
    Swxx::SparseVector{T},
    b::Vector{T},
    p_f::Vector{T},
    λ_seq::Vector{T},
    K::Int64,
    UD_invUt::Matrix{T},
    U::Matrix{T},
    w::T,
    eigvals::Vector{T},
    verbose::Bool,
    criterion,
    earlystop::Bool,
    irls_tol::T,
    irls_maxiter::Int64
) where T

    # Initialize array to store output for each λ
    betas = zeros(length(β), K)
    pct_dev = zeros(T, K)
    dev_ratio = convert(T, NaN)
    fitted_means = zeros(length(y), K)
    μ = zeros(length(y))

    # Define size of predictors
    k, p = size(Xstar, 2), size(Gstar, 2)

    # Loop through sequence of λ
    i = 0
    for _ = 1:K
        # Next iterate
        i += 1
        converged = false
        
        # Current value of λ
        λ = λ_seq[i]

        # Initialize objective function and save previous deviance ratio
        dev = loss = convert(T, Inf)
        last_dev_ratio = dev_ratio

        # Iterative weighted least squares (IRLS)
        for irls in 1:irls_maxiter

            # Update b
            b, r, Ytilde, loss = update_b(b, r, Ytilde, y, UD_invUt, U, eigvals, loss, sum(p_f[β.nzind] .* abs.(β.nzval)), λ)

            # Run coordinate descent inner loop to update β
            β, r = cd_lasso(r, X, G; family = Binomial(), Ytilde = Ytilde, y = y, w = w, β = β, Swxx = Swxx, b = b, U = U, eigvals = eigvals, p_f = p_f, λ = λ, criterion = criterion, k = k, p = p)

            # Update μ
            μ = updateμ(r, Ytilde)
            
            # Update deviance and loss function
            prev_loss = loss
            dev = LogisticDeviance(b, U, eigvals, y, μ)
            loss = dev/2 + λ * sum(p_f[β.nzind] .* abs.(β.nzval))
            
            # If loss function did not decrease, take a half step to ensure convergence
            if loss > prev_loss + length(μ)*eps(prev_loss)
                verbose && println("step-halving because loss=$loss > $prev_loss + $(length(μ)*eps(prev_loss)) = length(μ)*eps(prev_loss)")
                #= s = 1.0
                d = β - β_last
                while loss > prev_loss
                    s /= 2
                    β = β_last + s * d
                    μ = updateμ(r, Ytilde, UD_inv) 
                    dev = LogisticDeviance(r, w, y, μ, UD_inv_sq)
                    loss = dev/2 + λ * sum(p_f[β.nzind] .* abs.(β.nzval))
                end =#
            end 

            # Update working response and residuals
            Ytilde, r = wrkresp(y, r, μ, Ytilde)

            # Check termination conditions
            converged = abs(loss - prev_loss) < irls_tol * loss
            converged && verbose && println("Number of IRLS iterations = $irls for updating β at $i th value of λ.")
            converged && break 

        end
        @assert converged "IRLS failed to converge in $irls_maxiter iterations at λ = $λ"

        # Store ouput from irls loop
        betas[:, i] = convert(Vector{Float64}, β)
        dev_ratio = dev/nulldev
        pct_dev[i] = 1 - dev_ratio
        fitted_means[:, i] = μ

        # Test whether we should continue
        earlystop && ((last_dev_ratio - dev_ratio < MIN_DEV_FRAC_DIFF && length(β.nzind) > sum(p_f .==0)) || pct_dev[i] > MAX_DEV_FRAC) && break
    end

    return(betas = view(betas, :, 1:i), pct_dev = pct_dev[1:i], λ = λ_seq[1:i], fitted_values = view(fitted_means, :, 1:i))
end

function pglmm_fit(
    ::Normal,
    Ytilde::Vector{T},
    y::Vector{Int64},
    X::Matrix{T},
    G::SubArray{T, 2, SnpLinAlg{T}, Tuple{Vector{Int64}, UnitRange{Int64}}},
    nulldev::T,
    r::Vector{T},
    β::SparseVector{T},
    Swxx::SparseVector{T},
    b::Vector{T},
    p_f::Vector{T},
    λ_seq::Vector{T},
    K::Int64,
    UD_invUt::Matrix{T},
    U::Matrix{T},
    w::T,
    eigvals::Vector{T},
    verbose::Bool,
    criterion,
    earlystop::Bool,
    irls_tol::T,
    irls_maxiter::Int64
) where T

    # Initialize array to store output for each λ
    betas = zeros(length(β), K)
    pct_dev = zeros(Float64, K)
    dev_ratio = convert(Float64, NaN)
    residuals = zeros(length(y), K)

    # Define size of predictors
    k, p = size(Xstar, 2), size(Gstar, 2)

    # Loop through sequence of λ
    i = 0
    for _ = 1:K
        # Next iterate
        i += 1

        # Print current iteration
        verbose && println("i = ", i)
        
        # Current value of λ
        λ = λ_seq[i]

        # Save previous deviance ratio
        last_dev_ratio = dev_ratio

        # Update b
        b, r = update_b(b, r, UD_invUt)

        # Run coordinate descent inner loop to update β
        β, r = cd_lasso(r, X, G; family = Normal(), Ytilde = Ytilde, y = y, w = w, β = β, Swxx = Swxx, b = b, U = U, eigvals = eigvals, p_f = p_f, λ = λ, criterion = criterion, k = k, p = p)

        # Update deviance
        dev = NormalDeviance(b, U, w, r, eigvals)

        # Store ouput
        betas[:, i] = convert(Vector{Float64}, β)
        dev_ratio = dev/nulldev
        pct_dev[i] = 1 - dev_ratio
        residuals[:, i] = r

        # Test whether we should continue
        earlystop && ((last_dev_ratio - dev_ratio < MIN_DEV_FRAC_DIFF && length(β.nzind) > sum(p_f .==0))|| pct_dev[i] > MAX_DEV_FRAC) && break
    end

    return(betas = view(betas, :, 1:i), pct_dev = pct_dev[1:i], λ = λ_seq[1:i], fitted_values = view(residuals, :, 1:i))
end

# Function to update b
function update_b(
    # positional arguments
    b::Vector{T},
    r::Vector{T},
    Ytilde::Vector{T},
    y::Vector{Int64},
    UD_invUt::Matrix{T},
    U::Matrix{T},
    eigvals::Vector{T},
    loss::T,
    L1_β::T,
    λ::T,
    irls_maxiter::Integer = 30,
    irls_tol::Real=1e-8
    ) where T
    
    converged = false
    for irls_iter in 1:irls_maxiter
        # Update b and residuals
        last_b = b
        b = UD_invUt * (r + last_b)
        r += last_b - b

        # Update μ
        μ = updateμ(r, Ytilde)

        # Update working response and residuals
        Ytilde, r = wrkresp(y, r, μ, Ytilde)

        # Update deviance and loss function
        prev_loss = loss
        dev = LogisticDeviance(b, U, eigvals, y, μ)
        loss = dev/2 + λ * L1_β

        # Check termination conditions
        converged = abs(loss - prev_loss) < irls_tol * loss
        converged && break
    end
    # Assess convergence of IRLS
    @assert converged "IRLS updates for b failed to converge in $irls_maxiter iterations at λ = $λ."

    return(b, r, Ytilde, loss)
end

function update_b(
    # positional arguments
    last_b::Vector{T},
    r::Vector{T},
    UD_invUt::Matrix{T}
    ) where T

    # Update b and residuals
    b = UD_invUt * (r + last_b)
    r += last_b - b

    return(b, r)
end

# Function to perform coordinate descent with a lasso penalty
function cd_lasso(
    # positional arguments
    r::Vector{T},
    X::Matrix{T},
    G::SubArray{T, 2, SnpLinAlg{T}, Tuple{Vector{Int64}, UnitRange{Int64}}};
    #keywords arguments
    family::UnivariateDistribution,
    β::SparseVector{T},
    Swxx::SparseVector{T},
    b::Vector{T},
    eigvals::Vector{T},
    Ytilde::Vector{T},
    y::Union{Vector{Int64},Vector{T}},
    w::T, 
    U::Matrix{T},
    p_f::Vector{T}, 
    λ::T,
    cd_maxiter::Integer = 100000,
    cd_tol::Real=1e-8,
    criterion,
    k::Float64,
    p::Float64
    ) where T

    converged = false
    maxΔ = zero(T)
    loss = Inf

    for cd_iter in 1:cd_maxiter
        # At first iteration, perform one coordinate cycle and 
        # record active set of coefficients that are nonzero
        if cd_iter == 1 || converged
            # Non-genetic covariates
            for j in 1:k
                Xj = view(X, :, j)
                v = compute_grad(Xj, w, r)
                λj = λ * p_f[j]

                last_β = β[j]
                if last_β != 0
                    v += last_β * Swxxj[j]
                else
                    # Adding a new variable to the model
                    abs(v) < λj && continue
		            Swxx[j] = compute_Swxx(Xj, w)
                end
                new_β = softtreshold(v, λj) / Swxx[j]
                r += Xj * (last_β - new_β)

                maxΔ = max(maxΔ, Swxx[j] * (last_β - new_β)^2)
                β[j] = new_β
            end

            # Genetic covariates
            for j in 1:p
                Gj = view(G, :, j)
                v = compute_grad(Gj, w, r)
                λj = λ * p_f[k + j]

                last_β = β[k + j]
                if last_β != 0
                    v += last_β * Swxx[k+j]
                else
                    # Adding a new variable to the model
                    abs(v) < λj && continue
		            Swxx[k+j] = compute_Swxx(Gj, w)
                end
                new_β = softtreshold(v, λj) / Swxx[k+j]
                r += Gj * (last_β - new_β)

                maxΔ = max(maxΔ, Swxx[k+j] * (last_β - new_β)^2)
                β[k + j] = new_β
            end

            # Check termination condition at last iteration
            if criterion == :obj
                # Update μ
                μ = updateμ(r, Ytilde)

                # Update deviance and loss function
                prev_loss = loss
                dev = model_dev(family, b, U, w, r, eigvals, y, μ)
                loss = dev/2 + λ * sum(p_f[β.nzind] .* abs.(β.nzval))

                # Check termination condition
                converged && abs(loss - prev_loss) < cd_tol * loss && break

            elseif criterion == :coef
                converged && maxΔ < cd_tol && break
            end
        end

        # Cycle over coefficients in active set only until convergence
        maxΔ = zero(T)
        
        # Non-genetic covariates
        for j in β.nzind[β.nzind .<= k]
            last_β = β[j]
            last_β == 0 && continue
            
            Xj = view(X, :, j)
            v = compute_grad(Xj, w, r) + last_β * Swxx[j]
            new_β = softtreshold(v, λ * p_f[j]) / Swxx[j]
            r += Xj * (last_β - new_β)

            maxΔ = max(maxΔ, Swxx[j] * (last_β - new_β)^2)
            β[j] = new_β
        end

        # Genetic predictors
        for j in β.nzind[β.nzind .> k]
            last_β = β[j]
            last_β == 0 && continue
            
            Gj = view(G, :, j-k)
            v = compute_grad(Gj, w, r) + last_β * Swxx[j]
            new_β = softtreshold(v, λ * p_f[j]) / Swxx[j]
            r += Gj * (last_β - new_β)

            maxΔ = max(maxΔ, Swxx[j] * (last_β - new_β)^2)
            β[j] = new_β
        end

        # Check termination condition before last iteration
        if criterion == :obj
            # Update μ
            μ = updateμ(r, Ytilde)

            # Update deviance and loss function
            prev_loss = loss
            dev = model_dev(family, b, U, w, r, eigvals, y, μ)
            loss = dev/2 + λ * sum(p_f[β.nzind] .* abs.(β.nzval))

            # Check termination condition
            converged = abs(loss - prev_loss) < cd_tol * loss 

        elseif criterion == :coef
            converged = maxΔ < cd_tol
        end
    end

    # Assess convergence of coordinate descent
    @assert converged "Coordinate descent failed to converge in $cd_maxiter iterations at λ = $λ"

    return(β, r)

end

modeltype(::Normal) = "Least Squares GLMNet"
modeltype(::Binomial) = "Logistic"

eigenweights(::Normal, eigvals::Vector{Float64}, φ::Float64) = Diagonal(eigvals ./(φ .+ eigvals))
eigenweights(::Binomial, eigvals::Vector{Float64}, kargs...) = Diagonal(eigvals ./(4 .+ eigvals))

struct pglmmPath{F<:Distribution, A<:AbstractArray, B<:AbstractArray}
    family::F
    a0::A                                       # intercept values for each solution
    betas::B                                    # coefficient values for each solution
    null_dev::Float64                           # Null deviance of the model
    pct_dev::Vector{Float64}                    # R^2 values for each solution
    lambda::Vector{Float64}                     # lamda values corresponding to each solution
    npasses::Int                                # actual number of passes over the
                                                # data for all lamda values
    fitted_values                               # fitted_values
    y::Union{Vector{Int64}, Vector{Float64}}    # eigenvalues vector
    UD_invUt::Matrix{Float64}                   # eigenvectors matrix times diagonal weights matrix
    τ::Vector{Float64}                          # estimated variance components
    intercept::Bool                             # boolean for intercept
end

function show(io::IO, g::pglmmPath)
    df = [length(findall(x -> x != 0, vec(view(g.betas, :,k)))) for k in 1:size(g.betas, 2)]
    println(io, "$(modeltype(g.family)) Solution Path ($(size(g.betas, 2)) solutions for $(size(g.betas, 1)) predictors):") #in $(g.npasses) passes):"
    print(io, CoefTable(Union{Vector{Int},Vector{Float64}}[df, g.pct_dev, g.lambda], ["df", "pct_dev", "λ"], []))
end

# Function to compute sequence of values for λ
function lambda_seq(
    r::Vector{T}, 
    X::Matrix{T},
    G::SubArray{T, 2, SnpLinAlg{T}, Tuple{Vector{Int64}, UnitRange{Int64}}}, 
    w::T,
    p_fX::Vector{T},
    p_fG::Vector{T},
    K::Integer = 100
    ) where T

    λ_min_ratio = (length(r) < size(G, 2) ? 1e-2 : 1e-4)
    λ_max = lambda_max(X, r, w, p_fX)
    λ_max = lambda_max(G, r, w, p_fG, λ_max)
    λ_min = λ_max * λ_min_ratio
    λ_step = log(λ_min_ratio)/(K - 1)
    λ_seq = exp.(collect(log(λ_max):λ_step:log(λ_min)))
    
    return λ_seq
end


# Function to compute λ_max
function lambda_max(X::Union{AbstractMatrix{T}, SnpLinAlg{T}}, r::AbstractVector{T}, w::T, p_f::AbstractVector{T}, λ_max::T = zero(T)) where T

    seq = findall(x-> x != 0, p_f)
    for j in seq
	    Xj = view(X, :,j)
        x = abs(w * dot(Xj, r))
        if x > λ_max
            λ_max = x
        end
    end
    return(λ_max)
end

# Define softtreshold function
function  softtreshold(z::Float64, γ::Float64) :: Float64
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
    r::Vector{T},
    μ::Vector{T},
    Ytilde::Vector{T}
) where T
    η = GLM.linkfun.(LogitLink(), μ)
    r += η + 4 * (y - μ) - Ytilde
    Ytilde = η + 4 * (y - μ)
    return(Ytilde, r)
end

# Function to update linear predictor and mean at each iteration
const PMIN = 1e-5
const PMAX = 1-1e-5
function updateμ(r::Vector{T}, Ytilde::Vector{T}) where T
    η = Ytilde - r
    μ = GLM.linkinv.(LogitLink(), η)
    μ = [μ[i] < PMIN ? PMIN : μ[i] > PMAX ? PMAX : μ[i] for i in 1:length(μ)]
    return(μ)
end

# Functions to calculate deviance
model_dev(::Binomial, b::Vector{T}, U::AbstractMatrix{T}, w::T, r::Vector{T}, eigvals::Vector{T}, y::Vector{Int64}, μ::Vector{Float64}) where T = LogisticDeviance(b, U, eigvals, y, μ)
model_dev(::Normal, b::Vector{T}, U::AbstractMatrix{T}, w::T, r::Vector{T}, eigvals::Vector{T}, kargs...) where T = NormalDeviance(b, U, w, r, eigvals)

function LogisticDeviance(b::Vector{T}, U::AbstractMatrix{T}, eigvals::Vector{T}, y::Vector{Int64}, μ::Vector{T}) where T
    bstar = b' * U
    -2 * sum(y .* log.(μ ./ (1 .- μ)) .+ log.(1 .- μ)) + dot(bstar, Diagonal(eigvals), bstar')
end

function NormalDeviance(b::Vector{T}, U::AbstractMatrix{T}, w::T, r::Vector{T}, eigvals::Vector{T}) where T
    bstar = b' * U
    w * dot(r, r) + dot(bstar, Diagonal(eigvals), bstar')
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
    η = path.a0[s]' .+ X * path.betas[:,s]

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
                  outtype = :response
                 ) where T
    
    #--------------------------------------------------------------
    # Read covariate file
    #--------------------------------------------------------------
    covdf = CSV.read(covfile, DataFrame)

    if !isnothing(covrowinds)
        covdf = covdf[covrowinds,:]
    end 

    if !isnothing(covrowinds)
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

    @assert nX == nrow(GRM) "GRM and covariates matrix must have same number of rows."
    
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
        
        # Read genotype and calculate mean and standard deviation
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
    @assert nG == nrow(GRM) "GRM and genotype matrix must have same number of rows."
    
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
    η = path.a0[s]' .+ X * path.betas[1:k,s] .+ G * path.betas[(k+1):p,s]

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
    p, K = size(path.betas)
    df = path.intercept .+ [length(findall(x -> x != 0, vec(view(path.betas, :, k)))) for k in 1:K] .+ length(path.τ)

    # Define GIC criterion
    if criterion == :BIC
        a_n = log(n)
    elseif criterion == :AIC
        a_n = 2
    elseif criterion == :HDBIC
        a_n = log(log(n)) * log(p)
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
        s = vec(std(X, dims = 1, corrected = false))
        if any(s .== zero(T))
            !intercept && @warn("One predictor is a constant, hence it can't been standardized!")
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

function compute_grad(Xj::AbstractVector{T}, w::AbstractVector{T}, r::AbstractVector{T}) where T
    dot(Xj, Diagonal(w), r)
end

function compute_grad(Xj::AbstractVector{T}, w::T, r::AbstractVector{T}) where T
    w * dot(Xj, r)
end

function compute_Swxx(Xj::AbstractVector{T}, w::AbstractVector{T}) where T
    dot(Xj, Diagonal(w), Xj)
end

function compute_Swxx(Xj::AbstractVector{T}, w::T) where T
    w * dot(Xj, Xj)
end