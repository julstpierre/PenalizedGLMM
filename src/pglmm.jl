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
- `irwls_maxiter::Integer = 300 (default)`: maximum number of Newton iterations for the IRWLS loop.
- `criterion = :coef (default)`: criterion for coordinate descent convergence.
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
    irwls_tol::Float64 = 1e-7,
    irwls_maxiter::Integer = 500,
    K_::Union{Nothing, Integer} = nothing,
    verbose::Bool = false,
    standardize_X::Bool = true,
    standardize_G::Bool = true,
    standardize_weights::Bool = false,
    criterion = :coef,
    earlystop::Bool = true,
    kwargs...
    )

    # Read genotype file
    if !isnothing(plinkfile)
        # read PLINK files
        geno = SnpArray(plinkfile * ".bed")
        
        # Convert genotype file to matrix, convert to additive model (default) and impute
        snpinds = isnothing(snpinds) ? (1:size(geno, 2)) : snpinds 
        geneticrowinds = isnothing(geneticrowinds) ? (1:size(geno, 1)) : geneticrowinds
        G = convert(Matrix{Float64}, @view(geno[geneticrowinds, snpinds]), model = snpmodel, impute = true)
    elseif !isnothing(snpfile)
        # read CSV file
        geno = CSV.read(snpfile, DataFrame)
        
        # Convert genotype file to matrix, convert to additive model (default) and impute
        snpinds = isnothing(snpinds) ? (1:size(geno, 2)) : snpinds 
        geneticrowinds = isnothing(geneticrowinds) ? (1:size(geno, 1)) : geneticrowinds
        G = geno[geneticrowinds, snpinds]
    end

    # Initialize number of subjects and predictors (including intercept)
    (n, p), k = size(G), size(nullmodel.X, 2)
    @assert n == length(nullmodel.y) "Genotype matrix and y must have same number of rows"

    # standardize non-genetic covariates and genetic predictors
    intercept = all(nullmodel.X[:,1] .== 1)
    X, muX, sX = standardizeX(nullmodel.X, standardize_X, intercept)
    G, muG, sG = standardizeX(G, standardize_G)

    # Initialize β and penalty factors
    β = sparse([nullmodel.α; zeros(p)])
    p_f = [zeros(k); ones(p)]

    # Spectral decomposition of sum(τ * V)
    eigvals, U = eigen(nullmodel.τV)

    # Define (normalized) weights for each observation
    w = eigenweights(nullmodel.family, eigvals, nullmodel.φ)
    w_n = standardize_weights ? w / sum(w) : w

    # Initialize working variable and mean vector and initialize null deviance 
    # based on model with intercept only and no random effects
    y = nullmodel.y
    if nullmodel.family == Binomial()
        μ, ybar = GLM.linkinv.(LogitLink(), nullmodel.η), mean(y)
        Ytilde = nullmodel.η + 4 * (y - μ)
        nulldev = -2 * sum(y * log(ybar / (1 - ybar)) .+ log(1 - ybar))
    elseif nullmodel.family == Normal()
        Ytilde = y
        nulldev = var(y) * (n - 1) / nullmodel.φ
    end

    # Calculate U * Diagonal(w)
    UD_inv = U * Diagonal(w)

    # Transform X and G
	Xstar = Umul!(U, X)
    Gstar = Umul!(U, G)

    # Transform Y
    Ystar = Array{Float64}(undef, n)
	mul!(Ystar, U', Ytilde)

    # Initialize residuals
    r = Ystar - Xstar * β.nzval

    # Sequence of λ
    λ_seq, K = lambda_seq(Ystar, Xstar, Gstar; weights = w_n, p_f = p_f)
    K = isnothing(K_) ? K : K_
    
    # Fit penalized model
    path = pglmm_fit(nullmodel.family, Ytilde, y, Xstar, Gstar, nulldev, r, w, β, p_f, λ_seq, K, UD_inv, U, verbose, irwls_tol, irwls_maxiter, criterion, earlystop, intercept)

    # Return coefficients on original scale
    if !isempty(sX) & !isempty(sG)
        lmul!(inv(Diagonal(vec([sX; sG]))), path.betas)
        path.a0 .-= vec([muX muG] * path.betas)
    elseif !isempty(sX)
        k = k - intercept
        path.betas[1:k,:] = lmul!(inv(Diagonal(vec(sX))), path.betas[1:k,:])
        path.a0 .-=  vec(muX * path.betas[1:k,:])
    elseif !isempty(sG)
        k = k - intercept
        path.betas[(k+1):end,:] = lmul!(inv(Diagonal(vec(sG))), path.betas[(k+1):end,:])
        path.a0 .-=  vec(muG * path.betas[(k+1):end,:])
    end

    # Return lasso path
    pglmmPath(nullmodel.family, path.a0, path.betas, nulldev, path.pct_dev, path.λ, 0, path.fitted_values, y, UD_inv)
end

# Controls early stopping criteria with automatic λ
const MIN_DEV_FRAC_DIFF = 1e-5
const MAX_DEV_FRAC = 0.999

# Function to fit a penalised mixed model
function pglmm_fit(
    ::Binomial,
    Ytilde::Vector{Float64},
    y::Vector{Int64},
    Xstar::Matrix{Float64},
    Gstar::Matrix{Float64},
    nulldev::Float64,
    r::Vector{Float64},
    w::Vector{Float64},
    β::SparseVector{Float64, Int64},
    p_f::Vector{Float64},
    λ_seq::Vector{Float64},
    K::Int64,
    UD_inv::Matrix{Float64},
    U::Matrix{Float64},
    verbose::Bool,
    irwls_tol::Float64,
    irwls_maxiter::Int64,
    criterion,
    earlystop::Bool,
    intercept::Bool
)
    # Initialize array to store output for each λ
    betas = zeros(size(β, 1), K)
    pct_dev = zeros(Float64, K)
    dev_ratio = convert(Float64, NaN)
    fitted_means = zeros(length(y), K)
    μ = zeros(length(y))

    # Loop through sequence of λ
    i = 0
    for _ = 1:K
        # Next iterate
        i += 1
        converged = false
        
        # Current value of λ
        λ = λ_seq[i]

        # Initialize objective function and save previous deviance ratio
        dev = loss = convert(Float64, Inf)
        last_dev_ratio = dev_ratio
        
        # Penalized iterative weighted least squares (IWLS)
        for irwls in 1:irwls_maxiter

            # Run coordinate descent inner loop to update β and r
            β, r = cd_lasso(r, Xstar, Gstar; family = Binomial(), Ytilde = Ytilde, y = y, w = w, UD_inv = UD_inv, β = β, p_f = p_f, λ = λ, criterion = criterion)

            # Update μ
            μ = updateμ(r, Ytilde, UD_inv)
            
            # Update deviance and loss function
            prev_loss = loss
            dev = LogisticDeviance(y, r, w, μ)
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
                    dev = LogisticDeviance(y, r, w, μ)
                    loss = dev/2 + λ * sum(p_f[β.nzind] .* abs.(β.nzval))
                end =#
            end 

            # Update working response and residuals
            Ytilde, r = wrkresp(y, r, μ, Ytilde, U)

            # Check termination conditions
            converged = abs(loss - prev_loss) < irwls_tol * loss
            converged && verbose && println("Number of irwls iterations = $irwls at $i th value of λ.")
            converged && break 

        end
        @assert converged "IRWLS failed to converge in $irwls_maxiter iterations at λ = $λ"

        # Store ouput from IRWLS loop
        betas[:, i] = convert(Vector{Float64}, β)
        dev_ratio = dev/nulldev
        pct_dev[i] = 1 - dev_ratio
        fitted_means[:, i] = μ

        # Test whether we should continue
        earlystop && ((last_dev_ratio - dev_ratio < MIN_DEV_FRAC_DIFF && length(β.nzind) > sum(p_f .==0)) || pct_dev[i] > MAX_DEV_FRAC) && break
    end

    if intercept
        return(a0 = view(betas, 1, 1:i), betas = view(betas, 2:sizeX[2], 1:i), pct_dev = pct_dev[1:i], λ = λ_seq[1:i], fitted_values = view(fitted_means, :, 1:i))
    else
        return(a0 = zeros(i), betas = view(betas, :, 1:i), pct_dev = pct_dev[1:i], λ = λ_seq[1:i], fitted_values = view(fitted_means, :, 1:i))
    end
end

function pglmm_fit(
    ::Normal,
    Ytilde::Vector{Float64},
    y::Vector{Float64},
    Xstar::Matrix{Float64},
    Gstar::Matrix{Float64},
    nulldev::Float64,
    r::Vector{Float64},
    w::Vector{Float64},
    β::SparseVector{Float64, Int64},
    p_f::Vector{Float64},
    λ_seq::Vector{Float64},
    K::Int64,
    UD_inv::Matrix{Float64},
    U::Matrix{Float64},
    verbose::Bool,
    irwls_tol::Float64,
    irwls_maxiter::Int64,
    earlystop::Bool,
    intercept::Bool  
)
    # Initialize array to store output for each λ
    betas = zeros(size(β, 1), K)
    pct_dev = zeros(Float64, K)
    dev_ratio = convert(Float64, NaN)
    residuals = zeros(length(y), K)

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

        # Run coordinate descent inner loop to update β and r
        β, r = cd_lasso(r, Xstar, Gstar; family = Normal(), β = β, p_f = p_f, λ = λ)

        # Update deviance
        dev = NormalDeviance(r, w)

        # Store ouput
        betas[:, i] = convert(Vector{Float64}, β)
        dev_ratio = dev/nulldev
        pct_dev[i] = 1 - dev_ratio
        residuals[:, i] = r

        # Test whether we should continue
        earlystop && ((last_dev_ratio - dev_ratio < MIN_DEV_FRAC_DIFF && length(β.nzind) > sum(p_f .==0))|| pct_dev[i] > MAX_DEV_FRAC) && break
    end

    if intercept
        return(a0 = view(betas, 1, 1:i), betas = view(betas, 2:sizeX[2], 1:i), pct_dev = pct_dev[1:i], λ = λ_seq[1:i], fitted_values = view(fitted_means, :, 1:i))
    else
        return(a0 = zeros(i), betas = view(betas, :, 1:i), pct_dev = pct_dev[1:i], λ = λ_seq[1:i], fitted_values = view(fitted_means, :, 1:i))
    end
end

# Function to perform coordinate descent with a lasso penalty
function cd_lasso(
    # positional arguments
    r::Vector{Float64},
    X::Matrix{Float64},
    G::Matrix{Float64};
    #keywords arguments
    family::UnivariateDistribution,
    β::SparseVector{Float64},
    Ytilde::Vector{Float64},
    y::Vector{Int64},
    w::Vector{Float64}, 
    UD_inv::Matrix{Float64},
    p_f::Vector{Float64} = ones(size(X, 2)), 
    λ::Float64,
    cd_maxiter::Integer = 100000,
    cd_tol::Real=1e-8,
    criterion
    )

    converged = false
    maxΔ = zero(Float64)
    loss = Inf
    k = size(X, 2)

    for cd_iter in 1:cd_maxiter
        # At firs iteration, perform one coordinate cycle and 
        # record active set of coefficients that are nonzero
        if cd_iter == 1 || converged
            # Non-genetic covariates
            for j in 1:k
                v = dot(view(X, :, j), Diagonal(w), r)
                λj = λ * p_f[j]
                Swxxj = dot(view(X, :, j), Diagonal(w), view(X, :, j))

                last_β = β[j]
                if last_β != 0
                    v += last_β * Swxxj
                else
                    # Adding a new variable to the model
                    abs(v) < λj && continue
                end
                new_β = softtreshold(v, λj) / Swxxj
                r += view(X, :, j) * (last_β - new_β)

                maxΔ = max(maxΔ, Swxxj * (last_β - new_β)^2)
                β[j] = new_β
            end

            # Genetic covariates
            for j in 1:size(G, 2)
                v = dot(view(G, :, j), Diagonal(w), r)
                λj = λ * p_f[k + j]
                Swxxj = dot(view(G, :, j), Diagonal(w), view(G, :, j))

                last_β = β[k + j]
                if last_β != 0
                    v += last_β * Swxxj
                else
                    # Adding a new variable to the model
                    abs(v) < λj && continue
                end
                new_β = softtreshold(v, λj) / Swxxj
                r += view(G, :, j) * (last_β - new_β)

                maxΔ = max(maxΔ, Swxxj * (last_β - new_β)^2)
                β[k + j] = new_β
            end

            # Check termination condition at last iteration
            if criterion == :obj
                # Update μ
                μ = updateμ(r, Ytilde, UD_inv)

                # Update deviance and loss function
                prev_loss = loss
                dev = model_dev(family, y, r, w, μ)
                loss = dev/2 + λ * sum(p_f[β.nzind] .* abs.(β.nzval))

                # Check termination condition
                converged && abs(loss - prev_loss) < cd_tol * loss && break

            elseif criterion == :coef
                converged && maxΔ < cd_tol && break
            end
        end

        # Cycle over coefficients in active set only until convergence
        maxΔ = zero(Float64)
        
        # Non-genetic covariates
        for j in β.nzind[β.nzind .<= k]
            last_β = β[j]
            last_β == 0 && continue
            
            Swxxj = dot(view(X, :, j), Diagonal(w), view(X, :, j))
            v = dot(view(X, :, j), Diagonal(w), r) + last_β * Swxxj
            new_β = softtreshold(v, λ * p_f[j]) / Swxxj
            r += view(X, :, j) * (last_β - new_β)

            maxΔ = max(maxΔ, Swxxj * (last_β - new_β)^2)
            β[j] = new_β
        end

        # Genetic predictors
        for j in β.nzind[β.nzind .> k]
            last_β = β[j]
            last_β == 0 && continue
            
            Swxxj = dot(view(G, :, j-k), Diagonal(w), view(G, :, j-k))
            v = dot(view(G, :, j-k), Diagonal(w), r) + last_β * Swxxj
            new_β = softtreshold(v, λ * p_f[j]) / Swxxj
            r += view(G, :, j-k) * (last_β - new_β)

            maxΔ = max(maxΔ, Swxxj * (last_β - new_β)^2)
            β[j] = new_β
        end

        # Check termination condition before last iteration
        if criterion == :obj
            # Update μ
            μ = updateμ(r, Ytilde, UD_inv)

            # Update deviance and loss function
            prev_loss = loss
            dev = model_dev(family, y, r, w, μ)
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

eigenweights(::Normal, eigvals::Vector{Float64}, φ::Float64) = (1 ./(φ .+ eigvals))
eigenweights(::Binomial, eigvals::Vector{Float64}, φ::Float64) = (1 ./(4 .+ eigvals))

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
    UD_inv::Matrix{Float64}                     # eigenvectors matrix times diagonal weights matrix
end

function show(io::IO, g::pglmmPath)
    df = [length(findall(x -> x != 0, vec(view(g.betas, :,k)))) for k in 1:size(g.betas, 2)]
    println(io, "$(modeltype(g.family)) Solution Path ($(size(g.betas, 2)) solutions for $(size(g.betas, 1)) predictors):") #in $(g.npasses) passes):"
    print(io, CoefTable(Union{Vector{Int},Vector{Float64}}[df, g.pct_dev, g.lambda], ["df", "pct_dev", "λ"], []))
end

# Function to compute sequence of values for λ
function lambda_seq(
    y::Vector{Float64}, 
    X::Matrix{Float64},
    G::Matrix{Float64}; 
    weights::Vector{Float64},
    p_f::Vector{Float64},
    K::Integer = 100
    )

    λ_min_ratio = (length(y) < size(G, 2) ? 1e-2 : 1e-4)
    λ_max = lambda_max(X, y, weights, p_f[1:size(X, 2)])
    λ_max = lambda_max(G, y, weights, p_f[(size(X, 2) + 1):end], λ_max)
    λ_min = λ_max * λ_min_ratio
    λ_step = log(λ_min_ratio)/(K - 1)
    λ_seq = exp.(collect(log(λ_max):λ_step:log(λ_min)))
    
    return(λ_seq, length(λ_seq))
end

# Function to compute λ_max
function lambda_max(X::AbstractMatrix{T}, y::AbstractVector{T}, w::AbstractVector{T}, p_f::AbstractVector{T}, λ_max::T = zero(T)) where T

    wY = w .* y
    seq = findall(x-> x .!= 0, p_f)
    for j in seq
        x = abs(dot(view(X, :,j), wY))
        if x > λ_max
            λ_max = x
        end
    end
    return(λ_max)
end

# Define Softtreshold function
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
    r::Vector{Float64},
    μ::Vector{Float64},
    Ytilde::Vector{Float64},
    U::Matrix{Float64}
)
    η = GLM.linkfun.(LogitLink(), μ)
    r += U' * (η + 4 * (y - μ) - Ytilde)
    Ytilde = η + 4 * (y - μ)
    return(Ytilde, r)
end

# Function to update linear predictor and mean at each iteration
const PMIN = 1e-5
const PMAX = 1-1e-5
function updateμ(r::Vector{Float64}, Ytilde::Vector{Float64}, UD_inv::Matrix{Float64})
    η = Ytilde - 4 * UD_inv * r
    μ = GLM.linkinv.(LogitLink(), η)
    μ = [μ[i] < PMIN ? PMIN : μ[i] > PMAX ? PMAX : μ[i] for i in 1:length(μ)]
    return(μ)
end

# Functions to calculate deviance
model_dev(::Binomial, y::Vector{Int64}, r::Vector{Float64}, w::Vector{Float64}, μ::Vector{Float64}) = LogisticDeviance(y, r, w, μ)
model_dev(::Normal, y::Vector{Int64}, r::Vector{Float64}, w::Vector{Float64}, μ::Vector{Float64}) = NormalDeviance(r, w)

function LogisticDeviance(y::Vector{Int64}, r::Vector{Float64}, w::Vector{Float64}, μ::Vector{Float64})
    -2 * sum(y .* log.(μ ./ (1 .- μ)) .+ log.(1 .- μ)) + sum(w .* (1 .- 4 * w) .* r.^2)
end

function NormalDeviance(r::Vector{Float64}, w::Vector{Float64})
    dot(w, r.^2)
end

# Standardize predictors for lasso
function standardizeX(X::AbstractMatrix{T}, standardize::Bool, intercept::Bool = false) where T
    if standardize
        mu = intercept ? mean(X[:,2:end], dims = 1) : mean(X, dims = 1) 
        s = intercept ? vec(std(X[:,2:end], dims = 1, corrected = false)) : vec(std(X, dims = 1, corrected = false))
        @assert all(s .!= 0) "One predictor is a constant, hence it cannot be standardize."
        if intercept
			for j in 2:size(X,2), i in 1:size(X, 1) 
                @inbounds X[i,j] = (X[i,j] .- mu[j-1]) / s[j-1]
            end
		else
			for j in 1:size(X,2), i in 1:size(X, 1) 
                @inbounds X[i,j] = (X[i,j] .- mu[j]) / s[j]
            end
		end
    else
        mu = []; s = []
    end

    X, mu, s
end

# Predict phenotype for normal trait
function predict(path::pglmmPath, 
                  X::AbstractMatrix{T}, 
                  grmfile::AbstractString; 
                  grmrowinds::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                  grmcolinds::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                  s::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                  fixed_effects_only::Bool = false,
                  outtype = :response
                 ) where T

    # read file containing the m x (N-m) kinship matrix between m test and (N-m) training subjects
    Kins = open(GzipDecompressorStream, grmfile, "r") do stream
        Symmetric(Matrix(CSV.read(stream, DataFrame)))
    end

    if !isnothing(grmrowinds)
        Kins = Kins[grmrowinds, :]
    end

    if !isnothing(grmcolinds)
        Kins = Kins[:, grmcolinds]
    end

    # Number of predictions to compute. User can provide index s for which to provide predictions, 
    # rather than computing predictions for the whole path.
    s = isnothing(s) ? (1:size(path.betas, 2)) : s

    # Linear predictor
    η = path.a0[s]' .+ X * path.betas[:,s]

    if fixed_effects_only == false
        if path.family == Binomial()
            η += Kins * (path.y .- path.fitted_values[:,s])
        elseif path.family == Normal()
            η += Kins * path.UD_inv * path.fitted_values[:,s]
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
    df = [length(findall(x -> x != 0, vec(view(path.betas, :, k)))) for k in 1:K]

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

# In-place multiplication of predictor matrix with eigenvectors matrix U'
function Umul!(U::AbstractMatrix{T}, X::AbstractMatrix{T}; K::Integer = 1) where T
    n, p = size(X)
    b = similar(X, n, K)
    jseq = collect(1:K:p)[1:(end-1)]

    @inbounds for j in jseq
        X[:, j:(j+K-1)] = mul!(b, U', view(X, :, j:(j+K-1)))
    end

    b = similar(X, n, length((last(jseq) + K):p))
    X[:, (last(jseq) + K):p] = mul!(b, U', view(X, :, (last(jseq) + K):p))
    return(X)
end