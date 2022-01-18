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
    plinkfile::AbstractString;
    # keyword arguments
    snpmodel = ADDITIVE_MODEL,
    snpinds::Union{Nothing,AbstractVector{<:Integer}} = nothing,
    geneticrowinds::Union{Nothing,AbstractVector{<:Integer}} = nothing,
    irwls_tol::Float64 = 1e-7,
    irwls_maxiter::Integer = 500,
    K_::Union{Nothing, Integer} = nothing,
    verbose::Bool = false,
    standardize_X::Bool = true,
    standardize_G::Bool = true,
    standardize_weight::Bool = false,
    criterion = :coef,
    earlystop::Bool = true,
    kwargs...
    )

    # read PLINK files
    geno = SnpArray(plinkfile * ".bed")
    
    # Convert genotype file to matrix, convert to additive model (default) and impute
    snpinds = isnothing(snpinds) ? (1:size(geno, 2)) : snpinds 
    geneticrowinds = isnothing(geneticrowinds) ? (1:size(geno, 1)) : geneticrowinds
    G = convert(Matrix{Float64}, @view(geno[geneticrowinds, snpinds]), model = snpmodel, impute = true)

    # Initialize number of subjects and predictors (including intercept)
    (n, p), k = size(G), size(nullmodel.X, 2)
    @assert n == length(nullmodel.y) "Genotype matrix and y must have same number of rows"

    # standardize non-genetic covariates and genetic predictors
    X, muX, sX = standardizeX(nullmodel.X, standardize_X, all(nullmodel.X[:,1] .== 1))
    G, muG, sG = standardizeX(G, standardize_G)

    # Initialize β and penalty factors
    β = sparse([nullmodel.α; zeros(p)])
    p_f = [zeros(k); ones(p)]

    # Spectral decomposition of sum(τ * V)
    eigvals, U = eigen(nullmodel.τV)

    # Define (normalized) weights for each observation
    w = weight(nullmodel.family, eigvals, nullmodel.φ)
    w_n = standardize_weight ? w / sum(w) : w

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

    # Calculate U * Diagonal(w) and define Ut = U'
    UD_inv = U * Diagonal(w)
    Ut = Matrix(U')

    # Transform X and Y
    Xstar = Ut * [X  G]
    Ystar = Ut * Ytilde

    # Define weighted sum of squares
    wXstar = w_n .* Xstar
    Swxx = vec(sum(wXstar .* Xstar, dims = 1))

    # Initialize residuals
    r = Ystar - view(Xstar, :, β.nzind) * β.nzval

    # Sequence of λ
    λ_seq, K = lambda_seq(Ystar, Xstar; weights = w_n, penalty_factor = p_f)
    K = isnothing(K_) ? K : K_
    
    # Fit penalized model
    path = pglmm_fit(nullmodel.family, Ytilde, y, Xstar, nulldev, r, w, β, p_f, λ_seq, K, UD_inv, Ut, wXstar, Swxx, verbose, irwls_tol, irwls_maxiter, criterion, earlystop)

    # Return coefficients on original scale
    if !isempty(sX)
        path.betas[2:k,:] = lmul!(inv(Diagonal(vec(sX))), path.betas[2:k,:])
        path.betas[1,:] -=  vec(muX * path.betas[2:k,:])
    end

    if !isempty(sG)
        path.betas[(k+1):end,:] = lmul!(inv(Diagonal(vec(sG))), path.betas[(k+1):end,:])
        path.betas[1,:] -=  vec(muG * path.betas[(k+1):end,:])
    end

    # Return lasso path
    pglmmPath(nullmodel.family, path.betas[1,:], path.betas[2:end,:], nulldev, path.pct_dev, path.λ, 0, path.residuals, path.fitted_means, U, eigvals, nullmodel.φ)
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
    nulldev::Float64,
    r::Vector{Float64},
    w::Vector{Float64},
    β::SparseVector{Float64, Int64},
    p_f::Vector{Float64},
    λ_seq::Vector{Float64},
    K::Int64,
    UD_inv::Matrix{Float64},
    Ut::Matrix{Float64},
    wXstar::Matrix{Float64},
    Swxx::Vector{Float64},
    verbose::Bool,
    irwls_tol::Float64,
    irwls_maxiter::Int64,
    criterion,
    earlystop::Bool  
)
    # Initialize array to store output for each λ
    sizeX = size(Xstar)
    betas = zeros(sizeX[2], K)
    pct_dev = zeros(Float64, K)
    dev_ratio = convert(Float64, NaN)
    residuals = zeros(sizeX[1], K)
    fitted_means = zeros(sizeX[1], K)
    μ = zeros(sizeX[1])

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
            β, r = cd_lasso(r, Xstar, wXstar, Swxx; family = Binomial(), Ytilde = Ytilde, y = y, w = w, UD_inv = UD_inv, β = β, p_f = p_f, λ = λ, criterion = criterion)

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
            Ytilde, r = wrkresp(y, r, μ, Ytilde, Ut)

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
        residuals[:, i] = r
        fitted_means[:, i] = μ

        # Test whether we should continue
        earlystop && ((last_dev_ratio - dev_ratio < MIN_DEV_FRAC_DIFF && length(β.nzind) > sum(p_f .==0)) || pct_dev[i] > MAX_DEV_FRAC) && break
    end

    return(betas = view(betas, :, 1:i), pct_dev = pct_dev[1:i], λ = λ_seq[1:i], residuals = view(residuals, :, 1:i), fitted_means = view(fitted_means, :, 1:i))
end

function pglmm_fit(
    ::Normal,
    Ytilde::Vector{Float64},
    y::Vector{Float64},
    Xstar::Matrix{Float64},
    nulldev::Float64,
    r::Vector{Float64},
    w::Vector{Float64},
    β::SparseVector{Float64, Int64},
    p_f::Vector{Float64},
    λ_seq::Vector{Float64},
    K::Int64,
    UD_inv::Matrix{Float64},
    Ut::Matrix{Float64},
    wXstar::Matrix{Float64},
    Swxx::Vector{Float64},
    verbose::Bool,
    irwls_tol::Float64,
    irwls_maxiter::Int64,
    earlystop::Bool  
)
    # Initialize array to store output for each λ
    betas = zeros(size(Xstar, 2), K)
    pct_dev = zeros(Float64, K)
    dev_ratio = convert(Float64, NaN)
    residuals = zeros(size(Xstar, 1), K)

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
        β, r = cd_lasso(r, Xstar, wXstar, Swxx; family = Normal(), β = β, p_f = p_f, λ = λ)

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

    return(betas = view(betas, :, 1:i), pct_dev = pct_dev[1:i], λ = λ_seq[1:i], residuals = view(residuals, :, 1:i), fitted_means = missing)
end

# Function to perform coordinate descent with a lasso penalty
function cd_lasso(
    # positional arguments
    r::Vector{Float64},
    X::Matrix{Float64},
    wX::Matrix{Float64}, 
    Swxx::Vector{Float64};
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
  
    for cd_iter in 1:cd_maxiter
        # At firs iteration, perform one coordinate cycle and 
        # record active set of coefficients that are nonzero
        if cd_iter == 1 || converged
            for j in 1:size(X, 2)
                v = dot(view(wX, :, j), r)
                λj = λ * p_f[j]

                last_β = β[j]
                if last_β != 0
                    v += last_β * Swxx[j]
                else
                    # Adding a new variable to the model
                    abs(v) < λj && continue
                end
                new_β = softtreshold(v, λj) / Swxx[j]
                r += view(X, :, j) * (last_β - new_β)

                maxΔ = max(maxΔ, Swxx[j] * (last_β - new_β)^2)
                β[j] = new_β
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

        for j in β.nzind
            last_β = β[j]
            last_β == 0 && continue
            
            v = dot(view(wX, :, j), r) + last_β * Swxx[j]
            new_β = softtreshold(v, λ * p_f[j]) / Swxx[j]
            r += view(X, :, j) * (last_β - new_β)

            maxΔ = max(maxΔ, Swxx[j] * (last_β - new_β)^2)
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

weight(::Normal, eigvals::Vector{Float64}, φ::Float64) = (1 ./(φ .+ eigvals))
weight(::Binomial, eigvals::Vector{Float64}, φ::Float64) = (1 ./(4 .+ eigvals))
struct pglmmPath{F<:Distribution, A<:AbstractArray, B<:AbstractArray}
    family::F
    a0::A                            # intercept values for each solution
    betas::B                         # coefficient values for each solution
    null_dev::Float64                # Null deviance of the model
    pct_dev::Vector{Float64}         # R^2 values for each solution
    lambda::Vector{Float64}          # lamda values corresponding to each solution
    npasses::Int                     # actual number of passes over the
                                     # data for all lamda values
    residuals                        # residuals
    fitted_means                     # fitted_means
    U::Matrix{Float64}               # eigenvectors matrix
    eigvals::Vector{Float64}         # eigenvalues vector
    φ::Float64                     # dispersion parameter
end

function show(io::IO, g::pglmmPath)
    df = [length(findall(x -> x != 0, vec(view(g.betas, :,k)))) for k in 1:size(g.betas, 2)]
    println(io, "$(modeltype(g.family)) Solution Path ($(size(g.betas, 2)) solutions for $(size(g.betas, 1)) predictors):") #in $(g.npasses) passes):"
    print(io, CoefTable(Union{Vector{Int},Vector{Float64}}[df, g.pct_dev, g.lambda], ["df", "pct_dev", "λ"], []))
end

# Function to compute sequence of values for λ
function lambda_seq(
    y::Vector{Float64}, 
    X::Matrix{Float64}; 
    weights::Vector{Float64},
    penalty_factor::Vector{Float64},
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
    Ut::Matrix{Float64}
)
    η = GLM.linkfun.(LogitLink(), μ)
    r += Ut * (η + 4 * (y - μ) - Ytilde)
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
        s = intercept ? std(X[:,2:end], dims = 1, corrected = false) : std(X, dims = 1, corrected = false)
        @assert all(s .!= 0) "One predictor is a constant, hence it cannot be standardize."
        X = intercept ? [X[:,1] (X[:,2:end] .- mu) ./ s] : (X .- mu) ./ s
    else
        mu = []; s = []
    end

    X, mu, s
end

# Predict phenotype for binary trait
function predict(path::pglmmPath{Binomial{Float64}, Vector{Float64}, Matrix{Float64}}, 
                 X::AbstractMatrix{T}, 
                 grmfile::AbstractString; 
                 grmrowinds::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                 grmcolinds::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                 s::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                 outtype = :response
                ) where T
    
    # read file containing the m x (N-m) kinship matrix between m test and (N-m) training subjects
    Kins = open(GzipDecompressorStream, grmfile, "r") do stream
        Symmetric(Matrix(CSV.read(stream, DataFrame)))
    end

    if !ismissing(grmrowinds)
        Kins = Kins[grmrowinds, :]
    end

    if !ismissing(grmcolinds)
        Kins = Kins[:, grmcolinds]
    end

    # Number of predictions to compute. User can provide index s for which to provide predictions, 
    # rather than computing predictions for the whole path.
    s = ismissing(s) ? (1:size(path.betas, 2)) : s

    # Variance-covariance of the training data
    W = [Diagonal(path.fitted_means[:,i] .* (1 .- path.fitted_means[:,i])) for i in s]
    D = Diagonal(path.eigvals)
    Σ_inv = [inv(16 .* path.U' * W[i] * path.U + D) for i in 1:length(W)]

    # Random effect for test data
    r = path.residuals[:, s]
    b = [Kins * path.U * Σ_inv[i] * r[:,i] for i in 1:length(Σ_inv)] |> x-> reduce(hcat, x)

    # Linear predictor
    η = path.a0[s]' .+ X * path.betas[:,s] + b
    if outtype == :response
        return(η)
    elseif outtype == :prob
        return(GLM.linkinv.(LogitLink(), η))
    end
end

# Predict phenotype for normal trait
function predict(path::pglmmPath{Normal{Float64}, Vector{Float64}, Matrix{Float64}}, 
                 X::AbstractMatrix{T}, 
                 grmfile::AbstractString; 
                 grmrowinds::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                 grmcolinds::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                 s::Union{Nothing,AbstractVector{<:Integer}} = nothing
                ) where T

    # read file containing the m x (N-m) kinship matrix between m test and (N-m) training subjects
    Kins = open(GzipDecompressorStream, grmfile, "r") do stream
        Symmetric(Matrix(CSV.read(stream, DataFrame)))
    end

    if !ismissing(grmrowinds)
        Kins = Kins[grmrowinds, :]
    end

    if !ismissing(grmcolinds)
        Kins = Kins[:, grmcolinds]
    end

    # Number of predictions to compute. User can provide index s for which to provide predictions, 
    # rather than computing predictions for the whole path.
    s = ismissing(s) ? (1:size(path.betas, 2)) : s

    # Linear predictor
    UDinv = path.U * Diagonal(weight(Normal(), path.eigvals, path.φ))
    η = path.a0[s]' .+ X * path.betas[:,s] + Kins * UDinv * path.residuals[:,s]
    return(η)
end 

# GIC penalty parameter
function GIC(path::pglmmPath, criterion)
    
    # Obtain number of rows (n), predictors (p) and λ values (K)
    n = size(path.residuals, 1)
    p, K = size(path.betas)
    df = [length(findall(x -> x != 0, vec(view(path.betas, :, k)))) for k in K]

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
    return(path.betas[:, argmin(GIC)])
end