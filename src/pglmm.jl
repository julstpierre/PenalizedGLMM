"""
    pglmm(nullformula, covfile, geneticfile; kwargs...)
    pglmm(nullformula, df, geneticfile; kwargs...)
# Positional arguments 
- `nullformula::FormulaTerm`: formula for the null model.
- `covfile::AbstractString`: covariate file (csv) with one header line, including the phenotype.  
- `df::DataFrame`: DataFrame containing response and regressors for null model.
- `plinkfile::AbstractString`: PLINK file name containing genetic information,
    without the .bed, .fam, or .bim extensions. Moreover, bed, bim, and fam file with 
    the same `geneticfile` prefix need to exist.
- `grmfile::AbstractString`: GRM file name.
# Keyword arguments
- `covrowinds::Union{Nothing,AbstractVector{<:Integer}}`: sample indices for covariate file.
- `family::UnivariateDistribution:` `Binomial()` (default)   
- `link::GLM.Link`: `LogitLink()` (default).
- `snpmodel`: `ADDITIVE_MODEL` (default), `DOMINANT_MODEL`, or `RECESSIVE_MODEL`.
- `snpinds::Union{Nothing,AbstractVector{<:Integer}}`: SNP indices for bed/vcf file.
- `geneticrowinds::Union{Nothing,AbstractVector{<:Integer}}`: sample indices for bed/vcf file.
"""
function pglmm(
    # positional arguments
    nullformula::FormulaTerm,
    covfile::AbstractString,
    plinkfile::AbstractString,
    grmfile::AbstractString;
    # keyword arguments
    family::UnivariateDistribution = Binomial(),
    link::GLM.Link = LogitLink(),
    M::Union{Nothing, Vector{Any}} = nothing,
    tol::Float64 = 10^-6,
    maxiter::Integer = 10,
    kwargs...
    )

    #--------------------------------------------------------------
    # Read input files
    #--------------------------------------------------------------
    # read covariate file
    covdf = CSV.read(covfile, DataFrame)

    # read PLINK files
    geno = SnpArray(plinkfile * ".bed")

    # read grm file
    grm = open(GzipDecompressorStream, grmfile, "r") do stream
        Matrix(CSV.read(stream, DataFrame))
    end

    # Initialize number of subjects and genetic predictors
    n, p = size(grm)

    #--------------------------------------------------------------
    # Estimation of variance components under H0
    #--------------------------------------------------------------
    # fit null GLM
    nullfit = glm(nullformula, covdf, family, link)
    
    # Define the design matrix
    X = modelmatrix(nullfit)

    # Obtain initial values for alpha
    alpha0 = coef(nullfit)

    # Obtain initial values for Ytilde
    y = response(nullformula, covdf)
    mu = predict(nullfit)
    eta = GLM.linkfun.(link, mu)
    Ytilde = eta + dg(link, mu) .* (y - mu)

    # Number of variance components in the model
    if isnothing(M) 
        K = 1
        V = push!(Any[], grm)
    else 
        K = 1 + size(M, 1)
        V = push!(M, grm)
    end

    # For Normal family, dispersion parameter needs to be estimated
    if family == Normal()
        K = K + 1
        V = push!(V, Diagonal(ones(n)))
    end

    # Obtain initial values for variance components
    theta0 = fill(var(Ytilde) / K, K)

    # Initialize Sigma and Projection matrix
    Sigma_inv, P = Sigma_P_mats(theta0, family, link, mu, V, X)

    # Update variance components
    theta0 = theta0 + 2 * n^-1  * theta0.^2 .* dql_R(Ytilde, P, V, K)

    # Update Sigma and Projection matrix
    Sigma_inv, P = Sigma_P_mats(theta0, family, link, mu, V, X)
    
    # Initialize number of steps
    nsteps = 1

    # Iterate until convergence
    while true
        # Update variance components estimates
        theta = max.(theta0 + AI(Ytilde, P, V, K) \ dql_R(Ytilde, P, V, K), 10^-6 * var(Ytilde))
        Sigma_inv, P = Sigma_P_mats(theta, family, link, mu, V, X)
        
        # Update mean estimates
        alpha = (X' * Sigma_inv * X) \ X' * Sigma_inv * Ytilde
        b = sum(theta .* V) * Sigma_inv * (Ytilde - X * alpha)

        # Update working response
        eta = X * alpha + b
        mu = GLM.linkinv.(link, eta)
        Ytilde = eta + dg(link, mu) .* (y - mu)

        # Check termination conditions
        if  2 * max(norm(alpha - alpha0) / (norm(alpha) + norm(alpha0)), 
                    norm(theta - theta0) / (norm(theta) + norm(theta0))) <= tol || nsteps > maxiter break
        else
            theta0 = theta
            alpha0 = alpha
            nsteps += 1
        end
    end

    return(alpha, b, theta)
end