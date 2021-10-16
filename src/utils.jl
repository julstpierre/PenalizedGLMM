# Define the derivative of link function g at the mean value mu
function dg(link::GLM.Link, mu::Array{Float64})
    if link == LogitLink()
        1 ./ (mu .* (1 .- mu))
    elseif link == IdentityLink()
        1
    end
end

# Define the weights for link function g, variance function v(mu) at the mean value mu
function weights(family::UnivariateDistribution, link::GLM.Link, mu::Array{Float64}, phi::Float64)
    if family == Binomial() && link == LogitLink()
        Diagonal(mu .* (1 .- mu))
    elseif family == Normal() && link == IdentityLink()
        1/phi * Diagonal(ones(n))
    end
end

# Define Sigma and Projection matrices
function Sigma_P_mats(theta::Vector{Float64}, 
                      family::UnivariateDistribution, 
                      link::GLM.Link, 
                      mu::Vector{Float64},
                      V::Vector{Any},
                      X::Matrix{Float64})
    Sigma_inv = inv(weights(family, link, mu, theta[end])^-1 + sum(theta .* V))
    P = Sigma_inv - Sigma_inv * X * (X' * Sigma_inv * X)^-1 * X' * Sigma_inv

    return(Sigma_inv, P)
end

# Define the partial derivative of the restricted quasi-likelihood with respect to variance components
function dql_R(Ytilde::Vector{Float64}, P::Matrix{Float64}, V::Vector{Any}, K::Integer)
    [1 / 2 * (Ytilde' * P * V[k] * P * Ytilde - tr(P * V[k])) for k in 1:K]
end

# Define the average information matrix AI
function AI(Ytilde::Vector{Float64}, P::Matrix{Float64}, V::Vector{Any}, K::Integer)
    AI = Array{Float64}(undef, K, K)
    for k in 1:K
        for l in 1:K
            AI[k, l] = 1 / 2 * Ytilde' * P * V[k] * P * V[l] * P * Ytilde
        end
    end

    return AI
end