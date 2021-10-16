using LinearAlgebra

# Create structure
struct Distribution
    score_function::Function
    hessian_function::Function
end    

# Binomial with logit link (canonical)
function binomial_score(y, x, beta)
    x' * (y - exp.(x * beta)./(1 .+ exp.(x * beta)))
end
function binomial_hessian(x, beta, Phi)
    - x' * (inv(Diagonal(exp.(x * beta)./(1 .+ exp.(x * beta)).^2)) + Phi)^-1 * x
end

binomial = Distribution(binomial_score,
                        binomial_hessian)

# Function to return score at beta
score(y, x, dist::Distribution, beta) = dist.score_function(y, x, beta)

# Functions to return hessian at beta
hessian(x, dist::Distribution, beta, Phi) = dist.hessian_function(x, beta, Phi)

# Fisher's scoring method
function Fisher_scoring(y, x, dist::Distribution, beta; Phi, method, B, alpha = 1, bfgsdamptol = 0.2)
    """
    Some nice documentation here.
    """
    # Caculate score at current iterate
    S = score(y, x, dist, beta)

    # Caculate inverse of hessian at current iterate (if needed)
    if method == "Newton"
        H = hessian(x, dist, beta, Phi)
        return(beta - alpha * H \ S, nothing)
    elseif method == "LB"
        return(beta - alpha * B * S, B)
    elseif method == "BFGS"
        s = alpha * B * S
        beta = beta - s
        z = S - score(y, x, dist, beta)
        B = B - B * (z * z') * B / (z' * B * z) + s * s' / (z' * s)
        return(beta, B)
    end
end

# Implement Fisher's scoring method to find MLE of beta in a glmm
function glmm_fit(y, x, dist::Distribution, Phi; beta = nothing, opttol=10^-6, method = "Newton", weights = 0.25)
    """
    Some nice documentation here.
    """
    # Size of data
    n, p = size(x)

    # Initial value for beta
    if isnothing(beta) beta = zeros(p) end

    # Initialize iteration counter
    k = 0

    # Evaluate score function
    S = score(y, x, dist, beta)

    # Precompute approximate inverse hessian if method is "LB" or "BFGS"
    if method == "Newton"
        B = nothing
    elseif method in("LB", "BFGS")
        B = - inv(x' * inv(1/weights * Diagonal(ones(n)) + Phi) * x)
    end

    # Evaluate score norm
    norm_S = norm(S)

    # Store initial score norm
    norm_S0 = norm_S
    
    # Iteration loop
    while true
        # Check termination conditions
        if norm_S <= opttol * max(norm_S0, 1) break
        else 
            # Update iterate
            beta, B = Fisher_scoring(y, x, dist, beta; Phi, method, B)

            # Evaluate score norm
            S = score(y, x, dist, beta)
            norm_S = norm(S)

            # Increment iteration counter
            k += 1
        end    
    end

    return round.(beta, digits = 3), k
end