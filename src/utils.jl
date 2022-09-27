# Inverse the random effects covariance matrix using Sherman-Morison formula
function sherman(A::Symmetric{T, Matrix{T}}) where T

        n = size(A, 1)
        I, R = 0.5 * Diagonal(ones(n)), Matrix(Diagonal(ones(n)))
        y = 0.5 * view(A, :, 1)
    
	for j = 1:n
		R[:, 1:j] -= view(y * view(I, j, :)' / (1 + view(I, j, :)' * y), :, j) * view(R, j, 1:j)'
		y = R * 0.5 * view(A, :, min(j + 1, n))
	end

	return R * A
end

# In-place multiplication of predictor matrix with eigenvectors matrix U'
function Umul!(U::AbstractMatrix{T}, X::AbstractMatrix{T}; K::Integer = 1000) where T
    n, p = size(X)
    b = similar(X, n, K)
    jseq = collect(1:K:p)[1:(end-1)]

    @inbounds for j in jseq
        X[:, j:(j+K-1)] = mul!(b, U', view(X, :, j:(j+K-1)))
    end

    lastt = 0
    if (length(jseq)>0)
            lastt = last(jseq)
    end

    b = similar(X, n, length((lastt + K):p))
    X[:, (lastt + K):p] = mul!(b, U', view(X, :, (lastt + K):p))
    return(X)
end