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

# Convert sparse matrix into BlockDiagonal matrix
function BlockDiagonal(A::SparseMatrixCSC)
    rows, cols, vals = findnz(A)
    i = 1
    V = Matrix{Float64}[]

    while i <= last(cols)
        ix = rows[findall(cols .== i)]
        iy = findall(!isnothing, indexin(rows, ix))
        V_ = reshape(vals[iy], length(ix), Int(length(iy)/length(ix)))
        push!(V, V_)
        i += size(V_, 2)
    end
    
    BlockDiagonal(V)
end

# Read sparse GRM from .rds object and convert into BlockDiagonal matrix
function read_sparse_grm(grmfile::AbstractString, indvs::AbstractVector)
    GRM_sp = sparse(rcopy(R"as.matrix(readRDS($grmfile))"))
    GRM_ids = rcopy(R"colnames(readRDS($grmfile))")
    grmrowinds = indexin(intersect(GRM_ids, indvs), GRM_ids)

    # Convert sparse GRM to BD matrix using only training subjects
    GRM = BlockDiagonal(GRM_sp[grmrowinds, grmrowinds])

    return(GRM, GRM_ids[grmrowinds])

end

# Function to perform half-vectorization operator
function vech(A::AbstractMatrix{T}) where T
    m = LinAlg.checksquare(A)
    v = Vector{T}((m*(m+1))>>1)
    k = 0
    for j = 1:m, i = j:m
        @inbounds v[k += 1] = A[i,j]
    end
    return v
end