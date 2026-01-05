# CAMPS.jl/src/gf2.jl
# GF(2) linear algebra operations for bond dimension prediction

#==============================================================================#
# OVERVIEW
#
# The key theorem from Liu & Clark (arXiv:2412.17209):
#
# For t twisted Paulis P[1], ..., P[t], construct the GF(2) matrix M where:
#     M[k, j] = 1 if P[k] has X or Y at qubit j (i.e., xbit(P[k])[j])
#     M[k, j] = 0 if P[k] has I or Z at qubit j
#
# Then the final bond dimension is:
#     χ = 2^(t - rank(M))
#
# where rank is computed over GF(2) (binary field, addition = XOR).
#
# Intuition:
# - Each row of M corresponds to a T-gate (twisted Pauli)
# - Linearly dependent rows (in GF(2)) correspond to "redundant" T-gates
#   that can be disentangled using OFD
# - rank(M) = number of "truly entangling" T-gates
# - t - rank(M) = number of T-gates that increase bond dimension
#==============================================================================#

#==============================================================================#
# GF(2) MATRIX CONSTRUCTION
#==============================================================================#

"""
    build_gf2_matrix(twisted_paulis::Vector{<:PauliOperator}) -> Matrix{Bool}

Build the GF(2) matrix from a list of twisted Paulis.

The matrix M has dimensions (t × n) where t is the number of twisted Paulis
and n is the number of qubits. Entry M[k, j] = 1 if twisted Pauli k has X or Y
at qubit j (i.e., xbit is true).

# Arguments
- `twisted_paulis::Vector{<:PauliOperator}`: Vector of twisted Pauli operators

# Returns
- `Matrix{Bool}`: GF(2) matrix where M[k,j] = xbit(twisted_paulis[k])[j]

# Example
```julia
P1 = P"XZI"  # xbit = [1, 0, 0]
P2 = P"ZYI"  # xbit = [0, 1, 0]
P3 = P"YYI"  # xbit = [1, 1, 0]

M = build_gf2_matrix([P1, P2, P3])
# M = [1 0 0;
#      0 1 0;
#      1 1 0]
# Row 3 = Row 1 XOR Row 2 (in GF(2)), so rank = 2
```

# Theory
This matrix encodes which T-gates "overlap" in terms of having X or Y on the
same qubits. The rank determines how many T-gates are truly independent and
thus contribute to bond dimension growth.
"""
function build_gf2_matrix(twisted_paulis::Vector{<:PauliOperator})::Matrix{Bool}
    if isempty(twisted_paulis)
        return Matrix{Bool}(undef, 0, 0)
    end
    
    t = length(twisted_paulis)
    n = nqubits(twisted_paulis[1])
    
    # Verify all paulis have same number of qubits
    for P in twisted_paulis
        if nqubits(P) != n
            throw(ArgumentError("All twisted Paulis must have the same number of qubits"))
        end
    end
    
    # Build matrix using QuantumClifford's xbit function
    M = zeros(Bool, t, n)
    for k in 1:t
        # xbit returns a view into the Pauli's internal storage
        # Convert to Bool array for the matrix
        xb = xbit(twisted_paulis[k])
        for j in 1:n
            M[k, j] = xb[j]
        end
    end
    
    return M
end

"""
    build_gf2_matrix_from_xbits(xbits::Vector{BitVector}) -> Matrix{Bool}

Build the GF(2) matrix directly from precomputed x-bit vectors.

This is useful when you've already extracted the x-bits and don't need
to recompute them.

# Arguments
- `xbits::Vector{BitVector}`: Vector of x-bit vectors

# Returns
- `Matrix{Bool}`: GF(2) matrix
"""
function build_gf2_matrix_from_xbits(xbits::Vector{BitVector})::Matrix{Bool}
    if isempty(xbits)
        return Matrix{Bool}(undef, 0, 0)
    end
    
    t = length(xbits)
    n = length(xbits[1])
    
    M = zeros(Bool, t, n)
    for k in 1:t
        M[k, :] = xbits[k]
    end
    
    return M
end

#==============================================================================#
# GF(2) RANK COMPUTATION
#==============================================================================#

"""
    gf2_rank(M::Matrix{Bool}) -> Int

Compute the rank of a binary matrix over GF(2).

Uses QuantumClifford's gf2_gausselim! for Gaussian elimination, then counts
non-zero rows in the resulting row echelon form.

# Arguments
- `M::Matrix{Bool}`: Binary matrix

# Returns
- `Int`: Rank over GF(2)

# Example
```julia
M = Bool[1 0 1;
         0 1 1;
         1 1 0]
# Row 3 = Row 1 XOR Row 2 in GF(2)
gf2_rank(M)  # Returns 2
```

# Note
This function copies M before elimination to preserve the original matrix.
"""
function gf2_rank(M::Matrix{Bool})::Int
    if isempty(M)
        return 0
    end
    
    t, n = size(M)
    
    # Copy matrix since gf2_gausselim! modifies in place
    M_copy = copy(M)
    
    # Perform Gaussian elimination over GF(2)
    # This is provided by QuantumClifford
    gf2_gausselim!(M_copy)
    
    # Count non-zero rows
    rank = 0
    for row in 1:t
        if any(M_copy[row, :])
            rank += 1
        end
    end
    
    return rank
end

"""
    gf2_rank!(M::Matrix{Bool}) -> Int

Compute GF(2) rank, modifying M in place.

Same as `gf2_rank` but doesn't copy the matrix. M will be in row echelon form
after this call.

# Arguments
- `M::Matrix{Bool}`: Binary matrix (modified to row echelon form)

# Returns
- `Int`: Rank over GF(2)
"""
function gf2_rank!(M::Matrix{Bool})::Int
    if isempty(M)
        return 0
    end
    
    t, n = size(M)
    
    # Perform Gaussian elimination over GF(2)
    gf2_gausselim!(M)
    
    # Count non-zero rows
    rank = 0
    for row in 1:t
        if any(M[row, :])
            rank += 1
        end
    end
    
    return rank
end

#==============================================================================#
# BOND DIMENSION PREDICTION
#==============================================================================#

"""
    predict_bond_dimension(twisted_paulis::Vector{<:PauliOperator}) -> Int

Predict the final bond dimension from twisted Paulis using GF(2) theory.

The prediction formula is: χ = 2^(t_eff - rank(M))
where:
- t_eff is the number of NON-DIAGONAL twisted Paulis (those with at least one X or Y)
- rank is computed over GF(2)

# Arguments
- `twisted_paulis::Vector{<:PauliOperator}`: Vector of twisted Pauli operators

# Returns
- `Int`: Predicted bond dimension

# Theory
Pure Z Paulis (diagonal rotations with xbit = [0,0,...,0]) do NOT increase MPS bond
dimension because diagonal operations don't entangle qubits. Only twisted Paulis
with X or Y components (non-zero xbit) contribute to bond dimension growth.

The GF(2) matrix M only has non-zero rows for non-diagonal Paulis, so:
- t_eff = number of non-zero rows in M = number of non-diagonal twisted Paulis
- rank(M) = number of linearly independent non-diagonal twisted Paulis
- χ = 2^(t_eff - rank) = bond dimension from entangling T-gates

# Example
```julia
# T-gates without Hadamard: twisted Paulis are Z_k (diagonal)
# xbit = [0,0,...,0] for all → M is all zeros → t_eff = 0, rank = 0 → χ = 1

# T-gates after Hadamard: twisted Paulis are X_k (non-diagonal)
# xbit = e_k (unit vector) → M = identity → t_eff = t, rank = min(t,n) → χ = 2^max(0,t-n)
```
"""
function predict_bond_dimension(twisted_paulis::Vector{<:PauliOperator})::Int
    if isempty(twisted_paulis)
        return 1
    end

    M = build_gf2_matrix(twisted_paulis)

    # Count non-zero rows (non-diagonal twisted Paulis)
    # Pure Z Paulis have all-zero xbits and don't increase bond dimension
    t_eff = count(row -> any(M[row, :]), 1:size(M, 1))

    if t_eff == 0
        return 1  # All pure Z Paulis → no entanglement → bond dim = 1
    end

    r = gf2_rank(M)

    # Bond dimension formula: χ = 2^(t_eff - rank)
    # Cap at 2^62 to avoid Int64 overflow (2^63 is negative in signed int)
    exponent = t_eff - r
    if exponent <= 0
        return 1
    elseif exponent >= 62
        return typemax(Int) ÷ 2  # Return a very large value that won't overflow
    else
        return 2^exponent
    end
end

"""
    predict_bond_dimension(M::Matrix{Bool}) -> Int

Predict bond dimension directly from GF(2) matrix.

# Arguments
- `M::Matrix{Bool}`: GF(2) matrix (t × n)

# Returns
- `Int`: Predicted bond dimension

# Note
Pure Z rows (all-zero rows) are excluded from the effective T-count since
diagonal rotations don't increase MPS bond dimension.
"""
function predict_bond_dimension(M::Matrix{Bool})::Int
    if isempty(M)
        return 1
    end

    # Count non-zero rows (non-diagonal twisted Paulis)
    t_eff = count(row -> any(M[row, :]), 1:size(M, 1))

    if t_eff == 0
        return 1  # All pure Z Paulis → no entanglement → bond dim = 1
    end

    r = gf2_rank(M)

    # Bond dimension formula: χ = 2^(t_eff - rank)
    # Cap at 2^62 to avoid Int64 overflow (2^63 is negative in signed int)
    exponent = t_eff - r
    if exponent <= 0
        return 1
    elseif exponent >= 62
        return typemax(Int) ÷ 2  # Return a very large value that won't overflow
    else
        return 2^exponent
    end
end

#==============================================================================#
# DISENTANGLABILITY ANALYSIS
#==============================================================================#

"""
    count_disentanglable(twisted_paulis::Vector{<:PauliOperator}, 
                         free_qubits::BitVector) -> Int

Count how many twisted Paulis are disentanglable given current free qubits.

A twisted Pauli P is disentanglable if there exists a free qubit j such that
xbit(P)[j] = true (P has X or Y at position j).

# Arguments
- `twisted_paulis::Vector{<:PauliOperator}`: Twisted Paulis to check
- `free_qubits::BitVector`: Which qubits are currently free

# Returns
- `Int`: Number of disentanglable Paulis
"""
function count_disentanglable(twisted_paulis::Vector{<:PauliOperator}, 
                               free_qubits::BitVector)::Int
    count = 0
    for P in twisted_paulis
        if can_disentangle(P, free_qubits)
            count += 1
        end
    end
    return count
end

"""
    can_disentangle(P::PauliOperator, free_qubits::BitVector) -> Bool

Check if a twisted Pauli can be disentangled using OFD.

# Arguments
- `P::PauliOperator`: Twisted Pauli
- `free_qubits::BitVector`: Which qubits are free

# Returns
- `Bool`: true if P can be disentangled
"""
function can_disentangle(P::PauliOperator, free_qubits::BitVector)::Bool
    xb = xbit(P)
    for j in eachindex(free_qubits)
        if free_qubits[j] && xb[j]
            return true
        end
    end
    return false
end

"""
    find_disentangling_qubit(P::PauliOperator, free_qubits::BitVector) -> Union{Int, Nothing}

Find a free qubit that can be used to disentangle P via OFD.

# Arguments
- `P::PauliOperator`: Twisted Pauli
- `free_qubits::BitVector`: Which qubits are free

# Returns
- `Union{Int, Nothing}`: Qubit index if disentanglable, nothing otherwise
"""
function find_disentangling_qubit(P::PauliOperator, free_qubits::BitVector)::Union{Int, Nothing}
    xb = xbit(P)
    for j in eachindex(free_qubits)
        if free_qubits[j] && xb[j]
            return j
        end
    end
    return nothing
end

#==============================================================================#
# ANALYSIS UTILITIES
#==============================================================================#

"""
    analyze_gf2_structure(twisted_paulis::Vector{<:PauliOperator}) -> NamedTuple

Analyze the GF(2) structure of twisted Paulis.

# Arguments
- `twisted_paulis::Vector{<:PauliOperator}`: Twisted Paulis

# Returns
- `NamedTuple` with fields:
  - `t::Int`: Total number of T-gates
  - `t_eff::Int`: Number of non-diagonal T-gates (with X or Y components)
  - `n_pure_z::Int`: Number of pure Z Paulis (diagonal, don't increase bond dim)
  - `n::Int`: Number of qubits
  - `rank::Int`: GF(2) rank
  - `nullity::Int`: t_eff - rank (bond dimension exponent)
  - `predicted_chi::Int`: 2^nullity
  - `independent_rows::Vector{Int}`: Indices of linearly independent rows
"""
function analyze_gf2_structure(twisted_paulis::Vector{<:PauliOperator})
    if isempty(twisted_paulis)
        return (t=0, t_eff=0, n_pure_z=0, n=0, rank=0, nullity=0, predicted_chi=1, independent_rows=Int[])
    end

    t = length(twisted_paulis)
    n = nqubits(twisted_paulis[1])

    M = build_gf2_matrix(twisted_paulis)
    M_copy = copy(M)

    # Count non-zero rows (non-diagonal Paulis that actually increase bond dim)
    t_eff = count(row -> any(M[row, :]), 1:size(M, 1))
    n_pure_z = t - t_eff

    # Perform elimination and track pivots
    r = gf2_rank!(M_copy)

    # Find which original rows became pivot rows
    # This requires re-running elimination with tracking
    independent_rows = find_independent_rows(M)

    # Nullity uses t_eff, not t, because pure Z Paulis don't increase bond dim
    nullity = max(0, t_eff - r)
    predicted_chi = nullity >= 62 ? (typemax(Int) ÷ 2) : 2^nullity

    return (
        t=t,
        t_eff=t_eff,
        n_pure_z=n_pure_z,
        n=n,
        rank=r,
        nullity=nullity,
        predicted_chi=predicted_chi,
        independent_rows=independent_rows
    )
end

"""
    find_independent_rows(M::Matrix{Bool}) -> Vector{Int}

Find indices of linearly independent rows in a GF(2) matrix.

Uses row reduction with proper permutation tracking to identify which original
rows are linearly independent.

# Arguments
- `M::Matrix{Bool}`: Binary matrix

# Returns
- `Vector{Int}`: Indices of linearly independent rows (in original matrix ordering)

# Algorithm
Performs Gaussian elimination over GF(2) while tracking row swaps using a
permutation vector. The pivot rows after elimination correspond to the
original rows that form a linearly independent set.
"""
function find_independent_rows(M::Matrix{Bool})::Vector{Int}
    if isempty(M)
        return Int[]
    end

    t, n = size(M)
    M_work = copy(M)

    # Track the original row index for each current row position
    # row_perm[i] = original index of the row currently in position i
    row_perm = collect(1:t)

    pivot_col = 1
    current_row = 1

    while current_row <= t && pivot_col <= n
        # Find pivot: look for a 1 in this column at or below current row
        pivot_row = 0
        for r in current_row:t
            if M_work[r, pivot_col]
                pivot_row = r
                break
            end
        end

        if pivot_row == 0
            # No pivot in this column, move to next column
            pivot_col += 1
        else
            # Found pivot - swap rows if needed
            if pivot_row != current_row
                # Swap in the matrix
                M_work[current_row, :], M_work[pivot_row, :] =
                    M_work[pivot_row, :], M_work[current_row, :]
                # Swap in permutation tracking
                row_perm[current_row], row_perm[pivot_row] =
                    row_perm[pivot_row], row_perm[current_row]
            end

            # Eliminate all other rows (above and below for full reduction)
            for r in 1:t
                if r != current_row && M_work[r, pivot_col]
                    M_work[r, :] .⊻= M_work[current_row, :]
                end
            end

            current_row += 1
            pivot_col += 1
        end
    end

    # The first (current_row - 1) rows are pivot rows
    # Their original indices are in row_perm[1:(current_row-1)]
    rank = current_row - 1
    independent_original_indices = row_perm[1:rank]

    # Sort to return indices in ascending order (more intuitive)
    return sort(independent_original_indices)
end

"""
    find_independent_rows_with_basis(M::Matrix{Bool}) -> Tuple{Vector{Int}, Matrix{Bool}}

Find independent rows and return both the indices and the reduced echelon form.

# Arguments
- `M::Matrix{Bool}`: Binary matrix

# Returns
- `Tuple{Vector{Int}, Matrix{Bool}}`: (indices of independent rows, row echelon form)
"""
function find_independent_rows_with_basis(M::Matrix{Bool})::Tuple{Vector{Int}, Matrix{Bool}}
    if isempty(M)
        return (Int[], Matrix{Bool}(undef, 0, 0))
    end

    t, n = size(M)
    M_work = copy(M)
    row_perm = collect(1:t)

    pivot_col = 1
    current_row = 1

    while current_row <= t && pivot_col <= n
        pivot_row = 0
        for r in current_row:t
            if M_work[r, pivot_col]
                pivot_row = r
                break
            end
        end

        if pivot_row == 0
            pivot_col += 1
        else
            if pivot_row != current_row
                M_work[current_row, :], M_work[pivot_row, :] =
                    M_work[pivot_row, :], M_work[current_row, :]
                row_perm[current_row], row_perm[pivot_row] =
                    row_perm[pivot_row], row_perm[current_row]
            end

            for r in 1:t
                if r != current_row && M_work[r, pivot_col]
                    M_work[r, :] .⊻= M_work[current_row, :]
                end
            end

            current_row += 1
            pivot_col += 1
        end
    end

    rank = current_row - 1
    independent_indices = sort(row_perm[1:rank])

    return (independent_indices, M_work)
end

#==============================================================================#
# GF(2) NULL SPACE
#==============================================================================#

"""
    gf2_null_space(M::Matrix{Bool}) -> Vector{BitVector}

Compute a basis for the null space of M over GF(2).

The null space vectors indicate which linear combinations of rows sum to zero.

# Arguments
- `M::Matrix{Bool}`: Binary matrix (t × n)

# Returns
- `Vector{BitVector}`: Basis vectors for null space (each has length t)

# Theory
If v is in the null space (M' * v = 0 where * is GF(2) multiplication),
then the sum of rows M[i,:] for which v[i]=1 is zero (mod 2).

This identifies "redundant" T-gates: if twisted Paulis P[i₁], P[i₂], P[i₃]
are such that their x-bits XOR to zero, then one of them is redundant
(its disentangling cancels with the others).
"""
function gf2_null_space(M::Matrix{Bool})::Vector{BitVector}
    if isempty(M)
        return BitVector[]
    end
    
    t, n = size(M)
    
    # Augment M with identity to track row operations
    # [M | I] → row reduce → [R | T]
    # Null space of M is spanned by rows of T where corresponding R row is zero
    
    augmented = zeros(Bool, t, n + t)
    augmented[:, 1:n] = M
    for i in 1:t
        augmented[i, n + i] = true
    end
    
    # Row reduce
    gf2_gausselim!(augmented)
    
    # Find zero rows in left part, extract corresponding right part
    null_basis = BitVector[]
    for row in 1:t
        if !any(augmented[row, 1:n])
            # This row is zero in M part
            # The right part gives a null space vector
            push!(null_basis, BitVector(augmented[row, (n+1):(n+t)]))
        end
    end
    
    return null_basis
end

#==============================================================================#
# INCREMENTAL RANK UPDATE
#==============================================================================#

"""
    incremental_rank_update(M::Matrix{Bool}, new_row::BitVector) -> Tuple{Int, Bool}

Compute rank after adding a new row, without recomputing from scratch.

# Arguments
- `M::Matrix{Bool}`: Current GF(2) matrix (already in row echelon form)
- `new_row::BitVector`: New row to add

# Returns
- `Tuple{Int, Bool}`: (new rank, whether new row is independent)

# Note
This assumes M is already in row echelon form from previous gf2_gausselim! call.
"""
function incremental_rank_update(M::Matrix{Bool}, new_row::BitVector)::Tuple{Int, Bool}
    if isempty(M)
        return any(new_row) ? (1, true) : (0, false)
    end
    
    t, n = size(M)
    
    # Reduce new_row using existing rows
    reduced = copy(new_row)
    for row in 1:t
        if any(M[row, :])
            # Find pivot column
            pivot_col = findfirst(M[row, :])
            if !isnothing(pivot_col) && reduced[pivot_col]
                reduced .⊻= M[row, :]
            end
        end
    end
    
    # If reduced row is non-zero, it's independent
    is_independent = any(reduced)
    current_rank = sum(any(M[row, :]) for row in 1:t)
    new_rank = current_rank + (is_independent ? 1 : 0)
    
    return (new_rank, is_independent)
end
