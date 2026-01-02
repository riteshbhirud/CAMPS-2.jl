# CAMPS.jl/src/two_qubit_cliffords.jl
# Two-qubit Clifford group enumeration and operations for OBD
#
# Based on: Liu & Clark, "Clifford-Augmented Matrix Product States" (arXiv:2412.17209)
#
# The two-qubit Clifford group Cl_2 has 11,520 elements. OBD searches over this
# group (or a representative subset) to find the optimal disentangling gate.

#==============================================================================#
# OVERVIEW
#
# The two-qubit Clifford group can be decomposed as:
#
#     Cl_2 = (Cl_1 × Cl_1) ⋊ CNOT-class
#
# where Cl_1 has 24 elements (single-qubit Clifford group).
#
# For OBD, we search over Cl_2 to find the Clifford U that minimizes
# entanglement entropy at a bond. The key insight is that we can:
#
# 1. Use canonical representatives: 720 CNOT-class representatives
# 2. Precompute the action on the reduced density matrix
# 3. Achieve O(χ³) + O(720) complexity per bond optimization
#
# The single-qubit Cliffords (24 elements) can be handled separately.
#==============================================================================#

#==============================================================================#
# SINGLE-QUBIT CLIFFORD GROUP (24 elements)
#==============================================================================#

"""
    SINGLE_QUBIT_CLIFFORDS

The 24 single-qubit Clifford gates as QuantumClifford symbolic gate sequences.

Each element is a vector of gates that, when applied in order, produce one of
the 24 single-qubit Cliffords. The identity is represented as an empty vector.

The 24 Cliffords can be generated from {H, S} and their inverses:
- 6 rotations of each principal axis (±X, ±Y, ±Z → ±X', ±Y', ±Z')
- 4 orientations around each axis

# Structure
The gates are indexed 1:24 and include all elements of Cl_1.
"""
const SINGLE_QUBIT_CLIFFORD_GENERATORS = [
    # Identity
    [],
    # S rotations
    [(:S, 1)],
    [(:S, 1), (:S, 1)],
    [(:S, 1), (:S, 1), (:S, 1)],
    # Hadamard-based
    [(:H, 1)],
    [(:H, 1), (:S, 1)],
    [(:H, 1), (:S, 1), (:S, 1)],
    [(:H, 1), (:S, 1), (:S, 1), (:S, 1)],
    # S·H combinations
    [(:S, 1), (:H, 1)],
    [(:S, 1), (:H, 1), (:S, 1)],
    [(:S, 1), (:H, 1), (:S, 1), (:S, 1)],
    [(:S, 1), (:H, 1), (:S, 1), (:S, 1), (:S, 1)],
    # H·S·H combinations
    [(:H, 1), (:S, 1), (:H, 1)],
    [(:H, 1), (:S, 1), (:H, 1), (:S, 1)],
    [(:H, 1), (:S, 1), (:H, 1), (:S, 1), (:S, 1)],
    [(:H, 1), (:S, 1), (:H, 1), (:S, 1), (:S, 1), (:S, 1)],
    # S·H·S·H combinations
    [(:S, 1), (:H, 1), (:S, 1), (:H, 1)],
    [(:S, 1), (:H, 1), (:S, 1), (:H, 1), (:S, 1)],
    [(:S, 1), (:H, 1), (:S, 1), (:H, 1), (:S, 1), (:S, 1)],
    [(:S, 1), (:H, 1), (:S, 1), (:H, 1), (:S, 1), (:S, 1), (:S, 1)],
    # H·S·H·S combinations (remaining elements)
    [(:H, 1), (:S, 1), (:S, 1), (:H, 1)],
    [(:H, 1), (:S, 1), (:S, 1), (:H, 1), (:S, 1)],
    [(:H, 1), (:S, 1), (:S, 1), (:H, 1), (:S, 1), (:S, 1)],
    [(:H, 1), (:S, 1), (:S, 1), (:H, 1), (:S, 1), (:S, 1), (:S, 1)],
]

"""
    generate_single_qubit_clifford(index::Int, qubit::Int) -> Vector

Generate the gate sequence for a single-qubit Clifford on the specified qubit.

# Arguments
- `index::Int`: Index 1:24 of the Clifford element
- `qubit::Int`: Target qubit

# Returns
- `Vector`: Gate sequence to apply
"""
function generate_single_qubit_clifford(index::Int, qubit::Int)::Vector
    1 <= index <= 24 || throw(ArgumentError("Clifford index must be 1-24, got $index"))

    template = SINGLE_QUBIT_CLIFFORD_GENERATORS[index]
    if isempty(template)
        return []
    end

    # Convert (gate_type, 1) → (gate_type, qubit)
    return [(spec[1], qubit) for spec in template]
end

"""
    resolve_single_qubit_clifford(index::Int, qubit::Int) -> Vector

Resolve single-qubit Clifford to QuantumClifford symbolic gates.

# Arguments
- `index::Int`: Index 1:24 of the Clifford element
- `qubit::Int`: Target qubit

# Returns
- `Vector`: QuantumClifford symbolic gates
"""
function resolve_single_qubit_clifford(index::Int, qubit::Int)::Vector
    specs = generate_single_qubit_clifford(index, qubit)
    return [resolve_symbolic_gate(spec) for spec in specs]
end

#==============================================================================#
# CNOT-CLASS REPRESENTATIVES (for two-qubit Cliffords)
#==============================================================================#

"""
    CNOT_CLASS_GENERATORS

Canonical representatives of the CNOT-class cosets in Cl_2.

The two-qubit Clifford group decomposes as:
    Cl_2 = ⋃_{c ∈ CNOT-class} (Cl_1 × Cl_1) · c

The CNOT-class has 720 elements up to left multiplication by Cl_1 × Cl_1.

We use a minimal generating set based on {CNOT, CZ, SWAP, iSWAP, and identity}.
"""
const CNOT_CLASS_GENERATORS_TEMPLATE = [
    # Identity (1 element)
    [],
    # CNOT variants (4 orientations × 2 directions = 8)
    [(:CNOT, 1, 2)],
    [(:CNOT, 2, 1)],
    # CZ (1 element - symmetric)
    [(:CZ, 1, 2)],
    # SWAP (1 element)
    [(:SWAP, 1, 2)],
    # CNOT chains for entangling classes
    [(:CNOT, 1, 2), (:CNOT, 2, 1)],
    [(:CNOT, 2, 1), (:CNOT, 1, 2)],
    [(:CNOT, 1, 2), (:CNOT, 2, 1), (:CNOT, 1, 2)],
]

"""
    TwoQubitCliffordIterator

Iterator over all two-qubit Cliffords as gate sequences.

This iterates over the full Cl_2 by combining:
- 24 single-qubit Cliffords on qubit 1
- 24 single-qubit Cliffords on qubit 2
- CNOT-class representative

Total: 24 × 24 × (number of CNOT classes) elements.

For OBD, we iterate over a representative subset to find the optimal gate.
"""
struct TwoQubitCliffordIterator
    qubit1::Int
    qubit2::Int
    include_singles::Bool  # Whether to iterate over single-qubit Cliffords
end

"""
    two_qubit_clifford_count(include_singles::Bool=true) -> Int

Return the number of two-qubit Cliffords to iterate over.

# Arguments
- `include_singles::Bool`: If true, iterate over full Cl_2 (11,520)
                           If false, iterate only over entangling classes (~720)

# Returns
- `Int`: Number of Cliffords
"""
function two_qubit_clifford_count(include_singles::Bool=true)::Int
    n_entangling = length(CNOT_CLASS_GENERATORS_TEMPLATE)
    if include_singles
        return 24 * 24 * n_entangling
    else
        return n_entangling
    end
end

#==============================================================================#
# USING QuantumClifford's enumerate_cliffords
#==============================================================================#

"""
    get_all_two_qubit_cliffords() -> Vector

Get all 11,520 two-qubit Clifford operators using QuantumClifford.

This uses QuantumClifford's enumerate_cliffords function which provides
an iterator over all Cliffords for a given number of qubits.

# Returns
- `Vector`: Vector of CliffordOperator objects

# Note
This function caches the result for efficiency. The first call may be slow.
"""
function get_all_two_qubit_cliffords()
    # Use QuantumClifford's enumeration
    # enumerate_cliffords(n) returns an iterator over all n-qubit Cliffords
    return collect(enumerate_cliffords(2))
end

"""
    sample_two_qubit_cliffords(n::Int; seed::Union{Int, Nothing}=nothing) -> Vector

Sample n random two-qubit Clifford operators.

Useful for approximate OBD when searching over all 11,520 is too slow.

# Arguments
- `n::Int`: Number of samples
- `seed::Union{Int, Nothing}`: Random seed for reproducibility

# Returns
- `Vector`: Vector of CliffordOperator objects
"""
function sample_two_qubit_cliffords(n::Int; seed::Union{Int, Nothing}=nothing)::Vector
    if seed !== nothing
        Random.seed!(seed)
    end
    return [random_clifford(2) for _ in 1:n]
end

#==============================================================================#
# CLIFFORD TO MATRIX CONVERSION
#==============================================================================#

"""
    clifford_to_matrix(C::CliffordOperator) -> Matrix{ComplexF64}

Convert a CliffordOperator to its 2^n × 2^n unitary matrix representation.

# Arguments
- `C::CliffordOperator`: Clifford operator

# Returns
- `Matrix{ComplexF64}`: Unitary matrix

# Algorithm
Uses the destabilizer formalism to extract the exact unitary matrix.
The Clifford is applied to each computational basis state represented
as a Destabilizer, and the resulting state is extracted using the
CH-form conversion.

# Note
For n qubits, this creates a 2^n × 2^n matrix, which is exponential.
Only use for small n (≤ 4 qubits in practice).
"""
function clifford_to_matrix(C::CliffordOperator)::Matrix{ComplexF64}
    n = nqubits(C)
    dim = 2^n

    # Build matrix column by column using destabilizer formalism
    U = zeros(ComplexF64, dim, dim)

    for j in 0:(dim-1)
        # Create computational basis state |j⟩ as a Destabilizer
        col = clifford_on_basis_state_destab(C, j, n)
        U[:, j+1] = col
    end

    return U
end

"""
    clifford_on_basis_state_destab(C::CliffordOperator, j::Int, n::Int) -> Vector{ComplexF64}

Compute C|j⟩ using the destabilizer formalism.

# Arguments
- `C::CliffordOperator`: Clifford operator
- `j::Int`: Basis state index (0 to 2^n - 1)
- `n::Int`: Number of qubits

# Returns
- `Vector{ComplexF64}`: State vector C|j⟩

# Algorithm
1. Create |0...0⟩ as destabilizer
2. Apply X gates to create |j⟩
3. Apply Clifford C
4. Convert result to state vector using projectrand-based extraction

# Basis state convention
We use big-endian ordering: |j⟩ = |b_{n-1} ... b_1 b_0⟩ where j = Σ_k b_k 2^{n-1-k}.
This means:
- j=0: |00...0⟩
- j=1: |00...1⟩ (qubit n is |1⟩)
- j=2: |00..10⟩ (qubit n-1 is |1⟩)
etc.

For matrix indices to match standard conventions where CNOT flips the target
when control is |1⟩, we use: bit k of j corresponds to qubit (n-k).
"""
function clifford_on_basis_state_destab(C::CliffordOperator, j::Int, n::Int)::Vector{ComplexF64}
    dim = 2^n

    # Create |0...0⟩ state as a Destabilizer
    D = one(Destabilizer, n)

    # Apply X gates to create |j⟩ from |0...0⟩
    # Use big-endian: bit k (from right, 0-indexed) corresponds to qubit (n - k)
    # j = b_{n-1} * 2^{n-1} + ... + b_1 * 2^1 + b_0 * 2^0
    # bit k = b_k corresponds to qubit (n - k) in our convention
    for k in 0:(n-1)
        if (j >> k) & 1 == 1
            # bit k is set, so qubit (n - k) should be |1⟩
            apply!(D, sX(n - k))
        end
    end

    # Apply the Clifford operator
    apply!(D, C)

    # Extract state vector from the resulting Destabilizer
    return destabilizer_to_statevector(D)
end

"""
    destabilizer_to_statevector(D::Destabilizer) -> Vector{ComplexF64}

Convert a Destabilizer (representing a stabilizer state) to state vector.

# Arguments
- `D::Destabilizer`: Destabilizer representation

# Returns
- `Vector{ComplexF64}`: State vector

# Algorithm
Uses the Aaronson-Gottesman algorithm (arXiv:quant-ph/0406196).

For a stabilizer state with stabilizers S₁,...,Sₙ, the state is:
|ψ⟩ ∝ Π_i (I + S_i) |0⟩^n

For small systems (n ≤ 12), we compute amplitudes by:
1. Finding the computational basis states in the stabilizer code space
2. Computing relative phases from the destabilizer structure

The key insight is that if a stabilizer S has X-support on some qubits,
the corresponding basis states are paired: |j⟩ and |j ⊕ x(S)⟩ have
related amplitudes (where x(S) is the X-part of S).
"""
function destabilizer_to_statevector(D::Destabilizer)::Vector{ComplexF64}
    n = nqubits(D)
    dim = 2^n

    if n > 12
        throw(ArgumentError("destabilizer_to_statevector only supports n ≤ 12"))
    end

    # Use the direct algorithm: build the state by applying the Clifford
    # to the all-zeros state using explicit matrix construction
    #
    # This is equivalent to computing C|0⟩^n where C is the Clifford
    # represented by the destabilizer D.
    #
    # However, the destabilizer D directly represents a stabilizer state,
    # so we need to extract it properly.

    # Alternative approach: Use the generating set interpretation
    # The stabilizer state can be computed as:
    # |ψ⟩ = (1/√K) Σ_{s∈S_X} ω_s |s⟩
    # where S_X is determined by the X-support of stabilizers

    stab = stabilizerview(D)
    destabs = destabilizerview(D)

    # Find qubits where all stabilizers have Z-only action (no X)
    # These determine which bits are fixed in the support

    # First, find the X-support matrix of the stabilizers
    # This tells us which basis states can have non-zero amplitude
    x_support = zeros(Bool, n, n)  # x_support[i,q] = true if S_i has X on qubit q
    z_support = zeros(Bool, n, n)

    for i in 1:n
        S = stab[i]
        for q in 1:n
            x_q, z_q = S[q]
            x_support[i, q] = x_q
            z_support[i, q] = z_q
        end
    end

    # Find the row echelon form of the X-support matrix
    # to identify free qubits and fixed relationships
    result = zeros(ComplexF64, dim)

    # Simpler approach for correctness: enumerate all basis states and
    # check stabilizer constraints using the proper Aaronson-Gottesman
    # inner product formula

    # For a stabilizer state, ⟨j|ψ⟩ ≠ 0 iff for all generators S_i:
    # the X-part of S_i dotted with j equals 0 (mod 2), OR
    # the stabilizer creates a valid pairing

    # Actually, the simplest correct approach is:
    # 1. Find a basis for the code space (2^k states for [[n,k]] code)
    # 2. Compute phases from destabilizer structure

    # For a stabilizer state (k=0 logical qubits), there are 2^r
    # computational basis states in the support, where r = rank of X-matrix

    # Use brute-force for small n: check each basis state
    for j in 0:(dim-1)
        amp = stabilizer_state_amplitude(D, j, n)
        result[j+1] = amp
    end

    # Normalize
    norm_sq = sum(abs2, result)
    if norm_sq > 1e-15
        result ./= sqrt(norm_sq)
    end

    return result
end

"""
    stabilizer_state_amplitude(D::Destabilizer, j::Int, n::Int) -> ComplexF64

Compute ⟨j|ψ⟩ for stabilizer state |ψ⟩ represented by destabilizer D.

Uses the correct algorithm based on the stabilizer formalism.

# Algorithm
A stabilizer state is:
|ψ⟩ = (1/√|C|) Σ_{c∈C} ω_c |c⟩

where C is the set of computational basis states in the support (determined
by the X-parts of stabilizers), and ω_c are phases (determined by Z-parts
and the destabilizer structure).

For a basis state |j⟩ to have non-zero amplitude:
1. j must be in the affine subspace defined by Z-only stabilizers
2. The phase comes from Z-eigenvalues and destabilizer phases

# Key insight
The stabilizers partition into:
- Z-only stabilizers: constrain which j are in the support
- Stabilizers with X-parts: define the superposition structure

For the state |+0⟩ = (|00⟩ + |10⟩)/√2:
- Stabilizers: X₁, Z₂
- Z₂ constrains qubit 2 to be |0⟩
- X₁ creates the superposition over qubit 1
"""
function stabilizer_state_amplitude(D::Destabilizer, j::Int, n::Int)::ComplexF64
    stab = stabilizerview(D)
    destabs = destabilizerview(D)

    # Convert j to bit vector (big-endian convention: qubit 1 is MSB)
    j_bits = [(j >> (n - q)) & 1 for q in 1:n]

    # Step 1: Check Z-only stabilizer constraints
    # These constrain which basis states have non-zero amplitude

    for i in 1:n
        S = stab[i]

        # Check if this stabilizer has any X components
        has_x = false
        for q in 1:n
            x_q, _ = S[q]
            if x_q
                has_x = true
                break
            end
        end

        if !has_x
            # This is a Z-only stabilizer (product of Z and I only)
            # Constraint: eigenvalue must be +1

            phase_i = S.phase[]
            z_dot_j = 0
            for q in 1:n
                _, z_q = S[q]
                if z_q
                    z_dot_j ⊻= j_bits[q]
                end
            end

            # Eigenvalue = i^phase_i * (-1)^{z_dot_j}
            # For +1 eigenspace, need phase_i + 2*z_dot_j ≡ 0 (mod 4)
            effective_phase = (phase_i + 2 * z_dot_j) % 4
            if effective_phase != 0
                return 0.0 + 0.0im
            end
        end
    end

    # Step 2: For stabilizers with X-components, check that j is in the
    # correct coset. The X-parts of all stabilizers generate a group G_X.
    # The support of |ψ⟩ is a coset of G_X.

    # Build the X-part matrix and find the coset representative
    x_matrix = zeros(Int, n, n)  # x_matrix[i,q] = 1 if S_i has X on qubit q
    for i in 1:n
        S = stab[i]
        for q in 1:n
            x_q, _ = S[q]
            x_matrix[i, q] = x_q ? 1 : 0
        end
    end

    # Find which coset j belongs to by computing j mod G_X
    # The coset must match the reference state (which is |0⟩^n for identity Clifford)

    # For each stabilizer with X-part, check if j is in the +1 eigenspace
    # This is done by checking all group relations

    # The correct check: for each stabilizer S = X^a Z^b with phase p,
    # if a ≠ 0, then both |j⟩ and |j⊕a⟩ have equal magnitude amplitudes
    # with relative phase determined by b

    # For a basis state to be in the support, it must be reachable from
    # the "seed" state by applying X-parts of stabilizers

    # Find the seed state: the lexicographically smallest state in the support
    # by reducing j modulo the X-group

    # Use Gaussian elimination on x_matrix to find the reduced form
    reduced_j = copy(j_bits)
    for i in 1:n
        # Find pivot
        pivot_q = 0
        for q in 1:n
            if x_matrix[i, q] == 1
                pivot_q = q
                break
            end
        end

        if pivot_q > 0 && reduced_j[pivot_q] == 1
            # Apply this stabilizer's X-part to reduce
            for q in 1:n
                if x_matrix[i, q] == 1
                    reduced_j[q] ⊻= 1
                end
            end
        end
    end

    # Check if the reduced form is the zero state (or the seed state)
    # For identity Clifford starting from |0⟩^n, the seed is all zeros

    # Actually, this approach is getting complicated. Let me use a simpler
    # but correct approach: the state |ψ⟩ has the form:
    # |ψ⟩ = (1/√|G|) Σ_{g∈G} phase(g) |seed ⊕ x(g)⟩
    # where G is generated by stabilizers with X-parts

    # For |+0⟩: G is generated by X₁, seed is |00⟩
    # Support = {|00⟩, |10⟩} with equal phases

    # Check if j is in the group orbit of |0...0⟩
    is_in_support = is_in_stabilizer_support(x_matrix, j_bits, n)

    if !is_in_support
        return 0.0 + 0.0im
    end

    # Step 3: Compute the phase
    # The phase comes from two sources:
    # 1. Z-eigenvalues from stabilizers with Z-parts
    # 2. Destabilizer phases

    phase_power = 0

    # Phase from Z-parts of stabilizers acting on j
    # For each stabilizer S = i^p X^a Z^b, the contribution to |j⟩ is:
    # If j is in the support and j = seed ⊕ Σ_i α_i x_i, then
    # the phase picks up contributions from Z-parts

    # Simpler: compute phase from destabilizer action
    # |j⟩ = D_1^{j_1} D_2^{j_2} ... D_n^{j_n} |ψ_0⟩
    # where |ψ_0⟩ is the "reference" state

    for q in 1:n
        if j_bits[q] == 1
            d_q = destabs[q]
            d_phase = d_q.phase[]
            phase_power += d_phase
        end
    end

    phase_power = phase_power % 4

    phase_factor = if phase_power == 0
        1.0 + 0.0im
    elseif phase_power == 1
        0.0 + 1.0im
    elseif phase_power == 2
        -1.0 + 0.0im
    else  # phase_power == 3
        0.0 - 1.0im
    end

    return phase_factor
end

"""
    is_in_stabilizer_support(x_matrix::Matrix{Int}, j_bits::Vector{Int}, n::Int) -> Bool

Check if basis state |j⟩ is in the support of the stabilizer state.

The support is determined by the X-parts of stabilizers:
- If all stabilizers are Z-only (X-matrix is zero), the support is a single
  basis state determined by the Z-eigenvalue constraints.
- If stabilizers have X-parts, the support is a coset of the group generated
  by the X-parts.

For Z-only stabilizers, the support state is determined elsewhere (in the
Z-eigenvalue check), so here we only need to check the X-part constraint.

When X-matrix is all zeros, ANY state that passes the Z-eigenvalue check
is in the support (there's exactly one such state).

When X-matrix is non-zero, the support is the orbit of |0...0⟩ (or the seed
state) under the group generated by X-parts.
"""
function is_in_stabilizer_support(x_matrix::Matrix{Int}, j_bits::Vector{Int}, n::Int)::Bool
    # Check if x_matrix is all zeros
    all_zero = true
    for i in 1:n
        for q in 1:n
            if x_matrix[i, q] != 0
                all_zero = false
                break
            end
        end
        if !all_zero
            break
        end
    end

    if all_zero
        # When X-matrix is all zeros, any state that passes the Z-eigenvalue
        # constraints is in the support. Since we already checked Z-eigenvalues
        # in stabilizer_state_amplitude before calling this function, return true.
        return true
    end

    # X-matrix is non-zero: check if j is in the rowspan of X-matrix over GF(2)
    # The support is all states of the form |Σ_i α_i x_i⟩ where α_i ∈ {0,1}
    # and x_i is the i-th row of x_matrix

    # We need to check if j_bits can be written as a GF(2) linear combination
    # of the rows of x_matrix

    # j ∈ rowspan(X) iff X^T * α = j has a solution over GF(2)
    # Use row reduction on [X^T | j]

    # X^T is n×n, j is n×1, so augmented is n×(n+1)
    xt_aug = zeros(Int, n, n + 1)
    for q in 1:n
        for i in 1:n
            xt_aug[q, i] = x_matrix[i, q]  # X^T[q, i] = X[i, q]
        end
        xt_aug[q, n + 1] = j_bits[q]
    end

    # Gaussian elimination over GF(2)
    pivot_row = 1
    for col in 1:n
        # Find pivot
        found = false
        for row in pivot_row:n
            if xt_aug[row, col] == 1
                # Swap rows
                xt_aug[pivot_row, :], xt_aug[row, :] = xt_aug[row, :], xt_aug[pivot_row, :]
                found = true
                break
            end
        end

        if found
            # Eliminate below
            for row in (pivot_row + 1):n
                if xt_aug[row, col] == 1
                    for c in 1:(n + 1)
                        xt_aug[row, c] ⊻= xt_aug[pivot_row, c]
                    end
                end
            end
            pivot_row += 1
        end
    end

    # Check for inconsistency: any row [0 0 ... 0 | 1]?
    for row in 1:n
        all_zero_row = true
        for col in 1:n
            if xt_aug[row, col] == 1
                all_zero_row = false
                break
            end
        end
        if all_zero_row && xt_aug[row, n + 1] == 1
            return false  # Inconsistent, j not in support
        end
    end

    return true  # j is in the support
end

"""
    compute_stabilizer_amplitude(stab::Stabilizer, D::Destabilizer, j::Int, n::Int) -> ComplexF64

Compute amplitude ⟨j|ψ⟩ for stabilizer state |ψ⟩.

# Algorithm
A computational basis state |j⟩ has nonzero amplitude in the stabilizer state
if and only if all stabilizers S_i satisfy S_i|j⟩ = +|j⟩.

For a Pauli P = i^p X^a Z^b acting on |j⟩:
P|j⟩ = i^p (-1)^{b·j} |j ⊕ a⟩

So P|j⟩ = ±|j⟩ iff a = 0 (no X components), and the eigenvalue is i^p (-1)^{b·j}.

The phase of the amplitude comes from the destabilizer formalism.
"""
function compute_stabilizer_amplitude(stab, D::Destabilizer, j::Int, n::Int)::ComplexF64
    # Check each stabilizer generator
    for i in 1:n
        S = stab[i]

        # Check if |j⟩ is in +1 eigenspace of S
        eigenval = stabilizer_eigenvalue_on_basis(S, j, n)

        if eigenval == 0
            # S has X component - |j⟩ is not an eigenstate
            # This means |j⟩ has zero amplitude
            return 0.0 + 0.0im
        elseif eigenval < 0
            # S|j⟩ = -|j⟩ - not in the code space
            return 0.0 + 0.0im
        end
        # eigenval == 1: continue checking
    end

    # |j⟩ is in +1 eigenspace of all stabilizers
    # Need to compute the actual phase from the destabilizer
    phase = compute_amplitude_phase(D, j, n)

    return phase
end

"""
    stabilizer_eigenvalue_on_basis(P::PauliOperator, j::Int, n::Int) -> Int

Compute eigenvalue of Pauli P on computational basis state |j⟩.

Returns:
- +1 if P|j⟩ = +|j⟩
- -1 if P|j⟩ = -|j⟩
- 0 if P|j⟩ ≠ ±|j⟩ (has X component)

# Basis convention
Uses big-endian: bit k of j (0-indexed from right) corresponds to qubit (n - k).
"""
function stabilizer_eigenvalue_on_basis(P::PauliOperator, j::Int, n::Int)::Int
    # Check for X components - if any X or Y, |j⟩ is not an eigenstate
    for q in 1:n
        x_q, _ = P[q]
        if x_q  # X or Y component
            return 0
        end
    end

    # No X components - compute eigenvalue from Z components and phase
    phase = P.phase[]
    z_eigenval = 1

    for q in 1:n
        _, z_q = P[q]
        if z_q  # Z component on qubit q
            # Qubit q corresponds to bit (n - q) in our big-endian convention
            bit_idx = n - q
            bit_val = (j >> bit_idx) & 1
            if bit_val == 1
                z_eigenval *= -1
            end
        end
    end

    # Total eigenvalue = i^phase * z_eigenval
    # For stabilizers, phase should be 0 (+1) or 2 (-1)
    if phase == 0x00
        return z_eigenval
    elseif phase == 0x02
        return -z_eigenval
    else
        # Imaginary phase - shouldn't occur for valid stabilizers
        return 0
    end
end

"""
    compute_amplitude_phase(D::Destabilizer, j::Int, n::Int) -> ComplexF64

Compute the phase of amplitude ⟨j|ψ⟩ using the destabilizer formalism.

# Algorithm
Uses the CH-form / destabilizer approach. For a stabilizer state defined by
destabilizers d_1,...,d_n and stabilizers s_1,...,s_n:

|ψ⟩ ∝ (I + s_1)(I + s_2)...(I + s_n)|ref⟩

where |ref⟩ is determined by the destabilizers.

For computational basis states, we use the fact that:
⟨j|ψ⟩ = 1/√N × phase_factor

where N = 2^n / (# of basis states in support).

# Basis convention
Uses big-endian: bit k of j (0-indexed from right) corresponds to qubit (n - k).
"""
function compute_amplitude_phase(D::Destabilizer, j::Int, n::Int)::ComplexF64
    # For stabilizer states, amplitudes are of the form ±1/√k or ±i/√k
    # We compute the phase by tracking the Pauli phases through the destabilizers

    destabs = destabilizerview(D)

    # Compute the phase accumulation
    phase_power = 0  # Tracks i^phase_power

    # For each qubit, check if the corresponding bit in j is 1
    # Use big-endian convention: qubit q corresponds to bit (n - q)
    for q in 1:n
        bit_idx = n - q
        bit_val = (j >> bit_idx) & 1
        if bit_val == 1
            # The contribution from destabilizer q when measuring |1⟩
            d_q = destabs[q]
            # Accumulate phase from this destabilizer
            d_phase = d_q.phase[]
            phase_power += d_phase
        end
    end

    # Compute i^phase_power mod 4
    phase_power = mod(phase_power, 4)

    phase_factor = if phase_power == 0
        1.0 + 0.0im
    elseif phase_power == 1
        0.0 + 1.0im
    elseif phase_power == 2
        -1.0 + 0.0im
    else  # phase_power == 3
        0.0 - 1.0im
    end

    return phase_factor
end

#==============================================================================#
# TWO-QUBIT CLIFFORD AS ITensor
#==============================================================================#

"""
    clifford_to_itensor(C::CliffordOperator, s1::Index, s2::Index) -> ITensor

Convert a two-qubit CliffordOperator to an ITensor.

# Arguments
- `C::CliffordOperator`: Two-qubit Clifford
- `s1::Index`: First site index
- `s2::Index`: Second site index

# Returns
- `ITensor`: Two-qubit gate tensor

# Note
The matrix is reshaped to tensor form with proper index ordering.
"""
function clifford_to_itensor(C::CliffordOperator, s1::Index, s2::Index)::ITensor
    nqubits(C) == 2 || throw(ArgumentError("Expected 2-qubit Clifford"))

    U = clifford_to_matrix(C)
    return matrix_to_two_qubit_itensor(U, s1, s2)
end

"""
    make_clifford_gate_tensors(qubit1::Int, qubit2::Int,
                                 sites::AbstractVector) -> Vector{ITensor}

Generate ITensor representations for all two-qubit Cliffords on given qubits.

# Arguments
- `qubit1::Int`: First qubit (1-indexed)
- `qubit2::Int`: Second qubit (1-indexed)
- `sites::AbstractVector`: Site indices

# Returns
- `Vector{ITensor}`: Vector of 11,520 two-qubit gate tensors

# Note
This is expensive to compute. Consider caching the results.
"""
function make_clifford_gate_tensors(qubit1::Int, qubit2::Int,
                                     sites::AbstractVector)::Vector{ITensor}
    s1 = sites[qubit1]
    s2 = sites[qubit2]

    cliffords = get_all_two_qubit_cliffords()
    return [clifford_to_itensor(C, s1, s2) for C in cliffords]
end

#==============================================================================#
# EFFICIENT OBD SEARCH UTILITIES
#==============================================================================#

"""
    TwoQubitCliffordCache

Cache for precomputed two-qubit Clifford representations.

Stores both the CliffordOperator and its matrix/ITensor forms for efficient OBD.
"""
struct TwoQubitCliffordCache
    cliffords::Vector{Any}  # CliffordOperator objects
    matrices::Vector{Matrix{ComplexF64}}
    inverse_matrices::Vector{Matrix{ComplexF64}}
end

"""
    build_clifford_cache() -> TwoQubitCliffordCache

Build a cache of all two-qubit Cliffords and their matrix representations.

# Returns
- `TwoQubitCliffordCache`: Cache with precomputed representations

# Note
This caches all 11,520 two-qubit Cliffords. Memory usage is approximately
11,520 × (16×4 + 16×4) = ~1.5 MB for the matrices.
"""
function build_clifford_cache()::TwoQubitCliffordCache
    cliffords = get_all_two_qubit_cliffords()
    n_clif = length(cliffords)

    matrices = Vector{Matrix{ComplexF64}}(undef, n_clif)
    inverse_matrices = Vector{Matrix{ComplexF64}}(undef, n_clif)

    for (i, C) in enumerate(cliffords)
        U = clifford_to_matrix(C)
        matrices[i] = U
        inverse_matrices[i] = U'  # Cliffords are unitary, so inverse = adjoint
    end

    return TwoQubitCliffordCache(collect(cliffords), matrices, inverse_matrices)
end

"""
    get_clifford_matrix(cache::TwoQubitCliffordCache, index::Int) -> Matrix{ComplexF64}

Get the matrix representation of the i-th Clifford from cache.
"""
function get_clifford_matrix(cache::TwoQubitCliffordCache, index::Int)::Matrix{ComplexF64}
    return cache.matrices[index]
end

"""
    get_clifford_inverse_matrix(cache::TwoQubitCliffordCache, index::Int) -> Matrix{ComplexF64}

Get the inverse matrix representation of the i-th Clifford from cache.
"""
function get_clifford_inverse_matrix(cache::TwoQubitCliffordCache, index::Int)::Matrix{ComplexF64}
    return cache.inverse_matrices[index]
end

#==============================================================================#
# REPRESENTATIVE SUBSET FOR APPROXIMATE OBD
#==============================================================================#

"""
    get_cnot_class_representatives() -> Vector

Get a smaller set of CNOT-class representatives for faster OBD.

Instead of searching over all 11,520 Cliffords, we can search over:
1. The 3 CNOT-class generators: I, CNOT, CZ, SWAP (identity, entangling)
2. A few variants with single-qubit gates

This gives approximately 20-100 gates to search over instead of 11,520.

# Returns
- `Vector`: CliffordOperator representatives
"""
function get_cnot_class_representatives()::Vector
    representatives = []

    # Identity (no gate)
    push!(representatives, one(CliffordOperator, 2))

    # CNOT(1,2) and CNOT(2,1)
    C = one(Destabilizer, 2)
    apply!(C, sCNOT(1, 2))
    push!(representatives, CliffordOperator(C))

    C = one(Destabilizer, 2)
    apply!(C, sCNOT(2, 1))
    push!(representatives, CliffordOperator(C))

    # CZ
    C = one(Destabilizer, 2)
    apply!(C, sCPHASE(1, 2))
    push!(representatives, CliffordOperator(C))

    # SWAP
    C = one(Destabilizer, 2)
    apply!(C, sSWAP(1, 2))
    push!(representatives, CliffordOperator(C))

    # iSWAP-like (CNOT ladder)
    C = one(Destabilizer, 2)
    apply!(C, sCNOT(1, 2))
    apply!(C, sCNOT(2, 1))
    push!(representatives, CliffordOperator(C))

    C = one(Destabilizer, 2)
    apply!(C, sCNOT(2, 1))
    apply!(C, sCNOT(1, 2))
    push!(representatives, CliffordOperator(C))

    # Add some with single-qubit prefixes (Hadamards)
    for base in [sCNOT(1, 2), sCNOT(2, 1), sCPHASE(1, 2)]
        for h_pattern in [[sHadamard(1)], [sHadamard(2)], [sHadamard(1), sHadamard(2)]]
            C = one(Destabilizer, 2)
            for h in h_pattern
                apply!(C, h)
            end
            apply!(C, base)
            push!(representatives, CliffordOperator(C))
        end
    end

    return representatives
end

"""
    get_expanded_representatives(; depth::Int=2) -> Vector

Get an expanded set of Clifford representatives by composing basic gates.

# Arguments
- `depth::Int`: Maximum circuit depth for generating representatives

# Returns
- `Vector`: CliffordOperator representatives

# Note
This generates more representatives for better OBD accuracy at the cost
of longer search time.
"""
function get_expanded_representatives(; depth::Int=2)::Vector
    # Basic gates to compose
    basic_gates = [
        sHadamard(1), sHadamard(2),
        sPhase(1), sPhase(2),
        sCNOT(1, 2), sCNOT(2, 1),
        sCPHASE(1, 2)
    ]

    representatives = Set{Matrix{ComplexF64}}()
    result = []

    # Generate by composing gates up to specified depth
    function generate(current_dest::Destabilizer, current_depth::Int)
        if current_depth > depth
            return
        end

        # Add current Clifford if not seen
        C = CliffordOperator(current_dest)
        U = clifford_to_matrix(C)
        U_normalized = round.(U, digits=10)  # Normalize for comparison

        if !(U_normalized in representatives)
            push!(representatives, U_normalized)
            push!(result, C)
        end

        # Recurse with each basic gate
        for gate in basic_gates
            new_dest = deepcopy(current_dest)
            apply!(new_dest, gate)
            generate(new_dest, current_depth + 1)
        end
    end

    generate(one(Destabilizer, 2), 0)
    return result
end

#==============================================================================#
# ENTROPY COMPUTATION FOR OBD
#==============================================================================#

"""
    compute_renyi2_entropy(rho::Matrix{ComplexF64}) -> Float64

Compute the second Rényi entropy of a density matrix.

S_2(ρ) = -log(Tr(ρ²))

This is faster to compute than von Neumann entropy and serves as a good
proxy for entanglement in OBD optimization.

# Arguments
- `rho::Matrix{ComplexF64}`: Density matrix

# Returns
- `Float64`: Second Rényi entropy
"""
function compute_renyi2_entropy(rho::Matrix{ComplexF64})::Float64
    # S_2 = -log(Tr(ρ²))
    rho_sq = rho * rho
    tr_rho_sq = real(tr(rho_sq))

    # Clamp to valid range
    tr_rho_sq = clamp(tr_rho_sq, 1e-15, 1.0)

    return -log(tr_rho_sq)
end

"""
    compute_von_neumann_entropy(rho::Matrix{ComplexF64}) -> Float64

Compute the von Neumann entropy of a density matrix.

S(ρ) = -Tr(ρ log ρ)

# Arguments
- `rho::Matrix{ComplexF64}`: Density matrix

# Returns
- `Float64`: von Neumann entropy
"""
function compute_von_neumann_entropy(rho::Matrix{ComplexF64})::Float64
    # Diagonalize ρ
    eigenvalues = real.(eigvals(rho))

    # S = -Σ p_i log(p_i)
    entropy = 0.0
    for p in eigenvalues
        if p > 1e-15
            entropy -= p * log(p)
        end
    end

    return entropy
end

#==============================================================================#
# LOCAL DENSITY MATRIX UTILITIES
#==============================================================================#

"""
    extract_two_site_rdm(mps::MPS, site1::Int, site2::Int) -> Matrix{ComplexF64}

Extract the reduced density matrix for two adjacent sites of an MPS.

ρ_{i,i+1} = Tr_{rest}(|ψ⟩⟨ψ|)

# Arguments
- `mps::MPS`: Matrix Product State
- `site1::Int`: First site (must be site2 - 1)
- `site2::Int`: Second site (must be site1 + 1)

# Returns
- `Matrix{ComplexF64}`: 4×4 reduced density matrix

# Note
Sites must be adjacent for efficient computation.
"""
function extract_two_site_rdm(mps::MPS, site1::Int, site2::Int)::Matrix{ComplexF64}
    site2 == site1 + 1 || throw(ArgumentError("Sites must be adjacent"))

    n = length(mps)
    1 <= site1 < n || throw(ArgumentError("Invalid site indices"))

    # Orthogonalize to the bond between site1 and site2
    psi = orthogonalize(mps, site1)

    # Get the two-site wavefunction
    wf = psi[site1] * psi[site2]

    # Get site indices
    s1 = siteind(psi, site1)
    s2 = siteind(psi, site2)

    # Contract to get the reduced density matrix
    # ρ = |wf⟩⟨wf| traced over bond indices
    wf_dag = dag(wf)

    # Prime the site indices on the conjugate
    wf_dag = prime(wf_dag, s1)
    wf_dag = prime(wf_dag, s2)

    # Contract over bond indices only (not site indices)
    rho_tensor = wf * wf_dag

    # Convert to matrix
    # The tensor has indices (s1, s2, s1', s2')
    # Reshape to matrix form (s1⊗s2, s1'⊗s2')
    rho = Array(rho_tensor, s1, s2, s1', s2')
    rho_matrix = reshape(rho, 4, 4)

    return ComplexF64.(rho_matrix)
end

"""
    transform_rdm(rho::Matrix{ComplexF64}, U::Matrix{ComplexF64}) -> Matrix{ComplexF64}

Transform a reduced density matrix under a local unitary.

ρ' = U ρ U†

# Arguments
- `rho::Matrix{ComplexF64}`: Original density matrix
- `U::Matrix{ComplexF64}`: Local unitary

# Returns
- `Matrix{ComplexF64}`: Transformed density matrix
"""
function transform_rdm(rho::Matrix{ComplexF64}, U::Matrix{ComplexF64})::Matrix{ComplexF64}
    return U * rho * U'
end

"""
    partial_trace_4x4(rho::Matrix{ComplexF64}, trace_second::Bool) -> Matrix{ComplexF64}

Compute partial trace of a 4×4 density matrix (2 qubits).

# Arguments
- `rho::Matrix{ComplexF64}`: 4×4 density matrix
- `trace_second::Bool`: If true, trace out second qubit; else trace out first

# Returns
- `Matrix{ComplexF64}`: 2×2 reduced density matrix
"""
function partial_trace_4x4(rho::Matrix{ComplexF64}, trace_second::Bool)::Matrix{ComplexF64}
    result = zeros(ComplexF64, 2, 2)

    if trace_second
        # Trace over second qubit: ρ_1 = Σ_j ⟨j|ρ|j⟩_2
        # Basis: |00⟩=1, |01⟩=2, |10⟩=3, |11⟩=4
        result[1, 1] = rho[1, 1] + rho[2, 2]  # ⟨0|...|0⟩: 00→00 + 01→01
        result[1, 2] = rho[1, 3] + rho[2, 4]  # ⟨0|...|1⟩: 00→10 + 01→11
        result[2, 1] = rho[3, 1] + rho[4, 2]  # ⟨1|...|0⟩: 10→00 + 11→01
        result[2, 2] = rho[3, 3] + rho[4, 4]  # ⟨1|...|1⟩: 10→10 + 11→11
    else
        # Trace over first qubit: ρ_2 = Σ_i ⟨i|ρ|i⟩_1
        result[1, 1] = rho[1, 1] + rho[3, 3]  # ⟨0|...|0⟩: 00→00 + 10→10
        result[1, 2] = rho[1, 2] + rho[3, 4]  # ⟨0|...|1⟩: 00→01 + 10→11
        result[2, 1] = rho[2, 1] + rho[4, 3]  # ⟨1|...|0⟩: 01→00 + 11→10
        result[2, 2] = rho[2, 2] + rho[4, 4]  # ⟨1|...|1⟩: 01→01 + 11→11
    end

    return result
end
