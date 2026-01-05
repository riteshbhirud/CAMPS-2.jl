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
For each computational basis state |j⟩, compute C|j⟩ by:
1. Express |j⟩ as X^j |0⟩ where X^j applies X to qubits with bit=1
2. Use C X^j C† = P̃^j (conjugated Pauli) to get C|j⟩ = P̃^j C|0⟩
3. C|0⟩ is computed from the stabilizer structure

# Note
For n qubits, this creates a 2^n × 2^n matrix, which is exponential.
Only use for small n (≤ 4 qubits in practice).
"""
function clifford_to_matrix(C::CliffordOperator)::Matrix{ComplexF64}
    n = nqubits(C)
    dim = 2^n

    if n > 12
        throw(ArgumentError("clifford_to_matrix only supports n ≤ 12"))
    end

    # Build the unitary matrix by computing C|j⟩ for each basis state j
    # We use a hybrid approach:
    # 1. For column 0 (C|0⟩), use destabilizer_to_statevector (works correctly)
    # 2. For other columns, use the conjugated-X formula with proper phase tracking

    U = zeros(ComplexF64, dim, dim)

    # Column 0: C|0⟩
    D0 = one(Destabilizer, n)
    apply!(D0, C)
    col0 = destabilizer_to_statevector(D0)
    U[:, 1] = col0

    # For columns j > 0: C|j⟩ = C X_{bits} |0⟩ = (C X_k1 C†)(C X_k2 C†)... C|0⟩
    # The key is to track phases properly when applying conjugated X operators

    for j in 1:(dim-1)
        col = copy(col0)

        # For each bit set in j, apply the conjugated X operator
        # Use little-endian: bit k corresponds to qubit (k+1)
        for bit in 0:(n-1)
            if (j >> bit) & 1 == 1
                qubit = bit + 1  # little-endian: bit 0 → qubit 1

                # Compute C X_qubit C†
                X_q = single_x(n, qubit)
                stab = Stabilizer([X_q])
                apply!(stab, C)
                P = stab[1]

                # Apply P to col with correct phase
                # The phase includes: generator phase + Y phase from P acting on states
                apply_pauli_to_statevector_with_y_phase!(col, P, n)
            end
        end

        U[:, j+1] = col
    end

    return U
end

"""
    apply_pauli_to_statevector_with_y_phase!(psi::Vector{ComplexF64}, P::PauliOperator, n::Int)

Apply Pauli operator P to state vector psi in-place, correctly handling Y = iXZ phase.

For each basis state |j⟩ with amplitude ψ_j:
P|j⟩ = (generator_phase) × (Y_phase) × (Z_phase) × |j ⊕ x_P⟩

where:
- generator_phase: i^{P.phase[]}
- Y_phase: i for each qubit with Y (X=1, Z=1)
- Z_phase: (-1)^{z·j} where z is Z-pattern and j is basis state
"""
function apply_pauli_to_statevector_with_y_phase!(psi::Vector{ComplexF64}, P::PauliOperator, n::Int)
    dim = 2^n
    psi_new = zeros(ComplexF64, dim)

    for j in 0:(dim-1)
        if abs(psi[j+1]) < 1e-15
            continue
        end

        # Compute x_P (X pattern) and phases
        # Use little-endian: qubit q corresponds to bit (q-1)
        x_P = 0
        z_phase_count = 0
        y_phase_count = 0

        for q in 1:n
            x_q, z_q = P[q]
            if x_q
                x_P |= 1 << (q - 1)  # little-endian: qubit q → bit (q-1)
            end
            if z_q
                j_q = (j >> (q - 1)) & 1  # little-endian
                if j_q == 1
                    z_phase_count += 1
                end
            end
            if x_q && z_q  # Y component
                y_phase_count += 1
            end
        end

        # Total phase: generator + Z + Y
        # Generator phase is in i^k where k = P.phase[]
        # Z phase is (-1)^count = i^{2*count}
        # Y phase is i^count
        total_phase_power = (P.phase[] + 2 * z_phase_count + y_phase_count) % 4

        phase = if total_phase_power == 0
            1.0 + 0.0im
        elseif total_phase_power == 1
            0.0 + 1.0im
        elseif total_phase_power == 2
            -1.0 + 0.0im
        else
            0.0 - 1.0im
        end

        j_new = j ⊻ x_P
        psi_new[j_new + 1] = phase * psi[j + 1]
    end

    psi .= psi_new
end

"""
    destabilizer_to_statevector_with_phase(D::Destabilizer, C::CliffordOperator, input_j::Int, n::Int)

Compute the state vector for C|j⟩ with proper phase tracking.

The destabilizer D represents the state C|j⟩, but may have lost global phase.
We recover the correct phases by tracking how the Clifford acts on X operators.
"""
function destabilizer_to_statevector_with_phase(D::Destabilizer, C::CliffordOperator, input_j::Int, n::Int)::Vector{ComplexF64}
    dim = 2^n

    # Check if C is a diagonal Clifford (maps computational basis to itself)
    # A Clifford is diagonal if C X_k C† has X component only on qubit k
    # (i.e., X_k → ±Y_k or ±X_k, not X_k → something involving other qubits)

    is_diagonal = true
    for k in 1:n
        X_k = single_x(n, k)
        stab_x = Stabilizer([X_k])
        apply!(stab_x, C)
        P = stab_x[1]

        # Check that P has X only on qubit k
        for q in 1:n
            x_q, _ = P[q]
            if x_q && q != k
                is_diagonal = false
                break
            end
        end
        if !is_diagonal
            break
        end
    end

    if is_diagonal
        # For diagonal Cliffords, the basis state is unchanged, but phase changes
        # The phase for |j⟩ → e^{iθ_j}|j⟩ can be computed from how C transforms X operators

        # C X_k C† tells us how phase changes when qubit k is |1⟩
        # For S gate: S X S† = Y, so when qubit is |1⟩, we pick up phase i

        # Compute phase correction for this basis state
        phase_power = 0
        for bit in 0:(n-1)
            if (input_j >> bit) & 1 == 1
                qubit = n - bit
                # See how C transforms X on this qubit
                X_q = single_x(n, qubit)
                stab_x = Stabilizer([X_q])
                apply!(stab_x, C)
                P = stab_x[1]

                # Count Y components to determine phase
                for q in 1:n
                    x_q, z_q = P[q]
                    if x_q && z_q  # Y component
                        phase_power += 1
                    end
                end

                # Also account for the generator phase of the transformed Pauli
                phase_power += P.phase[]
            end
        end

        phase_power = phase_power % 4
        phase_correction = if phase_power == 0
            1.0 + 0.0im
        elseif phase_power == 1
            0.0 + 1.0im
        elseif phase_power == 2
            -1.0 + 0.0im
        else
            0.0 - 1.0im
        end

        # For diagonal Cliffords, output state is same basis state with phase
        result = zeros(ComplexF64, dim)
        result[input_j + 1] = phase_correction
        return result
    end

    # For non-diagonal Cliffords, use destabilizer_to_statevector
    # which correctly identifies which basis states have nonzero amplitudes
    return destabilizer_to_statevector(D)
end

"""
    clifford_on_zero_state(C::CliffordOperator, n::Int) -> Vector{ComplexF64}

Compute C|0...0⟩ for a Clifford operator.

The result is a stabilizer state, which we convert to a state vector.
"""
function clifford_on_zero_state(C::CliffordOperator, n::Int)::Vector{ComplexF64}
    # Apply C to |0...0⟩ represented as a Destabilizer
    D = one(Destabilizer, n)
    apply!(D, C)

    # Convert to state vector
    return destabilizer_to_statevector(D)
end

"""
    apply_conjugated_x!(psi::Vector{ComplexF64}, C::CliffordOperator, qubit::Int, n::Int)

Apply the operator C X_qubit C† to state vector psi (in-place).

C X_qubit C† is a Pauli operator (since Cliffords map Paulis to Paulis).
We compute it and apply it to psi.
"""
function apply_conjugated_x!(psi::Vector{ComplexF64}, C::CliffordOperator, qubit::Int, n::Int)
    # Compute C X_qubit C†
    # Create X on the specified qubit
    X_q = single_x(n, qubit)

    # Conjugate by C: wrap in Stabilizer and apply C
    # apply!(stab, C) computes C · X · C†
    stab = Stabilizer([X_q])
    apply!(stab, C)  # C · X · C†

    # Extract the conjugated Pauli
    P_conj = stab[1]

    # Apply this Pauli to the state vector
    apply_pauli_to_statevector!(psi, P_conj, n)
end

"""
    apply_pauli_to_statevector!(psi::Vector{ComplexF64}, P::PauliOperator, n::Int)

Apply Pauli operator P to state vector psi (in-place).

P = i^p X^a Z^b where p is the phase, a is the X-pattern, b is the Z-pattern.
"""
function apply_pauli_to_statevector!(psi::Vector{ComplexF64}, P::PauliOperator, n::Int)
    dim = 2^n

    # Extract phase
    phase_power = P.phase[]
    global_phase = if phase_power == 0
        1.0 + 0.0im
    elseif phase_power == 1
        0.0 + 1.0im
    elseif phase_power == 2
        -1.0 + 0.0im
    else  # phase_power == 3
        0.0 - 1.0im
    end

    # Extract X and Z patterns
    x_bits = zeros(Int, n)
    z_bits = zeros(Int, n)
    for q in 1:n
        x_q, z_q = P[q]
        x_bits[q] = x_q ? 1 : 0
        z_bits[q] = z_q ? 1 : 0
    end

    # P|j⟩ = global_phase * (-1)^{z·j} |j ⊕ x⟩
    # where z·j is the dot product of z_bits with j_bits

    new_psi = zeros(ComplexF64, dim)

    for j in 0:(dim-1)
        # Extract j's bits (little-endian: qubit q corresponds to bit q-1)
        j_bits = [(j >> (q - 1)) & 1 for q in 1:n]

        # Compute z · j
        z_dot_j = 0
        for q in 1:n
            z_dot_j += z_bits[q] * j_bits[q]
        end
        z_phase = (z_dot_j % 2 == 0) ? 1.0 : -1.0

        # Compute j ⊕ x (the output index)
        out_bits = [j_bits[q] ⊻ x_bits[q] for q in 1:n]
        out_idx = 0
        for q in 1:n
            out_idx += out_bits[q] * (1 << (q - 1))  # little-endian
        end

        # new_psi[out_idx] = global_phase * z_phase * psi[j]
        new_psi[out_idx + 1] += global_phase * z_phase * psi[j + 1]
    end

    psi .= new_psi
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

We use little-endian: bit k of j corresponds to qubit (k+1).
"""
function clifford_on_basis_state_destab(C::CliffordOperator, j::Int, n::Int)::Vector{ComplexF64}
    dim = 2^n

    # Create |0...0⟩ state as a Destabilizer
    D = one(Destabilizer, n)

    # Apply X gates to create |j⟩ from |0...0⟩
    # Use little-endian: bit k (0-indexed) corresponds to qubit (k + 1)
    for k in 0:(n-1)
        if (j >> k) & 1 == 1
            # bit k is set, so qubit (k + 1) should be |1⟩
            apply!(D, sX(k + 1))
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

    stab = stabilizerview(D)

    # Algorithm: Find basis states in the stabilizer code space by enumeration
    # For each basis state |j⟩, check if S_i|j⟩ = ±|j⟩ matches the stabilizer phase
    # States with X components in stabilizers can have multiple basis states

    psi = zeros(ComplexF64, dim)

    # First, find which basis states are eigenstates of all stabilizers
    # and track their eigenvalues
    for j in 0:(dim-1)
        is_eigenstate = true
        total_phase = 1.0 + 0.0im

        for i in 1:n
            S = stab[i]
            gen_phase = S.phase[]  # 0 = +1, 2 = -1

            # Check if |j⟩ is an eigenstate of the Pauli part of S
            # and compute the eigenvalue
            x_pattern = 0
            has_x = false
            z_eigenval = 1

            for q in 1:n
                x_q, z_q = S[q]
                if x_q
                    has_x = true
                    x_pattern |= 1 << (q - 1)  # little-endian
                end
                if z_q
                    j_q = (j >> (q - 1)) & 1  # little-endian
                    if j_q == 1
                        z_eigenval *= -1
                    end
                end
            end

            if has_x
                # S has X component, so |j⟩ is only an eigenstate if
                # j ⊕ x_pattern = j, which means x_pattern = 0
                # But x_pattern ≠ 0, so |j⟩ is not an eigenstate
                # unless this is an entangled state situation
                is_eigenstate = false
                break
            else
                # Pure Z stabilizer: eigenvalue is z_eigenval
                # Check if it matches the expected sign from gen_phase
                expected_eigenval = gen_phase == 0 ? 1 : -1
                if z_eigenval != expected_eigenval
                    is_eigenstate = false
                    break
                end
            end
        end

        if is_eigenstate
            psi[j + 1] = 1.0
        end
    end

    # Check if we found any eigenstates
    norm_val = norm(psi)
    if norm_val > 1e-15
        psi ./= norm_val
        return psi
    end

    # If no pure Z eigenstates found, this is an entangled state
    # Use the destabilizer-based algorithm
    # Find a computational basis state in the code by examining destabilizers

    # For entangled states, use a different approach:
    # Start from a "seed" state and apply (I + S)/2 projectors
    # The seed should be found using the destabilizers

    # Algorithm from Aaronson-Gottesman:
    # 1. Use destabilizers to identify a basis state in the code
    # 2. Apply stabilizers to find all basis states and their phases

    destab = destabilizerview(D)

    # Find the "graph" of basis states connected by X operators in stabilizers
    # For the Bell state (+XX, +ZZ), both |00⟩ and |11⟩ are in the code

    # Simpler approach: enumerate and check all basis states
    # For each state, apply all stabilizers and see if it maps back to code space

    # Even simpler: Use the projector method but start from a superposition
    # that spans the code space

    # Actually, the correct approach for entangled states:
    # Initialize all amplitudes, then iteratively project

    # Reset and try uniform superposition
    psi = ones(ComplexF64, dim) / sqrt(dim)

    for i in 1:n
        S = stab[i]
        gen_phase = S.phase[]

        # Apply projector (I + S)/2 or (I - |S|)/2
        psi_new = zeros(ComplexF64, dim)

        for j in 0:(dim-1)
            if abs(psi[j+1]) < 1e-15
                continue
            end

            # Add the identity part
            psi_new[j + 1] += psi[j + 1]

            # Compute S|j⟩ (including generator phase)
            x_S = 0
            z_phase_count = 0
            y_phase_count = 0

            for q in 1:n
                x_q, z_q = S[q]
                if x_q
                    x_S |= 1 << (q - 1)  # little-endian
                end
                if z_q
                    j_q = (j >> (q - 1)) & 1  # little-endian
                    if j_q == 1
                        z_phase_count += 1
                    end
                end
                if x_q && z_q
                    y_phase_count += 1
                end
            end

            # Total phase including generator phase
            total_phase_power = (gen_phase + 2 * z_phase_count + y_phase_count) % 4

            phase = if total_phase_power == 0
                1.0 + 0.0im
            elseif total_phase_power == 1
                0.0 + 1.0im
            elseif total_phase_power == 2
                -1.0 + 0.0im
            else
                0.0 - 1.0im
            end

            j_new = j ⊻ x_S
            psi_new[j_new + 1] += phase * psi[j + 1]
        end

        psi = psi_new / 2
    end

    # Normalize
    norm_val = norm(psi)
    if norm_val > 1e-15
        psi ./= norm_val
    end

    return psi
end

"""
    stabilizer_state_amplitude(D::Destabilizer, j::Int, n::Int) -> ComplexF64

Compute ⟨j|ψ⟩ for stabilizer state |ψ⟩ represented by destabilizer D.

Uses the correct algorithm based on the stabilizer formalism:
|ψ⟩ = (1/2^n) Σ_{g ∈ G} g |0...0⟩

where G is the stabilizer group (all 2^n combinations of n generators).

For computational basis state |j⟩:
⟨j|ψ⟩ = (1/2^n) Σ_{g: x(g)=j} phase(g)

where x(g) is the X-part of g (determines which basis state g|0⟩ maps to).
"""
function stabilizer_state_amplitude(D::Destabilizer, j::Int, n::Int)::ComplexF64
    stab = stabilizerview(D)

    # j's bits in little-endian (qubit q corresponds to bit q-1)
    j_bits = [(j >> (q - 1)) & 1 for q in 1:n]

    # Sum over all 2^n combinations of the n generators
    # g = S_1^{a_1} S_2^{a_2} ... S_n^{a_n} for a_i ∈ {0,1}
    total_phase = 0.0 + 0.0im

    for mask in 0:(2^n - 1)
        # Compute the X-pattern and phase for this group element
        x_pattern = zeros(Int, n)
        # Accumulate phase: starts at +1 (phase_power = 0)
        phase_power = 0

        # We multiply stabilizers in order, tracking anticommutation phases
        # When P1 * P2 where they anticommute, we get an extra factor of i

        # Build the product incrementally
        # Track both X and Z parts to compute anticommutation
        running_x = zeros(Int, n)
        running_z = zeros(Int, n)

        for i in 1:n
            if (mask >> (i-1)) & 1 == 1
                S = stab[i]

                # Extract X and Z parts of S
                s_x = zeros(Int, n)
                s_z = zeros(Int, n)
                for q in 1:n
                    x_q, z_q = S[q]
                    s_x[q] = x_q ? 1 : 0
                    s_z[q] = z_q ? 1 : 0
                end

                # Compute anticommutation phase when multiplying running * S
                # The phase from P1 * P2 is i^{2 * Σ_q (x1_q * z2_q)}
                # when P1 = i^{p1} X^{a1} Z^{b1} and P2 = i^{p2} X^{a2} Z^{b2}
                # Actually: P1 * P2 = i^{p1+p2} * (-1)^{a1·b2} * X^{a1⊕a2} Z^{b1⊕b2}
                # The factor (-1)^{a1·b2} = i^{2*a1·b2}
                anticommute_phase = 0
                for q in 1:n
                    anticommute_phase += running_x[q] * s_z[q]
                end
                phase_power += 2 * anticommute_phase

                # Add the generator's own phase
                phase_power += S.phase[]

                # Update running product
                for q in 1:n
                    running_x[q] ⊻= s_x[q]
                    running_z[q] ⊻= s_z[q]
                end
            end
        end

        # The X-pattern of the final product
        x_pattern = running_x

        # Check if x_pattern matches j
        if x_pattern == j_bits
            # Need to account for:
            # 1. The Z-part acting on |0...0⟩ - Z|0⟩ = |0⟩ (no phase needed)
            # 2. The Y = iXZ convention - each qubit with both X=1 and Z=1 contributes i
            #
            # The formula for P|0...0⟩ where P is a Pauli string with X-part x and Z-part z:
            # P|0...0⟩ = (phase_byte) × (i^{#Y}) × |x⟩
            # where #Y is the count of qubits with both X=1 and Z=1.
            #
            # This is because:
            # - Z|0⟩ = |0⟩ (eigenvalue +1)
            # - X|0⟩ = |1⟩
            # - Y|0⟩ = iXZ|0⟩ = iX|0⟩ = i|1⟩

            # Count Y components (both X and Z set)
            y_count = 0
            for q in 1:n
                if running_x[q] == 1 && running_z[q] == 1
                    y_count += 1
                end
            end
            phase_power += y_count

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
            total_phase += phase_factor
        end
    end

    # Normalize by 2^n (the size of the stabilizer group)
    return total_phase / (2^n)
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
Uses little-endian: bit k of j (0-indexed) corresponds to qubit (k + 1).
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
            # Qubit q corresponds to bit (q - 1) in little-endian convention
            bit_idx = q - 1
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
Uses little-endian: bit k of j (0-indexed from right) corresponds to qubit (k + 1).
"""
function compute_amplitude_phase(D::Destabilizer, j::Int, n::Int)::ComplexF64
    # For stabilizer states, amplitudes are of the form ±1/√k or ±i/√k
    # We compute the phase by tracking the Pauli phases through the destabilizers

    destabs = destabilizerview(D)

    # Compute the phase accumulation
    phase_power = 0  # Tracks i^phase_power

    # For each qubit, check if the corresponding bit in j is 1
    # Use little-endian convention: qubit q corresponds to bit (q - 1)
    for q in 1:n
        bit_idx = q - 1
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
