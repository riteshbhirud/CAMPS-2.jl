# CAMPS.jl/src/clifford_interface.jl
# Interface with QuantumClifford.jl for Clifford algebra operations

#==============================================================================#
# IMPORTS (these are imported in the main CAMPS.jl module)
# We use: PauliOperator, Stabilizer, Destabilizer, CliffordOperator
#         single_x, single_y, single_z, nqubits, xbit, zbit
#         apply!, inv, sHadamard, sPhase, sInvPhase, sX, sY, sZ
#         sCNOT, sCPHASE, sSWAP, gf2_gausselim!
#==============================================================================#

#==============================================================================#
# CLIFFORD INITIALIZATION
#==============================================================================#

"""
    initialize_clifford(n::Int) -> Destabilizer

Create an identity Clifford operator for n qubits.

The identity Clifford is represented as a Destabilizer initialized from
the stabilizer state |0⟩^⊗n (which has stabilizers Z₁, Z₂, ..., Zₙ).

We use Destabilizer (not MixedDestabilizer) because:
1. We're tracking a Clifford unitary, which is always full-rank
2. Destabilizer can be converted to CliffordOperator for Pauli conjugation
3. MixedDestabilizer is meant for mixed states, not unitary tracking

# Arguments
- `n::Int`: Number of qubits

# Returns
- `Destabilizer`: Identity Clifford operator

# Example
```julia
C = initialize_clifford(5)  # Identity on 5 qubits
```
"""
function initialize_clifford(n::Int)
    # one(Destabilizer, n) creates the identity destabilizer for n qubits
    # representing the state |0⟩^⊗n with stabilizers Z₁, Z₂, ..., Zₙ
    # and destabilizers X₁, X₂, ..., Xₙ
    return one(Destabilizer, n)
end

#==============================================================================#
# TWISTED PAULI COMPUTATION
#==============================================================================#

"""
    commute_pauli_through_clifford(P::PauliOperator, C::Destabilizer) -> PauliOperator

Compute the twisted Pauli P̃ = C† · P · C.

This is the core operation for CAMPS: when a rotation R_P(θ) is applied after
Clifford C, we have C · R_P(θ) = R_{P̃}(θ) · C where P̃ = C† · P · C.

# How it works
1. Convert the Destabilizer C to a CliffordOperator
2. Compute the inverse C⁻¹ = C†
3. Apply: apply!(stab, C_inv) computes C⁻¹ · P · (C⁻¹)† = C† · P · C

# Arguments
- `P::PauliOperator`: Original Pauli operator
- `C::Destabilizer`: Accumulated Clifford operator (as Destabilizer tableau)

# Returns
- `PauliOperator`: Twisted Pauli P̃ = C† · P · C

# Example
```julia
C = initialize_clifford(2)
apply!(C, sHadamard(1))  # C = H₁

P = single_z(2, 1)  # Z on qubit 1
P_twisted = commute_pauli_through_clifford(P, C)
# P_twisted = H† · Z · H = X (on qubit 1)
```
"""
function commute_pauli_through_clifford(P::PauliOperator, C::Destabilizer)::PauliOperator
    # Wrap P in a single-row Stabilizer
    stab = Stabilizer([P])

    # Convert Destabilizer to CliffordOperator for inversion
    # This works because Destabilizer represents a pure state (full rank)
    C_op = CliffordOperator(C)

    # Compute inverse: C⁻¹ = C†
    C_inv = inv(C_op)

    # apply!(stab, cliff) computes cliff · stab · cliff†
    # So apply!(stab, C_inv) computes C_inv · P · C_inv† = C† · P · C
    apply!(stab, C_inv)

    # Extract the transformed Pauli
    return stab[1]
end

"""
    axis_to_pauli(axis::Symbol, qubit::Int, n::Int) -> PauliOperator

Create a single-qubit Pauli operator for the given rotation axis.

# Arguments
- `axis::Symbol`: Rotation axis (:X, :Y, or :Z)
- `qubit::Int`: Target qubit (1-indexed)
- `n::Int`: Total number of qubits

# Returns
- `PauliOperator`: Single-qubit Pauli on specified qubit

# Example
```julia
P = axis_to_pauli(:Z, 2, 5)  # Z on qubit 2 of 5-qubit system
```
"""
function axis_to_pauli(axis::Symbol, qubit::Int, n::Int)::PauliOperator
    if axis == :X
        return single_x(n, qubit)
    elseif axis == :Y
        return single_y(n, qubit)
    elseif axis == :Z
        return single_z(n, qubit)
    else
        throw(ArgumentError("Unknown axis: $axis. Expected :X, :Y, or :Z"))
    end
end

#==============================================================================#
# CLIFFORD GATE APPLICATION
#==============================================================================#

"""
    apply_clifford_gate!(C::Destabilizer, gate) -> Destabilizer

Apply a QuantumClifford symbolic gate to the accumulated Clifford.

This modifies C in-place to represent C_new = gate · C.

# Arguments
- `C::Destabilizer`: Accumulated Clifford (modified in-place)
- `gate`: QuantumClifford symbolic gate (e.g., sHadamard(1), sCNOT(1,2))

# Returns
- `Destabilizer`: The modified Clifford (same object as input)

# Example
```julia
C = initialize_clifford(3)
apply_clifford_gate!(C, sHadamard(1))
apply_clifford_gate!(C, sCNOT(1, 2))
# C now represents CNOT(1,2) · H(1)
```
"""
function apply_clifford_gate!(C::Destabilizer, gate)::Destabilizer
    apply!(C, gate)
    return C
end

"""
    apply_clifford_gates!(C::Destabilizer, gates::Vector) -> Destabilizer

Apply a sequence of QuantumClifford symbolic gates to the accumulated Clifford.

Gates are applied in order: C_new = gates[end] · ... · gates[2] · gates[1] · C

# Arguments
- `C::Destabilizer`: Accumulated Clifford (modified in-place)
- `gates::Vector`: Vector of QuantumClifford symbolic gates

# Returns
- `Destabilizer`: The modified Clifford
"""
function apply_clifford_gates!(C::Destabilizer, gates::Vector)::Destabilizer
    for gate in gates
        apply!(C, gate)
    end
    return C
end

"""
    apply_inverse_gates!(C::Destabilizer, gates::Vector) -> Destabilizer

Apply the inverse of each gate in reverse order.

If gates = [g₁, g₂, g₃], this applies g₃⁻¹ · g₂⁻¹ · g₁⁻¹ to C.
This is equivalent to applying (g₁ · g₂ · g₃)⁻¹.

Used in OFD: after building disentangler D = [d₁, d₂, ...], we need to
apply D⁻¹ to the accumulated Clifford.

# Arguments
- `C::Destabilizer`: Accumulated Clifford (modified in-place)
- `gates::Vector`: Vector of QuantumClifford symbolic gates

# Returns
- `Destabilizer`: The modified Clifford

# Example
```julia
# If D = [sCNOT(1,2), sCPHASE(1,3)], then:
# apply_inverse_gates!(C, D) applies sCPHASE(1,3)⁻¹ · sCNOT(1,2)⁻¹
# which equals (sCNOT(1,2) · sCPHASE(1,3))⁻¹ = D⁻¹
```
"""
function apply_inverse_gates!(C::Destabilizer, gates::Vector)::Destabilizer
    for gate in reverse(gates)
        apply!(C, inv(gate))
    end
    return C
end

#==============================================================================#
# SYMBOLIC GATE CONVERSION
#==============================================================================#

"""
    resolve_symbolic_gate(gate_spec::Tuple) -> Any

Convert a gate specification tuple to a QuantumClifford symbolic gate.

This is used to convert the placeholder tuples from Phase 1 CliffordGate
constructors to actual QuantumClifford gates.

# Arguments
- `gate_spec::Tuple`: Gate specification like (:H, 1) or (:CNOT, 1, 2)

# Returns
- QuantumClifford symbolic gate

# Supported gates
- `(:H, q)` → `sHadamard(q)`
- `(:S, q)` → `sPhase(q)`
- `(:Sdag, q)` → `sInvPhase(q)`
- `(:X, q)` → `sX(q)`
- `(:Y, q)` → `sY(q)`
- `(:Z, q)` → `sZ(q)`
- `(:CNOT, c, t)` → `sCNOT(c, t)`
- `(:CZ, q1, q2)` → `sCPHASE(q1, q2)`
- `(:SWAP, q1, q2)` → `sSWAP(q1, q2)`
"""
function resolve_symbolic_gate(gate_spec::Tuple)
    gate_type = gate_spec[1]
    
    if gate_type == :H
        return sHadamard(gate_spec[2])
    elseif gate_type == :S
        return sPhase(gate_spec[2])
    elseif gate_type == :Sdag
        return sInvPhase(gate_spec[2])
    elseif gate_type == :X
        return sX(gate_spec[2])
    elseif gate_type == :Y
        return sY(gate_spec[2])
    elseif gate_type == :Z
        return sZ(gate_spec[2])
    elseif gate_type == :CNOT
        return sCNOT(gate_spec[2], gate_spec[3])
    elseif gate_type == :CZ
        return sCPHASE(gate_spec[2], gate_spec[3])
    elseif gate_type == :SWAP
        return sSWAP(gate_spec[2], gate_spec[3])
    else
        throw(ArgumentError("Unknown gate type: $gate_type"))
    end
end

"""
    resolve_clifford_gate(cg::CliffordGate) -> Vector

Convert a CliffordGate to a vector of QuantumClifford symbolic gates.

# Arguments
- `cg::CliffordGate`: CliffordGate with tuple specifications

# Returns
- `Vector`: Vector of QuantumClifford symbolic gates
"""
function resolve_clifford_gate(cg::CliffordGate)::Vector
    return [resolve_symbolic_gate(spec) for spec in cg.gates if spec isa Tuple]
end

"""
    apply_clifford_gate_to_state!(state::CAMPSState, cg::CliffordGate) -> CAMPSState

Apply a CliffordGate to a CAMPSState.

This resolves the gate specifications to QuantumClifford gates and applies
them to the accumulated Clifford. The MPS is unchanged.

# Arguments
- `state::CAMPSState`: CAMPS state (modified in-place)
- `cg::CliffordGate`: Clifford gate to apply

# Returns
- `CAMPSState`: The modified state
"""
function apply_clifford_gate_to_state!(state::CAMPSState, cg::CliffordGate)::CAMPSState
    for spec in cg.gates
        if spec isa Tuple
            gate = resolve_symbolic_gate(spec)
            apply!(state.clifford, gate)
        else
            # Already a QuantumClifford gate
            apply!(state.clifford, spec)
        end
    end
    return state
end

#==============================================================================#
# PAULI OPERATOR UTILITIES
#==============================================================================#

"""
    get_pauli_at(P::PauliOperator, j::Int) -> Symbol

Get the Pauli symbol at position j of a PauliOperator.

# Arguments
- `P::PauliOperator`: Pauli operator
- `j::Int`: Qubit index (1-indexed)

# Returns
- `Symbol`: :I, :X, :Y, or :Z
"""
function get_pauli_at(P::PauliOperator, j::Int)::Symbol
    x, z = P[j]
    return xz_to_symbol(x, z)
end

"""
    get_pauli_phase(P::PauliOperator) -> ComplexF64

Get the phase of a PauliOperator as a complex number.

# Arguments
- `P::PauliOperator`: Pauli operator

# Returns
- `ComplexF64`: +1, +i, -1, or -i
"""
function get_pauli_phase(P::PauliOperator)::ComplexF64
    return phase_to_complex(P.phase[])
end

"""
    pauli_weight(P::PauliOperator) -> Int

Count the number of non-identity Paulis in the operator.

# Arguments
- `P::PauliOperator`: Pauli operator

# Returns
- `Int`: Number of qubits where P is not identity
"""
function pauli_weight(P::PauliOperator)::Int
    n = nqubits(P)
    weight = 0
    for j in 1:n
        x, z = P[j]
        if x || z  # Not identity
            weight += 1
        end
    end
    return weight
end

"""
    pauli_support(P::PauliOperator) -> Vector{Int}

Get the indices of qubits where P is not identity.

# Arguments
- `P::PauliOperator`: Pauli operator

# Returns
- `Vector{Int}`: Indices of non-identity positions
"""
function pauli_support(P::PauliOperator)::Vector{Int}
    n = nqubits(P)
    support = Int[]
    for j in 1:n
        x, z = P[j]
        if x || z
            push!(support, j)
        end
    end
    return support
end

"""
    has_x_or_y(P::PauliOperator, j::Int) -> Bool

Check if qubit j has X or Y (i.e., xbit is true).

This is the key check for OFD disentanglability.

# Arguments
- `P::PauliOperator`: Pauli operator
- `j::Int`: Qubit index

# Returns
- `Bool`: true if P has X or Y at position j
"""
function has_x_or_y(P::PauliOperator, j::Int)::Bool
    return xbit(P)[j]
end

"""
    get_xbit_vector(P::PauliOperator) -> BitVector

Get the x-bit vector of a Pauli operator.

xbit[j] = true if P has X or Y at position j.
This is used for building the GF(2) matrix.

# Arguments
- `P::PauliOperator`: Pauli operator

# Returns
- `BitVector`: x-bits for each qubit
"""
function get_xbit_vector(P::PauliOperator)::BitVector
    return BitVector(xbit(P))
end

"""
    get_zbit_vector(P::PauliOperator) -> BitVector

Get the z-bit vector of a Pauli operator.

zbit[j] = true if P has Y or Z at position j.

# Arguments
- `P::PauliOperator`: Pauli operator

# Returns
- `BitVector`: z-bits for each qubit
"""
function get_zbit_vector(P::PauliOperator)::BitVector
    return BitVector(zbit(P))
end

#==============================================================================#
# CLIFFORD STATE QUERIES
#==============================================================================#

"""
    clifford_nqubits(C::Destabilizer) -> Int

Get the number of qubits in a Clifford operator.
"""
function clifford_nqubits(C::Destabilizer)::Int
    return nqubits(C)
end

#==============================================================================#
# PAULI STRING CREATION
#==============================================================================#

"""
    create_pauli_string(paulis::Vector{Symbol}, n::Int) -> PauliOperator

Create a Pauli string from a vector of symbols.

Uses QuantumClifford's PauliOperator constructor with explicit x and z bit vectors.

# Arguments
- `paulis::Vector{Symbol}`: Pauli symbols [:I, :X, :Y, :Z] for each qubit
- `n::Int`: Number of qubits (must equal length(paulis))

# Returns
- `PauliOperator`: The constructed Pauli string

# Example
```julia
P = create_pauli_string([:X, :Y, :Z], 3)  # X ⊗ Y ⊗ Z
```
"""
function create_pauli_string(paulis::Vector{Symbol}, n::Int)::PauliOperator
    length(paulis) == n || throw(ArgumentError("paulis length must equal n"))

    # Build x and z bit vectors from symbols
    # QuantumClifford encoding: I=(0,0), X=(1,0), Y=(1,1), Z=(0,1)
    xbits = falses(n)
    zbits = falses(n)

    for (j, σ) in enumerate(paulis)
        x, z = symbol_to_xz(σ)
        xbits[j] = x
        zbits[j] = z
    end

    # Use QuantumClifford's PauliOperator constructor
    # PauliOperator(phase, xbits, zbits) where phase is UInt8
    # Phase 0x00 = +1 (we start with positive phase)
    # Note: n is inferred from the length of the bit vectors
    return PauliOperator(0x00, BitVector(xbits), BitVector(zbits))
end

"""
    create_pauli_string_with_phase(paulis::Vector{Symbol}, n::Int, phase::ComplexF64) -> PauliOperator

Create a Pauli string from symbols with a specified phase.

# Arguments
- `paulis::Vector{Symbol}`: Pauli symbols [:I, :X, :Y, :Z] for each qubit
- `n::Int`: Number of qubits
- `phase::ComplexF64`: Overall phase (+1, +i, -1, or -i)

# Returns
- `PauliOperator`: The constructed Pauli string with phase
"""
function create_pauli_string_with_phase(paulis::Vector{Symbol}, n::Int, phase::ComplexF64)::PauliOperator
    length(paulis) == n || throw(ArgumentError("paulis length must equal n"))

    xbits = falses(n)
    zbits = falses(n)

    for (j, σ) in enumerate(paulis)
        x, z = symbol_to_xz(σ)
        xbits[j] = x
        zbits[j] = z
    end

    phase_byte = complex_to_phase(phase)
    # Note: n is inferred from the length of the bit vectors
    return PauliOperator(phase_byte, BitVector(xbits), BitVector(zbits))
end

#==============================================================================#
# DEBUGGING UTILITIES
#==============================================================================#

"""
    pauli_to_string(P::PauliOperator) -> String

Convert a PauliOperator to a human-readable string.

# Arguments
- `P::PauliOperator`: Pauli operator

# Returns
- `String`: String like "+XYZ" or "-iXX_Z"
"""
function pauli_to_string(P::PauliOperator)::String
    n = nqubits(P)
    
    # Phase prefix
    phase = P.phase[]
    prefix = if phase == 0x00
        "+"
    elseif phase == 0x01
        "+i"
    elseif phase == 0x02
        "-"
    else  # 0x03
        "-i"
    end
    
    # Pauli string
    chars = Char[]
    for j in 1:n
        σ = get_pauli_at(P, j)
        push!(chars, σ == :I ? '_' : first(string(σ)))
    end
    
    return prefix * String(chars)
end
