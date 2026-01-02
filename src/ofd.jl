# CAMPS.jl/src/ofd.jl
# Optimization-Free Disentangling (OFD) algorithm implementation
#
# Based on: Liu & Clark, "Clifford-Augmented Matrix Product States" (arXiv:2412.17209)
#
# OFD is an algebraic construction that achieves optimal disentangling when a
# "free qubit" is available. The key insight is that controlled-Pauli gates
# can transform a rotation on a free qubit into a product state.

#==============================================================================#
# OVERVIEW
#
# The OFD algorithm works as follows:
#
# Given a twisted Pauli P̃ = C†PC where P is the original rotation axis,
# we want to apply the rotation R_{P̃}(θ) = αI + βP̃ to the MPS |ψ⟩.
#
# If there exists a "free qubit" i (qubit in |0⟩ state) where P̃[i] ∈ {X, Y},
# we can use the following disentangler:
#
#     D = ∏_{j≠i, P̃[j]≠I} (CP̃[j])_{i,j}
#
# where (CP)_{i,j} is a controlled-P gate with control i and target j.
#
# After applying D:
#     D · R_{P̃}(θ) = R_{P̃[i]}(θ) ⊗ I_{N\i} · D
#
# The rotation now acts only on the control qubit i, transforming it from
# |0⟩ to the magic state |m⟩ = α|0⟩ + β|1⟩ (for X) or α|0⟩ + iβ|1⟩ (for Y).
#
# This keeps the bond dimension at 1 for this T-gate application.
#==============================================================================#

#==============================================================================#
# OFD DISENTANGLER CONSTRUCTION
#==============================================================================#

"""
    build_disentangler_gates(P::PauliOperator, control_qubit::Int) -> Vector

Build the sequence of controlled-Pauli gates that form the OFD disentangler.

The disentangler is:
    D = ∏_{j≠i, P[j]≠I} (CP[j])_{i,j}

where i is the control qubit and (CP)_{i,j} is a controlled-P gate.

# Arguments
- `P::PauliOperator`: The twisted Pauli operator
- `control_qubit::Int`: The free qubit to use as control (must have X or Y at this position)

# Returns
- `Vector`: Vector of QuantumClifford symbolic gates [(CP)_{i,j1}, (CP)_{i,j2}, ...]

# Theory
The controlled-Pauli gates are:
- CX (CNOT) when P[j] = X
- CY when P[j] = Y
- CZ when P[j] = Z

These are all Clifford gates, so the disentangler D is a Clifford circuit.

# Example
```julia
P = P"XZY"  # X on qubit 1, Z on qubit 2, Y on qubit 3
gates = build_disentangler_gates(P, 1)  # Use qubit 1 as control
# Returns [sCNOT(1,2)†, controlled-Y(1,3)] or equivalent decomposition
# Wait, CZ is symmetric. And for CX with control on X, we need CNOT.
# Actually: if control qubit has X, and target has Z, we apply CZ.
```

# Note
The gates are returned in the order they should be applied.
"""
function build_disentangler_gates(P::PauliOperator, control_qubit::Int)::Vector
    n = nqubits(P)
    gates = []

    # Verify control qubit has X or Y
    σ_control = get_pauli_at(P, control_qubit)
    if σ_control ∉ (:X, :Y)
        throw(ArgumentError("Control qubit must have X or Y in the twisted Pauli, got $σ_control"))
    end

    # Build controlled-Pauli gates for each target qubit j where P[j] ≠ I
    for j in 1:n
        if j == control_qubit
            continue  # Skip control qubit
        end

        σ_j = get_pauli_at(P, j)
        if σ_j == :I
            continue  # No gate needed for identity
        end

        # Build the appropriate controlled gate
        # Control is on qubit `control_qubit`, target is on qubit `j`
        gate = build_controlled_pauli_gate(σ_j, control_qubit, j)
        push!(gates, gate)
    end

    return gates
end

"""
    build_controlled_pauli_gate(σ::Symbol, control::Int, target::Int)

Build a controlled-σ gate with the specified control and target qubits.

# Arguments
- `σ::Symbol`: Target Pauli (:X, :Y, or :Z)
- `control::Int`: Control qubit index
- `target::Int`: Target qubit index

# Returns
- QuantumClifford symbolic gate

# Implementation
- CX (CNOT): sCNOT(control, target)
- CZ: sCPHASE(control, target) (symmetric)
- CY: Decomposed as S†·CX·S on target, i.e., (I⊗S†)·CNOT·(I⊗S)

# Note on CY decomposition
CY = (I ⊗ S†) · CNOT · (I ⊗ S)
This is because S·X·S† = Y, so:
    |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ Y
  = |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ S·X·S†
  = (I⊗S†)·(|0⟩⟨0|⊗I + |1⟩⟨1|⊗X)·(I⊗S)
  = (I⊗S†)·CNOT·(I⊗S)
"""
function build_controlled_pauli_gate(σ::Symbol, control::Int, target::Int)
    if σ == :X
        # CX = CNOT
        return sCNOT(control, target)
    elseif σ == :Z
        # CZ = CPHASE (symmetric)
        return sCPHASE(control, target)
    elseif σ == :Y
        # CY = S†_target · CNOT · S_target
        # Return as a vector of gates
        return [sInvPhase(target), sCNOT(control, target), sPhase(target)]
    else
        throw(ArgumentError("Invalid Pauli symbol: $σ"))
    end
end

"""
    flatten_gate_sequence(gates::Vector) -> Vector

Flatten a sequence of gates that may contain nested vectors (from CY decomposition).

# Arguments
- `gates::Vector`: Potentially nested vector of gates

# Returns
- `Vector`: Flat vector of individual gates
"""
function flatten_gate_sequence(gates::Vector)::Vector
    result = []
    for g in gates
        if g isa Vector
            append!(result, g)
        else
            push!(result, g)
        end
    end
    return result
end

#==============================================================================#
# OFD APPLICATION
#==============================================================================#

"""
    apply_ofd!(state::CAMPSState, P_twisted::PauliOperator, θ::Real,
               control_qubit::Int) -> CAMPSState

Apply OFD (Optimization-Free Disentangling) for a non-Clifford rotation.

This is the core OFD algorithm:
1. Build the disentangler D from the twisted Pauli
2. Apply D† to the accumulated Clifford (C → C·D†)
3. Apply the single-qubit rotation on the control qubit to the MPS
4. Mark the control qubit as "magic" (no longer free)
5. Update remaining twisted Paulis for subsequent gates

# Arguments
- `state::CAMPSState`: CAMPS state (modified in-place)
- `P_twisted::PauliOperator`: Twisted Pauli P̃ = C†PC
- `θ::Real`: Rotation angle
- `control_qubit::Int`: Free qubit to use for disentangling

# Returns
- `CAMPSState`: The modified state

# Theory
After OFD, the state factorizes as:
    |ψ'⟩ = |m⟩_i ⊗ |ψ_{N\\i}⟩

where |m⟩ = R_{P̃[i]}(θ)|0⟩ is the magic state on the control qubit.

For T gate (θ = π/4) on a qubit with X:
    |m⟩ = cos(π/8)|0⟩ - i·sin(π/8)|1⟩

# Example
```julia
state = CAMPSState(4)
initialize!(state)

# Apply H on qubit 1 (makes Z→X)
apply_clifford_gate!(state.clifford, sHadamard(1))

# For T gate on qubit 1 after H:
P_twisted = compute_twisted_pauli(state, :Z, 1)  # Returns X on qubit 1
control = find_disentangling_qubit(P_twisted, state.free_qubits)  # Returns 1

apply_ofd!(state, P_twisted, π/4, control)
```
"""
function apply_ofd!(state::CAMPSState, P_twisted::PauliOperator, θ::Real,
                    control_qubit::Int)::CAMPSState
    ensure_initialized!(state)

    n = state.n_qubits

    # Validate control qubit
    if !state.free_qubits[control_qubit]
        throw(ArgumentError("Control qubit $control_qubit is not free"))
    end

    σ_control = get_pauli_at(P_twisted, control_qubit)
    if σ_control ∉ (:X, :Y)
        throw(ArgumentError("Control qubit must have X or Y in twisted Pauli, got $σ_control"))
    end

    # Step 1: Build the disentangler gates
    disentangler_gates = build_disentangler_gates(P_twisted, control_qubit)
    disentangler_flat = flatten_gate_sequence(disentangler_gates)

    # Step 2: Apply D† to the accumulated Clifford
    # D† is applied by applying gates in reverse order with inverses
    # Note: We want C_new = C · D†, which is achieved by apply_inverse_gates!
    # But the paper convention is that we're updating C to absorb D†
    # So we apply D† = (g_1 · g_2 · ... · g_k)† = g_k† · ... · g_2† · g_1†
    apply_inverse_gates!(state.clifford, disentangler_flat)

    # Step 3: Apply the single-qubit rotation to the MPS on the control qubit
    # The rotation is R_{σ_control}(θ) where σ_control is X or Y
    # For the MPS in |0⟩^⊗n state initially, this creates the magic state
    apply_local_rotation!(state.mps, state.sites, control_qubit, σ_control, Float64(θ))

    # Step 4: Mark the control qubit as magic (no longer free)
    mark_as_magic!(state, control_qubit)

    # Step 5: Record the twisted Pauli for GF(2) analysis
    add_twisted_pauli!(state, P_twisted)

    return state
end

"""
    try_apply_ofd!(state::CAMPSState, P_twisted::PauliOperator, θ::Real)
        -> Tuple{Bool, CAMPSState}

Try to apply OFD if a suitable free qubit exists.

This function checks if OFD is possible and applies it if so.

# Arguments
- `state::CAMPSState`: CAMPS state (may be modified)
- `P_twisted::PauliOperator`: Twisted Pauli operator
- `θ::Real`: Rotation angle

# Returns
- `Tuple{Bool, CAMPSState}`: (success, modified state)
  - If success=true, OFD was applied
  - If success=false, the state is unchanged (needs fallback to OBD or direct)

# Example
```julia
success, state = try_apply_ofd!(state, P_twisted, π/4)
if !success
    # Fall back to OBD or direct application
    apply_twisted_rotation!(state.mps, state.sites, P_twisted, θ)
end
```
"""
function try_apply_ofd!(state::CAMPSState, P_twisted::PauliOperator, θ::Real)::Tuple{Bool, CAMPSState}
    # Check if OFD is possible
    control = find_disentangling_qubit(P_twisted, state.free_qubits)

    if control === nothing
        return (false, state)
    end

    # Apply OFD
    apply_ofd!(state, P_twisted, θ, control)
    return (true, state)
end

#==============================================================================#
# OFD FOR T-GATE SPECIFICALLY
#==============================================================================#

"""
    apply_t_gate_ofd!(state::CAMPSState, qubit::Int) -> Tuple{Bool, CAMPSState}

Apply a T gate using OFD if possible.

This is a convenience function specifically for T gates (the most common
non-Clifford gate in fault-tolerant circuits).

# Arguments
- `state::CAMPSState`: CAMPS state
- `qubit::Int`: Target qubit for the T gate

# Returns
- `Tuple{Bool, CAMPSState}`: (success, modified state)

# Process
1. Compute twisted Pauli: P̃ = C† Z_qubit C
2. Try to find a free qubit with X or Y in P̃
3. If found, apply OFD; otherwise return false

# Example
```julia
state = CAMPSState(4)
initialize!(state)

# Apply Hadamard to make T-gate disentanglable
apply_clifford_gate!(state.clifford, sHadamard(1))

# Now apply T gate on qubit 1
success, state = apply_t_gate_ofd!(state, 1)
@assert success  # Should succeed because H maps Z→X
```
"""
function apply_t_gate_ofd!(state::CAMPSState, qubit::Int)::Tuple{Bool, CAMPSState}
    ensure_initialized!(state)

    # T gate is Rz(π/4), so we compute twisted Pauli for Z on the target qubit
    P_twisted = compute_twisted_pauli(state, :Z, qubit)

    # Try OFD with T gate angle
    return try_apply_ofd!(state, P_twisted, π/4)
end

"""
    apply_tdag_gate_ofd!(state::CAMPSState, qubit::Int) -> Tuple{Bool, CAMPSState}

Apply a T† gate using OFD if possible.

Same as apply_t_gate_ofd! but with angle -π/4.

# Arguments
- `state::CAMPSState`: CAMPS state
- `qubit::Int`: Target qubit for the T† gate

# Returns
- `Tuple{Bool, CAMPSState}`: (success, modified state)
"""
function apply_tdag_gate_ofd!(state::CAMPSState, qubit::Int)::Tuple{Bool, CAMPSState}
    ensure_initialized!(state)

    P_twisted = compute_twisted_pauli(state, :Z, qubit)
    return try_apply_ofd!(state, P_twisted, -π/4)
end

#==============================================================================#
# BATCH OFD (OFDS ALGORITHM)
#==============================================================================#

"""
    OFDSResult

Result of applying OFDS (OFD Sequential) algorithm to multiple T-gates.

# Fields
- `num_applied::Int`: Number of T-gates successfully applied via OFD
- `num_failed::Int`: Number of T-gates that couldn't use OFD
- `final_free_qubits::Int`: Number of remaining free qubits
- `final_magic_qubits::Int`: Number of magic qubits
- `predicted_chi::Int`: Predicted bond dimension from GF(2) theory
- `actual_chi::Int`: Actual maximum bond dimension
"""
struct OFDSResult
    num_applied::Int
    num_failed::Int
    final_free_qubits::Int
    final_magic_qubits::Int
    predicted_chi::Int
    actual_chi::Int
end

"""
    apply_ofds!(state::CAMPSState, t_gates::Vector{Int}) -> OFDSResult

Apply OFDS (OFD Sequential) algorithm to a sequence of T-gates.

This implements Algorithm 2 from Liu & Clark (arXiv:2412.17209):
For each T-gate, compute the twisted Pauli and apply OFD if possible.

# Arguments
- `state::CAMPSState`: CAMPS state (modified in-place)
- `t_gates::Vector{Int}`: Target qubits for T-gates (in order of application)

# Returns
- `OFDSResult`: Statistics about the OFD application

# Algorithm
```
for qubit in t_gates:
    P̃ = compute_twisted_pauli(state, :Z, qubit)
    control = find_disentangling_qubit(P̃, state.free_qubits)
    if control is not nothing:
        apply_ofd!(state, P̃, π/4, control)
    else:
        # OFD failed - would need OBD fallback
        record failure
```

# Note
This function only uses OFD. For gates where OFD fails, the caller should
use a fallback strategy (OBD or direct application).

# Example
```julia
state = CAMPSState(10)
initialize!(state)

# Apply H layer (makes Z→X on all qubits)
for q in 1:10
    apply_clifford_gate!(state.clifford, sHadamard(q))
end

# Apply T gates on first 5 qubits
result = apply_ofds!(state, [1, 2, 3, 4, 5])
# All should succeed because each qubit has X after H
```
"""
function apply_ofds!(state::CAMPSState, t_gates::Vector{Int})::OFDSResult
    ensure_initialized!(state)

    num_applied = 0
    num_failed = 0

    for qubit in t_gates
        success, _ = apply_t_gate_ofd!(state, qubit)
        if success
            num_applied += 1
        else
            num_failed += 1
            # Note: In a complete implementation, we would fall back to OBD here
            # For now, we just record the failure
        end
    end

    return OFDSResult(
        num_applied,
        num_failed,
        num_free_qubits(state),
        num_magic_qubits(state),
        get_predicted_bond_dimension(state),
        get_bond_dimension(state)
    )
end

#==============================================================================#
# TWISTED PAULI UPDATE AFTER OFD
#==============================================================================#

"""
    update_twisted_pauli_after_ofd(P::PauliOperator, D_gates::Vector,
                                    control::Int) -> PauliOperator

Update a twisted Pauli after OFD disentangler has been applied.

When we apply OFD with disentangler D, the accumulated Clifford becomes C' = C·D†.
Subsequent twisted Paulis must account for this change:

    P̃'_new = (C')† · P · C' = D · C† · P · C · D† = D · P̃ · D†

# Arguments
- `P::PauliOperator`: Original twisted Pauli
- `D_gates::Vector`: Gates that form the disentangler D
- `control::Int`: Control qubit used in OFD

# Returns
- `PauliOperator`: Updated twisted Pauli

# Note
This function is used when we need to update the twisted Paulis of future
gates after applying OFD. In many cases, we recompute from scratch instead.
"""
function update_twisted_pauli_after_ofd(P::PauliOperator, D_gates::Vector,
                                         control::Int)::PauliOperator
    # Create a temporary Destabilizer to compute D · P · D†
    n = nqubits(P)
    temp_dest = initialize_clifford(n)

    # Apply D gates
    for gate in D_gates
        if gate isa Vector
            for g in gate
                apply!(temp_dest, g)
            end
        else
            apply!(temp_dest, gate)
        end
    end

    # Conjugate P through D: D · P · D† = D† P D (since we want conjugation)
    # Actually, we want to compute how P transforms under D†CD = C'
    # This is P → D P D†

    return commute_pauli_through_clifford(P, temp_dest)
end

#==============================================================================#
# OFD ANALYSIS UTILITIES
#==============================================================================#

"""
    analyze_ofd_applicability(state::CAMPSState, qubit::Int) -> NamedTuple

Analyze whether OFD can be applied for a T-gate on the specified qubit.

# Arguments
- `state::CAMPSState`: Current CAMPS state
- `qubit::Int`: Target qubit for the T-gate

# Returns
- `NamedTuple` with fields:
  - `can_apply::Bool`: Whether OFD can be applied
  - `twisted_pauli::PauliOperator`: The computed twisted Pauli
  - `control_qubit::Union{Int, Nothing}`: Best control qubit (if any)
  - `pauli_weight::Int`: Weight of the twisted Pauli
  - `free_qubits_with_xy::Vector{Int}`: Free qubits that could serve as control

# Example
```julia
analysis = analyze_ofd_applicability(state, 1)
if analysis.can_apply
    println("OFD possible using qubit \$(analysis.control_qubit)")
else
    println("OFD not possible, twisted Pauli has weight \$(analysis.pauli_weight)")
end
```
"""
function analyze_ofd_applicability(state::CAMPSState, qubit::Int)
    ensure_initialized!(state)

    # Compute twisted Pauli
    P_twisted = compute_twisted_pauli(state, :Z, qubit)

    # Find all free qubits with X or Y
    n = state.n_qubits
    free_with_xy = Int[]
    xb = xbit(P_twisted)

    for j in 1:n
        if state.free_qubits[j] && xb[j]
            push!(free_with_xy, j)
        end
    end

    # Determine if OFD is possible
    can_apply = !isempty(free_with_xy)
    control = can_apply ? first(free_with_xy) : nothing

    return (
        can_apply = can_apply,
        twisted_pauli = P_twisted,
        control_qubit = control,
        pauli_weight = pauli_weight(P_twisted),
        free_qubits_with_xy = free_with_xy
    )
end

"""
    count_ofd_applicable(state::CAMPSState, t_gates::Vector{Int}) -> NamedTuple

Count how many T-gates can be handled by OFD in a given sequence.

This performs a "dry run" analysis without modifying the state.

# Arguments
- `state::CAMPSState`: Current CAMPS state (not modified)
- `t_gates::Vector{Int}`: Target qubits for T-gates

# Returns
- `NamedTuple` with fields:
  - `total::Int`: Total number of T-gates
  - `ofd_possible::Int`: Number that can use OFD
  - `ofd_impossible::Int`: Number that cannot use OFD
  - `details::Vector{NamedTuple}`: Per-gate analysis

# Note
This creates a copy of the state for analysis, so the original is not modified.
"""
function count_ofd_applicable(state::CAMPSState, t_gates::Vector{Int})
    # Create a lightweight copy for simulation
    # We only need to track Clifford and free_qubits
    temp_clifford = deepcopy(state.clifford)
    temp_free = copy(state.free_qubits)

    ofd_possible = 0
    details = []

    for (i, qubit) in enumerate(t_gates)
        # Compute twisted Pauli with current Clifford
        P = axis_to_pauli(:Z, qubit, state.n_qubits)
        P_twisted = commute_pauli_through_clifford(P, temp_clifford)

        # Check if disentanglable
        control = find_disentangling_qubit(P_twisted, temp_free)
        can_apply = control !== nothing

        push!(details, (
            gate_index = i,
            qubit = qubit,
            can_ofd = can_apply,
            control = control,
            pauli_string = pauli_to_string(P_twisted)
        ))

        if can_apply
            ofd_possible += 1
            # Simulate OFD: update Clifford with D† and mark qubit as magic
            D_gates = build_disentangler_gates(P_twisted, control)
            D_flat = flatten_gate_sequence(D_gates)
            apply_inverse_gates!(temp_clifford, D_flat)
            temp_free[control] = false
        end
    end

    return (
        total = length(t_gates),
        ofd_possible = ofd_possible,
        ofd_impossible = length(t_gates) - ofd_possible,
        details = details
    )
end

#==============================================================================#
# OFD GATE SEQUENCE GENERATION
#==============================================================================#

"""
    generate_ofd_circuit(P_twisted::PauliOperator, control::Int,
                          θ::Real) -> Vector{Gate}

Generate the complete gate sequence for OFD application.

Returns the gates that should be applied to achieve OFD:
1. Disentangler D (Clifford gates)
2. Single-qubit rotation on control qubit

# Arguments
- `P_twisted::PauliOperator`: Twisted Pauli operator
- `control::Int`: Control qubit
- `θ::Real`: Rotation angle

# Returns
- `Vector{Gate}`: Sequence of CAMPS Gate objects

# Note
The returned gates can be applied to the original (untwisted) circuit.
"""
function generate_ofd_circuit(P_twisted::PauliOperator, control::Int,
                               θ::Real)::Vector{Gate}
    gates = Gate[]

    σ_control = get_pauli_at(P_twisted, control)
    n = nqubits(P_twisted)

    # Build disentangler gates as CliffordGate objects
    for j in 1:n
        if j == control
            continue
        end

        σ_j = get_pauli_at(P_twisted, j)
        if σ_j == :I
            continue
        end

        # Create the appropriate controlled gate
        if σ_j == :X
            push!(gates, CNOTGate(control, j))
        elseif σ_j == :Z
            push!(gates, CZGate(control, j))
        elseif σ_j == :Y
            # CY = S†_j · CNOT · S_j
            push!(gates, SdagGate(j))
            push!(gates, CNOTGate(control, j))
            push!(gates, SGate(j))
        end
    end

    # Add the rotation on the control qubit
    if σ_control == :X
        push!(gates, RxGate(control, Float64(θ)))
    elseif σ_control == :Y
        push!(gates, RyGate(control, Float64(θ)))
    end

    return gates
end

#==============================================================================#
# MAGIC STATE PROPERTIES
#==============================================================================#

"""
    t_gate_magic_state() -> Tuple{ComplexF64, ComplexF64}

Return the amplitudes (α, β) for the T-gate magic state.

The T-gate magic state is: |T⟩ = α|0⟩ + β|1⟩
where α = cos(π/8) and β = -i·sin(π/8).

# Returns
- `Tuple{ComplexF64, ComplexF64}`: (α, β) amplitudes

# Properties
- |α|² + |β|² = 1 (normalized)
- |α|² ≈ 0.854, |β|² ≈ 0.146
- The state has magic (non-stabilizer) correlations
"""
function t_gate_magic_state()::Tuple{ComplexF64, ComplexF64}
    α = cos(π/8)
    β = -im * sin(π/8)
    return (ComplexF64(α), ComplexF64(β))
end

"""
    magic_state_vector(σ::Symbol, θ::Real) -> Vector{ComplexF64}

Compute the magic state vector for a rotation R_σ(θ) on |0⟩.

|m⟩ = R_σ(θ)|0⟩ = (cos(θ/2)I - i·sin(θ/2)σ)|0⟩

# Arguments
- `σ::Symbol`: Rotation axis (:X, :Y, or :Z)
- `θ::Real`: Rotation angle

# Returns
- `Vector{ComplexF64}`: 2-element state vector [⟨0|m⟩, ⟨1|m⟩]

# Examples
For T gate (σ=:Z, θ=π/4):
    |m⟩ = e^{-iπ/8}|0⟩

For X rotation from |0⟩:
    |m⟩ = cos(θ/2)|0⟩ - i·sin(θ/2)|1⟩
"""
function magic_state_vector(σ::Symbol, θ::Real)::Vector{ComplexF64}
    c = cos(θ/2)
    s = sin(θ/2)

    if σ == :X
        # Rx(θ)|0⟩ = cos(θ/2)|0⟩ - i·sin(θ/2)|1⟩
        return ComplexF64[c, -im*s]
    elseif σ == :Y
        # Ry(θ)|0⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩
        return ComplexF64[c, s]
    elseif σ == :Z
        # Rz(θ)|0⟩ = e^{-iθ/2}|0⟩
        return ComplexF64[exp(-im*θ/2), 0]
    else
        throw(ArgumentError("Invalid axis: $σ"))
    end
end
