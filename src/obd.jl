# CAMPS.jl/src/obd.jl
# Optimization-Based Disentangling (OBD) algorithm implementation
#
# Based on: Liu & Clark, "Clifford-Augmented Matrix Product States" (arXiv:2412.17209)
#
# OBD is a search-based approach that finds the optimal two-qubit Clifford gate
# to minimize entanglement entropy at each bond. It's used as a fallback when
# OFD cannot be applied (no suitable free qubit available).

#==============================================================================#
# OVERVIEW
#
# The OBD algorithm works as follows:
#
# 1. For each bond in the MPS (from left to right, then right to left):
#    a. Extract the two-site reduced density matrix ρ_{i,i+1}
#    b. Search over all two-qubit Cliffords U ∈ Cl_2
#    c. Find U that minimizes S(Tr_j(U ρ U†)) - the entropy after tracing
#    d. Apply U to the MPS and update the accumulated Clifford
#
# 2. Repeat sweeps until convergence (entropy improvement < threshold)
#
# Key insight: Clifford gates preserve stabilizer structure, so the
# optimization is over a finite discrete group (11,520 elements for 2 qubits).
#
# Complexity: O(sweeps × n × 11520 × χ³) where χ is bond dimension.
#==============================================================================#

#==============================================================================#
# OBD SINGLE BOND OPTIMIZATION
#==============================================================================#

"""
    find_optimal_clifford_for_bond(mps::MPS, bond::Int, sites::AbstractVector;
                                    use_full_search::Bool=false,
                                    cache::Union{TwoQubitCliffordCache, Nothing}=nothing)
        -> Tuple{Int, Float64, Float64}

Find the optimal two-qubit Clifford to minimize entanglement at a bond.

# Arguments
- `mps::MPS`: Matrix Product State
- `bond::Int`: Bond index (between sites bond and bond+1)
- `sites::AbstractVector`: Site indices
- `use_full_search::Bool`: If true, search all 11,520 Cliffords; else use representatives
- `cache::Union{TwoQubitCliffordCache, Nothing}`: Precomputed Clifford cache

# Returns
- `Tuple{Int, Float64, Float64}`: (best_index, initial_entropy, final_entropy)

# Algorithm
1. Extract the two-site RDM ρ
2. For each Clifford U, compute entropy of Tr_2(U ρ U†)
3. Return the Clifford that gives minimum entropy

# Note
Uses second Rényi entropy for efficiency: S_2(ρ) = -log(Tr(ρ²))
"""
function find_optimal_clifford_for_bond(mps::MPS, bond::Int, sites::AbstractVector;
                                         use_full_search::Bool=false,
                                         cache::Union{TwoQubitCliffordCache, Nothing}=nothing)
    n = length(mps)
    (bond < 1 || bond >= n) && throw(ArgumentError("Invalid bond index"))

    site1 = bond
    site2 = bond + 1

    # Extract two-site reduced density matrix
    rho = extract_two_site_rdm(mps, site1, site2)

    # Compute initial entropy (for comparison)
    rho_1 = partial_trace_4x4(rho, true)  # Trace out second qubit
    initial_entropy = compute_renyi2_entropy(rho_1)

    # Get Cliffords to search over
    if use_full_search
        if cache !== nothing
            cliffords_matrices = cache.matrices
        else
            cliffords = get_all_two_qubit_cliffords()
            cliffords_matrices = [clifford_to_matrix(C) for C in cliffords]
        end
    else
        representatives = get_cnot_class_representatives()
        cliffords_matrices = [clifford_to_matrix(C) for C in representatives]
    end

    # Search for optimal Clifford
    best_index = 1
    best_entropy = initial_entropy

    for (i, U) in enumerate(cliffords_matrices)
        # Transform ρ under U
        rho_transformed = transform_rdm(rho, U)

        # Trace out second qubit to get ρ_1
        rho_1_transformed = partial_trace_4x4(rho_transformed, true)

        # Compute entropy
        entropy = compute_renyi2_entropy(rho_1_transformed)

        if entropy < best_entropy
            best_entropy = entropy
            best_index = i
        end
    end

    return (best_index, initial_entropy, best_entropy)
end

"""
    apply_clifford_to_mps!(mps::MPS, C::CliffordOperator, site1::Int, site2::Int,
                            sites::AbstractVector;
                            max_bond::Int=1024, cutoff::Float64=1e-15) -> MPS

Apply a two-qubit Clifford operator to an MPS at adjacent sites.

# Arguments
- `mps::MPS`: Matrix Product State (modified in-place)
- `C::CliffordOperator`: Two-qubit Clifford to apply
- `site1::Int`: First site (must be site2 - 1)
- `site2::Int`: Second site (must be site1 + 1)
- `sites::AbstractVector`: Site indices
- `max_bond::Int`: Maximum bond dimension
- `cutoff::Float64`: SVD cutoff

# Returns
- `MPS`: Modified MPS
"""
function apply_clifford_to_mps!(mps::MPS, C::CliffordOperator, site1::Int, site2::Int,
                                 sites::AbstractVector;
                                 max_bond::Int=1024, cutoff::Float64=1e-15)::MPS
    U = clifford_to_matrix(C)
    gate = matrix_to_two_qubit_itensor(U, sites[site1], sites[site2])
    return apply_two_qubit_gate!(mps, gate, site1, site2; max_bond=max_bond, cutoff=cutoff)
end

"""
    apply_clifford_index_to_mps!(mps::MPS, clifford_index::Int, site1::Int, site2::Int,
                                  sites::AbstractVector;
                                  cache::Union{TwoQubitCliffordCache, Nothing}=nothing,
                                  use_full_search::Bool=false,
                                  max_bond::Int=1024, cutoff::Float64=1e-15) -> MPS

Apply the i-th Clifford to an MPS at adjacent sites.

# Arguments
- `mps::MPS`: Matrix Product State (modified)
- `clifford_index::Int`: Index of the Clifford
- `site1::Int`: First site
- `site2::Int`: Second site
- `sites::AbstractVector`: Site indices
- `cache::Union{TwoQubitCliffordCache, Nothing}`: Precomputed cache
- `use_full_search::Bool`: Whether full search was used (determines index interpretation)
- `max_bond::Int`: Maximum bond dimension
- `cutoff::Float64`: SVD cutoff

# Returns
- `MPS`: Modified MPS
"""
function apply_clifford_index_to_mps!(mps::MPS, clifford_index::Int, site1::Int, site2::Int,
                                       sites::AbstractVector;
                                       cache::Union{TwoQubitCliffordCache, Nothing}=nothing,
                                       use_full_search::Bool=false,
                                       max_bond::Int=1024, cutoff::Float64=1e-15)::MPS
    # Get the Clifford matrix
    if use_full_search
        if cache !== nothing
            U = cache.matrices[clifford_index]
        else
            cliffords = get_all_two_qubit_cliffords()
            U = clifford_to_matrix(cliffords[clifford_index])
        end
    else
        representatives = get_cnot_class_representatives()
        U = clifford_to_matrix(representatives[clifford_index])
    end

    gate = matrix_to_two_qubit_itensor(U, sites[site1], sites[site2])
    return apply_two_qubit_gate!(mps, gate, site1, site2; max_bond=max_bond, cutoff=cutoff)
end

#==============================================================================#
# OBD SWEEP
#==============================================================================#

"""
    OBDSweepResult

Result of a single OBD sweep.

# Fields
- `initial_max_entropy::Float64`: Maximum entropy before sweep
- `final_max_entropy::Float64`: Maximum entropy after sweep
- `entropy_reduction::Float64`: Reduction in max entropy
- `applied_cliffords::Vector{Tuple{Int, Int, Int}}`: (bond, clifford_index, site1) for each applied Clifford
- `bond_entropies::Vector{Float64}`: Entropy at each bond after sweep
"""
struct OBDSweepResult
    initial_max_entropy::Float64
    final_max_entropy::Float64
    entropy_reduction::Float64
    applied_cliffords::Vector{Tuple{Int, Int, Int}}
    bond_entropies::Vector{Float64}
end

"""
    obd_sweep!(mps::MPS, sites::AbstractVector, clifford::Destabilizer;
               use_full_search::Bool=false,
               cache::Union{TwoQubitCliffordCache, Nothing}=nothing,
               max_bond::Int=1024, cutoff::Float64=1e-15,
               direction::Symbol=:left_to_right) -> Tuple{OBDSweepResult, Destabilizer}

Perform a single OBD sweep over all bonds.

# Arguments
- `mps::MPS`: Matrix Product State (modified)
- `sites::AbstractVector`: Site indices
- `clifford::Destabilizer`: Accumulated Clifford operator (modified)
- `use_full_search::Bool`: Search all 11,520 vs representatives only
- `cache::Union{TwoQubitCliffordCache, Nothing}`: Precomputed Clifford cache
- `max_bond::Int`: Maximum bond dimension
- `cutoff::Float64`: SVD cutoff
- `direction::Symbol`: :left_to_right or :right_to_left

# Returns
- `Tuple{OBDSweepResult, Destabilizer}`: (sweep result, updated Clifford)

# Algorithm
Sweep through bonds, optimizing each one:
1. Find optimal Clifford U for each bond
2. Apply U to MPS: |ψ⟩ → U|ψ⟩
3. Update Clifford: C → C·U† (since we want U†·C·|ψ⟩ = C'·|ψ'⟩)
"""
function obd_sweep!(mps::MPS, sites::AbstractVector, clifford::Destabilizer;
                    use_full_search::Bool=false,
                    cache::Union{TwoQubitCliffordCache, Nothing}=nothing,
                    max_bond::Int=1024, cutoff::Float64=1e-15,
                    direction::Symbol=:left_to_right)::Tuple{OBDSweepResult, Destabilizer}

    n = length(mps)
    n >= 2 || throw(ArgumentError("MPS must have at least 2 sites"))

    # Compute initial entropies
    initial_entropies = Float64[entanglement_entropy(mps, bond) for bond in 1:(n-1)]
    initial_max = maximum(initial_entropies)

    # Track applied Cliffords
    applied = Tuple{Int, Int, Int}[]

    # Get Cliffords for Clifford update
    if use_full_search
        all_cliffords = cache !== nothing ? cache.cliffords : get_all_two_qubit_cliffords()
    else
        all_cliffords = get_cnot_class_representatives()
    end

    # Determine bond order
    bond_order = if direction == :left_to_right
        1:(n-1)
    else
        (n-1):-1:1
    end

    for bond in bond_order
        site1 = bond
        site2 = bond + 1

        # Find optimal Clifford
        best_idx, _, _ = find_optimal_clifford_for_bond(mps, bond, sites;
                                                         use_full_search=use_full_search,
                                                         cache=cache)

        # Skip identity (index 1 for representatives, may vary for full search)
        if best_idx == 1
            continue  # No improvement from identity
        end

        push!(applied, (bond, best_idx, site1))

        # Apply Clifford to MPS
        apply_clifford_index_to_mps!(mps, best_idx, site1, site2, sites;
                                      cache=cache, use_full_search=use_full_search,
                                      max_bond=max_bond, cutoff=cutoff)

        # Update accumulated Clifford: C → C·U†
        # The Clifford operator U from the search needs to be inverted
        U_clifford = all_cliffords[best_idx]

        # Apply U† to the accumulated Clifford
        # Since we applied U to |ψ⟩, we need C → C·U† to maintain C'|ψ'⟩ = C·U†·U|ψ⟩ = C|ψ⟩
        # This is done by applying inv(U) to the Clifford
        apply_clifford_to_destabilizer!(clifford, U_clifford, site1, site2; inverse=true)
    end

    # Compute final entropies
    final_entropies = Float64[entanglement_entropy(mps, bond) for bond in 1:(n-1)]
    final_max = maximum(final_entropies)

    result = OBDSweepResult(
        initial_max,
        final_max,
        initial_max - final_max,
        applied,
        final_entropies
    )

    return (result, clifford)
end

"""
    apply_clifford_to_destabilizer!(D::Destabilizer, C::CliffordOperator,
                                     site1::Int, site2::Int; inverse::Bool=false)

Apply a two-qubit Clifford to the accumulated Destabilizer.

# Arguments
- `D::Destabilizer`: Accumulated Clifford (modified)
- `C::CliffordOperator`: Two-qubit Clifford to apply
- `site1::Int`: First qubit (in the full n-qubit system)
- `site2::Int`: Second qubit
- `inverse::Bool`: If true, apply C†; if false, apply C

# Note
This embeds the 2-qubit Clifford into the n-qubit system by only acting
on the specified qubits.
"""
function apply_clifford_to_destabilizer!(D::Destabilizer, C::CliffordOperator,
                                          site1::Int, site2::Int; inverse::Bool=false)
    # Get the gates that produce C from the Destabilizer representation
    # For a two-qubit Clifford, we can decompose it into single-qubit and CNOT gates

    # Since C is a CliffordOperator, we can apply it via its stabilizer tableau
    # But we need to embed it into the n-qubit space

    # The approach: build a sequence of gates that realizes C on qubits (site1, site2)
    # This requires decomposing C into a gate sequence

    # For now, use a decomposition approach via canonical form
    gates = decompose_two_qubit_clifford(C, site1, site2)

    if inverse
        # Apply gates in reverse order with inverses
        for gate in reverse(gates)
            apply!(D, inv(gate))
        end
    else
        for gate in gates
            apply!(D, gate)
        end
    end

    return D
end

"""
    decompose_two_qubit_clifford(C::CliffordOperator, q1::Int, q2::Int) -> Vector

Decompose a two-qubit Clifford into a sequence of elementary gates on qubits q1, q2.

# Arguments
- `C::CliffordOperator`: Two-qubit Clifford
- `q1::Int`: First qubit index in larger system
- `q2::Int`: Second qubit index in larger system

# Returns
- `Vector`: Sequence of QuantumClifford symbolic gates

# Algorithm
Uses the canonical decomposition:
    C = (L₁ ⊗ L₂) · E · (R₁ ⊗ R₂)
where L₁, L₂, R₁, R₂ are single-qubit Cliffords and E is an entangling gate.

We use stabilizer tableau comparison for exact matching.
"""
function decompose_two_qubit_clifford(C::CliffordOperator, q1::Int, q2::Int)::Vector
    # Use cached decomposition if available
    if haskey(TWO_QUBIT_CLIFFORD_DECOMPOSITION_CACHE, C)
        return remap_decomposition(TWO_QUBIT_CLIFFORD_DECOMPOSITION_CACHE[C], q1, q2)
    end

    # Try to find decomposition via systematic search
    result = find_clifford_decomposition(C, q1, q2)

    if result !== nothing
        return result
    end

    # Last resort: use matrix-based synthesis
    return synthesize_clifford_from_matrix(C, q1, q2)
end

"""
    find_clifford_decomposition(C::CliffordOperator, q1::Int, q2::Int) -> Union{Vector, Nothing}

Find a gate decomposition for a two-qubit Clifford by comparing stabilizer tableaux.
"""
function find_clifford_decomposition(C::CliffordOperator, q1::Int, q2::Int)::Union{Vector, Nothing}
    # The entangling gate classes that span all two-qubit Cliffords
    # Based on the canonical form: any 2-qubit Clifford needs at most 3 CNOTs
    entangling_classes = [
        [],                                    # Identity class
        [sCNOT(1, 2)],                        # Single CNOT
        [sCNOT(2, 1)],
        [sCPHASE(1, 2)],                      # CZ
        [sSWAP(1, 2)],                        # SWAP
        [sCNOT(1, 2), sCNOT(2, 1)],           # Two CNOTs
        [sCNOT(2, 1), sCNOT(1, 2)],
        [sCNOT(1, 2), sCNOT(2, 1), sCNOT(1, 2)],  # Three CNOTs (covers iSWAP class)
    ]

    # Search over canonical decomposition: (L1 ⊗ L2) · E · (R1 ⊗ R2)
    for entangling in entangling_classes
        result = search_with_entangling_class(C, entangling, q1, q2)
        if result !== nothing
            return result
        end
    end

    return nothing
end

"""
    search_with_entangling_class(C::CliffordOperator, entangling::Vector, q1::Int, q2::Int)

Search for single-qubit Cliffords that complete the decomposition.
"""
function search_with_entangling_class(C::CliffordOperator, entangling::Vector, q1::Int, q2::Int)::Union{Vector, Nothing}
    # Build the entangling part on 2 qubits
    E_dest = one(Destabilizer, 2)
    for g in entangling
        apply!(E_dest, g)
    end
    E = CliffordOperator(E_dest)

    # We need to find L1, L2, R1, R2 such that:
    # C = (L1 ⊗ L2) · E · (R1 ⊗ R2)
    #
    # Rearranging: (L1† ⊗ L2†) · C = E · (R1 ⊗ R2)
    #
    # Strategy: For each combination of R1, R2, compute E · (R1 ⊗ R2)
    # then find L1, L2 that complete it

    for r1_idx in 1:24
        for r2_idx in 1:24
            # Build R = R1 ⊗ R2 on 2 qubits
            R_dest = one(Destabilizer, 2)
            for g in resolve_single_qubit_clifford_local(r1_idx, 1)
                apply!(R_dest, g)
            end
            for g in resolve_single_qubit_clifford_local(r2_idx, 2)
                apply!(R_dest, g)
            end
            R = CliffordOperator(R_dest)

            # Compute E · R
            ER_dest = one(Destabilizer, 2)
            apply!(ER_dest, R)
            apply!(ER_dest, E)
            ER = CliffordOperator(ER_dest)

            # Now find L1, L2 such that L · ER = C
            # Equivalently: L = C · ER†
            ER_inv = inv(ER)

            # Compute C · ER†
            target_L_dest = one(Destabilizer, 2)
            apply!(target_L_dest, ER_inv)
            apply!(target_L_dest, C)
            target_L = CliffordOperator(target_L_dest)

            # Check if target_L is a tensor product of single-qubit Cliffords
            l1_idx, l2_idx = decompose_as_tensor_product(target_L)

            if l1_idx !== nothing && l2_idx !== nothing
                # Found a candidate decomposition - verify it before returning
                gates = Vector{Any}()

                # Apply R1, R2
                append!(gates, resolve_single_qubit_clifford_local(r1_idx, 1))
                append!(gates, resolve_single_qubit_clifford_local(r2_idx, 2))

                # Apply entangling gates
                append!(gates, entangling)

                # Apply L1, L2
                append!(gates, resolve_single_qubit_clifford_local(l1_idx, 1))
                append!(gates, resolve_single_qubit_clifford_local(l2_idx, 2))

                # Verify the decomposition is correct before returning
                D_verify = one(Destabilizer, 2)
                for g in gates
                    apply!(D_verify, g)
                end
                C_verify = CliffordOperator(D_verify)

                if cliffords_equal(C, C_verify)
                    # Verified correct - remap to actual qubit indices
                    return remap_gates(gates, q1, q2)
                end
                # Otherwise continue searching
            end
        end
    end

    return nothing
end

"""
    decompose_as_tensor_product(C::CliffordOperator) -> Tuple{Union{Int, Nothing}, Union{Int, Nothing}}

Check if a 2-qubit Clifford is a tensor product of single-qubit Cliffords.
Returns (l1_idx, l2_idx) if successful, (nothing, nothing) otherwise.
"""
function decompose_as_tensor_product(C::CliffordOperator)::Tuple{Union{Int, Nothing}, Union{Int, Nothing}}
    # A tensor product L1 ⊗ L2 has the property that:
    # - The action on qubit 1 is independent of qubit 2
    # - The action on qubit 2 is independent of qubit 1

    # Check by comparing with all 24×24 tensor products
    for l1_idx in 1:24
        for l2_idx in 1:24
            # Build L1 ⊗ L2
            L_dest = one(Destabilizer, 2)
            for g in resolve_single_qubit_clifford_local(l1_idx, 1)
                apply!(L_dest, g)
            end
            for g in resolve_single_qubit_clifford_local(l2_idx, 2)
                apply!(L_dest, g)
            end
            L = CliffordOperator(L_dest)

            # Compare stabilizer tableaux
            if cliffords_equal(C, L)
                return (l1_idx, l2_idx)
            end
        end
    end

    return (nothing, nothing)
end

"""
    cliffords_equal(C1::CliffordOperator, C2::CliffordOperator) -> Bool

Check if two Clifford operators are equal by comparing their matrix representations.
Two Cliffords are equal if their matrices are equal up to global phase.
"""
function cliffords_equal(C1::CliffordOperator, C2::CliffordOperator)::Bool
    n = nqubits(C1)
    nqubits(C2) == n || return false

    # Compare matrix representations (up to global phase)
    U1 = clifford_to_matrix(C1)
    U2 = clifford_to_matrix(C2)

    return is_equivalent_up_to_phase(U1, U2)
end

"""
    make_single_x(n::Int, q::Int) -> PauliOperator

Create X on qubit q in an n-qubit system.
"""
function make_single_x(n::Int, q::Int)::PauliOperator
    return make_single_pauli(n, q, :X)
end

"""
    make_single_z(n::Int, q::Int) -> PauliOperator

Create Z on qubit q in an n-qubit system.
"""
function make_single_z(n::Int, q::Int)::PauliOperator
    return make_single_pauli(n, q, :Z)
end

"""
    make_single_pauli(n::Int, q::Int, p::Symbol) -> PauliOperator

Create a single-qubit Pauli operator in an n-qubit system.
"""
function make_single_pauli(n::Int, q::Int, p::Symbol)::PauliOperator
    xs = falses(n)
    zs = falses(n)

    if p == :X
        xs[q] = true
    elseif p == :Y
        xs[q] = true
        zs[q] = true
    elseif p == :Z
        zs[q] = true
    end

    return PauliOperator(0x00, xs, zs)
end

"""
    resolve_single_qubit_clifford_local(index::Int, qubit::Int) -> Vector

Resolve single-qubit Clifford to gates on local qubit index (1 or 2).
"""
function resolve_single_qubit_clifford_local(index::Int, qubit::Int)::Vector
    specs = generate_single_qubit_clifford(index, qubit)
    return [resolve_symbolic_gate(spec) for spec in specs]
end

"""
    remap_gates(gates::Vector, q1::Int, q2::Int) -> Vector

Remap gates from local indices (1, 2) to actual qubit indices (q1, q2).
"""
function remap_gates(gates::Vector, q1::Int, q2::Int)::Vector
    result = []

    for g in gates
        push!(result, remap_gate(g, q1, q2))
    end

    return result
end

"""
    remap_gate(g, q1::Int, q2::Int)

Remap a single gate from local indices to actual indices.
"""
function remap_gate(g, q1::Int, q2::Int)
    # Handle different gate types
    if g isa typeof(sHadamard(1))
        target = g.q
        new_target = target == 1 ? q1 : q2
        return sHadamard(new_target)
    elseif g isa typeof(sPhase(1))
        target = g.q
        new_target = target == 1 ? q1 : q2
        return sPhase(new_target)
    elseif g isa typeof(sCNOT(1, 2))
        ctrl = g.q1
        targ = g.q2
        new_ctrl = ctrl == 1 ? q1 : q2
        new_targ = targ == 1 ? q1 : q2
        return sCNOT(new_ctrl, new_targ)
    elseif g isa typeof(sCPHASE(1, 2))
        q1_local = g.q1
        q2_local = g.q2
        new_q1 = q1_local == 1 ? q1 : q2
        new_q2 = q2_local == 1 ? q1 : q2
        return sCPHASE(new_q1, new_q2)
    elseif g isa typeof(sSWAP(1, 2))
        return sSWAP(q1, q2)
    elseif g isa typeof(sX(1))
        target = g.q
        new_target = target == 1 ? q1 : q2
        return sX(new_target)
    elseif g isa typeof(sY(1))
        target = g.q
        new_target = target == 1 ? q1 : q2
        return sY(new_target)
    elseif g isa typeof(sZ(1))
        target = g.q
        new_target = target == 1 ? q1 : q2
        return sZ(new_target)
    elseif g isa typeof(sInvPhase(1))
        target = g.q
        new_target = target == 1 ? q1 : q2
        return sInvPhase(new_target)
    else
        # For other gate types, try to access qubit field directly
        @warn "Unknown gate type in remap: $(typeof(g)), returning as-is"
        return g
    end
end

"""
    remap_decomposition(decomp::Vector, q1::Int, q2::Int) -> Vector

Remap a cached decomposition to the actual qubit indices.
"""
function remap_decomposition(decomp::Vector, q1::Int, q2::Int)::Vector
    return remap_gates(decomp, q1, q2)
end

"""
    synthesize_clifford_from_matrix(C::CliffordOperator, q1::Int, q2::Int) -> Vector

Synthesize a Clifford decomposition using matrix-based approach as fallback.
"""
function synthesize_clifford_from_matrix(C::CliffordOperator, q1::Int, q2::Int)::Vector
    # Get the matrix representation
    U = clifford_to_matrix(C)

    # Common matrices for quick lookup
    I4 = Matrix{ComplexF64}(LinearAlgebra.I, 4, 4)

    # Check for identity (allowing global phase)
    if is_equivalent_up_to_phase(U, I4)
        return []
    end

    # CNOT(1,2)
    CNOT_12 = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0]
    if is_equivalent_up_to_phase(U, CNOT_12)
        return [sCNOT(q1, q2)]
    end

    # CNOT(2,1)
    CNOT_21 = ComplexF64[1 0 0 0; 0 0 0 1; 0 0 1 0; 0 1 0 0]
    if is_equivalent_up_to_phase(U, CNOT_21)
        return [sCNOT(q2, q1)]
    end

    # CZ
    CZ_mat = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 -1]
    if is_equivalent_up_to_phase(U, CZ_mat)
        return [sCPHASE(q1, q2)]
    end

    # SWAP
    SWAP_mat = ComplexF64[1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1]
    if is_equivalent_up_to_phase(U, SWAP_mat)
        return [sSWAP(q1, q2)]
    end

    # If we get here, use full brute force search (expensive but guaranteed)
    return brute_force_decomposition(C, q1, q2)
end

"""
    is_equivalent_up_to_phase(U1::Matrix, U2::Matrix) -> Bool

Check if two unitary matrices are equal up to a global phase.
"""
function is_equivalent_up_to_phase(U1::Matrix, U2::Matrix)::Bool
    # Find first non-zero element in U1
    phase_idx = findfirst(x -> abs(x) > 1e-10, U1)
    if phase_idx === nothing
        return all(abs.(U2) .< 1e-10)
    end

    # Get the phase difference
    if abs(U2[phase_idx]) < 1e-10
        return false
    end

    phase = U1[phase_idx] / U2[phase_idx]

    # Check if U1 ≈ phase * U2
    return isapprox(U1, phase * U2, atol=1e-10)
end

"""
    brute_force_decomposition(C::CliffordOperator, q1::Int, q2::Int) -> Vector

Brute-force search over all canonical decompositions.
Guaranteed to find a decomposition for any valid 2-qubit Clifford.
"""
function brute_force_decomposition(C::CliffordOperator, q1::Int, q2::Int)::Vector
    # Entangling gate classes
    entangling_classes = [
        [],
        [sCNOT(1, 2)],
        [sCNOT(2, 1)],
        [sCPHASE(1, 2)],
        [sSWAP(1, 2)],
        [sCNOT(1, 2), sCNOT(2, 1)],
        [sCNOT(2, 1), sCNOT(1, 2)],
        [sCNOT(1, 2), sCNOT(2, 1), sCNOT(1, 2)],
        [sCNOT(2, 1), sCNOT(1, 2), sCNOT(2, 1)],
    ]

    for entangling in entangling_classes
        for l1_idx in 1:24
            for l2_idx in 1:24
                for r1_idx in 1:24
                    for r2_idx in 1:24
                        # Build candidate Clifford
                        cand_dest = one(Destabilizer, 2)

                        # Apply R1 ⊗ R2
                        for g in resolve_single_qubit_clifford_local(r1_idx, 1)
                            apply!(cand_dest, g)
                        end
                        for g in resolve_single_qubit_clifford_local(r2_idx, 2)
                            apply!(cand_dest, g)
                        end

                        # Apply entangling gates
                        for g in entangling
                            apply!(cand_dest, g)
                        end

                        # Apply L1 ⊗ L2
                        for g in resolve_single_qubit_clifford_local(l1_idx, 1)
                            apply!(cand_dest, g)
                        end
                        for g in resolve_single_qubit_clifford_local(l2_idx, 2)
                            apply!(cand_dest, g)
                        end

                        cand = CliffordOperator(cand_dest)

                        if cliffords_equal(C, cand)
                            # Found it! Build the gate sequence
                            gates = Vector{Any}()
                            append!(gates, resolve_single_qubit_clifford_local(r1_idx, 1))
                            append!(gates, resolve_single_qubit_clifford_local(r2_idx, 2))
                            append!(gates, entangling)
                            append!(gates, resolve_single_qubit_clifford_local(l1_idx, 1))
                            append!(gates, resolve_single_qubit_clifford_local(l2_idx, 2))

                            return remap_gates(gates, q1, q2)
                        end
                    end
                end
            end
        end
    end

    # This should never happen for a valid 2-qubit Clifford
    @warn "Could not decompose Clifford - this may indicate a bug"
    return []
end

# Global cache for decompositions (built lazily)
const TWO_QUBIT_CLIFFORD_DECOMPOSITION_CACHE = Dict{CliffordOperator, Vector}()

#==============================================================================#
# FULL OBD ALGORITHM
#==============================================================================#

"""
    OBDResult

Result of full OBD optimization.

# Fields
- `num_sweeps::Int`: Number of sweeps performed
- `converged::Bool`: Whether optimization converged
- `initial_max_entropy::Float64`: Initial maximum bond entropy
- `final_max_entropy::Float64`: Final maximum bond entropy
- `sweep_results::Vector{OBDSweepResult}`: Results from each sweep
- `total_cliffords_applied::Int`: Total number of Cliffords applied
"""
struct OBDResult
    num_sweeps::Int
    converged::Bool
    initial_max_entropy::Float64
    final_max_entropy::Float64
    sweep_results::Vector{OBDSweepResult}
    total_cliffords_applied::Int
end

"""
    obd!(state::CAMPSState; max_sweeps::Int=10,
         improvement_threshold::Float64=1e-10,
         use_full_search::Bool=false,
         cache::Union{TwoQubitCliffordCache, Nothing}=nothing) -> OBDResult

Apply full OBD optimization to a CAMPS state.

This is the main OBD entry point for reducing entanglement in the MPS.

# Arguments
- `state::CAMPSState`: CAMPS state (modified in-place)
- `max_sweeps::Int`: Maximum number of bidirectional sweeps
- `improvement_threshold::Float64`: Stop if improvement < this
- `use_full_search::Bool`: Search all 11,520 Cliffords vs representatives
- `cache::Union{TwoQubitCliffordCache, Nothing}`: Precomputed cache

# Returns
- `OBDResult`: Optimization results

# Algorithm
1. Perform left-to-right sweep
2. Perform right-to-left sweep
3. Repeat until convergence or max_sweeps reached

# Example
```julia
state = CAMPSState(10)
initialize!(state)
# ... apply some gates that increase entanglement ...

result = obd!(state; max_sweeps=5)
println("Reduced max entropy from \$(result.initial_max_entropy) to \$(result.final_max_entropy)")
```
"""
function obd!(state::CAMPSState; max_sweeps::Int=10,
              improvement_threshold::Float64=1e-10,
              use_full_search::Bool=false,
              cache::Union{TwoQubitCliffordCache, Nothing}=nothing)::OBDResult

    ensure_initialized!(state)

    n = state.n_qubits
    if n < 2
        return OBDResult(0, true, 0.0, 0.0, OBDSweepResult[], 0)
    end

    sweep_results = OBDSweepResult[]
    initial_max = max_entanglement_entropy(state.mps)
    total_cliffords = 0
    converged = false

    for sweep_num in 1:max_sweeps
        # Left-to-right sweep
        result_lr, _ = obd_sweep!(state.mps, state.sites, state.clifford;
                                   use_full_search=use_full_search,
                                   cache=cache,
                                   max_bond=state.max_bond,
                                   cutoff=state.cutoff,
                                   direction=:left_to_right)
        push!(sweep_results, result_lr)
        total_cliffords += length(result_lr.applied_cliffords)

        # Right-to-left sweep
        result_rl, _ = obd_sweep!(state.mps, state.sites, state.clifford;
                                   use_full_search=use_full_search,
                                   cache=cache,
                                   max_bond=state.max_bond,
                                   cutoff=state.cutoff,
                                   direction=:right_to_left)
        push!(sweep_results, result_rl)
        total_cliffords += length(result_rl.applied_cliffords)

        # Check convergence
        improvement = result_lr.entropy_reduction + result_rl.entropy_reduction
        if improvement < improvement_threshold
            converged = true
            break
        end
    end

    final_max = max_entanglement_entropy(state.mps)

    return OBDResult(
        length(sweep_results) ÷ 2,  # Each iteration has 2 sweeps
        converged,
        initial_max,
        final_max,
        sweep_results,
        total_cliffords
    )
end

#==============================================================================#
# OBD FOR ROTATION GATES (FALLBACK FROM OFD)
#==============================================================================#

"""
    apply_rotation_with_obd!(state::CAMPSState, P_twisted::PauliOperator, θ::Real;
                              obd_sweeps::Int=2,
                              use_full_search::Bool=false) -> CAMPSState

Apply a non-Clifford rotation using OBD for entanglement reduction.

This is the fallback when OFD cannot be applied (no free qubit with X/Y).

# Arguments
- `state::CAMPSState`: CAMPS state (modified)
- `P_twisted::PauliOperator`: Twisted Pauli operator
- `θ::Real`: Rotation angle
- `obd_sweeps::Int`: Number of OBD sweeps after rotation
- `use_full_search::Bool`: Use full Clifford search

# Returns
- `CAMPSState`: Modified state

# Algorithm
1. Apply the twisted rotation directly to MPS (may increase bond dimension)
2. Run OBD to reduce entanglement
3. Record the twisted Pauli for GF(2) tracking
"""
function apply_rotation_with_obd!(state::CAMPSState, P_twisted::PauliOperator, θ::Real;
                                   obd_sweeps::Int=2,
                                   use_full_search::Bool=false)::CAMPSState
    ensure_initialized!(state)

    # Apply the twisted rotation directly to MPS
    apply_twisted_rotation!(state.mps, state.sites, P_twisted, Float64(θ);
                            max_bond=state.max_bond, cutoff=state.cutoff)

    # Record twisted Pauli
    add_twisted_pauli!(state, P_twisted)

    # Run OBD to reduce entanglement
    if obd_sweeps > 0
        obd!(state; max_sweeps=obd_sweeps, use_full_search=use_full_search)
    end

    return state
end

#==============================================================================#
# HYBRID OFD/OBD APPLICATION
#==============================================================================#

"""
    apply_rotation_hybrid!(state::CAMPSState, axis::Symbol, qubit::Int, θ::Real;
                            strategy::DisentanglingStrategy=HybridStrategy()) -> CAMPSState

Apply a non-Clifford rotation using the specified disentangling strategy.

This is the main entry point for applying non-Clifford gates with disentangling.

# Arguments
- `state::CAMPSState`: CAMPS state (modified)
- `axis::Symbol`: Rotation axis (:X, :Y, or :Z)
- `qubit::Int`: Target qubit
- `θ::Real`: Rotation angle
- `strategy::DisentanglingStrategy`: OFD, OBD, Hybrid, or NoDisentangling

# Returns
- `CAMPSState`: Modified state

# Example
```julia
state = CAMPSState(5)
initialize!(state)
apply_clifford_gate!(state.clifford, sHadamard(1))

# Apply T gate with hybrid strategy
apply_rotation_hybrid!(state, :Z, 1, π/4)
```
"""
function apply_rotation_hybrid!(state::CAMPSState, axis::Symbol, qubit::Int, θ::Real;
                                 strategy::DisentanglingStrategy=HybridStrategy())::CAMPSState
    ensure_initialized!(state)

    # Compute twisted Pauli
    P_twisted = compute_twisted_pauli(state, axis, qubit)

    # Apply based on strategy
    if strategy isa OFDStrategy
        success, _ = try_apply_ofd!(state, P_twisted, θ)
        if !success
            @warn "OFD failed for rotation on qubit $qubit, applying directly"
            apply_twisted_rotation!(state.mps, state.sites, P_twisted, Float64(θ);
                                    max_bond=state.max_bond, cutoff=state.cutoff)
            add_twisted_pauli!(state, P_twisted)
        end

    elseif strategy isa OBDStrategy
        apply_rotation_with_obd!(state, P_twisted, θ;
                                  obd_sweeps=strategy.max_sweeps,
                                  use_full_search=false)

    elseif strategy isa HybridStrategy
        success, _ = try_apply_ofd!(state, P_twisted, θ)
        if !success
            apply_rotation_with_obd!(state, P_twisted, θ;
                                      obd_sweeps=strategy.obd_sweeps_on_failure,
                                      use_full_search=false)
        end

    elseif strategy isa NoDisentangling
        apply_twisted_rotation!(state.mps, state.sites, P_twisted, Float64(θ);
                                max_bond=state.max_bond, cutoff=state.cutoff)
        add_twisted_pauli!(state, P_twisted)

    else
        throw(ArgumentError("Unknown strategy type: $(typeof(strategy))"))
    end

    return state
end

"""
    apply_t_gate_hybrid!(state::CAMPSState, qubit::Int;
                          strategy::DisentanglingStrategy=HybridStrategy()) -> CAMPSState

Apply a T gate using the specified disentangling strategy.

Convenience function for T gates (the most common non-Clifford gate).

# Arguments
- `state::CAMPSState`: CAMPS state
- `qubit::Int`: Target qubit
- `strategy::DisentanglingStrategy`: Disentangling strategy

# Returns
- `CAMPSState`: Modified state
"""
function apply_t_gate_hybrid!(state::CAMPSState, qubit::Int;
                               strategy::DisentanglingStrategy=HybridStrategy())::CAMPSState
    return apply_rotation_hybrid!(state, :Z, qubit, π/4; strategy=strategy)
end

"""
    apply_tdag_gate_hybrid!(state::CAMPSState, qubit::Int;
                             strategy::DisentanglingStrategy=HybridStrategy()) -> CAMPSState

Apply a T† gate using the specified disentangling strategy.

# Arguments
- `state::CAMPSState`: CAMPS state
- `qubit::Int`: Target qubit
- `strategy::DisentanglingStrategy`: Disentangling strategy

# Returns
- `CAMPSState`: Modified state
"""
function apply_tdag_gate_hybrid!(state::CAMPSState, qubit::Int;
                                  strategy::DisentanglingStrategy=HybridStrategy())::CAMPSState
    return apply_rotation_hybrid!(state, :Z, qubit, -π/4; strategy=strategy)
end

#==============================================================================#
# BOND ENTROPY UTILITIES
#==============================================================================#

"""
    get_entropy_profile(state::CAMPSState) -> Vector{Float64}

Get the entanglement entropy at each bond of the CAMPS state.

# Arguments
- `state::CAMPSState`: CAMPS state

# Returns
- `Vector{Float64}`: Entropy at each bond [S₁, S₂, ..., S_{n-1}]
"""
function get_entropy_profile(state::CAMPSState)::Vector{Float64}
    ensure_initialized!(state)
    return entanglement_entropy_all_bonds(state.mps)
end

"""
    get_bond_dimension_profile(state::CAMPSState) -> Vector{Int}

Get the bond dimension at each bond of the CAMPS state.

# Arguments
- `state::CAMPSState`: CAMPS state

# Returns
- `Vector{Int}`: Bond dimension at each bond [χ₁, χ₂, ..., χ_{n-1}]
"""
function get_bond_dimension_profile(state::CAMPSState)::Vector{Int}
    ensure_initialized!(state)
    return all_bond_dimensions(state.mps)
end

"""
    estimate_obd_improvement(state::CAMPSState; use_full_search::Bool=false) -> NamedTuple

Estimate how much OBD could reduce entanglement without actually applying it.

# Arguments
- `state::CAMPSState`: CAMPS state (not modified)
- `use_full_search::Bool`: Use full Clifford search

# Returns
- `NamedTuple` with fields:
  - `current_max_entropy::Float64`: Current maximum bond entropy
  - `estimated_reduction::Float64`: Estimated entropy reduction
  - `best_bond::Int`: Bond with highest potential improvement
"""
function estimate_obd_improvement(state::CAMPSState; use_full_search::Bool=false)
    ensure_initialized!(state)

    n = state.n_qubits
    if n < 2
        return (current_max_entropy=0.0, estimated_reduction=0.0, best_bond=0)
    end

    current_entropies = get_entropy_profile(state)
    current_max = maximum(current_entropies)

    best_reduction = 0.0
    best_bond = 1

    for bond in 1:(n-1)
        _, initial, final = find_optimal_clifford_for_bond(state.mps, bond, state.sites;
                                                            use_full_search=use_full_search)
        reduction = initial - final
        if reduction > best_reduction
            best_reduction = reduction
            best_bond = bond
        end
    end

    return (
        current_max_entropy = current_max,
        estimated_reduction = best_reduction,
        best_bond = best_bond
    )
end
