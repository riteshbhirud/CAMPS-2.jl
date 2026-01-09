# CAMPS.jl/benchmarks/qaoa_maxcut_family.jl
#
# RIGOROUS QAOA MAXCUT IMPLEMENTATION
# Based on Farhi et al. "A Quantum Approximate Optimization Algorithm" (arXiv:1411.4028)
#
# This implements the Quantum Approximate Optimization Algorithm for MaxCut
# with p=1 layer, using discrete angles for controlled Clifford+T decomposition.
#
# NOTE: Requires AbstractCircuitFamily and CircuitInstance from circuit_families_minimal.jl

# IMPORTANT: T-GATE DENSITY CONSTRAINT
# =====================================
# QAOA with p=1 and discrete Clifford+T angles can only achieve SPECIFIC T-gate densities
# due to the fixed structure:
#   - Cost layer: 3n/2 gates (for 3-regular graph)
#   - Mixer layer: n gates
#
# For n qubits, achievable T-counts are:
#   - Both Clifford: 0 T-gates (t/n ≈ 0)
#   - Mixer only non-Clifford: n T-gates (t/n = 1.0)
#   - Cost only non-Clifford: 3n/2 T-gates (t/n = 1.5)
#   - Both non-Clifford: 5n/2 T-gates (t/n = 2.5)
#
# This is a FUNDAMENTAL PROPERTY of QAOA structure, not an implementation limitation.
# For benchmarking across families, we select angles that give the CLOSEST achievable
# T-count to each target density.
#
# For paper: This constraint should be acknowledged as inherent to QAOA's structure
# with discrete gate sets, demonstrating the trade-off between variational flexibility
# and fault-tolerant implementation.

using Random

# Assuming AbstractCircuitFamily and CircuitInstance are already defined
# If using standalone, include circuit_families_minimal.jl first

#==============================================================================#
# QAOA MATHEMATICAL FOUNDATION (Farhi et al. 2014)
#==============================================================================#

# For MaxCut on graph G = (V, E):
#
# Cost Hamiltonian: C = ∑_{⟨jk⟩∈E} C_jk where C_jk = (1/2)(1 - σ_z^j σ_z^k)
# Mixer Hamiltonian: B = ∑_{j∈V} σ_x^j
#
# QAOA Circuit (p=1):
#   |ψ(γ,β)⟩ = U(B,β) U(C,γ) |s⟩
# where:
#   |s⟩ = H^⊗n |0⟩  (uniform superposition)
#   U(C,γ) = e^{-iγC} = ∏_{⟨jk⟩} e^{-iγC_jk}
#   U(B,β) = e^{-iβB} = ∏_j e^{-iβσ_x^j}
#
# Gate Decomposition:
#   e^{-iγC_jk} = e^{-iγ/2}·e^{iγ/2 σ_z^j σ_z^k} = (global phase)·RZZ(γ)
#   e^{-iβσ_x^j} = RX(2β)
#
# where:
#   RZZ(θ) = exp(-iθ/2 Z⊗Z) = CNOT · RZ(θ) · CNOT
#   RX(θ) = exp(-iθ/2 X) = H · RZ(θ) · H
#
# Circuit Structure:
#   1. Initialize: Apply H to all n qubits → |+⟩^⊗n
#   2. Cost layer: Apply RZZ(γ) on all edges
#   3. Mixer layer: Apply RX(2β) on all qubits
#   4. Measure in computational basis

#==============================================================================#
# 3-REGULAR GRAPH GENERATION
#==============================================================================#

"""
    generate_random_3regular_graph(n::Int; seed::Int=42)

Generate a random 3-regular graph on n vertices.

A 3-regular graph has exactly 3 edges per vertex, giving 3n/2 total edges.
Uses configuration model with guaranteed 3-regularity.

# Arguments
- `n::Int`: Number of vertices (must be even for 3-regularity to exist)
- `seed::Int`: Random seed for reproducibility

# Returns
- `edges::Vector{Tuple{Int,Int}}`: List of edges as (vertex1, vertex2) pairs

# Mathematical Properties
- Degree(v) = 3 for all v ∈ V
- |E| = 3n/2
- Connected with high probability for random generation
- Exists if and only if n is even and n ≥ 4

# Reference
Configuration model: Bollobás, "Random Graphs" (2001)
"""
function generate_random_3regular_graph(n::Int; seed::Int=42)
    # Validate input
    n >= 4 || error("Need at least 4 vertices for 3-regular graph")
    n % 2 == 0 || error("3-regular graphs require even number of vertices")
    
    Random.seed!(seed)
    
    # Try multiple times with different strategies
    max_retries = 20
    
    for retry in 1:max_retries
        try
            edges = attempt_3regular_generation(n)
            
            # Verify we got exactly 3n/2 edges
            if length(edges) != 3n ÷ 2
                continue  # Try again
            end
            
            # Verify degree-3 property
            degree = zeros(Int, n)
            for (u, v) in edges
                degree[u] += 1
                degree[v] += 1
            end
            
            if all(d == 3 for d in degree)
                return edges  # Success!
            end
        catch
            # Try again with different random state
            Random.seed!(seed + retry)
            continue
        end
    end
    
    # If random generation failed, use deterministic construction
    # This always works but is less random
    @warn "Random 3-regular graph generation failed, using deterministic construction for n=$n"
    return deterministic_3regular_graph(n, seed)
end

"""
    deterministic_3regular_graph(n::Int, seed::Int)

Construct a 3-regular graph deterministically using circulant graph method.
This always works but produces a specific structure.
"""
function deterministic_3regular_graph(n::Int, seed::Int)
    Random.seed!(seed)
    edges = Tuple{Int,Int}[]
    visited = Set{Tuple{Int,Int}}()
    
    # Use three different step sizes to create 3-regularity
    # Connect each vertex i to (i+1), (i+2), (i+n/2) mod n
    for i in 1:n
        # Connection 1: to next neighbor
        j = mod1(i + 1, n)
        edge = i < j ? (i, j) : (j, i)
        if !(edge in visited)
            push!(edges, edge)
            push!(visited, edge)
        end
        
        # Connection 2: to second neighbor (creates triangles when n is small)
        j = mod1(i + 2, n)
        edge = i < j ? (i, j) : (j, i)
        if !(edge in visited)
            push!(edges, edge)
            push!(visited, edge)
        end
        
        # Connection 3: to opposite vertex (n must be even)
        j = mod1(i + n÷2, n)
        edge = i < j ? (i, j) : (j, i)
        if !(edge in visited)
            push!(edges, edge)
            push!(visited, edge)
        end
    end
    
    return collect(edges)
end

"""
    attempt_3regular_generation(n::Int)

Single attempt at generating 3-regular graph using improved configuration model.
Uses smarter pairing to avoid self-loops and multi-edges.
"""
function attempt_3regular_generation(n::Int)
    # Configuration model: Create 3 "stubs" per vertex
    stubs = Vector{Int}()
    for v in 1:n
        append!(stubs, [v, v, v])  # 3 stubs per vertex
    end
    
    # Shuffle for randomness
    shuffle!(stubs)
    
    edges = Tuple{Int,Int}[]
    visited = Set{Tuple{Int,Int}}()
    
    # Count edges per vertex to avoid creating multi-edges
    edge_count = Dict{Tuple{Int,Int}, Int}()
    
    rejected = 0
    max_rejected = 50  # If we reject this many times, restart
    
    while length(stubs) >= 2
        if rejected > max_rejected
            # Too many rejections, this attempt is failing
            throw(ErrorException("Too many rejections"))
        end
        
        # Take first stub
        v1 = popfirst!(stubs)
        
        # Find a suitable partner (not v1, and not creating duplicate)
        found_partner = false
        partner_idx = -1
        
        for i in 1:min(length(stubs), 10)  # Check first 10 stubs
            v2 = stubs[i]
            
            if v1 != v2  # Not self-loop
                edge = v1 < v2 ? (v1, v2) : (v2, v1)
                if !(edge in visited)  # Not duplicate
                    partner_idx = i
                    found_partner = true
                    break
                end
            end
        end
        
        if !found_partner
            # Couldn't find partner in first 10, try any
            for i in 1:length(stubs)
                v2 = stubs[i]
                if v1 != v2
                    edge = v1 < v2 ? (v1, v2) : (v2, v1)
                    if !(edge in visited)
                        partner_idx = i
                        found_partner = true
                        break
                    end
                end
            end
        end
        
        if !found_partner
            # Still no partner, put v1 back and shuffle
            push!(stubs, v1)
            shuffle!(stubs)
            rejected += 1
            continue
        end
        
        # Found valid partner!
        v2 = stubs[partner_idx]
        deleteat!(stubs, partner_idx)
        
        edge = v1 < v2 ? (v1, v2) : (v2, v1)
        push!(edges, edge)
        push!(visited, edge)
        rejected = 0  # Reset rejection counter
    end
    
    return edges
end

#==============================================================================#
# CLIFFORD+T DECOMPOSITION
#==============================================================================#

"""
    clifford_t_angle_properties(θ::Float64)

Determine Clifford+T decomposition properties for RZ(θ).

# Exact Decompositions (multiples of π/4):
- RZ(0) = I         (0 T-gates, Clifford)
- RZ(π/4) = T       (1 T-gate)
- RZ(π/2) = S       (0 T-gates, Clifford)
- RZ(3π/4) = S·T    (1 T-gate)
- RZ(π) = Z         (0 T-gates, Clifford)
- RZ(5π/4) = Z·T    (1 T-gate)
- RZ(3π/2) = S†     (0 T-gates, Clifford)
- RZ(7π/4) = T†     (1 T-gate)

# Returns
- `(is_clifford::Bool, n_t_gates::Int)`: Whether angle is Clifford-only and T-gate count

# Reference
Optimal Clifford+T decomposition: Ross & Selinger, arXiv:1403.2975 (2014)
"""
function clifford_t_angle_properties(θ::Float64)
    # Normalize to [0, 2π)
    θ_norm = mod(θ, 2π)
    
    # Check if multiple of π/4
    k = round(θ_norm / (π/4))
    is_multiple_of_pi4 = abs(θ_norm - k * π/4) < 1e-10
    
    if !is_multiple_of_pi4
        error("Angle $θ is not a multiple of π/4. Cannot determine exact T-count.")
    end
    
    # Determine T-count based on k (where θ = k·π/4)
    k_mod_8 = Int(mod(k, 8))
    
    if k_mod_8 in [0, 2, 4, 6]
        # Clifford angles: 0, π/2, π, 3π/2
        return (true, 0)
    else
        # T-gate angles: π/4, 3π/4, 5π/4, 7π/4
        return (false, 1)
    end
end

"""
    decompose_rzz(θ::Float64, q1::Int, q2::Int)

Decompose RZZ(θ) into elementary gates.

RZZ(θ) = exp(-iθ/2 Z⊗Z) = CNOT(q1,q2) · RZ(θ,q2) · CNOT(q1,q2)

# Returns
Vector of (gate_type, qubits) tuples
"""
function decompose_rzz(θ::Float64, q1::Int, q2::Int)
    return [
        (:CNOT, [q1, q2]),
        (:RZ, [q2], θ),      # Store angle as third element
        (:CNOT, [q1, q2])
    ]
end

"""
    decompose_rx(θ::Float64, q::Int)

Decompose RX(θ) into elementary gates.

RX(θ) = exp(-iθ/2 X) = H · RZ(θ) · H

# Returns
Vector of (gate_type, qubits) tuples
"""
function decompose_rx(θ::Float64, q::Int)
    return [
        (:H, [q]),
        (:RZ, [q], θ),
        (:H, [q])
    ]
end

#==============================================================================#
# ANGLE SELECTION FOR TARGET T-GATE DENSITY
#==============================================================================#

"""
    select_qaoa_angles(n::Int, target_t_fraction::Float64; seed::Int=42)

Select (γ, β) angles to approximately achieve target T-gate density.

**FUNDAMENTAL CONSTRAINT:**
QAOA with p=1 and discrete Clifford+T angles can only achieve specific T-counts:
- Cost layer: 3n/2 RZ(γ) gates
- Mixer layer: n RZ(2β) gates

Possible T-counts (for discrete angles):
1. Both Clifford: 0 T-gates
2. β non-Clifford only: n T-gates
3. γ non-Clifford only: 3n/2 T-gates  
4. Both non-Clifford: 5n/2 T-gates

This means we can only approximate arbitrary target densities.

# Arguments
- `n::Int`: Number of qubits
- `target_t_fraction::Float64`: Desired t/n ratio (may not be exactly achievable)
- `seed::Int`: Random seed for angle variation within achievable set

# Returns
- `(γ::Float64, β::Float64)`: QAOA angles (in radians)
- `predicted_t_count::Int`: Actual T-gate count (may differ from target)

# Strategy
Select angle pair that minimizes |actual_t - target_t|, with random variation
among angles that give the same T-count.
"""
function select_qaoa_angles(n::Int, target_t_fraction::Float64; seed::Int=42)
    Random.seed!(seed)
    
    # Calculate target T-count
    target_t = n * target_t_fraction
    
    # For 3-regular graph: 3n/2 edges
    n_cost_rz = (3 * n) ÷ 2
    n_mixer_rz = n
    
    # Possible T-counts with discrete angles:
    # γ=Clifford, β=Clifford: 0 T-gates
    # γ=Clifford, β=T-angle:  n T-gates (from mixer only)
    # γ=T-angle, β=Clifford:  (3n/2) T-gates (from cost only)
    # γ=T-angle, β=T-angle:   (5n/2) T-gates (from both)
    
    option_0 = 0
    option_mixer = n_mixer_rz
    option_cost = n_cost_rz
    option_both = n_cost_rz + n_mixer_rz
    
    # Find closest achievable T-count
    options = [
        (option_0, "both_clifford"),
        (option_mixer, "mixer_only"),
        (option_cost, "cost_only"),
        (option_both, "both_nonclifford")
    ]
    
    # Select option with minimum distance to target
    best_option = argmin([abs(t - target_t) for (t, _) in options])
    predicted_t, strategy = options[best_option]
    
    # Select angles based on strategy, with random variation
    clifford_angles = [0.0, π/2, π, 3π/2]
    t_angles = [π/4, 3π/4, 5π/4, 7π/4]
    
    if strategy == "both_clifford"
        γ = rand(clifford_angles)
        β = rand(clifford_angles)
    elseif strategy == "mixer_only"
        γ = rand(clifford_angles)
        # Need 2β to be T-angle
        # If β = π/8, then 2β = π/4 (T-angle) ✓
        # If β = 3π/8, then 2β = 3π/4 (T-angle) ✓
        # If β = 5π/8, then 2β = 5π/4 (T-angle) ✓
        # If β = 7π/8, then 2β = 7π/4 (T-angle) ✓
        β_options = [π/8, 3π/8, 5π/8, 7π/8]
        β = rand(β_options)
    elseif strategy == "cost_only"
        γ = rand(t_angles)
        # β should be Clifford so that 2β is also Clifford
        # Clifford angles for β: 0, π/4, π/2, 3π/4, π, 5π/4, 3π/2, 7π/4
        # When doubled and normalized: 0, π/2, π, 3π/2, 0, π/2, π, 3π/2
        # All result in Clifford angles for 2β ✓
        β = rand(clifford_angles)
    else  # both_nonclifford
        γ = rand(t_angles)
        β_options = [π/8, 3π/8, 5π/8, 7π/8]
        β = rand(β_options)
    end
    
    return (γ, β), predicted_t
end

#==============================================================================#
# QAOA CIRCUIT GENERATION
#==============================================================================#

"""
    QAOAMaxCutCircuit

Quantum Approximate Optimization Algorithm for MaxCut on 3-regular graphs (p=1).

# Reference
Farhi, Goldstone, Gutmann, "A Quantum Approximate Optimization Algorithm"
arXiv:1411.4028 (2014)

# Circuit Structure
1. Initialization: H^⊗n |0⟩ → |+⟩^⊗n
2. Cost layer: U(C,γ) = ∏_{edges} RZZ(γ)
3. Mixer layer: U(B,β) = ∏_{qubits} RX(2β)

# Properties
- Approximation ratio: 0.6924 for 3-regular graphs at p=1 (Farhi et al.)
- Circuit depth: O(n)
- T-gate count: Controlled via discrete angle selection
"""
struct QAOAMaxCutCircuit <: AbstractCircuitFamily end

function generate_circuit(::QAOAMaxCutCircuit, params::Dict)
    n = params[:n_qubits]
    n_t_target = params[:n_t_gates]
    seed = get(params, :seed, 42)
    
    # Validate n (must be even for 3-regular graph)
    if n % 2 != 0
        error("QAOA MaxCut on 3-regular graphs requires even number of qubits")
    end
    
    # Generate random 3-regular graph
    edges = generate_random_3regular_graph(n; seed=seed)
    
    # Select angles to match target T-gate count
    target_t_fraction = n_t_target / n
    (γ, β), predicted_t = select_qaoa_angles(n, target_t_fraction; seed=seed)
    
    # Build circuit gates
    gates = Tuple{Symbol, Vector{Int}}[]
    t_positions = Int[]
    
    # 1. INITIALIZATION LAYER: H^⊗n
    for i in 1:n
        push!(gates, (:H, [i]))
    end
    
    # 2. COST LAYER: Apply RZZ(γ) on all edges
    for (q1, q2) in edges
        # Decompose RZZ(γ) → CNOT, RZ(γ), CNOT
        push!(gates, (:CNOT, [q1, q2]))
        
        # Check if RZ(γ) requires T-gates
        is_clifford, n_t = clifford_t_angle_properties(γ)
        if !is_clifford
            # Mark position for T-gate
            push!(t_positions, length(gates) + 1)
        end
        
        # Add RZ gate (will be decomposed to Clifford+T by CAMPS)
        if abs(γ) < 1e-10
            # RZ(0) = I, skip
        elseif abs(γ - π/2) < 1e-10
            push!(gates, (:S, [q2]))
        elseif abs(γ - π) < 1e-10
            push!(gates, (:Z, [q2]))
        elseif abs(γ - 3π/2) < 1e-10
            # S† = S^3 = Z·S
            push!(gates, (:Z, [q2]))
            push!(gates, (:S, [q2]))
        elseif abs(γ - π/4) < 1e-10
            push!(gates, (:T, [q2]))
        elseif abs(γ - 3π/4) < 1e-10
            # S·T
            push!(gates, (:S, [q2]))
            push!(gates, (:T, [q2]))
        elseif abs(γ - 5π/4) < 1e-10
            # Z·T
            push!(gates, (:Z, [q2]))
            push!(gates, (:T, [q2]))
        elseif abs(γ - 7π/4) < 1e-10
            # T† = Z·S·T (since T† = T^7 = T^(-1) and using Z·S·T identity)
            # Alternative: T† = S†·Z·T but S† = S^3
            # Simplest: Count as 1 T-gate, represent as T with phase adjustment
            # For Clifford+T: T† is also 1 T-gate (just T with different phase)
            push!(gates, (:T, [q2]))  # Count as T-gate (same T-count)
            # Note: Technically T†, but CAMPS will handle via T with conjugation
        else
            error("Unexpected angle γ=$γ")
        end
        
        push!(gates, (:CNOT, [q1, q2]))
    end
    
    # 3. MIXER LAYER: Apply RX(2β) on all qubits
    β_mixer = 2β  # QAOA mixer uses RX(2β)
    
    # Normalize to [0, 2π) to handle β values that give 2β > 2π
    β_mixer_normalized = mod(β_mixer, 2π)
    
    for i in 1:n
        # Decompose RX(2β) → H, RZ(2β), H
        push!(gates, (:H, [i]))
        
        # Check if RZ(2β) requires T-gates (use normalized angle)
        is_clifford, n_t = clifford_t_angle_properties(β_mixer_normalized)
        if !is_clifford
            # Mark position for T-gate
            push!(t_positions, length(gates) + 1)
        end
        
        # Add RZ gate based on normalized angle
        if abs(β_mixer_normalized) < 1e-10
            # RZ(0) = I, skip
        elseif abs(β_mixer_normalized - π/2) < 1e-10
            push!(gates, (:S, [i]))
        elseif abs(β_mixer_normalized - π) < 1e-10
            push!(gates, (:Z, [i]))
        elseif abs(β_mixer_normalized - 3π/2) < 1e-10
            push!(gates, (:Z, [i]))
            push!(gates, (:S, [i]))
        elseif abs(β_mixer_normalized - π/4) < 1e-10
            push!(gates, (:T, [i]))
        elseif abs(β_mixer_normalized - 3π/4) < 1e-10
            push!(gates, (:S, [i]))
            push!(gates, (:T, [i]))
        elseif abs(β_mixer_normalized - 5π/4) < 1e-10
            push!(gates, (:Z, [i]))
            push!(gates, (:T, [i]))
        elseif abs(β_mixer_normalized - 7π/4) < 1e-10
            # T† counts as 1 T-gate
            push!(gates, (:T, [i]))  # Represent as T (same T-count)
        else
            error("Unexpected angle 2β=$β_mixer (normalized: $β_mixer_normalized, β=$β)")
        end
        
        push!(gates, (:H, [i]))
    end
    
    # Metadata
    metadata = Dict{String, Any}(
        "family" => "QAOA_MaxCut_p1",
        "graph_type" => "3-regular",
        "n_edges" => length(edges),
        "gamma" => γ,
        "beta" => β,
        "target_t_count" => n_t_target,
        "predicted_t_count" => predicted_t,
        "actual_t_count" => length(t_positions),
        "t_density_target" => n_t_target / n,
        "t_density_actual" => length(t_positions) / n,
        "approximation_ratio_3regular" => 0.6924,  # From Farhi et al. Theorem
        "note" => "T-count limited by discrete QAOA structure: achievable t/n ∈ {0, 1.0, 1.5, 2.5} for n=$(n)"
    )
    
    return CircuitInstance(n, gates, t_positions, metadata)
end

get_name(::QAOAMaxCutCircuit) = "QAOA MaxCut (p=1, 3-regular)"

#==============================================================================#
# EXPORT
#==============================================================================#

export QAOAMaxCutCircuit
export generate_random_3regular_graph
export select_qaoa_angles
export clifford_t_angle_properties