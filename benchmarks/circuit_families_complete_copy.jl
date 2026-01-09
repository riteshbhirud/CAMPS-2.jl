# CAMPS.jl/benchmarks/circuit_families_minimal.jl
#
#9 Exact Circuit Families
# 
# This version uses only QuantumClifford's public random_clifford() function
# and symbolic gate notation to avoid any API issues.

using CAMPS
using QuantumClifford
using Random

#==============================================================================#
# CIRCUIT REPRESENTATION
#==============================================================================#

abstract type AbstractCircuitFamily end

"""
CircuitInstance stores a circuit as a sequence of symbolic gate operations.
Gates are represented as (gate_type, qubits) tuples.
"""
struct CircuitInstance
    n_qubits::Int
    gates::Vector{Tuple{Symbol, Vector{Int}}}  # (:H, [1]) or (:CNOT, [1,2]) or (:random2q, [1,2])
    t_gate_positions::Vector{Int}  # Which gate indices are T-gates
    metadata::Dict{String, Any}
end

#==============================================================================#
# 1. RANDOM CLIFFORD+T (BRICK-WALL)
#==============================================================================#

"""
    RandomBrickwallCliffordT

Random Clifford+T circuits with brick-wall architecture (Liu & Clark baseline).

Uses TRUE uniform sampling from the 11,520-element 2-qubit Clifford group Cl_2
via QuantumClifford's random_clifford(2) function and SparseGate application.

This provides the exact baseline used in Liu & Clark "Clifford-Augmented Matrix
Product States" (arXiv:2412.17209) for testing OFD applicability.

Parameters:
- `n_qubits::Int`: Number of qubits
- `n_t_gates::Int`: Number of T-gates
- `clifford_depth::Int`: Complete brick-wall layers (LT=1 means 2 sub-layers)
- `seed::Int`: Random seed
"""
struct RandomBrickwallCliffordT <: AbstractCircuitFamily end

function generate_circuit(::RandomBrickwallCliffordT, params::Dict)
    n = params[:n_qubits]
    n_t = params[:n_t_gates]
    depth = params[:clifford_depth]
    seed = get(params, :seed, 42)
    
    Random.seed!(seed)
    
    gates = Tuple{Symbol, Vector{Int}}[]
    t_positions = Int[]
    
    for t_idx in 1:n_t
        # Brick-wall Clifford layers
        for layer in 1:depth
            offset = (layer - 1) % 2
            for i in (1 + offset):2:(n - 1)
                push!(gates, (:random2q, [i, i+1]))
            end
        end
        
        # Mark next position for T-gate
        push!(t_positions, length(gates) + 1)
        push!(gates, (:T, [rand(1:n)]))
    end
    
    metadata = Dict{String, Any}(
        "family" => "RandomBrickwallCliffordT",
        "clifford_depth" => depth,
        "n_t_gates" => n_t
    )
    
    return CircuitInstance(n, gates, t_positions, metadata)
end

get_name(::RandomBrickwallCliffordT) = "Random Clifford+T (Brick-wall)"

#==============================================================================#
# 2. RANDOM CLIFFORD+T (ALL-TO-ALL)
#==============================================================================#

"""
    RandomAllToAllCliffordT

Random Clifford+T with unrestricted connectivity.

Uses TRUE uniform sampling from Cl_2 via random_clifford(2) with SparseGate.
Tests connectivity effect on OFD compared to brick-wall baseline.

Parameters:
- `n_qubits::Int`: Number of qubits
- `n_t_gates::Int`: Number of T-gates  
- `clifford_layers::Int`: Clifford gates per T-gate
- `seed::Int`: Random seed
"""
struct RandomAllToAllCliffordT <: AbstractCircuitFamily end

function generate_circuit(::RandomAllToAllCliffordT, params::Dict)
    n = params[:n_qubits]
    n_t = params[:n_t_gates]
    layers = params[:clifford_layers]
    seed = get(params, :seed, 42)
    
    Random.seed!(seed)
    
    gates = Tuple{Symbol, Vector{Int}}[]
    t_positions = Int[]
    
    for t_idx in 1:n_t
        for _ in 1:layers
            q1, q2 = rand(1:n), rand(1:n)
            while q2 == q1
                q2 = rand(1:n)
            end
            push!(gates, (:random2q, [q1, q2]))
        end
        
        push!(t_positions, length(gates) + 1)
        push!(gates, (:T, [rand(1:n)]))
    end
    
    metadata = Dict{String, Any}(
        "family" => "RandomAllToAllCliffordT",
        "clifford_layers" => layers,
        "n_t_gates" => n_t
    )
    
    return CircuitInstance(n, gates, t_positions, metadata)
end

get_name(::RandomAllToAllCliffordT) = "Random Clifford+T (All-to-all)"

#==============================================================================#
# 3-5: ORACLE ALGORITHMS
#==============================================================================#

struct BernsteinVaziraniCircuit <: AbstractCircuitFamily end

function generate_circuit(::BernsteinVaziraniCircuit, params::Dict)
    n = params[:n_qubits]
    n_t = params[:n_t_gates]
    seed = get(params, :seed, 42)
    Random.seed!(seed)
    
    secret = rand(0:1, n)
    gates = Tuple{Symbol, Vector{Int}}[]
    t_positions = Int[]
    
    # Initial Hadamards
    for i in 1:n
        push!(gates, (:H, [i]))
    end
    
    # Oracle
    for i in 1:n
        if secret[i] == 1 && i < n
            push!(gates, (:CNOT, [i, n]))
        end
    end
    
    # T-gates
    for _ in 1:n_t
        push!(t_positions, length(gates) + 1)
        push!(gates, (:T, [rand(1:n)]))
    end
    
    # Final Hadamards
    for i in 1:n
        push!(gates, (:H, [i]))
    end
    
    return CircuitInstance(n, gates, t_positions, Dict("family" => "BernsteinVazirani"))
end

get_name(::BernsteinVaziraniCircuit) = "Bernstein-Vazirani"

struct SimonCircuit <: AbstractCircuitFamily end

function generate_circuit(::SimonCircuit, params::Dict)
    n = params[:n_qubits]
    n_t = params[:n_t_gates]
    seed = get(params, :seed, 42)
    Random.seed!(seed)
    
    n_half = n ÷ 2
    period = rand(0:1, n_half)
    gates = Tuple{Symbol, Vector{Int}}[]
    t_positions = Int[]
    
    for i in 1:n_half
        push!(gates, (:H, [i]))
    end
    
    for i in 1:n_half
        push!(gates, (:CNOT, [i, n_half + i]))
    end
    
    for i in 1:n_half
        if period[i] == 1
            for j in 1:n_half
                if j != i
                    push!(gates, (:CNOT, [i, n_half + j]))
                end
            end
        end
    end
    
    for _ in 1:n_t
        push!(t_positions, length(gates) + 1)
        push!(gates, (:T, [rand(1:n)]))
    end
    
    for i in 1:n_half
        push!(gates, (:H, [i]))
    end
    
    return CircuitInstance(n, gates, t_positions, Dict("family" => "Simon"))
end

get_name(::SimonCircuit) = "Simon's Algorithm"

struct DeutschJozsaCircuit <: AbstractCircuitFamily end

function generate_circuit(::DeutschJozsaCircuit, params::Dict)
    n = params[:n_qubits]
    n_t = params[:n_t_gates]
    ftype = get(params, :function_type, :balanced)
    seed = get(params, :seed, 42)
    Random.seed!(seed)
    
    gates = Tuple{Symbol, Vector{Int}}[]
    t_positions = Int[]
    n_input = n - 1
    
    for i in 1:n
        push!(gates, (:H, [i]))
    end
    
    if ftype == :constant && rand() < 0.5
        push!(gates, (:X, [n]))
    elseif ftype == :balanced
        for q in randperm(n_input)[1:(n_input÷2)]
            push!(gates, (:CNOT, [q, n]))
        end
        for _ in 1:n_t
            push!(t_positions, length(gates) + 1)
            push!(gates, (:T, [rand(1:n_input)]))
            push!(gates, (:H, [rand(1:n_input)]))
        end
    end
    
    for i in 1:n_input
        push!(gates, (:H, [i]))
    end
    
    return CircuitInstance(n, gates, t_positions, Dict("family" => "DeutschJozsa"))
end

get_name(::DeutschJozsaCircuit) = "Deutsch-Jozsa"

#==============================================================================#
# 6-9: ENTANGLED STATES
#==============================================================================#

struct GHZStateCircuit <: AbstractCircuitFamily end

function generate_circuit(::GHZStateCircuit, params::Dict)
    n = params[:n_qubits]
    n_t = params[:n_t_gates]
    seed = get(params, :seed, 42)
    Random.seed!(seed)
    
    gates = Tuple{Symbol, Vector{Int}}[]
    t_positions = Int[]
    
    push!(gates, (:H, [1]))
    for i in 2:n
        push!(gates, (:CNOT, [1, i]))
    end
    
    for _ in 1:n_t
        push!(t_positions, length(gates) + 1)
        push!(gates, (:T, [rand(1:n)]))
    end
    
    return CircuitInstance(n, gates, t_positions, Dict("family" => "GHZState"))
end

get_name(::GHZStateCircuit) = "GHZ State"

struct BellStateCircuit <: AbstractCircuitFamily end

function generate_circuit(::BellStateCircuit, params::Dict)
    n = params[:n_qubits]
    n_t = params[:n_t_gates]
    seed = get(params, :seed, 42)
    Random.seed!(seed)
    
    gates = Tuple{Symbol, Vector{Int}}[]
    t_positions = Int[]
    
    for i in 1:2:(n-1)
        push!(gates, (:H, [i]))
        push!(gates, (:CNOT, [i, i+1]))
    end
    
    if n % 2 == 1
        push!(gates, (:H, [n]))
    end
    
    for _ in 1:n_t
        push!(t_positions, length(gates) + 1)
        push!(gates, (:T, [rand(1:n)]))
    end
    
    return CircuitInstance(n, gates, t_positions, Dict("family" => "BellState"))
end

get_name(::BellStateCircuit) = "Bell State / EPR Pairs"

struct GraphStateCircuit <: AbstractCircuitFamily end

function generate_circuit(::GraphStateCircuit, params::Dict)
    n = params[:n_qubits]
    n_t = params[:n_t_gates]
    edge_prob = get(params, :edge_probability, 0.3)
    seed = get(params, :seed, 42)
    Random.seed!(seed)
    
    gates = Tuple{Symbol, Vector{Int}}[]
    t_positions = Int[]
    
    for i in 1:n
        push!(gates, (:H, [i]))
    end
    
    for i in 1:n, j in (i+1):n
        if rand() < edge_prob
            # CZ = H + CNOT + H
            push!(gates, (:H, [i]))
            push!(gates, (:CNOT, [i, j]))
            push!(gates, (:H, [i]))
        end
    end
    
    for _ in 1:n_t
        push!(t_positions, length(gates) + 1)
        push!(gates, (:T, [rand(1:n)]))
    end
    
    return CircuitInstance(n, gates, t_positions, Dict("family" => "GraphState", "edge_prob" => edge_prob))
end

get_name(::GraphStateCircuit) = "Graph State"

struct ClusterStateCircuit <: AbstractCircuitFamily end

function generate_circuit(::ClusterStateCircuit, params::Dict)
    n = params[:n_qubits]
    n_t = params[:n_t_gates]
    seed = get(params, :seed, 42)
    Random.seed!(seed)
    
    gates = Tuple{Symbol, Vector{Int}}[]
    t_positions = Int[]
    
    for i in 1:n
        push!(gates, (:H, [i]))
    end
    
    for i in 1:(n-1)
        push!(gates, (:H, [i]))
        push!(gates, (:CNOT, [i, i+1]))
        push!(gates, (:H, [i]))
    end
    
    for _ in 1:n_t
        push!(t_positions, length(gates) + 1)
        push!(gates, (:T, [rand(1:n)]))
    end
    
    return CircuitInstance(n, gates, t_positions, Dict("family" => "ClusterState"))
end

get_name(::ClusterStateCircuit) = "Cluster State (1D)"

#==============================================================================#
# REGISTRY
#==============================================================================#

function get_rigorous_circuit_families()
    return [
        RandomBrickwallCliffordT(),
        RandomAllToAllCliffordT(),
        BernsteinVaziraniCircuit(),
        SimonCircuit(),
        DeutschJozsaCircuit(),
        GHZStateCircuit(),
        BellStateCircuit(),
        GraphStateCircuit(),
        ClusterStateCircuit()
    ]
end

export AbstractCircuitFamily, CircuitInstance
export RandomBrickwallCliffordT, RandomAllToAllCliffordT
export BernsteinVaziraniCircuit, SimonCircuit, DeutschJozsaCircuit
export GHZStateCircuit, BellStateCircuit, GraphStateCircuit, ClusterStateCircuit
export generate_circuit, get_name
export get_rigorous_circuit_families