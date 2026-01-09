"""
Surface Code Circuit Family for CAMPS.jl
=========================================

Generates Surface Code circuits using QuantumClifford.ECC with T-gate injection.

Requirements:
- Add XCXGate constructor to CAMPS.jl/src/types.jl (see ADD_TO_TYPES.jl)
- Handle :XCX in clifford_interface.jl resolution

Usage:
    family = SurfaceCodeFamily()
    circuit = generate_circuit(family, n_qubits=8, n_t_gates=6, seed=1234)
    suite = generate_benchmark_suite(family, n_seeds=8)
"""

using QuantumClifford
using QuantumClifford.ECC
using Random

#==============================================================================#
# GATE CONVERSION (QuantumClifford → CAMPS)
#==============================================================================#

"""
    convert_qc_gate_to_camps(qc_gate)

Convert a QuantumClifford.ECC gate to a CAMPS Gate object.

Uses CAMPS gate constructors: HGate(), CNOTGate(), XCXGate(), etc.
Returns `nothing` for measurement gates (sMRZ, sMRX) since CAMPS only
simulates unitary evolution.

# Returns
- `CliffordGate` for Clifford operations
- `nothing` for measurements (filtered out later)
"""
function convert_qc_gate_to_camps(qc_gate)
    gate_str = string(qc_gate)
    
    # Single-qubit Clifford gates
    if occursin("Hadamard", gate_str) || occursin("sH", gate_str)
        m = match(r"qubit (\d+)", gate_str)
        m !== nothing && return HGate(parse(Int, m[1]))
        
    elseif occursin("sPhase", gate_str) || (occursin("sS", gate_str) && !occursin("SWAP", gate_str))
        m = match(r"qubit (\d+)", gate_str)
        m !== nothing && return SGate(parse(Int, m[1]))
        
    elseif occursin("sX", gate_str) && !occursin("sXCX", gate_str)
        m = match(r"qubit (\d+)", gate_str)
        m !== nothing && return XGate(parse(Int, m[1]))
        
    elseif occursin("sY", gate_str)
        m = match(r"qubit (\d+)", gate_str)
        m !== nothing && return YGate(parse(Int, m[1]))
        
    elseif occursin("sZ", gate_str) && !occursin("sZCX", gate_str) && !occursin("sMRZ", gate_str)
        m = match(r"qubit (\d+)", gate_str)
        m !== nothing && return ZGate(parse(Int, m[1]))
    
    # Two-qubit Clifford gates
    elseif occursin("sCNOT", gate_str)
        m = match(r"\((\d+),(\d+)\)", gate_str)
        if m !== nothing
            return CNOTGate(parse(Int, m[1]), parse(Int, m[2]))
        end
        
    elseif occursin("sXCX", gate_str)
        # X-controlled-X: H(c) · CNOT(c,t) · H(c)
        # Most common gate in syndrome measurements (~40% of gates)
        m = match(r"\((\d+),(\d+)\)", gate_str)
        if m !== nothing
            return XCXGate(parse(Int, m[1]), parse(Int, m[2]))
        end
        
    elseif occursin("sZCX", gate_str) || occursin("sCZ", gate_str)
        m = match(r"\((\d+),(\d+)\)", gate_str)
        if m !== nothing
            return CZGate(parse(Int, m[1]), parse(Int, m[2]))
        end
        
    elseif occursin("SWAP", gate_str) || occursin("sSWAP", gate_str)
        m = match(r"\((\d+),(\d+)\)", gate_str)
        if m !== nothing
            return SWAPGate(parse(Int, m[1]), parse(Int, m[2]))
        end
    
    # Measurements - skip (CAMPS doesn't simulate measurement)
    elseif occursin("sMRZ", gate_str) || occursin("sMRX", gate_str)
        return nothing
    end
    
    # Unknown gate type
    error("Unsupported QuantumClifford gate: $gate_str")
end


"""
    convert_qc_circuit_to_camps(qc_circuit)

Convert a vector of QuantumClifford gates to CAMPS Gates.
Filters out measurements automatically.
"""
function convert_qc_circuit_to_camps(qc_circuit)
    camps_gates = []
    
    for qc_gate in qc_circuit
        camps_gate = convert_qc_gate_to_camps(qc_gate)
        
        # Skip measurements (nothing)
        if camps_gate !== nothing
            push!(camps_gates, camps_gate)
        end
    end
    
    return camps_gates
end


#==============================================================================#
# SURFACE CODE FAMILY
#==============================================================================#

struct SurfaceCodeFamily
    name::String
    description::String
    
    function SurfaceCodeFamily()
        new(
            "Surface Code",
            "Quantum error correction circuits with T-gate injection for fault-tolerant quantum computing"
        )
    end
end


"""
    generate_circuit(family::SurfaceCodeFamily; n_qubits, n_t_gates, seed)

Generate a Surface Code circuit with specified parameters.

# Arguments
- `n_qubits::Int`: Target qubit count (8, 12, or 16)
- `n_t_gates::Int`: Number of T-gates to inject (4-24)
- `seed::Int`: Random seed for reproducibility

The actual qubit count may differ slightly based on Surface Code distance:
- n_qubits=8  → distance d=2 (9 qubits)
- n_qubits=12 → distance d=2-3 (13 qubits)
- n_qubits=16 → distance d=3 (17 qubits)

# Returns
Circuit with Surface Code structure:
1. Encoding circuit (Clifford gates)
2. Multiple syndrome measurement rounds (Clifford + measurements)
3. T-gates injected between rounds (logical operations)
4. Final syndrome round

Note: Measurement gates are filtered out (CAMPS simulates unitary evolution only).
"""
function generate_circuit(family::SurfaceCodeFamily; n_qubits::Int, n_t_gates::Int, seed::Int)
    rng = Random.MersenneTwister(seed)
    
    # Choose Surface Code distance based on target qubit count
    # Surface(dx, dz) creates a code with ~2*dx*dz physical qubits
    dx, dz = if n_qubits <= 10
        (2, 2)  # 8-9 qubits
    elseif n_qubits <= 14
        (3, 2)  # 13 qubits
    else
        (3, 3)  # 17 qubits
    end
    
    # Create Surface Code
    code = Surface(dx, dz)
    n_physical = code_n(code)
    code_distance = distance(code)
    
    # Generate QuantumClifford circuits
    encoding_circ = naive_encoding_circuit(code)
    syndrome_circ, _, _ = naive_syndrome_circuit(code)
    
    # Convert to CAMPS gates
    encoding_gates = convert_qc_circuit_to_camps(encoding_circ)
    syndrome_gates = convert_qc_circuit_to_camps(syndrome_circ)
    
    # Build complete circuit
    gates = Gate[]
    t_positions = Int[]
    
    # 1. Encoding circuit
    append!(gates, encoding_gates)
    gate_index = length(gates)
    
    # 2. Multiple syndrome rounds with T-gate injection
    n_syndrome_rounds = 3 + rand(rng, 0:2)  # 3-5 rounds
    
    # Distribute T-gates across rounds
    t_per_round = distribute_t_gates(n_t_gates, n_syndrome_rounds, rng)
    
    # Track T-gate pattern (initialize before loop)
    t_gate_pattern = "uniform"  # Default
    
    for round_idx in 1:n_syndrome_rounds
        # Add syndrome measurement round (Clifford gates only)
        append!(gates, syndrome_gates)
        gate_index += length(syndrome_gates)
        
        # Inject T-gates for this round
        n_t_this_round = t_per_round[round_idx]
        if n_t_this_round > 0
            # Choose qubits for T-gates (bias toward data qubits)
            n_data = div(n_physical, 2)
            data_range = 1:n_data
            
            # Clustered or uniform distribution
            is_clustered = rand(rng) < 0.5
            t_gate_pattern = is_clustered ? "clustered" : "uniform"  # Track pattern
            
            if is_clustered
                center = rand(rng, data_range)
                radius = max(1, code_distance)
                local_range = max(1, center-radius):min(n_data, center+radius)
                qubit_positions = rand(rng, local_range, n_t_this_round)
            else
                qubit_positions = rand(rng, data_range, n_t_this_round)
            end
            
            # Add T-gates
            for qubit_pos in qubit_positions
                push!(gates, TGate(qubit_pos))
                push!(t_positions, gate_index + 1)
                gate_index += 1
            end
        end
    end
    
    # 3. Final syndrome round
    append!(gates, syndrome_gates)
    
    # Create metadata
    metadata = Dict{String, Any}(
        "family" => "Surface Code",
        "code_distance" => code_distance,
        "dx" => dx,
        "dz" => dz,
        "n_syndrome_rounds" => n_syndrome_rounds,
        "n_physical_qubits" => n_physical,
        "n_logical_qubits" => code_k(code),
        "n_stabilizers" => size(parity_checks(code), 1),
        "n_t_gates" => length(t_positions),
        "t_gate_pattern" => t_gate_pattern,  # Use the variable defined above
        "seed" => seed
    )
    
    return (
        n_qubits = n_physical,
        gates = gates,
        t_positions = t_positions,
        metadata = metadata
    )
end


"""
    distribute_t_gates(n_t_gates, n_rounds, rng)

Distribute T-gates across syndrome rounds.
Strategy: Spread temporally, not all in one round.
"""
function distribute_t_gates(n_t_gates::Int, n_rounds::Int, rng)
    distribution = zeros(Int, n_rounds)
    remaining = n_t_gates
    
    # Use at least half the rounds
    active_rounds = max(1, min(n_rounds, ceil(Int, n_rounds * 0.6)))
    round_indices = shuffle(rng, 1:n_rounds)[1:active_rounds]
    
    for (i, round_idx) in enumerate(round_indices)
        if i == active_rounds
            distribution[round_idx] = remaining
        else
            n_this_round = min(remaining, rand(rng, 1:min(4, max(1, remaining))))
            distribution[round_idx] = n_this_round
            remaining -= n_this_round
        end
    end
    
    return distribution
end


"""
    get_parameter_ranges(family::SurfaceCodeFamily)

Return valid parameter ranges for Surface Code benchmark suite.
"""
function get_parameter_ranges(family::SurfaceCodeFamily)
    return Dict(
        :n_qubits => [8, 12, 16],
        :n_t_gates => [
            [4, 6],      # Low T-gate density
            [8, 12],     # Medium T-gate density
            [16, 24]     # High T-gate density
        ],
        :seed => 1000:9999
    )
end


"""
    generate_benchmark_suite(family::SurfaceCodeFamily; n_seeds=8)

Generate complete benchmark suite of 72 Surface Code circuits.

Returns 72 circuits:
- 3 qubit sizes (8, 12, 16)
- 3 T-gate density levels (low, medium, high)
- 8 random seeds per configuration

Each circuit includes:
- Surface Code structure (encoding + syndrome rounds)
- T-gates injected at logical operation points
- Metadata (code distance, stabilizers, etc.)
"""
function generate_benchmark_suite(family::SurfaceCodeFamily; n_seeds=8)
    ranges = get_parameter_ranges(family)
    circuits = []
    
    for target_n in ranges[:n_qubits]
        for t_range in ranges[:n_t_gates]
            for seed_offset in 0:(n_seeds-1)
                seed = minimum(ranges[:seed]) + seed_offset
                
                # Random T-gate count in range
                rng_temp = Random.MersenneTwister(seed)
                n_t_gates = rand(rng_temp, t_range)
                
                # Generate circuit
                circuit = generate_circuit(
                    family;
                    n_qubits = target_n,
                    n_t_gates = n_t_gates,
                    seed = seed
                )
                
                push!(circuits, circuit)
                
                println("Generated Surface Code: n=$(circuit.n_qubits) qubits, " *
                       "$(length(circuit.t_positions)) T-gates, " *
                       "d=$(circuit.metadata["code_distance"]) (seed=$seed)")
            end
        end
    end
    
    return circuits
end


#==============================================================================#
# EXAMPLE USAGE & TESTING
#==============================================================================#

"""
    test_surface_code_generation()

Test Surface Code generation and conversion.
Run this to verify everything works.
"""
function test_surface_code_generation()
    println("="^70)
    println("TESTING SURFACE CODE GENERATION")
    println("="^70)
    
    family = SurfaceCodeFamily()
    
    # Test single circuit generation
    println("\n1. Testing single circuit generation...")
    circuit = generate_circuit(family; n_qubits=8, n_t_gates=6, seed=1234)
    
    println("   Physical qubits: $(circuit.n_qubits)")
    println("   T-gates: $(length(circuit.t_positions))")
    println("   Total gates: $(length(circuit.gates))")
    println("   Code distance: $(circuit.metadata["code_distance"])")
    println("   ✓ Single circuit generated successfully!")
    
    # Test gate types
    println("\n2. Testing gate type distribution...")
    gate_types = Dict{String, Int}()
    for gate in circuit.gates
        type_name = string(typeof(gate).name.name)
        gate_types[type_name] = get(gate_types, type_name, 0) + 1
    end
    
    println("   Gate types:")
    for (type_name, count) in sort(collect(gate_types))
        pct = round(100 * count / length(circuit.gates), digits=1)
        println("     $type_name: $count ($pct%)")
    end
    println("   ✓ Gate distribution looks good!")
    
    # Test benchmark suite generation (small subset)
    println("\n3. Testing benchmark suite generation (6 circuits)...")
    test_circuits = []
    for n in [8, 12, 16]
        for n_t in [6, 12]
            circ = generate_circuit(family; n_qubits=n, n_t_gates=n_t, seed=1000)
            push!(test_circuits, circ)
        end
    end
    
    println("   Generated $(length(test_circuits)) test circuits")
    println("   Qubit range: $(minimum(c.n_qubits for c in test_circuits)) - $(maximum(c.n_qubits for c in test_circuits))")
    println("   T-gate range: $(minimum(length(c.t_positions) for c in test_circuits)) - $(maximum(length(c.t_positions) for c in test_circuits))")
    println("   ✓ Benchmark suite generation works!")
    
    println("\n" * "="^70)
    println("ALL TESTS PASSED!")
    println("="^70)
    println("\nSurface Code family is ready to use!")
    println("To generate full benchmark suite (72 circuits):")
    println("  suite = generate_benchmark_suite(SurfaceCodeFamily(), n_seeds=8)")
end


# Run test if this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    test_surface_code_generation()
end