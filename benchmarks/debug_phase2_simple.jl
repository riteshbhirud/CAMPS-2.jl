#!/usr/bin/env julia
# Simple debug - run from CAMPS.jl/benchmarks/
# Usage: julia --project=.. debug_phase2_simple.jl

using CAMPS
using QuantumClifford
using QuantumClifford.ECC
using Random

println("="^80)
println("PHASE 2 DEBUG - Testing Circuit Generation")
println("="^80)
println()

# Load families
println("Loading circuit families...")
include("circuit_families_complete.jl")
println("✓ Loaded")
println()

# Test 1: QAOA
println("[1/5] Testing QAOA MaxCut...")
try
    qaoa = QAOAMaxCutCircuit()
    params = Dict(:n_qubits => 8, :n_t_gates => 4, :seed => 1234)
    circuit = generate_circuit(qaoa, params)
    println("  ✓ Circuit generated: ", typeof(circuit))
    println("  ✓ Has n_qubits: ", hasfield(typeof(circuit), :n_qubits))
    println("  ✓ Has gates: ", hasfield(typeof(circuit), :gates))
    println("  ✓ n_qubits = ", circuit.n_qubits)
    println("  ✓ n_gates = ", length(circuit.gates))
catch e
    println("  ✗ FAILED!")
    println("  Error: ", e)
    println()
    showerror(stdout, e, catch_backtrace())
end
println()

# Test 2: Surface Code  
println("[2/5] Testing Surface Code...")
try
    surface = SurfaceCodeFamily()
    circuit = generate_circuit(surface; n_qubits=8, n_t_gates=4, seed=1234)
    println("  ✓ Circuit generated: ", typeof(circuit))
    println("  ✓ n_qubits = ", circuit.n_qubits)
    println("  ✓ n_gates = ", length(circuit.gates))
catch e
    println("  ✗ FAILED!")
    println("  Error: ", e)
    println()
    showerror(stdout, e, catch_backtrace())
end
println()

# Test 3: QFT
println("[3/5] Testing QFT...")
try
    qft = QFTFamily()
    circuit = generate_circuit(qft; n_qubits=4, density=:low, seed=1234)
    println("  ✓ Circuit generated: ", typeof(circuit))
    println("  ✓ n_qubits = ", circuit.n_qubits)
    println("  ✓ n_gates = ", length(circuit.gates))
catch e
    println("  ✗ FAILED!")
    println("  Error: ", e)
    println()
    showerror(stdout, e, catch_backtrace())
end
println()

# Test 4: Grover
println("[4/5] Testing Grover...")
try
    grover = GroverFamily()
    circuit = generate_circuit(grover; n_qubits=4, density=:full, seed=1234)
    println("  ✓ Circuit generated: ", typeof(circuit))
    println("  ✓ n_qubits = ", circuit.n_qubits)
    println("  ✓ n_gates = ", length(circuit.gates))
catch e
    println("  ✗ FAILED!")
    println("  Error: ", e)
    println()
    showerror(stdout, e, catch_backtrace())
end
println()

# Test 5: VQE
println("[5/5] Testing VQE...")
try
    vqe = VQEFamily()
    circuit = generate_circuit(vqe; n_qubits=4, layers=1, seed=1234)
    println("  ✓ Circuit generated: ", typeof(circuit))
    println("  ✓ n_qubits = ", circuit.n_qubits)
    println("  ✓ n_gates = ", length(circuit.gates))
catch e
    println("  ✗ FAILED!")
    println("  Error: ", e)
    println()
    showerror(stdout, e, catch_backtrace())
end
println()

println("="^80)
println("CIRCUIT GENERATION TESTS COMPLETE")
println("="^80)