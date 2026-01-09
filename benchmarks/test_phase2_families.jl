#!/usr/bin/env julia
# Test each Phase 2 family individually to find which one hangs

using CAMPS
using QuantumClifford
using QuantumClifford.ECC
using Random

println("Loading circuit families...")
include("circuit_families_complete.jl")
println("✓ Loaded")
println()

println("="^70)
println("TESTING PHASE 2 FAMILIES INDIVIDUALLY")
println("="^70)
println()

# Test QAOA
println("[1/5] Testing QAOA MaxCut...")
try
    qaoa = QAOAMaxCutCircuit()
    params = Dict(:n_qubits => 8, :n_t_gates => 8, :seed => 1234)
    
    println("  Generating circuit...")
    circuit = generate_circuit(qaoa, params)
    println("  ✓ Generated: $(circuit.n_qubits) qubits, $(length(circuit.gates)) gates")
    
    println("  Executing circuit...")
    state = CAMPSState(circuit.n_qubits; max_bond=2048)
    initialize!(state)
    
    for (i, (gate_type, qubits)) in enumerate(circuit.gates)
        if gate_type == :T
            apply_t_gate_ofd!(state, qubits[1])
        elseif gate_type == :H
            apply_gate!(state, CliffordGate([(:H, qubits[1])], [qubits[1]]))
        elseif gate_type == :CNOT
            apply_gate!(state, CliffordGate([(:CNOT, qubits[1], qubits[2])], [qubits[1], qubits[2]]))
        elseif gate_type == :X
            apply_gate!(state, CliffordGate([(:X, qubits[1])], [qubits[1]]))
        elseif gate_type == :RZZ || gate_type == :RX
            # Skip for now - these might need special handling
            println("    Skipping $(gate_type) gate")
        else
            println("    Unknown gate type: $gate_type")
        end
        
        if i % 20 == 0
            println("    Applied $i/$(length(circuit.gates)) gates...")
        end
    end
    
    chi = get_bond_dimension(state)
    println("  ✓ Execution complete: χ = $chi")
    println("  ✅ QAOA WORKS!")
catch e
    println("  ✗ QAOA FAILED: $e")
    println("  Stack trace:")
    showerror(stdout, e, catch_backtrace())
end
println()

# Test Surface Code
println("[2/5] Testing Surface Code...")
try
    surface = SurfaceCodeFamily()
    
    println("  Generating circuit...")
    circuit = generate_circuit(surface; n_qubits=9, n_t_gates=4, seed=1234)
    println("  ✓ Generated: $(circuit.n_qubits) qubits, $(length(circuit.gates)) gates")
    println("  Gate types: $(typeof(circuit.gates[1]))")
    
    println("  Executing circuit...")
    state = CAMPSState(circuit.n_qubits; max_bond=2048)
    initialize!(state)
    
    for (i, gate) in enumerate(circuit.gates)
        # Surface code returns actual Gate objects
        apply_gate!(state, gate)
        
        if i % 20 == 0
            println("    Applied $i/$(length(circuit.gates)) gates...")
        end
    end
    
    chi = get_bond_dimension(state)
    println("  ✓ Execution complete: χ = $chi")
    println("  ✅ SURFACE CODE WORKS!")
catch e
    println("  ✗ SURFACE CODE FAILED: $e")
    println("  Stack trace:")
    showerror(stdout, e, catch_backtrace())
end
println()

# Test QFT
println("[3/5] Testing QFT...")
try
    qft = QFTFamily()
    
    println("  Generating circuit...")
    circuit = generate_circuit(qft; n_qubits=4, density=:low, seed=1234)
    println("  ✓ Generated: $(circuit.n_qubits) qubits, $(length(circuit.gates)) gates")
    println("  Gate types: $(typeof(circuit.gates[1]))")
    
    println("  Executing circuit...")
    state = CAMPSState(circuit.n_qubits; max_bond=2048)
    initialize!(state)
    
    for (i, gate) in enumerate(circuit.gates)
        apply_gate!(state, gate)
        
        if i % 10 == 0
            println("    Applied $i/$(length(circuit.gates)) gates...")
        end
    end
    
    chi = get_bond_dimension(state)
    println("  ✓ Execution complete: χ = $chi")
    println("  ✅ QFT WORKS!")
catch e
    println("  ✗ QFT FAILED: $e")
    println("  Stack trace:")
    showerror(stdout, e, catch_backtrace())
end
println()


println()

# Test VQE
println("[5/5] Testing VQE...")
try
    vqe = VQEFamily()
    
    println("  Generating circuit...")
    circuit = generate_circuit(vqe; n_qubits=4, layers=1, seed=1234)
    println("  ✓ Generated: $(circuit.n_qubits) qubits, $(length(circuit.gates)) gates")
    println("  Gate types: $(typeof(circuit.gates[1]))")
    
    println("  Executing circuit...")
    state = CAMPSState(circuit.n_qubits; max_bond=2048)
    initialize!(state)
    
    for (i, gate) in enumerate(circuit.gates)
        apply_gate!(state, gate)
        
        if i % 10 == 0
            println("    Applied $i/$(length(circuit.gates)) gates...")
        end
    end
    
    chi = get_bond_dimension(state)
    println("  ✓ Execution complete: χ = $chi")
    println("  ✅ VQE WORKS!")
catch e
    println("  ✗ VQE FAILED: $e")
    println("  Stack trace:")
    showerror(stdout, e, catch_backtrace())
end
println()

println("="^70)
println("TESTING COMPLETE")
println("="^70)
println()
println("Check which families work and which fail/hang.")
println("If a test hangs forever, press Ctrl+C and note which one it was.")