#!/usr/bin/env julia
# Test CAMPS execution of Phase 2 circuits
# Run from CAMPS.jl/benchmarks/

using CAMPS
using QuantumClifford
using QuantumClifford.ECC
using Random

println("="^80)
println("TESTING CAMPS EXECUTION OF PHASE 2 CIRCUITS")
println("="^80)
println()

# Load families
include("circuit_families_complete.jl")

# Test running ONE Phase 2 circuit through full CAMPS execution
println("[Testing Surface Code with full CAMPS execution]")
println()

try
    # Generate circuit
    println("1. Generating Surface Code circuit...")
    surface = SurfaceCodeFamily()
    circuit = generate_circuit(surface; n_qubits=8, n_t_gates=4, seed=1234)
    
    println("   ✓ Generated")
    println("   n_qubits: ", circuit.n_qubits)
    println("   n_gates: ", length(circuit.gates))
    println("   n_t_gates: ", length(circuit.t_positions))
    println("   First gate type: ", typeof(circuit.gates[1]))
    println()
    
    # Extract components (EXACTLY as benchmark does)
    println("2. Extracting components (as benchmark does)...")
    n_qubits = circuit.n_qubits
    gates = circuit.gates
    t_positions = circuit.t_positions
    metadata = circuit.metadata
    
    println("   ✓ Extracted")
    println()
    
    # Initialize CAMPS
    println("3. Initializing CAMPS state...")
    state = CAMPSState(n_qubits; max_bond=2048)
    initialize!(state)
    
    println("   ✓ Initialized")
    println()
    
    # Apply gates (EXACTLY as benchmark does)
    println("4. Applying gates (as benchmark does)...")
    ofd_success = 0
    ofd_fail = 0
    
    for (idx, gate) in enumerate(gates)
        # Show first few gates
        if idx <= 3
            println("   Gate $idx: ", typeof(gate), " - ", gate)
        end
        
        # Check if symbolic gate (Phase 1) or CAMPS Gate (Phase 2)
        if gate isa Tuple
            println("   ERROR: Got tuple gate in Phase 2! $gate")
            break
        else
            # Phase 2: CAMPS Gate object
            if idx in t_positions
                # This is a T-gate - try OFD
                if gate isa RotationGate && gate.axis == :Z && abs(gate.angle - π/4) < 1e-10
                    success, _ = apply_t_gate_ofd!(state, gate.qubit)
                    if success
                        ofd_success += 1
                    else
                        apply_gate!(state, gate, strategy=OBDStrategy())
                        ofd_fail += 1
                    end
                elseif gate isa TGate
                    # Direct TGate
                    success, _ = apply_t_gate_ofd!(state, gate.qubit)
                    if success
                        ofd_success += 1
                    else
                        apply_gate!(state, gate, strategy=OBDStrategy())
                        ofd_fail += 1
                    end
                else
                    apply_gate!(state, gate)
                end
            else
                # Regular gate (Clifford)
                apply_gate!(state, gate)
            end
        end
        
        # Progress indicator
        if idx % 20 == 0
            println("   Applied $idx/$(length(gates)) gates...")
        end
    end
    
    println("   ✓ All gates applied")
    println()
    
    # Final metrics
    println("5. Computing final metrics...")
    final_chi = get_bond_dimension(state)
    final_nu = n_qubits - sum(state.free_qubits)
    final_S2 = max_entanglement_entropy(state.mps)
    ofd_rate = ofd_success / max(1, ofd_success + ofd_fail)
    
    println("   ✓ Metrics computed")
    println("   final_chi: ", final_chi)
    println("   final_nu: ", final_nu)
    println("   final_S2: ", final_S2)
    println("   ofd_rate: ", ofd_rate)
    println()
    
    println("✅ SUCCESS! Surface Code executes correctly through CAMPS")
    
catch e
    println()
    println("✗ FAILED!")
    println()
    println("Error type: ", typeof(e))
    println("Error message: ", e)
    println()
    println("Full stack trace:")
    showerror(stdout, e, catch_backtrace())
    println()
end

println()
println("="^80)