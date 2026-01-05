include("benchmarks/reproduce_figure4.jl")

using CAMPS
using Random
using QuantumClifford: random_clifford

function test_interleaving()
    state = CAMPSState(16; max_bond=2048)
    initialize!(state)

    Random.seed!(42)
    qubits = [rand(1:16) for _ in 1:16]
    clifford_seed = 12345

    println("Analyzing twisted Pauli complexity with interleaving:")
    println("="^60)

    ofd_count = 0
    fail_count = 0
    all_twisted_paulis = PauliOperator[]  # ← NEW! Track ALL twisted Paulis

    for i in 1:16
        apply_brick_wall_clifford_circuit!(state, 2; seed=clifford_seed+i)
        
        P = compute_twisted_pauli(state, :Z, qubits[i])
        push!(all_twisted_paulis, P)  # ← NEW! Store ALL (not just failures)
        
        control = find_disentangling_qubit(P, state.free_qubits)
        
        weight = count(q -> get_pauli_at(P, q) != :I, 1:16)
        has_xy = any(q -> get_pauli_at(P, q) in (:X, :Y), 1:16)
        
        if control !== nothing
            status = "OFD (free=$control)"
            ofd_count += 1
        else
            status = "FAIL"
            fail_count += 1
        end
        
        println("T-gate $i: weight=$weight, has_XY=$has_xy, $status")
        
        apply_t_gate_hybrid!(state, qubits[i]; strategy=HybridStrategy())
    end

    println("="^60)
    println("OFD succeeded: $ofd_count / 16")
    println("OFD failed: $fail_count / 16")
    println("Twisted Paulis in state (failures only): $(length(state.twisted_paulis))")
    println("Free qubits left: $(sum(state.free_qubits))")

    # Analyze ALL 16 twisted Paulis (not just the 4 failures!)
    if !isempty(all_twisted_paulis)  # ← CHANGED from state.twisted_paulis
        gf2 = analyze_gf2_structure(all_twisted_paulis)  # ← CHANGED
        println("\nGF(2) Analysis (ALL 16 twisted Paulis):")  # ← CHANGED
        println("  t_eff: $(gf2.t)")
        println("  rank:  $(gf2.rank)")  
        println("  nu:    $(gf2.nullity)")
    else
        println("\nNo twisted Paulis computed!")
    end

    println("\n" * "="^60)
    println("EXPECTED for correct implementation:")
    println("  OFD succeeded: ~10-12 / 16")
    println("  OFD failed: ~4-6 / 16")
    println("  Twisted Paulis (failures): ~4-6")
    println("  nu (from ALL 16): ~10-14")  # ← UPDATED
    println("="^60)
end

# Run the test
test_interleaving()