# CAMPS.jl/test/test_integration.jl
# Integration tests for Phase 2 - complete workflow tests

using Test
using CAMPS
using QuantumClifford

@testset "Phase 2 Integration Tests" begin
    
    @testset "Complete CAMPS workflow" begin
        # This test demonstrates the full Phase 2 capability:
        # 1. Create and initialize state
        # 2. Apply Clifford gates (update Clifford only)
        # 3. Compute twisted Paulis
        # 4. Apply twisted rotations to MPS
        # 5. Verify GF(2) predictions
        
        n = 4
        state = CAMPSState(n)
        initialize!(state)
        
        # Verify initial state
        @test get_bond_dimension(state) == 1
        @test num_free_qubits(state) == n
        @test num_magic_qubits(state) == 0
        @test num_twisted_paulis(state) == 0
        
        # Apply Hadamard layer (Clifford - doesn't touch MPS)
        for q in 1:n
            apply_clifford_gate!(state.clifford, sHadamard(q))
        end
        
        # Verify MPS unchanged
        @test get_bond_dimension(state) == 1
        
        # Compute what a T-gate on qubit 1 would do
        # After H layer: Z₁ → X₁
        P_twisted = compute_twisted_pauli(state, :Z, 1)
        @test get_pauli_at(P_twisted, 1) == :X
        @test pauli_weight(P_twisted) == 1
        
        # Check that X has xbit = 1 (can be disentangled)
        @test has_x_or_y(P_twisted, 1) == true
    end
    
    @testset "Twisted Pauli tracking" begin
        n = 3
        state = CAMPSState(n)
        initialize!(state)
        
        # Apply H on qubit 1
        apply_clifford_gate!(state.clifford, sHadamard(1))
        
        # Compute and record twisted Pauli for T on qubit 1
        P1 = compute_twisted_pauli(state, :Z, 1)
        add_twisted_pauli!(state, P1)
        @test num_twisted_paulis(state) == 1
        
        # Apply CNOT(1,2)
        apply_clifford_gate!(state.clifford, sCNOT(1, 2))
        
        # Compute twisted Pauli for T on qubit 2
        P2 = compute_twisted_pauli(state, :Z, 2)
        add_twisted_pauli!(state, P2)
        @test num_twisted_paulis(state) == 2
        
        # Check GF(2) prediction
        chi_predicted = get_predicted_bond_dimension(state)
        @test chi_predicted >= 1
    end
    
    @testset "MPS manipulation with twisted rotations" begin
        n = 2
        mps, sites = initialize_mps(n)
        
        # Initial state |00⟩
        @test get_mps_bond_dimension(mps) == 1
        
        # Apply X Pauli string to get |10⟩
        P_x = P"X_"
        apply_pauli_string!(mps, P_x, sites)
        @test sample_mps(mps) == [1, 0]
        
        # Create fresh state for rotation test
        mps2, sites2 = initialize_mps(n)
        
        # Apply twisted rotation with Z (diagonal - keeps product state)
        P_z = P"Z_"
        apply_twisted_rotation!(mps2, sites2, P_z, π/4)
        @test get_mps_bond_dimension(mps2) == 1
        
        # Apply twisted rotation with X (creates superposition)
        mps3, sites3 = initialize_mps(n)
        P_x2 = P"X_"
        apply_twisted_rotation!(mps3, sites3, P_x2, π/2)
        # Bond dimension might increase for X rotation on product state
        @test get_mps_bond_dimension(mps3) >= 1
    end
    
    @testset "GF(2) prediction validation" begin
        # Test case 1: Independent X Paulis (after H layer)
        # Should have rank = number of Paulis, chi = 1
        paulis_independent = [P"X___", P"_X__", P"__X_", P"___X"]
        chi1 = predict_bond_dimension(paulis_independent)
        @test chi1 == 1  # rank = 4, chi = 2^(4-4) = 1
        
        # Test case 2: All Z Paulis (no H layer)
        # xbit = 0 for Z, so rank = 0, chi = 2^t
        paulis_z = [P"Z___", P"_Z__", P"__Z_"]
        chi2 = predict_bond_dimension(paulis_z)
        @test chi2 == 8  # rank = 0, chi = 2^(3-0) = 8
        
        # Test case 3: Linearly dependent
        # XX_ XOR _XX = X_X (row 3 = row 1 XOR row 2)
        paulis_dependent = [P"XX_", P"_XX", P"X_X"]
        chi3 = predict_bond_dimension(paulis_dependent)
        @test chi3 == 2  # rank = 2, chi = 2^(3-2) = 2
    end
    
    @testset "Disentanglability analysis" begin
        n = 4
        
        # After H layer, Z → X, which has xbit = 1
        # So free qubits with X/Y can be used for OFD
        P_x1 = single_x(n, 1)  # X on qubit 1
        P_x2 = single_x(n, 2)  # X on qubit 2
        P_z1 = single_z(n, 1)  # Z on qubit 1
        
        free_qubits = BitVector([true, true, true, true])
        
        # X on qubit 1 can be disentangled using qubit 1
        @test can_disentangle(P_x1, free_qubits) == true
        @test find_disentangling_qubit(P_x1, free_qubits) == 1
        
        # Z has no X/Y component, cannot be disentangled
        @test can_disentangle(P_z1, free_qubits) == false
        @test find_disentangling_qubit(P_z1, free_qubits) === nothing
        
        # Count disentanglable
        paulis = [P_x1, P_x2, P_z1]
        @test count_disentanglable(paulis, free_qubits) == 2
    end
    
    @testset "Entanglement entropy" begin
        n = 4
        mps, sites = initialize_mps(n)

        # Product state has zero entropy
        # Use explicit CAMPS module prefix to avoid conflict with QuantumClifford.entanglement_entropy
        for bond in 1:(n-1)
            S = CAMPS.entanglement_entropy(mps, bond)
            @test S < 1e-10
        end

        # After creating entanglement, entropy should increase
        # (This would happen with OBD or non-trivial rotations in Phase 3)
        entropies = CAMPS.entanglement_entropy_all_bonds(mps)
        @test length(entropies) == n - 1
        @test CAMPS.max_entanglement_entropy(mps) < 1e-10
    end
    
    @testset "Sampling and probabilities" begin
        n = 3
        mps, sites = initialize_mps(n)
        
        # |000⟩ initial state
        @test mps_probability(mps, [0, 0, 0], sites) ≈ 1.0 atol=1e-10
        @test mps_probability(mps, [1, 0, 0], sites) ≈ 0.0 atol=1e-10
        
        @test mps_amplitude(mps, [0, 0, 0], sites) ≈ 1.0 atol=1e-10
        
        # Apply X to qubit 2: |010⟩
        apply_pauli_to_mps!(mps, sites, :X, 2)
        @test mps_probability(mps, [0, 1, 0], sites) ≈ 1.0 atol=1e-10
        @test mps_probability(mps, [0, 0, 0], sites) ≈ 0.0 atol=1e-10
    end
    
    @testset "Clifford gate resolution" begin
        # Test that our gate constructors resolve to correct QC gates
        h = HGate(1)
        gates = resolve_clifford_gate(h)
        @test length(gates) == 1
        @test gates[1] == sHadamard(1)
        
        cnot = CNOTGate(1, 2)
        gates2 = resolve_clifford_gate(cnot)
        @test gates2[1] == sCNOT(1, 2)
        
        # Test state application
        state = CAMPSState(3)
        initialize!(state)
        
        apply_clifford_gate_to_state!(state, h)
        P = compute_twisted_pauli(state, :Z, 1)
        @test get_pauli_at(P, 1) == :X  # H·Z·H† = X
    end
    
    @testset "Full Clifford+T simulation setup" begin
        # This demonstrates the Phase 2 capability to set up
        # everything needed for Phase 3's OFD/OBD
        
        n = 5
        state = CAMPSState(n; max_bond=64, cutoff=1e-12)
        initialize!(state)
        
        # Simulate circuit: H on all, then T gates
        # Phase 2 can do the H layer (Clifford)
        for q in 1:n
            apply_clifford_gate!(state.clifford, sHadamard(q))
        end
        
        # Phase 2 can compute and track twisted Paulis
        for q in 1:3  # 3 T-gates
            P = compute_twisted_pauli(state, :Z, q)
            add_twisted_pauli!(state, P)
            
            # Mark the qubit as magic (simulating OFD)
            mark_as_magic!(state, q)
            state.free_qubits[q] = false
        end
        
        # Verify state tracking
        @test num_twisted_paulis(state) == 3
        @test num_free_qubits(state) == 2  # 5 - 3 = 2
        @test num_magic_qubits(state) == 3
        
        # Phase 2 can predict bond dimension
        chi = get_predicted_bond_dimension(state)
        @test chi == 1  # Independent X paulis have rank 3, chi = 2^(3-3) = 1
        
        # Phase 2 has everything needed for Phase 3 to apply actual gates
        @test is_initialized(state)
        @test state.clifford !== nothing
        @test state.mps !== nothing
    end
end

@testset "Edge Cases" begin
    @testset "Single qubit system" begin
        state = CAMPSState(1)
        initialize!(state)
        
        @test get_bond_dimension(state) == 1
        
        apply_clifford_gate!(state.clifford, sHadamard(1))
        P = compute_twisted_pauli(state, :Z, 1)
        @test get_pauli_at(P, 1) == :X
    end
    
    @testset "Large system" begin
        n = 20
        state = CAMPSState(n)
        initialize!(state)
        
        @test n_qubits(state) == n
        @test get_bond_dimension(state) == 1
        
        # Apply Clifford layer
        for q in 1:n
            apply_clifford_gate!(state.clifford, sHadamard(q))
        end
        
        # Compute twisted Pauli
        P = compute_twisted_pauli(state, :Z, 10)
        @test pauli_weight(P) == 1
    end
    
    @testset "Empty twisted Pauli list" begin
        @test predict_bond_dimension(PauliOperator[]) == 1
        
        state = CAMPSState(3)
        initialize!(state)
        @test get_predicted_bond_dimension(state) == 1
    end
end
