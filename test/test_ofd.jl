# CAMPS.jl/test/test_ofd.jl
# Tests for OFD (Optimization-Free Disentangling) algorithm

using Test
using CAMPS
using QuantumClifford

@testset "OFD Tests" begin

    @testset "Disentangler Gate Construction" begin
        @testset "build_controlled_pauli_gate" begin
            # CX (CNOT)
            gate_x = build_controlled_pauli_gate(:X, 1, 2)
            @test gate_x == sCNOT(1, 2)

            # CZ
            gate_z = build_controlled_pauli_gate(:Z, 1, 2)
            @test gate_z == sCPHASE(1, 2)

            # CY (decomposed)
            gate_y = build_controlled_pauli_gate(:Y, 1, 2)
            @test gate_y isa Vector
            @test length(gate_y) == 3  # S†, CNOT, S
        end

        @testset "build_disentangler_gates" begin
            # Simple case: X on control qubit only
            P_x = P"X__"
            gates = build_disentangler_gates(P_x, 1)
            @test isempty(gates)  # No other qubits to control

            # X on qubit 1, Z on qubit 2 → need CZ(1,2)
            P_xz = P"XZ_"
            gates = build_disentangler_gates(P_xz, 1)
            gates_flat = flatten_gate_sequence(gates)
            @test length(gates_flat) == 1
            @test gates_flat[1] == sCPHASE(1, 2)

            # X on qubit 1, X on qubit 2 → need CNOT(1,2)
            P_xx = P"XX_"
            gates = build_disentangler_gates(P_xx, 1)
            gates_flat = flatten_gate_sequence(gates)
            @test length(gates_flat) == 1
            @test gates_flat[1] == sCNOT(1, 2)

            # More complex: XYZ (control on qubit 1)
            P_xyz = P"XYZ"
            gates = build_disentangler_gates(P_xyz, 1)
            gates_flat = flatten_gate_sequence(gates)
            # Should have CY(1,2) = [S†, CNOT, S] and CZ(1,3)
            @test length(gates_flat) == 4  # 3 for CY + 1 for CZ
        end

        @testset "flatten_gate_sequence" begin
            mixed = [[1, 2], 3, [4, 5, 6]]
            flat = flatten_gate_sequence(mixed)
            @test flat == [1, 2, 3, 4, 5, 6]

            simple = [1, 2, 3]
            @test flatten_gate_sequence(simple) == [1, 2, 3]

            @test flatten_gate_sequence([]) == []
        end
    end

    @testset "OFD Application" begin
        @testset "apply_ofd! basic" begin
            # Setup: 3 qubits, H on qubit 1, then T on qubit 1
            # After H, Z→X, so T gate twisted Pauli is X on qubit 1
            state = CAMPSState(3)
            initialize!(state)

            # Apply Hadamard to qubit 1
            apply_clifford_gate!(state.clifford, sHadamard(1))

            # Compute twisted Pauli for T gate on qubit 1
            P_twisted = compute_twisted_pauli(state, :Z, 1)
            @test get_pauli_at(P_twisted, 1) == :X

            # Apply OFD
            control = find_disentangling_qubit(P_twisted, state.free_qubits)
            @test control == 1

            apply_ofd!(state, P_twisted, π/4, control)

            # Verify state changes
            @test state.free_qubits[1] == false
            @test state.magic_qubits[1] == true
            @test num_twisted_paulis(state) == 1
            @test get_bond_dimension(state) == 1  # OFD keeps bond dim low
        end

        @testset "try_apply_ofd!" begin
            state = CAMPSState(4)
            initialize!(state)

            # Apply H layer
            for q in 1:4
                apply_clifford_gate!(state.clifford, sHadamard(q))
            end

            # Try OFD on qubit 1
            P_twisted = compute_twisted_pauli(state, :Z, 1)
            success, _ = try_apply_ofd!(state, P_twisted, π/4)
            @test success == true

            # Now qubit 1 is magic, try again on qubit 2
            P_twisted2 = compute_twisted_pauli(state, :Z, 2)
            success2, _ = try_apply_ofd!(state, P_twisted2, π/4)
            @test success2 == true

            # Check state
            @test num_magic_qubits(state) == 2
            @test num_free_qubits(state) == 2
        end

        @testset "OFD fails when no X/Y available" begin
            state = CAMPSState(3)
            initialize!(state)

            # No Hadamard - Z stays Z, which has no X component
            P_twisted = compute_twisted_pauli(state, :Z, 1)
            @test get_pauli_at(P_twisted, 1) == :Z

            # OFD should fail
            success, _ = try_apply_ofd!(state, P_twisted, π/4)
            @test success == false
            @test num_magic_qubits(state) == 0
        end
    end

    @testset "T-gate OFD convenience functions" begin
        @testset "apply_t_gate_ofd!" begin
            state = CAMPSState(5)
            initialize!(state)

            # Apply H layer
            for q in 1:5
                apply_clifford_gate!(state.clifford, sHadamard(q))
            end

            # Apply T gate via OFD
            success, _ = apply_t_gate_ofd!(state, 1)
            @test success == true
            @test is_magic(state, 1)
            @test !is_free(state, 1)
        end

        @testset "apply_tdag_gate_ofd!" begin
            state = CAMPSState(3)
            initialize!(state)
            apply_clifford_gate!(state.clifford, sHadamard(1))

            success, _ = apply_tdag_gate_ofd!(state, 1)
            @test success == true
        end
    end

    @testset "OFDS (Sequential OFD)" begin
        @testset "apply_ofds! all succeed" begin
            state = CAMPSState(5)
            initialize!(state)

            # H layer makes all Z→X
            for q in 1:5
                apply_clifford_gate!(state.clifford, sHadamard(q))
            end

            # Apply T gates on qubits 1-3
            result = apply_ofds!(state, [1, 2, 3])

            @test result.num_applied == 3
            @test result.num_failed == 0
            @test result.final_free_qubits == 2
            @test result.final_magic_qubits == 3
            @test result.actual_chi == 1  # OFD keeps bond dim at 1
        end

        @testset "apply_ofds! with entangling" begin
            state = CAMPSState(4)
            initialize!(state)

            # H on all
            for q in 1:4
                apply_clifford_gate!(state.clifford, sHadamard(q))
            end

            # CNOT layer
            apply_clifford_gate!(state.clifford, sCNOT(1, 2))
            apply_clifford_gate!(state.clifford, sCNOT(3, 4))

            # Now twisted Paulis have non-trivial structure
            result = apply_ofds!(state, [1, 2])

            @test result.num_applied >= 1  # Should succeed for some
        end
    end

    @testset "OFD Analysis" begin
        @testset "analyze_ofd_applicability" begin
            state = CAMPSState(4)
            initialize!(state)

            # Before H: Z stays Z, no OFD possible
            analysis1 = analyze_ofd_applicability(state, 1)
            @test analysis1.can_apply == false
            @test analysis1.control_qubit === nothing
            @test get_pauli_at(analysis1.twisted_pauli, 1) == :Z

            # After H: Z→X, OFD possible
            apply_clifford_gate!(state.clifford, sHadamard(1))
            analysis2 = analyze_ofd_applicability(state, 1)
            @test analysis2.can_apply == true
            @test analysis2.control_qubit == 1
            @test get_pauli_at(analysis2.twisted_pauli, 1) == :X
        end

        @testset "count_ofd_applicable" begin
            state = CAMPSState(6)
            initialize!(state)

            # H on first 3 qubits
            for q in 1:3
                apply_clifford_gate!(state.clifford, sHadamard(q))
            end

            # T gates on qubits 1-4
            analysis = count_ofd_applicable(state, [1, 2, 3, 4])

            @test analysis.total == 4
            @test analysis.ofd_possible == 3  # First 3 have H
            @test analysis.ofd_impossible == 1  # Qubit 4 has no H
        end
    end

    @testset "Generate OFD Circuit" begin
        @testset "generate_ofd_circuit" begin
            # X on qubit 1, Z on qubit 2
            P = P"XZ"
            circuit = generate_ofd_circuit(P, 1, π/4)

            # Should have CZ(1,2) and Rx on qubit 1
            @test length(circuit) >= 2
            @test circuit[1] isa CliffordGate  # CZ
            @test circuit[end] isa RotationGate  # Rx
            @test circuit[end].axis == :X
            @test circuit[end].angle ≈ π/4
        end
    end

    @testset "Magic State Properties" begin
        @testset "t_gate_magic_state" begin
            α, β = t_gate_magic_state()

            # Check normalization
            @test abs(α)^2 + abs(β)^2 ≈ 1.0 atol=1e-10

            # Check values
            @test abs(α) ≈ cos(π/8) atol=1e-10
            @test abs(β) ≈ sin(π/8) atol=1e-10

            # β should be pure imaginary
            @test abs(real(β)) < 1e-10
        end

        @testset "magic_state_vector" begin
            # X rotation
            v_x = magic_state_vector(:X, π/4)
            @test length(v_x) == 2
            @test abs(v_x[1])^2 + abs(v_x[2])^2 ≈ 1.0 atol=1e-10

            # Z rotation (stays in |0⟩ with phase)
            v_z = magic_state_vector(:Z, π/4)
            @test abs(v_z[2]) < 1e-10  # Still in |0⟩

            # Y rotation
            v_y = magic_state_vector(:Y, π/2)
            @test abs(v_y[1]) ≈ abs(v_y[2]) atol=1e-10  # Equal superposition
        end
    end

    @testset "Clifford Update Correctness" begin
        @testset "OFD preserves Clifford structure" begin
            # Apply OFD and verify the Clifford tableau is correctly updated
            state = CAMPSState(3)
            initialize!(state)

            # H on qubit 1
            apply_clifford_gate!(state.clifford, sHadamard(1))

            # Record initial Clifford effect on Z2
            P_z2_before = compute_twisted_pauli(state, :Z, 2)

            # Apply OFD for T on qubit 1
            P_t1 = compute_twisted_pauli(state, :Z, 1)
            apply_ofd!(state, P_t1, π/4, 1)

            # Compute twisted Pauli for Z2 after OFD
            P_z2_after = compute_twisted_pauli(state, :Z, 2)

            # For simple case (X on qubit 1 only), disentangler is identity
            # So P_z2 should be unchanged
            @test get_pauli_at(P_z2_before, 2) == get_pauli_at(P_z2_after, 2)
        end

        @testset "OFD with non-trivial disentangler" begin
            state = CAMPSState(3)
            initialize!(state)

            # Apply CNOT(1,2) first, then H on qubits 1 and 2
            # This gives Z1 → X1X2 (CNOT spreads the X after H transforms Z→X)
            apply_clifford_gate!(state.clifford, sCNOT(1, 2))
            apply_clifford_gate!(state.clifford, sHadamard(1))
            apply_clifford_gate!(state.clifford, sHadamard(2))

            # Now Z1 → X1X2 (has X on both qubits 1 and 2)
            P_t1 = compute_twisted_pauli(state, :Z, 1)
            @test get_pauli_at(P_t1, 1) == :X
            @test get_pauli_at(P_t1, 2) == :X

            # OFD should use qubit 1 as control and apply CNOT(1,2) as disentangler
            control = find_disentangling_qubit(P_t1, state.free_qubits)
            @test control == 1

            apply_ofd!(state, P_t1, π/4, control)

            # After OFD, subsequent twisted Paulis should account for D†
            # This is a more complex verification - just check state is valid
            @test is_magic(state, 1)
            @test is_initialized(state)
        end
    end

    @testset "GF(2) Prediction Consistency" begin
        @testset "OFD maintains rank structure" begin
            state = CAMPSState(4)
            initialize!(state)

            # H layer
            for q in 1:4
                apply_clifford_gate!(state.clifford, sHadamard(q))
            end

            # Apply T gates via OFD
            for q in 1:3
                success, _ = apply_t_gate_ofd!(state, q)
                @test success
            end

            # GF(2) prediction should match actual
            predicted = get_predicted_bond_dimension(state)
            actual = get_bond_dimension(state)

            # With independent X paulis, rank = 3, chi = 2^(3-3) = 1
            @test predicted == 1
            @test actual == 1
        end
    end

end
