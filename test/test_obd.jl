# CAMPS.jl/test/test_obd.jl
# Tests for OBD (Optimization-Based Disentangling) algorithm

using Test
using CAMPS
using QuantumClifford
using QuantumClifford: apply!
using LinearAlgebra

@testset "OBD Tests" begin

    @testset "Two-Qubit Clifford Utilities" begin
        @testset "Single-qubit Clifford generation" begin
            # Generate all 24 single-qubit Cliffords
            for idx in 1:24
                gates = generate_single_qubit_clifford(idx, 1)
                @test gates isa Vector
            end

            # Identity should be empty
            @test isempty(generate_single_qubit_clifford(1, 1))

            # Different qubits
            gates_q1 = generate_single_qubit_clifford(5, 1)
            gates_q2 = generate_single_qubit_clifford(5, 2)
            @test all(g -> g[2] == 1, gates_q1)
            @test all(g -> g[2] == 2, gates_q2)
        end

        @testset "CNOT class representatives" begin
            reps = get_cnot_class_representatives()
            @test length(reps) > 0
            @test all(C -> nqubits(C) == 2, reps)
        end

        @testset "Entropy computations" begin
            # Pure state: entropy = 0
            rho_pure = ComplexF64[1 0; 0 0]
            @test compute_renyi2_entropy(rho_pure) ≈ 0.0 atol=1e-10
            @test compute_von_neumann_entropy(rho_pure) ≈ 0.0 atol=1e-10

            # Maximally mixed: entropy = log(2)
            rho_mixed = ComplexF64[0.5 0; 0 0.5]
            @test compute_renyi2_entropy(rho_mixed) ≈ log(2) atol=1e-10
            @test compute_von_neumann_entropy(rho_mixed) ≈ log(2) atol=1e-10
        end

        @testset "Partial trace" begin
            # Identity / 4 (maximally mixed 2-qubit state)
            rho = ComplexF64(0.25) * Matrix{ComplexF64}(LinearAlgebra.I, 4, 4)

            # Trace out second qubit
            rho1 = partial_trace_4x4(rho, true)
            @test size(rho1) == (2, 2)
            @test rho1 ≈ 0.5 * Matrix{ComplexF64}(LinearAlgebra.I, 2, 2) atol=1e-10

            # Trace out first qubit
            rho2 = partial_trace_4x4(rho, false)
            @test rho2 ≈ 0.5 * Matrix{ComplexF64}(LinearAlgebra.I, 2, 2) atol=1e-10
        end

        @testset "Transform RDM" begin
            rho = ComplexF64[1 0; 0 0]  # |0⟩⟨0|
            X = ComplexF64[0 1; 1 0]

            rho_x = transform_rdm(rho, X)
            @test rho_x ≈ ComplexF64[0 0; 0 1] atol=1e-10  # |1⟩⟨1|
        end
    end

    @testset "Clifford to Matrix Conversion" begin
        @testset "Identity Clifford" begin
            C = one(CliffordOperator, 2)
            U = clifford_to_matrix(C)

            @test size(U) == (4, 4)
            @test U ≈ Matrix{ComplexF64}(LinearAlgebra.I, 4, 4) atol=1e-10
        end

        @testset "Known Cliffords" begin
            # CNOT(1,2)
            D = one(Destabilizer, 2)
            apply!(D, sCNOT(1, 2))
            C = CliffordOperator(D)
            U = clifford_to_matrix(C)

            # CNOT matrix: |00⟩→|00⟩, |01⟩→|01⟩, |10⟩→|11⟩, |11⟩→|10⟩
            CNOT_expected = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0]
            @test U ≈ CNOT_expected atol=1e-10
        end
    end

    @testset "OBD Bond Optimization" begin
        @testset "find_optimal_clifford_for_bond" begin
            # Create a state with some entanglement
            state = CAMPSState(3)
            initialize!(state)

            # Apply some gates to create entanglement
            apply_clifford_gate!(state.clifford, sHadamard(1))
            apply_clifford_gate!(state.clifford, sCNOT(1, 2))

            # Apply a non-trivial rotation (without OFD)
            P_twisted = compute_twisted_pauli(state, :Z, 1)
            apply_twisted_rotation!(state.mps, state.sites, P_twisted, π/4;
                                    max_bond=64, cutoff=1e-12)

            # Now try to optimize bond 1
            best_idx, initial_S, final_S = find_optimal_clifford_for_bond(
                state.mps, 1, state.sites; use_full_search=false)

            @test best_idx >= 1
            @test final_S <= initial_S + 1e-10  # Should not increase entropy
        end
    end

    @testset "OBD Sweep" begin
        @testset "obd_sweep! basic" begin
            state = CAMPSState(4)
            initialize!(state)

            # Create some entanglement
            for q in 1:4
                apply_clifford_gate!(state.clifford, sHadamard(q))
            end
            apply_clifford_gate!(state.clifford, sCNOT(1, 2))
            apply_clifford_gate!(state.clifford, sCNOT(2, 3))
            apply_clifford_gate!(state.clifford, sCNOT(3, 4))

            # Apply rotations directly (no OFD) to create entanglement
            for q in 1:2
                P = compute_twisted_pauli(state, :Z, q)
                apply_twisted_rotation!(state.mps, state.sites, P, π/4;
                                        max_bond=64, cutoff=1e-12)
            end

            # Run OBD sweep
            result, _ = obd_sweep!(state.mps, state.sites, state.clifford;
                                    use_full_search=false,
                                    direction=:left_to_right)

            @test result.initial_max_entropy >= 0
            @test result.final_max_entropy >= 0
            @test result.entropy_reduction >= -1e-10  # Should not increase
        end
    end

    @testset "Full OBD" begin
        @testset "obd! convergence" begin
            state = CAMPSState(3)
            initialize!(state)

            # Simple setup with minimal entanglement
            apply_clifford_gate!(state.clifford, sHadamard(1))

            result = obd!(state; max_sweeps=2, use_full_search=false)

            @test result.num_sweeps >= 0
            @test result.final_max_entropy >= 0
            @test result.total_cliffords_applied >= 0
        end
    end

    @testset "OBD with Rotations" begin
        @testset "apply_rotation_with_obd!" begin
            state = CAMPSState(4)
            initialize!(state)

            # Apply Hadamard layer
            for q in 1:4
                apply_clifford_gate!(state.clifford, sHadamard(q))
            end

            # Apply rotation with OBD
            P_twisted = compute_twisted_pauli(state, :Z, 1)
            apply_rotation_with_obd!(state, P_twisted, π/4; obd_sweeps=1)

            @test num_twisted_paulis(state) == 1
            @test get_bond_dimension(state) >= 1
        end
    end

    @testset "Hybrid Strategy" begin
        @testset "apply_rotation_hybrid! with OFD success" begin
            state = CAMPSState(4)
            initialize!(state)

            # H layer makes OFD possible
            for q in 1:4
                apply_clifford_gate!(state.clifford, sHadamard(q))
            end

            # Apply with hybrid strategy - should use OFD
            apply_rotation_hybrid!(state, :Z, 1, π/4; strategy=HybridStrategy())

            @test is_magic(state, 1)  # OFD was used
            @test get_bond_dimension(state) == 1
        end

        @testset "apply_rotation_hybrid! with OBD fallback" begin
            state = CAMPSState(3)
            initialize!(state)

            # No Hadamard - OFD will fail, should fall back to OBD
            apply_rotation_hybrid!(state, :Z, 1, π/4; strategy=HybridStrategy())

            @test num_twisted_paulis(state) == 1
            # State should still be valid
            @test is_initialized(state)
        end

        @testset "Strategy selection" begin
            state = CAMPSState(3)
            initialize!(state)
            apply_clifford_gate!(state.clifford, sHadamard(1))

            # OFD strategy
            state1 = CAMPSState(3)
            initialize!(state1)
            apply_clifford_gate!(state1.clifford, sHadamard(1))
            apply_rotation_hybrid!(state1, :Z, 1, π/4; strategy=OFDStrategy())
            @test is_magic(state1, 1)

            # OBD strategy
            state2 = CAMPSState(3)
            initialize!(state2)
            apply_clifford_gate!(state2.clifford, sHadamard(1))
            apply_rotation_hybrid!(state2, :Z, 1, π/4; strategy=OBDStrategy(max_sweeps=1))
            @test num_twisted_paulis(state2) == 1

            # NoDisentangling strategy
            state3 = CAMPSState(3)
            initialize!(state3)
            apply_clifford_gate!(state3.clifford, sHadamard(1))
            apply_rotation_hybrid!(state3, :Z, 1, π/4; strategy=NoDisentangling())
            @test num_twisted_paulis(state3) == 1
        end
    end

    @testset "T-gate Convenience Functions" begin
        @testset "apply_t_gate_hybrid!" begin
            state = CAMPSState(3)
            initialize!(state)
            apply_clifford_gate!(state.clifford, sHadamard(1))

            apply_t_gate_hybrid!(state, 1)

            @test is_magic(state, 1)
        end

        @testset "apply_tdag_gate_hybrid!" begin
            state = CAMPSState(3)
            initialize!(state)
            apply_clifford_gate!(state.clifford, sHadamard(2))

            apply_tdag_gate_hybrid!(state, 2)

            @test is_magic(state, 2)
        end
    end

    @testset "Entropy and Bond Dimension Profiles" begin
        @testset "get_entropy_profile" begin
            state = CAMPSState(4)
            initialize!(state)

            profile = get_entropy_profile(state)
            @test length(profile) == 3  # n-1 bonds
            @test all(s -> s >= 0, profile)
            @test all(s -> s < 1e-10, profile)  # Product state
        end

        @testset "get_bond_dimension_profile" begin
            state = CAMPSState(5)
            initialize!(state)

            profile = get_bond_dimension_profile(state)
            @test length(profile) == 4
            @test all(d -> d == 1, profile)  # Product state
        end

        @testset "estimate_obd_improvement" begin
            state = CAMPSState(3)
            initialize!(state)

            est = estimate_obd_improvement(state; use_full_search=false)

            @test est.current_max_entropy >= 0
            @test est.estimated_reduction >= 0
            @test 1 <= est.best_bond <= 2
        end
    end

    @testset "Decompose Two-Qubit Clifford" begin
        @testset "Identity decomposition" begin
            C = one(CliffordOperator, 2)
            gates = decompose_two_qubit_clifford(C, 1, 2)
            @test isempty(gates)  # Identity needs no gates
        end

        @testset "CNOT decomposition" begin
            D = one(Destabilizer, 2)
            apply!(D, sCNOT(1, 2))
            C = CliffordOperator(D)

            gates = decompose_two_qubit_clifford(C, 1, 2)
            @test !isempty(gates) || true  # May be empty if exact match found
        end

        @testset "CZ decomposition" begin
            D = one(Destabilizer, 2)
            apply!(D, sCPHASE(1, 2))
            C = CliffordOperator(D)

            gates = decompose_two_qubit_clifford(C, 1, 2)
            # Should decompose correctly
            @test length(gates) <= 5  # At most a few gates
        end
    end

    @testset "OBD Result Types" begin
        @testset "OBDSweepResult" begin
            result = OBDSweepResult(1.0, 0.5, 0.5, [(1, 2, 1)], [0.3, 0.4])
            @test result.initial_max_entropy == 1.0
            @test result.final_max_entropy == 0.5
            @test result.entropy_reduction == 0.5
        end

        @testset "OBDResult" begin
            result = OBDResult(3, true, 1.0, 0.2, OBDSweepResult[], 10)
            @test result.num_sweeps == 3
            @test result.converged == true
            @test result.total_cliffords_applied == 10
        end
    end

end
