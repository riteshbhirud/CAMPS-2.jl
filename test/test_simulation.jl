# CAMPS.jl/test/test_simulation.jl
# Tests for the main simulation interface

using Test
using CAMPS
using QuantumClifford

@testset "Simulation Tests" begin

    @testset "Gate Application" begin
        @testset "apply_gate! CliffordGate" begin
            state = CAMPSState(3)
            initialize!(state)

            # Apply Hadamard
            apply_gate!(state, HGate(1))
            P = compute_twisted_pauli(state, :Z, 1)
            @test get_pauli_at(P, 1) == :X

            # Apply CNOT
            apply_gate!(state, CNOTGate(1, 2))
            P2 = compute_twisted_pauli(state, :Z, 1)
            # After C = CNOT(1,2) · H1, twisted Z1 = C† Z1 C = X1
            # (CNOT doesn't affect Z on control, then H transforms Z to X)
            @test get_pauli_at(P2, 1) == :X
            @test get_pauli_at(P2, 2) == :I  # Z1 doesn't spread to qubit 2
        end

        @testset "apply_gate! RotationGate non-Clifford" begin
            state = CAMPSState(3)
            initialize!(state)
            apply_gate!(state, HGate(1))

            # T gate
            apply_gate!(state, TGate(1))
            @test num_twisted_paulis(state) == 1 || is_magic(state, 1)
        end

        @testset "apply_gate! RotationGate Clifford" begin
            state = CAMPSState(3)
            initialize!(state)

            # Rz(π) = Z gate (Clifford)
            apply_gate!(state, RzGate(1, π))

            # Should be treated as Clifford, no twisted Pauli recorded
            # (unless the implementation doesn't optimize this)
            @test is_initialized(state)
        end
    end

    @testset "rotation_to_clifford" begin
        # π/2 rotations
        @test rotation_to_clifford(RzGate(1, π/2)) isa CliffordGate  # S gate
        @test rotation_to_clifford(RxGate(1, π/2)) isa CliffordGate  # √X
        @test rotation_to_clifford(RyGate(1, π/2)) isa CliffordGate  # √Y

        # π rotations (Paulis)
        @test rotation_to_clifford(RzGate(1, π)) isa CliffordGate
        @test rotation_to_clifford(RxGate(1, π)) isa CliffordGate
        @test rotation_to_clifford(RyGate(1, π)) isa CliffordGate

        # Non-Clifford
        @test rotation_to_clifford(RzGate(1, π/4)) === nothing  # T gate
        @test rotation_to_clifford(RzGate(1, π/3)) === nothing
    end

    @testset "Circuit Simulation" begin
        @testset "simulate_circuit basic" begin
            # Simple circuit: H, CNOT, T
            circuit = Gate[
                HGate(1),
                HGate(2),
                CNOTGate(1, 2),
                TGate(1),
                TGate(2)
            ]

            result = simulate_circuit(circuit, 2)

            @test result.n_qubits == 2
            @test result.n_gates == 5
            @test result.n_clifford == 3  # 2 H + 1 CNOT
            @test result.n_non_clifford == 2  # 2 T
            @test result.final_bond_dim >= 1
            @test is_initialized(result.state)
        end

        @testset "simulate_circuit larger" begin
            circuit = Gate[
                # H layer
                HGate(1), HGate(2), HGate(3), HGate(4),
                # Entangling
                CNOTGate(1, 2), CNOTGate(3, 4),
                # T gates
                TGate(1), TGate(3),
                # More Cliffords
                CNOTGate(2, 3),
                SGate(1)
            ]

            result = simulate_circuit(circuit, 4; verbose=false)

            @test result.n_qubits == 4
            @test result.n_gates == 10
            @test result.n_non_clifford == 2
        end

        @testset "simulate_circuit with strategies" begin
            circuit = Gate[HGate(1), TGate(1)]

            # OFD strategy
            result_ofd = simulate_circuit(circuit, 2; strategy=OFDStrategy())
            @test result_ofd.final_bond_dim == 1  # OFD keeps it low

            # No disentangling
            result_none = simulate_circuit(circuit, 2; strategy=NoDisentangling())
            @test result_none.final_bond_dim >= 1
        end

        @testset "simulate_circuit empty" begin
            result = simulate_circuit(Gate[], 3)
            @test result.n_gates == 0
            @test result.final_bond_dim == 1
        end
    end

    @testset "Observable Extraction" begin
        @testset "sample_state" begin
            state = CAMPSState(3)
            initialize!(state)

            samples = sample_state(state; num_samples=10)
            @test length(samples) == 10
            @test all(s -> length(s) == 3, samples)
            @test all(s -> all(b -> b in [0, 1], s), samples)

            # Initial state |000⟩ should always sample [0,0,0]
            @test all(s -> s == [0, 0, 0], samples)
        end

        @testset "amplitude" begin
            state = CAMPSState(2)
            initialize!(state)

            # Initial state |00⟩
            @test abs(amplitude(state, [0, 0])) ≈ 1.0 atol=1e-10
            @test abs(amplitude(state, [1, 0])) ≈ 0.0 atol=1e-10
            @test abs(amplitude(state, [0, 1])) ≈ 0.0 atol=1e-10
            @test abs(amplitude(state, [1, 1])) ≈ 0.0 atol=1e-10
        end

        @testset "probability" begin
            state = CAMPSState(2)
            initialize!(state)

            @test probability(state, [0, 0]) ≈ 1.0 atol=1e-10
            @test probability(state, [1, 1]) ≈ 0.0 atol=1e-10
        end

        @testset "expectation_value" begin
            state = CAMPSState(2)
            initialize!(state)

            # ⟨00|Z1|00⟩ = 1
            @test expectation_value_z(state, 1) ≈ 1.0 atol=1e-10
            @test expectation_value_z(state, 2) ≈ 1.0 atol=1e-10

            # ⟨00|X1|00⟩ = 0
            @test expectation_value_x(state, 1) ≈ 0.0 atol=1e-10

            # After H: |+0⟩
            apply_gate!(state, HGate(1))
            @test expectation_value_x(state, 1) ≈ 1.0 atol=1e-10
            @test expectation_value_z(state, 1) ≈ 0.0 atol=1e-10
        end

        @testset "expectation_value Pauli string" begin
            state = CAMPSState(2)
            initialize!(state)

            P = P"ZZ"
            exp_val = expectation_value(state, P)
            @test real(exp_val) ≈ 1.0 atol=1e-10  # ⟨00|ZZ|00⟩ = 1
        end
    end

    @testset "Random Circuit Generation" begin
        @testset "random_clifford_t_circuit" begin
            circuit = random_clifford_t_circuit(5, 10; seed=42)

            @test length(circuit) > 10  # Should have Cliffords + T gates
            @test all(g -> g isa Gate, circuit)

            # Count T gates
            t_count = count(g -> g isa RotationGate && g.axis == :Z && abs(g.angle) ≈ π/4, circuit)
            @test t_count == 10
        end

        @testset "random_brickwork_circuit" begin
            circuit = random_brickwork_circuit(4, 3, 0.5; seed=123)

            @test length(circuit) > 0
            @test all(g -> g isa Gate, circuit)

            # Should be reproducible with same seed
            circuit2 = random_brickwork_circuit(4, 3, 0.5; seed=123)
            @test length(circuit) == length(circuit2)
        end
    end

    @testset "Circuit Analysis" begin
        @testset "analyze_circuit" begin
            circuit = Gate[
                HGate(1), SGate(2),
                CNOTGate(1, 2),
                TGate(1), TGate(2),
                RzGate(1, π/3)  # Non-T rotation
            ]

            analysis = analyze_circuit(circuit, 2)

            @test analysis.n_gates == 6
            @test analysis.n_clifford == 3  # H, S, CNOT
            @test analysis.n_t_gates == 2
            @test analysis.n_other_rotation == 1
            @test analysis.depth > 0
        end

        @testset "predict_bond_dimension_for_circuit" begin
            # Circuit with independent T gates after H layer
            circuit = Gate[
                HGate(1), HGate(2), HGate(3),
                TGate(1), TGate(2), TGate(3)
            ]

            prediction = predict_bond_dimension_for_circuit(circuit, 3; strategy=OFDStrategy())

            @test prediction.n_t_gates == 3
            @test prediction.predicted_chi == 1  # Independent, rank = 3
            @test prediction.gf2_rank == 3
        end

        @testset "predict with entangling" begin
            # Circuit with dependent T gates
            circuit = Gate[
                HGate(1), HGate(2),
                CNOTGate(1, 2),  # Makes Z1 → X1X2
                TGate(1), TGate(2)
            ]

            prediction = predict_bond_dimension_for_circuit(circuit, 2; strategy=OFDStrategy())
            @test prediction.n_t_gates == 2
        end
    end

    @testset "SimulationResult Display" begin
        @testset "show methods" begin
            result = SimulationResult(
                4, 10, 7, 3, 2, 1, 4, 2, 0.5,
                CAMPSState(4)
            )

            # Test that show doesn't error
            io = IOBuffer()
            show(io, result)
            @test length(take!(io)) > 0

            show(io, MIME"text/plain"(), result)
            @test length(take!(io)) > 0
        end
    end

    @testset "Full Simulation Workflow" begin
        @testset "Complete Clifford+T simulation" begin
            # Build a non-trivial circuit
            n = 5
            circuit = Gate[]

            # H layer
            for q in 1:n
                push!(circuit, HGate(q))
            end

            # Entangling layer
            for q in 1:(n-1)
                push!(circuit, CNOTGate(q, q+1))
            end

            # T layer
            for q in 1:3
                push!(circuit, TGate(q))
            end

            # More Cliffords
            push!(circuit, SGate(1))
            push!(circuit, SGate(2))

            # Simulate
            result = simulate_circuit(circuit, n; strategy=HybridStrategy())

            # Verify results
            @test result.n_qubits == n
            @test result.n_non_clifford == 3
            @test result.final_bond_dim >= 1
            @test result.final_entropy >= 0

            # Sample from result
            samples = sample_state(result.state; num_samples=5)
            @test length(samples) == 5
        end

        @testset "GF(2) prediction accuracy" begin
            # Test that GF(2) prediction matches actual for simple cases
            circuit = Gate[
                HGate(1), HGate(2), HGate(3), HGate(4),
                TGate(1), TGate(2), TGate(3)
            ]

            result = simulate_circuit(circuit, 4; strategy=OFDStrategy())

            # Independent X paulis: rank = 3, chi = 2^(3-3) = 1
            @test result.predicted_bond_dim == 1
            @test result.final_bond_dim == 1
        end
    end

    @testset "Edge Cases" begin
        @testset "Single qubit" begin
            circuit = Gate[HGate(1), TGate(1)]
            result = simulate_circuit(circuit, 1)

            @test result.n_qubits == 1
            @test result.final_bond_dim == 1
        end

        @testset "All Cliffords" begin
            circuit = Gate[
                HGate(1), SGate(2), CNOTGate(1, 2),
                HGate(2), CZGate(1, 2)
            ]

            result = simulate_circuit(circuit, 2)

            @test result.n_non_clifford == 0
            @test result.final_bond_dim == 1  # Clifford on product state
        end

        @testset "Many T gates" begin
            circuit = Gate[]
            for q in 1:4
                push!(circuit, HGate(q))
            end
            for _ in 1:10
                push!(circuit, TGate(rand(1:4)))
            end

            result = simulate_circuit(circuit, 4; strategy=HybridStrategy())
            @test result.n_non_clifford == 10
            @test is_initialized(result.state)
        end
    end

end
