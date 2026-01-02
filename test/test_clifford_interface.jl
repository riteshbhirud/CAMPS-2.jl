# CAMPS.jl/test/test_clifford_interface.jl
# Tests for clifford_interface.jl

using Test
using CAMPS
using QuantumClifford

@testset "Clifford Initialization" begin
    @testset "initialize_clifford" begin
        # Basic creation
        C = initialize_clifford(3)
        # Check that it's a Destabilizer type (it's parameterized: Destabilizer{T})
        # Use string matching on type name since isa can have issues with parameterized types
        @test occursin("Destabilizer", string(typeof(C)))
        @test nqubits(C) == 3

        # Different sizes
        for n in [1, 2, 5, 10]
            C = initialize_clifford(n)
            @test nqubits(C) == n
        end
    end
end

@testset "Twisted Pauli Computation" begin
    @testset "Identity Clifford" begin
        # With identity Clifford, twisted Pauli = original Pauli
        C = initialize_clifford(3)
        
        P_z1 = single_z(3, 1)
        P_twisted = commute_pauli_through_clifford(P_z1, C)
        
        # Z should remain Z (up to phase)
        @test get_pauli_at(P_twisted, 1) == :Z
        @test get_pauli_at(P_twisted, 2) == :I
        @test get_pauli_at(P_twisted, 3) == :I
    end
    
    @testset "Hadamard conjugation" begin
        # H · Z · H† = X
        C = initialize_clifford(1)
        apply_clifford_gate!(C, sHadamard(1))
        
        P_z = single_z(1, 1)
        P_twisted = commute_pauli_through_clifford(P_z, C)
        
        @test get_pauli_at(P_twisted, 1) == :X
    end
    
    @testset "Phase gate conjugation" begin
        # S · Z · S† = Z (Z is eigenstate of S)
        C = initialize_clifford(1)
        apply_clifford_gate!(C, sPhase(1))
        
        P_z = single_z(1, 1)
        P_twisted = commute_pauli_through_clifford(P_z, C)
        
        @test get_pauli_at(P_twisted, 1) == :Z
        
        # S · X · S† = Y
        P_x = single_x(1, 1)
        P_twisted_x = commute_pauli_through_clifford(P_x, C)
        
        @test get_pauli_at(P_twisted_x, 1) == :Y
    end
    
    @testset "CNOT propagation" begin
        # CNOT(1,2) · Z₂ · CNOT(1,2)† = Z₁Z₂
        C = initialize_clifford(2)
        apply_clifford_gate!(C, sCNOT(1, 2))
        
        P_z2 = single_z(2, 2)
        P_twisted = commute_pauli_through_clifford(P_z2, C)
        
        @test get_pauli_at(P_twisted, 1) == :Z
        @test get_pauli_at(P_twisted, 2) == :Z
        
        # CNOT(1,2) · Z₁ · CNOT(1,2)† = Z₁ (control unchanged)
        P_z1 = single_z(2, 1)
        P_twisted_z1 = commute_pauli_through_clifford(P_z1, C)
        
        @test get_pauli_at(P_twisted_z1, 1) == :Z
        @test get_pauli_at(P_twisted_z1, 2) == :I
    end
    
    @testset "Complex circuit" begin
        # Build a more complex Clifford circuit
        n = 3
        C = initialize_clifford(n)
        
        # H on all qubits
        for q in 1:n
            apply_clifford_gate!(C, sHadamard(q))
        end
        
        # CNOT ladder
        apply_clifford_gate!(C, sCNOT(1, 2))
        apply_clifford_gate!(C, sCNOT(2, 3))
        
        # After H layer, Z → X for all qubits
        # After CNOT(1,2), X₁ → X₁X₂
        # After CNOT(2,3), X₂ → X₂X₃
        
        # Test that twisted Paulis are non-trivial
        P_z1 = single_z(n, 1)
        P_twisted = commute_pauli_through_clifford(P_z1, C)
        
        # Should have non-trivial support
        @test pauli_weight(P_twisted) >= 1
    end
end

@testset "axis_to_pauli" begin
    n = 3
    
    @test get_pauli_at(axis_to_pauli(:X, 1, n), 1) == :X
    @test get_pauli_at(axis_to_pauli(:X, 1, n), 2) == :I
    
    @test get_pauli_at(axis_to_pauli(:Y, 2, n), 1) == :I
    @test get_pauli_at(axis_to_pauli(:Y, 2, n), 2) == :Y
    
    @test get_pauli_at(axis_to_pauli(:Z, 3, n), 3) == :Z
    
    @test_throws ArgumentError axis_to_pauli(:W, 1, n)
end

@testset "Clifford Gate Application" begin
    @testset "apply_clifford_gate!" begin
        C = initialize_clifford(2)
        
        # Apply Hadamard
        apply_clifford_gate!(C, sHadamard(1))
        
        # Verify by transforming Pauli
        P_z = single_z(2, 1)
        P_twisted = commute_pauli_through_clifford(P_z, C)
        @test get_pauli_at(P_twisted, 1) == :X
    end
    
    @testset "apply_clifford_gates!" begin
        C = initialize_clifford(2)
        
        # Apply multiple gates at once
        gates = [sHadamard(1), sHadamard(2), sCNOT(1, 2)]
        apply_clifford_gates!(C, gates)
        
        # This should be equivalent to applying them one by one
        C2 = initialize_clifford(2)
        apply_clifford_gate!(C2, sHadamard(1))
        apply_clifford_gate!(C2, sHadamard(2))
        apply_clifford_gate!(C2, sCNOT(1, 2))
        
        # Test by comparing twisted Paulis
        P = single_z(2, 1)
        @test commute_pauli_through_clifford(P, C) == commute_pauli_through_clifford(P, C2)
    end
    
    @testset "apply_inverse_gates!" begin
        C = initialize_clifford(2)
        
        # Apply gates then their inverse
        gates = [sHadamard(1), sCNOT(1, 2)]
        apply_clifford_gates!(C, gates)
        apply_inverse_gates!(C, gates)
        
        # Should be back to identity
        P = single_z(2, 1)
        P_twisted = commute_pauli_through_clifford(P, C)
        @test get_pauli_at(P_twisted, 1) == :Z
        @test get_pauli_at(P_twisted, 2) == :I
    end
end

@testset "Symbolic Gate Resolution" begin
    @testset "resolve_symbolic_gate" begin
        @test resolve_symbolic_gate((:H, 1)) == sHadamard(1)
        @test resolve_symbolic_gate((:S, 2)) == sPhase(2)
        @test resolve_symbolic_gate((:Sdag, 1)) == sInvPhase(1)
        @test resolve_symbolic_gate((:X, 3)) == sX(3)
        @test resolve_symbolic_gate((:Y, 1)) == sY(1)
        @test resolve_symbolic_gate((:Z, 2)) == sZ(2)
        @test resolve_symbolic_gate((:CNOT, 1, 2)) == sCNOT(1, 2)
        @test resolve_symbolic_gate((:CZ, 2, 3)) == sCPHASE(2, 3)
        @test resolve_symbolic_gate((:SWAP, 1, 2)) == sSWAP(1, 2)
        
        @test_throws ArgumentError resolve_symbolic_gate((:Unknown, 1))
    end
    
    @testset "resolve_clifford_gate" begin
        h = HGate(1)
        resolved = resolve_clifford_gate(h)
        @test length(resolved) == 1
        @test resolved[1] == sHadamard(1)
        
        cnot = CNOTGate(1, 2)
        resolved = resolve_clifford_gate(cnot)
        @test length(resolved) == 1
        @test resolved[1] == sCNOT(1, 2)
    end
end

@testset "Pauli Operator Utilities" begin
    @testset "get_pauli_at" begin
        P = P"IXYZ"
        @test get_pauli_at(P, 1) == :I
        @test get_pauli_at(P, 2) == :X
        @test get_pauli_at(P, 3) == :Y
        @test get_pauli_at(P, 4) == :Z
    end
    
    @testset "get_pauli_phase" begin
        @test get_pauli_phase(P"XYZ") ≈ 1.0 + 0.0im
        @test get_pauli_phase(P"-XYZ") ≈ -1.0 + 0.0im
        @test get_pauli_phase(P"iXYZ") ≈ 0.0 + 1.0im
        @test get_pauli_phase(P"-iXYZ") ≈ 0.0 - 1.0im
    end
    
    @testset "pauli_weight" begin
        @test pauli_weight(P"___") == 0
        @test pauli_weight(P"X__") == 1
        @test pauli_weight(P"XY_") == 2
        @test pauli_weight(P"XYZ") == 3
    end
    
    @testset "pauli_support" begin
        @test pauli_support(P"___") == Int[]
        @test pauli_support(P"X__") == [1]
        @test pauli_support(P"_Y_") == [2]
        @test pauli_support(P"X_Z") == [1, 3]
        @test pauli_support(P"XYZ") == [1, 2, 3]
    end
    
    @testset "has_x_or_y" begin
        P = P"IXYZ"
        @test has_x_or_y(P, 1) == false  # I
        @test has_x_or_y(P, 2) == true   # X
        @test has_x_or_y(P, 3) == true   # Y
        @test has_x_or_y(P, 4) == false  # Z
    end
    
    @testset "get_xbit_vector" begin
        P = P"IXYZ"
        xb = get_xbit_vector(P)
        @test xb == BitVector([false, true, true, false])
    end
    
    @testset "get_zbit_vector" begin
        P = P"IXYZ"
        zb = get_zbit_vector(P)
        @test zb == BitVector([false, false, true, true])
    end
end

@testset "pauli_to_string" begin
    @test pauli_to_string(P"X") == "+X"
    @test pauli_to_string(P"XYZ") == "+XYZ"
    @test pauli_to_string(P"-XYZ") == "-XYZ"
    @test pauli_to_string(P"iXYZ") == "+iXYZ"
    @test pauli_to_string(P"-iXYZ") == "-iXYZ"
    @test pauli_to_string(P"X_Z") == "+X_Z"
end

@testset "create_pauli_string" begin
    @testset "Basic creation" begin
        # Single qubit
        P = create_pauli_string([:X], 1)
        @test get_pauli_at(P, 1) == :X

        P = create_pauli_string([:Y], 1)
        @test get_pauli_at(P, 1) == :Y

        P = create_pauli_string([:Z], 1)
        @test get_pauli_at(P, 1) == :Z

        P = create_pauli_string([:I], 1)
        @test get_pauli_at(P, 1) == :I
    end

    @testset "Multi-qubit strings" begin
        P = create_pauli_string([:X, :Y, :Z], 3)
        @test get_pauli_at(P, 1) == :X
        @test get_pauli_at(P, 2) == :Y
        @test get_pauli_at(P, 3) == :Z

        # With identities
        P = create_pauli_string([:X, :I, :Z], 3)
        @test get_pauli_at(P, 1) == :X
        @test get_pauli_at(P, 2) == :I
        @test get_pauli_at(P, 3) == :Z

        # All identity
        P = create_pauli_string([:I, :I, :I], 3)
        @test get_pauli_at(P, 1) == :I
        @test get_pauli_at(P, 2) == :I
        @test get_pauli_at(P, 3) == :I
    end

    @testset "Phase" begin
        # Default phase is +1
        P = create_pauli_string([:X, :Y], 2)
        @test get_pauli_phase(P) ≈ 1.0 + 0.0im
    end

    @testset "create_pauli_string_with_phase" begin
        P = create_pauli_string_with_phase([:X, :Y], 2, -1.0 + 0.0im)
        @test get_pauli_at(P, 1) == :X
        @test get_pauli_at(P, 2) == :Y
        @test get_pauli_phase(P) ≈ -1.0 + 0.0im

        P = create_pauli_string_with_phase([:Z], 1, 0.0 + 1.0im)
        @test get_pauli_at(P, 1) == :Z
        @test get_pauli_phase(P) ≈ 0.0 + 1.0im
    end

    @testset "Error handling" begin
        @test_throws ArgumentError create_pauli_string([:X, :Y], 3)  # Length mismatch
        @test_throws ArgumentError create_pauli_string([:X], 2)      # Length mismatch
    end
end

@testset "CAMPSState Clifford Operations" begin
    @testset "apply_clifford_gate_to_state!" begin
        state = CAMPSState(3)
        initialize!(state)

        # Apply Hadamard using CliffordGate
        h = HGate(1)
        apply_clifford_gate_to_state!(state, h)

        # Verify by computing twisted Pauli
        P_twisted = compute_twisted_pauli(state, :Z, 1)
        @test get_pauli_at(P_twisted, 1) == :X
    end
end
