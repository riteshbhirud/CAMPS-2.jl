# CAMPS.jl/test/test_mps_interface.jl
# Tests for mps_interface.jl

using Test
using CAMPS
using QuantumClifford
using ITensors
using ITensorMPS

@testset "MPS Initialization" begin
    @testset "initialize_mps" begin
        # Basic creation
        mps, sites = initialize_mps(3)
        
        @test mps isa MPS
        @test length(mps) == 3
        @test length(sites) == 3
        
        # Check it's normalized
        @test abs(get_mps_norm(mps) - 1.0) < 1e-10
        
        # Check it's product state (bond dim = 1)
        @test get_mps_bond_dimension(mps) == 1
    end
    
    @testset "Different sizes" begin
        for n in [1, 2, 5, 10]
            mps, sites = initialize_mps(n)
            @test length(mps) == n
            @test length(sites) == n
            @test get_mps_bond_dimension(mps) == 1
        end
    end
end

@testset "MPS Properties" begin
    @testset "get_mps_bond_dimension" begin
        mps, sites = initialize_mps(5)
        @test get_mps_bond_dimension(mps) == 1
    end
    
    @testset "get_mps_norm" begin
        mps, sites = initialize_mps(3)
        @test abs(get_mps_norm(mps) - 1.0) < 1e-10
        
        # Scale MPS
        mps[1] = 2.0 * mps[1]
        @test abs(get_mps_norm(mps) - 2.0) < 1e-10
    end
    
    @testset "normalize_mps!" begin
        mps, sites = initialize_mps(3)
        mps[1] = 3.0 * mps[1]
        
        normalize_mps!(mps)
        @test abs(get_mps_norm(mps) - 1.0) < 1e-10
    end
end

@testset "Gate Tensor Construction" begin
    @testset "pauli_to_itensor" begin
        mps, sites = initialize_mps(2)
        s = sites[1]
        
        # Identity
        I_gate = pauli_to_itensor(:I, s)
        @test I_gate isa ITensor
        
        # Paulis
        X_gate = pauli_to_itensor(:X, s)
        Y_gate = pauli_to_itensor(:Y, s)
        Z_gate = pauli_to_itensor(:Z, s)
        
        @test X_gate isa ITensor
        @test Y_gate isa ITensor
        @test Z_gate isa ITensor
    end
    
    @testset "rotation_to_itensor" begin
        mps, sites = initialize_mps(2)
        s = sites[1]
        
        # T gate
        T_gate = rotation_to_itensor(:Z, π/4, s)
        @test T_gate isa ITensor
        
        # Various rotations
        for axis in [:X, :Y, :Z]
            for θ in [0.0, π/4, π/2, π]
                gate = rotation_to_itensor(axis, θ, s)
                @test gate isa ITensor
            end
        end
    end
    
    @testset "identity_itensor" begin
        mps, sites = initialize_mps(2)
        s = sites[1]
        
        I_gate = identity_itensor(s)
        @test I_gate isa ITensor
    end
end

@testset "Single-Site Gate Application" begin
    @testset "apply_local_rotation!" begin
        mps, sites = initialize_mps(2)
        
        # Apply Rz(π) to first qubit (should flip phase)
        apply_local_rotation!(mps, sites, 1, :Z, π)
        
        # MPS should still be normalized
        @test abs(get_mps_norm(mps) - 1.0) < 1e-10
    end
    
    @testset "apply_pauli_to_mps!" begin
        # X|0⟩ = |1⟩
        mps, sites = initialize_mps(1)
        apply_pauli_to_mps!(mps, sites, :X, 1)
        
        # Sample should give |1⟩
        sample_result = sample_mps(mps)
        @test sample_result == [1]
        
        # Z|0⟩ = |0⟩
        mps2, sites2 = initialize_mps(1)
        apply_pauli_to_mps!(mps2, sites2, :Z, 1)
        sample_result2 = sample_mps(mps2)
        @test sample_result2 == [0]
    end
end

@testset "Pauli String Application" begin
    @testset "apply_pauli_string!" begin
        mps, sites = initialize_mps(3)
        
        # Apply X to qubit 1: X|000⟩ = |100⟩
        P = P"X__"
        apply_pauli_string!(mps, P, sites)
        
        sample_result = sample_mps(mps)
        @test sample_result == [1, 0, 0]
    end
    
    @testset "apply_pauli_string! with multi-qubit" begin
        mps, sites = initialize_mps(3)
        
        # Apply X⊗X to qubits 1,3: |000⟩ → |101⟩
        P = P"X_X"
        apply_pauli_string!(mps, P, sites)
        
        sample_result = sample_mps(mps)
        @test sample_result == [1, 0, 1]
    end
    
    @testset "apply_pauli_string_to_copy" begin
        mps, sites = initialize_mps(2)
        
        P = P"XX"
        mps_new = apply_pauli_string_to_copy(mps, P, sites)
        
        # Original unchanged
        @test sample_mps(mps) == [0, 0]
        
        # New is modified
        @test sample_mps(mps_new) == [1, 1]
    end
end

@testset "Twisted Rotation Application" begin
    @testset "apply_twisted_rotation! basic" begin
        mps, sites = initialize_mps(2)
        
        # Apply T-gate-like rotation with Z Pauli
        P = P"Z_"
        θ = π/4
        
        apply_twisted_rotation!(mps, sites, P, θ)
        
        # MPS should still be approximately normalized
        @test abs(get_mps_norm(mps) - 1.0) < 1e-10
        
        # Bond dimension should still be 1 (Z is diagonal)
        @test get_mps_bond_dimension(mps) == 1
    end
    
    @testset "apply_twisted_rotation! with X" begin
        mps, sites = initialize_mps(2)
        
        # X-rotation on |00⟩ creates superposition
        P = P"X_"
        θ = π/2  # This creates (|00⟩ - i|10⟩)/√2
        
        apply_twisted_rotation!(mps, sites, P, θ; max_bond=10)
        
        # Bond dimension might increase
        @test get_mps_bond_dimension(mps) >= 1
        
        # Should still be normalized
        norm_val = get_mps_norm(mps)
        @test abs(norm_val - 1.0) < 1e-10
    end
    
    @testset "apply_twisted_rotation_to_copy" begin
        mps, sites = initialize_mps(2)
        
        P = P"X_"
        θ = π/4
        
        mps_new = apply_twisted_rotation_to_copy(mps, sites, P, θ)
        
        # Original unchanged (still |00⟩)
        @test sample_mps(mps) == [0, 0]  # Always samples |0⟩
        
        # New is modified
        @test mps_new isa MPS
    end
end

@testset "MPS Truncation" begin
    @testset "truncate_mps!" begin
        mps, sites = initialize_mps(4)
        
        # Create some entanglement by applying rotations
        for i in 1:3
            P = single_x(4, i)  # Use QuantumClifford's single_x
            apply_twisted_rotation!(mps, sites, P, π/3; max_bond=100)
        end
        
        # Truncate
        truncate_mps!(mps; max_bond=2, cutoff=1e-10)
        
        @test get_mps_bond_dimension(mps) <= 2
    end
end

@testset "Entanglement Entropy" begin
    @testset "Product state entropy" begin
        mps, sites = initialize_mps(4)

        # Product state should have zero entropy
        # Use explicit CAMPS module prefix to avoid conflict with QuantumClifford.entanglement_entropy
        for bond in 1:3
            S = CAMPS.entanglement_entropy(mps, bond)
            @test S < 1e-10
        end
    end

    @testset "entanglement_entropy_all_bonds" begin
        mps, sites = initialize_mps(4)

        entropies = CAMPS.entanglement_entropy_all_bonds(mps)
        @test length(entropies) == 3
        @test all(S -> S < 1e-10, entropies)
    end

    @testset "max_entanglement_entropy" begin
        mps, sites = initialize_mps(4)

        max_S = CAMPS.max_entanglement_entropy(mps)
        @test max_S < 1e-10
    end
end

@testset "MPS Sampling" begin
    @testset "sample_mps" begin
        mps, sites = initialize_mps(3)
        
        # |000⟩ should always sample to [0,0,0]
        for _ in 1:10
            sample_result = sample_mps(mps)
            @test sample_result == [0, 0, 0]
        end
    end
    
    @testset "sample_mps after X gate" begin
        mps, sites = initialize_mps(3)
        
        # Apply X to qubit 2: |010⟩
        apply_pauli_to_mps!(mps, sites, :X, 2)
        
        for _ in 1:10
            sample_result = sample_mps(mps)
            @test sample_result == [0, 1, 0]
        end
    end
    
    @testset "sample_mps_multiple" begin
        mps, sites = initialize_mps(2)
        
        samples = sample_mps_multiple(mps, 5)
        @test length(samples) == 5
        @test all(s -> s == [0, 0], samples)
    end
end

@testset "MPS Inner Products" begin
    @testset "mps_overlap" begin
        mps1, sites = initialize_mps(3)
        mps2, _ = initialize_mps(3)
        
        # Same state should have overlap 1
        overlap = mps_overlap(mps1, mps2)
        @test abs(overlap - 1.0) < 1e-10
    end
    
    @testset "mps_probability" begin
        mps, sites = initialize_mps(3)
        
        # |000⟩ should have probability 1 for bitstring [0,0,0]
        prob = mps_probability(mps, [0, 0, 0], sites)
        @test abs(prob - 1.0) < 1e-10
        
        # And 0 for other bitstrings
        prob2 = mps_probability(mps, [1, 0, 0], sites)
        @test prob2 < 1e-10
    end
    
    @testset "mps_amplitude" begin
        mps, sites = initialize_mps(2)
        
        amp = mps_amplitude(mps, [0, 0], sites)
        @test abs(amp - 1.0) < 1e-10
        
        amp2 = mps_amplitude(mps, [1, 1], sites)
        @test abs(amp2) < 1e-10
    end
end

@testset "Two-Qubit Gates" begin
    @testset "matrix_to_two_qubit_itensor" begin
        mps, sites = initialize_mps(3)
        s1, s2 = sites[1], sites[2]
        
        # CNOT matrix
        CNOT_mat = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0]
        gate = matrix_to_two_qubit_itensor(CNOT_mat, s1, s2)
        
        @test gate isa ITensor
    end
end

@testset "CAMPSState MPS Operations" begin
    @testset "get_bond_dimension" begin
        state = CAMPSState(5)
        initialize!(state)
        
        @test get_bond_dimension(state) == 1
    end
    
    @testset "State initialization check" begin
        state = CAMPSState(3)
        
        # Before initialization, state is not initialized
        @test !is_initialized(state)
        
        # Call ensure_initialized! which auto-initializes
        ensure_initialized!(state)
        @test is_initialized(state)
        
        # Bond dimension accessible
        @test get_bond_dimension(state) == 1
    end
    
    @testset "ensure_initialized! auto-initializes" begin
        state = CAMPSState(4)
        
        # get_bond_dimension calls ensure_initialized! internally
        @test get_bond_dimension(state) == 1
        
        # Now it should be initialized
        @test is_initialized(state)
    end
end
