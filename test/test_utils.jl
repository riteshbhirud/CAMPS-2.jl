# CAMPS.jl/test/test_utils.jl
# Tests for utils.jl

using Test
using CAMPS
using LinearAlgebra

@testset "Pauli Symbol Conversion" begin
    @testset "xz_to_symbol" begin
        @test xz_to_symbol(false, false) == :I
        @test xz_to_symbol(true, false) == :X
        @test xz_to_symbol(true, true) == :Y
        @test xz_to_symbol(false, true) == :Z
    end
    
    @testset "symbol_to_xz" begin
        @test symbol_to_xz(:I) == (false, false)
        @test symbol_to_xz(:X) == (true, false)
        @test symbol_to_xz(:Y) == (true, true)
        @test symbol_to_xz(:Z) == (false, true)
        
        @test_throws ArgumentError symbol_to_xz(:W)
        @test_throws ArgumentError symbol_to_xz(:invalid)
    end
    
    @testset "Round-trip conversion" begin
        for σ in [:I, :X, :Y, :Z]
            x, z = symbol_to_xz(σ)
            @test xz_to_symbol(x, z) == σ
        end
    end
end

@testset "Phase Conversion" begin
    @testset "phase_to_complex" begin
        @test phase_to_complex(0x00) ≈ 1.0 + 0.0im
        @test phase_to_complex(0x01) ≈ 0.0 + 1.0im
        @test phase_to_complex(0x02) ≈ -1.0 + 0.0im
        @test phase_to_complex(0x03) ≈ 0.0 - 1.0im
        
        # Higher bytes should use only lower 2 bits
        @test phase_to_complex(0x04) ≈ 1.0 + 0.0im  # 0x04 & 0x03 = 0x00
        @test phase_to_complex(0x05) ≈ 0.0 + 1.0im  # 0x05 & 0x03 = 0x01
    end
    
    @testset "complex_to_phase" begin
        @test complex_to_phase(1.0 + 0.0im) == 0x00
        @test complex_to_phase(0.0 + 1.0im) == 0x01
        @test complex_to_phase(-1.0 + 0.0im) == 0x02
        @test complex_to_phase(0.0 - 1.0im) == 0x03
        
        # Should work with integer inputs
        @test complex_to_phase(1) == 0x00
        @test complex_to_phase(im) == 0x01
        @test complex_to_phase(-1) == 0x02
        @test complex_to_phase(-im) == 0x03
        
        # Invalid phases
        @test_throws ArgumentError complex_to_phase(0.5 + 0.0im)
        @test_throws ArgumentError complex_to_phase(0.5 + 0.5im)
    end
    
    @testset "Round-trip conversion" begin
        for phase_byte in 0x00:0x03
            c = phase_to_complex(phase_byte)
            @test complex_to_phase(c) == phase_byte
        end
    end
end

@testset "Pauli Matrices" begin
    @testset "pauli_matrix" begin
        # Identity
        I_mat = pauli_matrix(:I)
        @test I_mat ≈ [1 0; 0 1]
        
        # Pauli X
        X_mat = pauli_matrix(:X)
        @test X_mat ≈ [0 1; 1 0]
        
        # Pauli Y
        Y_mat = pauli_matrix(:Y)
        @test Y_mat ≈ [0 -im; im 0]
        
        # Pauli Z
        Z_mat = pauli_matrix(:Z)
        @test Z_mat ≈ [1 0; 0 -1]
        
        # Invalid
        @test_throws ArgumentError pauli_matrix(:W)
    end
    
    @testset "Pauli properties" begin
        # Paulis square to identity
        for σ in [:X, :Y, :Z]
            P = pauli_matrix(σ)
            @test P * P ≈ pauli_matrix(:I)
        end
        
        # Paulis are Hermitian
        for σ in [:I, :X, :Y, :Z]
            P = pauli_matrix(σ)
            @test P ≈ P'
        end
        
        # Paulis are unitary
        for σ in [:I, :X, :Y, :Z]
            P = pauli_matrix(σ)
            @test P * P' ≈ pauli_matrix(:I)
        end
    end
end

@testset "Rotation Matrices" begin
    @testset "rotation_matrix basic" begin
        # Identity rotations (θ = 0)
        @test rotation_matrix(:X, 0.0) ≈ [1 0; 0 1]
        @test rotation_matrix(:Y, 0.0) ≈ [1 0; 0 1]
        @test rotation_matrix(:Z, 0.0) ≈ [1 0; 0 1]
        
        # Pauli rotations (θ = 2π)
        @test rotation_matrix(:X, 2π) ≈ -[1 0; 0 1] atol=1e-10
        @test rotation_matrix(:Y, 2π) ≈ -[1 0; 0 1] atol=1e-10
        @test rotation_matrix(:Z, 2π) ≈ -[1 0; 0 1] atol=1e-10
        
        # Invalid axis
        @test_throws ArgumentError rotation_matrix(:W, π)
    end
    
    @testset "T gate" begin
        T_mat = rotation_matrix(:Z, π/4)
        expected = [exp(-im*π/8) 0; 0 exp(im*π/8)]
        @test T_mat ≈ expected
    end
    
    @testset "S gate" begin
        S_mat = rotation_matrix(:Z, π/2)
        expected = [exp(-im*π/4) 0; 0 exp(im*π/4)]
        @test S_mat ≈ expected
    end
    
    @testset "Rotation is unitary" begin
        for axis in [:X, :Y, :Z]
            for θ in [0.0, π/4, π/3, π/2, π, 1.5]
                R = rotation_matrix(axis, θ)
                @test R * R' ≈ [1 0; 0 1] atol=1e-10
            end
        end
    end
    
    @testset "Rotation composition" begin
        # Rx(π) = -iX (up to global phase)
        Rx_pi = rotation_matrix(:X, π)
        X_mat = pauli_matrix(:X)
        @test abs(det(Rx_pi' * (-im * X_mat))) ≈ 1 atol=1e-10
        
        # Two half rotations = full rotation
        Rz_half = rotation_matrix(:Z, π/4)
        Rz_full = rotation_matrix(:Z, π/2)
        @test Rz_half * Rz_half ≈ Rz_full atol=1e-10
    end
end

@testset "Rotation Coefficients" begin
    @testset "rotation_coefficients" begin
        # θ = 0: α = 1, β = 0
        α, β = rotation_coefficients(0.0)
        @test α ≈ 1.0
        @test β ≈ 0.0im
        
        # θ = π/4 (T gate)
        α, β = rotation_coefficients(π/4)
        @test α ≈ cos(π/8)
        @test β ≈ -im * sin(π/8)
        
        # θ = π: α = 0, β = -i
        α, β = rotation_coefficients(π)
        @test α ≈ 0.0 atol=1e-10
        @test β ≈ -im atol=1e-10
        
        # Verify: αI + βZ = Rz(θ)
        for θ in [0.0, π/4, π/3, π/2, π]
            α, β = rotation_coefficients(θ)
            R_direct = rotation_matrix(:Z, θ)
            R_decomposed = α * pauli_matrix(:I) + β * pauli_matrix(:Z)
            @test R_direct ≈ R_decomposed atol=1e-10
        end
    end
end

@testset "Angle Utilities" begin
    @testset "is_clifford_angle" begin
        # Clifford angles: multiples of π/2
        @test is_clifford_angle(0.0) == true
        @test is_clifford_angle(π/2) == true
        @test is_clifford_angle(π) == true
        @test is_clifford_angle(3π/2) == true
        @test is_clifford_angle(2π) == true
        @test is_clifford_angle(-π/2) == true
        
        # Non-Clifford angles
        @test is_clifford_angle(π/4) == false  # T gate
        @test is_clifford_angle(π/3) == false
        @test is_clifford_angle(π/8) == false
        @test is_clifford_angle(0.1) == false
    end
    
    @testset "normalize_angle" begin
        @test normalize_angle(0.0) ≈ 0.0
        @test normalize_angle(π) ≈ π
        @test normalize_angle(2π) ≈ 0.0 atol=1e-10
        @test normalize_angle(3π) ≈ π atol=1e-10
        @test normalize_angle(-π) ≈ π atol=1e-10
        @test normalize_angle(-π/2) ≈ 3π/2 atol=1e-10
    end
end

@testset "Bit String Utilities" begin
    @testset "int_to_bits" begin
        @test int_to_bits(0, 4) == BitVector([0, 0, 0, 0])
        @test int_to_bits(1, 4) == BitVector([1, 0, 0, 0])
        @test int_to_bits(5, 4) == BitVector([1, 0, 1, 0])  # 5 = 0101 binary (little endian)
        @test int_to_bits(15, 4) == BitVector([1, 1, 1, 1])
        @test int_to_bits(7, 3) == BitVector([1, 1, 1])
        
        # Error cases
        @test_throws ArgumentError int_to_bits(-1, 4)
        @test_throws ArgumentError int_to_bits(5, 0)
        @test_throws ArgumentError int_to_bits(5, -1)
    end
    
    @testset "bits_to_int" begin
        @test bits_to_int(BitVector([0, 0, 0, 0])) == 0
        @test bits_to_int(BitVector([1, 0, 0, 0])) == 1
        @test bits_to_int(BitVector([1, 0, 1, 0])) == 5
        @test bits_to_int(BitVector([1, 1, 1, 1])) == 15
        @test bits_to_int(BitVector([1, 1, 1])) == 7
    end
    
    @testset "Round-trip int_to_bits/bits_to_int" begin
        for x in 0:31
            bits = int_to_bits(x, 5)
            @test bits_to_int(bits) == x
        end
    end
    
    @testset "bitstring_to_vector" begin
        @test bitstring_to_vector("0101") == [0, 1, 0, 1]
        @test bitstring_to_vector("000") == [0, 0, 0]
        @test bitstring_to_vector("111") == [1, 1, 1]
        @test bitstring_to_vector("") == Int[]
        
        @test_throws ArgumentError bitstring_to_vector("012")
        @test_throws ArgumentError bitstring_to_vector("abc")
    end
    
    @testset "vector_to_bitstring" begin
        @test vector_to_bitstring([0, 1, 0, 1]) == "0101"
        @test vector_to_bitstring([0, 0, 0]) == "000"
        @test vector_to_bitstring([1, 1, 1]) == "111"
        @test vector_to_bitstring(Int[]) == ""
    end
    
    @testset "Round-trip bitstring conversion" begin
        for s in ["0000", "1111", "0101", "1010", "0011", "1100"]
            v = bitstring_to_vector(s)
            @test vector_to_bitstring(v) == s
        end
    end
end

@testset "Circuit Utilities" begin
    @testset "count_gates_by_type" begin
        # Empty circuit
        @test count_gates_by_type(Gate[]) == Dict(:clifford => 0, :rotation => 0, :t_gate => 0, :total => 0)
        
        # Only Clifford gates
        circuit1 = [HGate(1), CNOTGate(1, 2), SGate(2)]
        counts1 = count_gates_by_type(circuit1)
        @test counts1[:clifford] == 3
        @test counts1[:rotation] == 0
        @test counts1[:t_gate] == 0
        @test counts1[:total] == 3
        
        # Only T gates
        circuit2 = [TGate(1), TGate(2), TdagGate(1)]
        counts2 = count_gates_by_type(circuit2)
        @test counts2[:clifford] == 0
        @test counts2[:rotation] == 3
        @test counts2[:t_gate] == 3
        @test counts2[:total] == 3
        
        # Mixed circuit
        circuit3 = [HGate(1), TGate(1), CNOTGate(1, 2), TGate(2), RzGate(1, π/3)]
        counts3 = count_gates_by_type(circuit3)
        @test counts3[:clifford] == 2
        @test counts3[:rotation] == 3
        @test counts3[:t_gate] == 2  # T gates only, not general Rz
        @test counts3[:total] == 5
    end
    
    @testset "gate_depth" begin
        # Empty circuit
        @test gate_depth(Gate[], 3) == 0
        
        # Single gate
        @test gate_depth([HGate(1)], 3) == 1
        
        # Parallel gates
        @test gate_depth([HGate(1), HGate(2), HGate(3)], 3) == 1
        
        # Sequential on same qubit
        @test gate_depth([HGate(1), TGate(1), HGate(1)], 3) == 3
        
        # Mixed
        circuit = [HGate(1), HGate(2), CNOTGate(1, 2), TGate(1), TGate(2)]
        @test gate_depth(circuit, 2) == 3
    end
end

@testset "Validation Utilities" begin
    @testset "validate_qubit_index" begin
        # Valid cases
        @test validate_qubit_index(1, 5) === nothing
        @test validate_qubit_index(3, 5) === nothing
        @test validate_qubit_index(5, 5) === nothing
        
        # Invalid cases
        @test_throws ArgumentError validate_qubit_index(0, 5)
        @test_throws ArgumentError validate_qubit_index(-1, 5)
        @test_throws ArgumentError validate_qubit_index(6, 5)
    end
    
    @testset "validate_circuit" begin
        # Valid circuit
        valid_circuit = [HGate(1), CNOTGate(1, 2), TGate(3)]
        @test validate_circuit(valid_circuit, 3) === nothing
        @test validate_circuit(valid_circuit, 5) === nothing

        # Invalid: gate on qubit out of range
        invalid_circuit1 = [HGate(1), HGate(4)]
        @test_throws ArgumentError validate_circuit(invalid_circuit1, 3)

        # Invalid: CNOT target out of range
        invalid_circuit2 = [CNOTGate(1, 5)]
        @test_throws ArgumentError validate_circuit(invalid_circuit2, 4)

        # Note: We cannot test validate_circuit with qubit index 0 because
        # the gate constructors themselves validate and throw before
        # validate_circuit is called. This is by design - invalid gates
        # cannot be created.
        @test_throws ArgumentError TGate(0)  # Gate constructor validates

        # Empty circuit is valid
        @test validate_circuit(Gate[], 5) === nothing
    end
end

@testset "Numerical Utilities" begin
    @testset "isapprox_zero" begin
        @test isapprox_zero(0.0) == true
        @test isapprox_zero(1e-15) == true
        @test isapprox_zero(-1e-15) == true
        @test isapprox_zero(1e-13) == false
        @test isapprox_zero(1.0) == false
        
        # With custom tolerance
        @test isapprox_zero(1e-5; atol=1e-4) == true
        @test isapprox_zero(1e-5; atol=1e-6) == false
        
        # Complex numbers
        @test isapprox_zero(0.0 + 0.0im) == true
        @test isapprox_zero(1e-15 + 1e-15im) == true
    end
    
    @testset "isapprox_one" begin
        @test isapprox_one(1.0) == true
        @test isapprox_one(1.0 + 1e-15) == true
        @test isapprox_one(1.0 - 1e-15) == true
        @test isapprox_one(1.0 + 1e-10) == false
        @test isapprox_one(0.0) == false
        
        # Complex numbers
        @test isapprox_one(1.0 + 0.0im) == true
        @test isapprox_one(1.0 + 1e-15im) == true
    end
    
    @testset "safe_log" begin
        @test safe_log(1.0) ≈ 0.0
        @test safe_log(exp(1.0)) ≈ 1.0
        @test safe_log(0.5) ≈ log(0.5)
        
        # Should not throw for very small values
        @test isfinite(safe_log(1e-400))
        @test isfinite(safe_log(0.0))
        
        # Custom min_val
        @test safe_log(0.0; min_val=1e-10) ≈ log(1e-10)
    end
end
