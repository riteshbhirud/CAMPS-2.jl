# CAMPS.jl/test/test_gf2.jl
# Tests for gf2.jl

using Test
using CAMPS
using QuantumClifford

@testset "GF(2) Matrix Construction" begin
    @testset "build_gf2_matrix basic" begin
        # Single Pauli
        M = build_gf2_matrix([P"X"])
        @test size(M) == (1, 1)
        @test M[1, 1] == true
        
        # Z has xbit = 0
        M = build_gf2_matrix([P"Z"])
        @test M[1, 1] == false
    end
    
    @testset "build_gf2_matrix multi-qubit" begin
        # IXYZ
        P = P"IXYZ"
        M = build_gf2_matrix([P])
        @test size(M) == (1, 4)
        @test M[1, :] == [false, true, true, false]  # I, X, Y, Z → 0, 1, 1, 0
    end
    
    @testset "build_gf2_matrix multiple paulis" begin
        paulis = [P"XZI", P"ZXI", P"YYI"]
        M = build_gf2_matrix(paulis)
        
        @test size(M) == (3, 3)
        
        # XZI → xbit = [1, 0, 0]
        @test M[1, :] == [true, false, false]
        
        # ZXI → xbit = [0, 1, 0]
        @test M[2, :] == [false, true, false]
        
        # YYI → xbit = [1, 1, 0]
        @test M[3, :] == [true, true, false]
    end
    
    @testset "build_gf2_matrix empty" begin
        M = build_gf2_matrix(PauliOperator[])
        @test size(M) == (0, 0)
    end
    
    @testset "build_gf2_matrix_from_xbits" begin
        xbits = [
            BitVector([true, false, false]),
            BitVector([false, true, false]),
            BitVector([true, true, false])
        ]
        M = build_gf2_matrix_from_xbits(xbits)
        
        @test size(M) == (3, 3)
        @test M[1, :] == [true, false, false]
        @test M[2, :] == [false, true, false]
        @test M[3, :] == [true, true, false]
    end
end

@testset "GF(2) Rank" begin
    @testset "gf2_rank identity" begin
        # Identity matrix has full rank
        M = Bool[true false false; false true false; false false true]
        @test gf2_rank(M) == 3
    end
    
    @testset "gf2_rank zero matrix" begin
        M = Bool[false false; false false; false false]
        @test gf2_rank(M) == 0
    end
    
    @testset "gf2_rank linearly dependent" begin
        # Row 3 = Row 1 XOR Row 2 in GF(2)
        M = Bool[true true false;
                 false true true;
                 true false true]
        @test gf2_rank(M) == 2
    end
    
    @testset "gf2_rank single row" begin
        M = Bool[true true false]
        @test gf2_rank(reshape(M, 1, 3)) == 1
        
        M_zero = Bool[false false false]
        @test gf2_rank(reshape(M_zero, 1, 3)) == 0
    end
    
    @testset "gf2_rank empty" begin
        @test gf2_rank(Matrix{Bool}(undef, 0, 0)) == 0
    end
    
    @testset "gf2_rank! modifies in place" begin
        M = Bool[true true false;
                 false true true;
                 true false true]
        M_copy = copy(M)
        
        r = gf2_rank!(M_copy)
        @test r == 2
        @test M_copy != M  # Modified
    end
end

@testset "Bond Dimension Prediction" begin
    @testset "predict_bond_dimension empty" begin
        @test predict_bond_dimension(PauliOperator[]) == 1
    end
    
    @testset "predict_bond_dimension all Z" begin
        # Z Paulis have xbit = 0, so rank = 0
        paulis = [P"Z", P"Z", P"Z"]
        # All rows are [0], rank = 0, χ = 2^(3-0) = 8
        @test predict_bond_dimension(paulis) == 8
    end
    
    @testset "predict_bond_dimension independent X" begin
        # Independent X Paulis on different qubits
        paulis = [P"X__", P"_X_", P"__X"]
        M = build_gf2_matrix(paulis)
        # M is identity matrix, rank = 3
        # χ = 2^(3-3) = 1
        @test predict_bond_dimension(paulis) == 1
    end
    
    @testset "predict_bond_dimension dependent" begin
        # X₁X₂, X₂X₃, X₁X₃ are linearly dependent
        # (X₁X₂) XOR (X₂X₃) = X₁X₃
        paulis = [P"XX_", P"_XX", P"X_X"]
        
        M = build_gf2_matrix(paulis)
        @test size(M) == (3, 3)
        
        # Check the matrix
        @test M[1, :] == [true, true, false]   # XX_ → 110
        @test M[2, :] == [false, true, true]   # _XX → 011
        @test M[3, :] == [true, false, true]   # X_X → 101
        
        # Row 3 = Row 1 XOR Row 2: 110 XOR 011 = 101 ✓
        # So rank = 2, χ = 2^(3-2) = 2
        @test predict_bond_dimension(paulis) == 2
    end
    
    @testset "predict_bond_dimension from matrix" begin
        M = Bool[true false; true false]  # Duplicate rows
        @test predict_bond_dimension(M) == 2  # rank = 1, χ = 2^(2-1) = 2
    end
end

@testset "Disentanglability Analysis" begin
    @testset "can_disentangle" begin
        P = P"XZI"  # X at position 1
        
        # Qubit 1 is free and has X
        @test can_disentangle(P, BitVector([true, false, false])) == true
        
        # Qubit 1 is not free
        @test can_disentangle(P, BitVector([false, true, true])) == false
        
        # Qubit 3 is free but has I
        @test can_disentangle(P, BitVector([false, false, true])) == false
    end
    
    @testset "find_disentangling_qubit" begin
        P = P"XYI"
        
        # Qubit 1 free → use qubit 1
        @test find_disentangling_qubit(P, BitVector([true, false, false])) == 1
        
        # Qubit 2 free → use qubit 2
        @test find_disentangling_qubit(P, BitVector([false, true, false])) == 2
        
        # Both free → returns first (qubit 1)
        @test find_disentangling_qubit(P, BitVector([true, true, false])) == 1
        
        # None available → nothing
        @test find_disentangling_qubit(P, BitVector([false, false, true])) === nothing
        
        # Z only Pauli
        P_z = P"ZZZ"
        @test find_disentangling_qubit(P_z, BitVector([true, true, true])) === nothing
    end
    
    @testset "count_disentanglable" begin
        paulis = [P"X__", P"_X_", P"ZZZ"]  # First two have X, third doesn't
        free = BitVector([true, true, true])
        
        @test count_disentanglable(paulis, free) == 2
        
        # Only qubit 3 free (but no Pauli has X/Y there)
        free2 = BitVector([false, false, true])
        @test count_disentanglable(paulis, free2) == 0
    end
end

@testset "GF(2) Structure Analysis" begin
    @testset "analyze_gf2_structure basic" begin
        paulis = [P"X_", P"_X"]
        
        result = analyze_gf2_structure(paulis)
        
        @test result.t == 2
        @test result.n == 2
        @test result.rank == 2
        @test result.nullity == 0
        @test result.predicted_chi == 1
    end
    
    @testset "analyze_gf2_structure with dependencies" begin
        # Linearly dependent: row 3 = row 1 XOR row 2
        paulis = [P"XX_", P"_XX", P"X_X"]
        
        result = analyze_gf2_structure(paulis)
        
        @test result.t == 3
        @test result.n == 3
        @test result.rank == 2
        @test result.nullity == 1
        @test result.predicted_chi == 2
    end
    
    @testset "analyze_gf2_structure empty" begin
        result = analyze_gf2_structure(PauliOperator[])
        
        @test result.t == 0
        @test result.n == 0
        @test result.rank == 0
        @test result.nullity == 0
        @test result.predicted_chi == 1
    end
end

@testset "GF(2) Null Space" begin
    @testset "gf2_null_space full rank" begin
        # Full rank → empty null space
        M = Bool[true false; false true]
        ns = gf2_null_space(M)
        @test isempty(ns)
    end
    
    @testset "gf2_null_space rank deficient" begin
        # Row 2 = Row 1 → null space has v = [1,1]
        M = Bool[true true; true true]
        ns = gf2_null_space(M)
        
        @test length(ns) == 1
        @test ns[1] == BitVector([true, true])
    end
    
    @testset "gf2_null_space 3 rows" begin
        # Row 3 = Row 1 XOR Row 2
        M = Bool[true true false;
                 false true true;
                 true false true]
        ns = gf2_null_space(M)
        
        @test length(ns) == 1
        # Null space should contain [1,1,1] (row1 XOR row2 XOR row3 = 0)
        @test ns[1] == BitVector([true, true, true])
    end
end

@testset "Incremental Rank Update" begin
    @testset "incremental_rank_update new independent row" begin
        M = Bool[true false; false true]  # rank 2
        new_row = BitVector([true, true])  # independent
        
        # Reduce to row echelon first
        M_echelon = copy(M)
        gf2_gausselim!(M_echelon)
        
        new_rank, is_indep = incremental_rank_update(M_echelon, new_row)
        
        # [1,1] is independent of identity rows
        # Actually [1,0] + [0,1] = [1,1] in GF(2), so it's dependent!
        @test new_rank == 2
        @test is_indep == false
    end
    
    @testset "incremental_rank_update new dependent row" begin
        M = Bool[true false; false true]
        new_row = BitVector([true, false])  # same as row 1
        
        M_echelon = copy(M)
        gf2_gausselim!(M_echelon)
        
        new_rank, is_indep = incremental_rank_update(M_echelon, new_row)
        
        @test is_indep == false
        @test new_rank == 2
    end
    
    @testset "incremental_rank_update empty matrix" begin
        M = Matrix{Bool}(undef, 0, 3)
        new_row = BitVector([true, false, true])
        
        new_rank, is_indep = incremental_rank_update(M, new_row)
        
        @test new_rank == 1
        @test is_indep == true
    end
    
    @testset "incremental_rank_update zero row" begin
        M = Bool[true false]
        new_row = BitVector([false, false])  # zero row
        
        M_echelon = copy(M)
        gf2_gausselim!(M_echelon)
        
        new_rank, is_indep = incremental_rank_update(reshape(M_echelon, 1, 2), new_row)
        
        @test is_indep == false
        @test new_rank == 1
    end
end

@testset "find_independent_rows" begin
    @testset "Identity matrix" begin
        M = Bool[true false false; false true false; false false true]
        rows = find_independent_rows(M)
        @test sort(rows) == [1, 2, 3]
    end

    @testset "All dependent rows" begin
        # All rows identical
        M = Bool[true true; true true; true true]
        rows = find_independent_rows(M)
        @test length(rows) == 1
        @test rows[1] ∈ [1, 2, 3]
    end

    @testset "Mixed independence with swaps" begin
        # Row 1: [0, 1, 0]
        # Row 2: [1, 0, 0]  <- will become pivot for column 1
        # Row 3: [1, 1, 0]  <- dependent (row2 XOR row1)
        M = Bool[false true false;
                 true false false;
                 true true false]
        rows = find_independent_rows(M)
        @test length(rows) == 2
        # Rows 1 and 2 are independent
        @test sort(rows) == [1, 2]
    end

    @testset "Specific case for permutation tracking" begin
        # This tests that we correctly track which ORIGINAL rows are independent
        # after row swaps during Gaussian elimination
        # Row 3 comes first as pivot but rows 1 and 2 are independent
        M = Bool[false false true;   # Row 1
                 false true false;   # Row 2
                 true false false]   # Row 3 - becomes first pivot
        rows = find_independent_rows(M)
        @test length(rows) == 3
        @test sort(rows) == [1, 2, 3]  # All are independent
    end

    @testset "Empty matrix" begin
        M = Matrix{Bool}(undef, 0, 0)
        @test find_independent_rows(M) == Int[]
    end

    @testset "find_independent_rows_with_basis" begin
        M = Bool[true true false;
                 false true true;
                 true false true]  # Row 3 = Row 1 XOR Row 2

        rows, echelon = find_independent_rows_with_basis(M)

        @test length(rows) == 2
        @test size(echelon) == size(M)
        # Echelon form should have at most 2 non-zero rows
        nonzero_rows = sum(any(echelon[r, :]) for r in 1:3)
        @test nonzero_rows == 2
    end
end

@testset "Integration: GF(2) with Twisted Paulis" begin
    @testset "After Hadamard layer" begin
        # After H on all qubits, Z → X
        # So t T-gates on different qubits give independent X Paulis
        # rank = t, χ = 2^0 = 1
        
        n = 5
        paulis = [single_x(n, k) for k in 1:n]  # X on each qubit
        
        @test predict_bond_dimension(paulis) == 1
    end
    
    @testset "All Z Paulis" begin
        # Without Hadamard, T-gates have Z twisted Paulis
        # xbit = 0 for all, so rank = 0
        # χ = 2^t
        
        n = 4
        t = 3
        paulis = [single_z(n, k) for k in 1:t]
        
        @test predict_bond_dimension(paulis) == 2^t
    end
    
    @testset "Mixed scenario" begin
        # Mix of X and Y Paulis
        paulis = [P"XY_", P"YX_", P"XX_"]
        
        # xbits:
        # XY_ → [1, 1, 0]
        # YX_ → [1, 1, 0]  (same!)
        # XX_ → [1, 1, 0]  (same!)
        
        # All rows identical → rank = 1
        # χ = 2^(3-1) = 4
        @test predict_bond_dimension(paulis) == 4
    end
end
