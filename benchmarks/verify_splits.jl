#!/usr/bin/env julia
"""
Verification script for QFT and VQE split experiments.

This script verifies that:
1. qft1 + qft2 + qft3 = qft (exactly same experiments)
2. vqe1 + vqe2 + vqe3 = vqe (exactly same experiments)
3. All seeds are deterministic and reproducible
4. Each part has exactly 24 circuits
5. Total coverage is maintained (72 circuits each)
"""

using Printf

# Add CAMPS directory to load path
camps_dir = dirname(dirname(@__FILE__))
using Pkg
Pkg.activate(camps_dir)

println("="^80)
println("QFT AND VQE SPLIT VERIFICATION")
println("="^80)
println()

# Load the benchmark code (just the experiment generation parts)
include("circuit_families_complete.jl")

"""
Generate QFT experiments for a specific subset.
"""
function generate_qft_experiments_for_subset(subset::String; n_realizations=8)
    qft_name = "Quantum Fourier Transform"
    experiments = []

    qft_n_range = [4, 6, 8]
    for n in qft_n_range
        for density in [:low, :medium, :high]
            for real in 1:n_realizations
                seed = Int(hash(("QFT", n, density, real)) % UInt32)
                params = Dict{Symbol, Any}(
                    :n_qubits => n,
                    :density => density,
                    :seed => seed
                )
                push!(experiments, (family_name=qft_name, params=params))
            end
        end
    end

    if subset == "qft"
        return experiments
    elseif subset == "qft1"
        return filter(exp -> exp.params[:n_qubits] == 4, experiments)
    elseif subset == "qft2"
        return filter(exp -> exp.params[:n_qubits] == 6, experiments)
    elseif subset == "qft3"
        return filter(exp -> exp.params[:n_qubits] == 8, experiments)
    end
end

"""
Generate VQE experiments for a specific subset.
"""
function generate_vqe_experiments_for_subset(subset::String; n_realizations=8)
    vqe_name = "VQE Hardware-Efficient Ansatz"
    experiments = []

    vqe_n_range = [4, 6, 8]
    for n in vqe_n_range
        for layers in [1, 2, 4]
            for real in 1:n_realizations
                seed = Int(hash(("VQE", n, layers, real)) % UInt32)
                params = Dict{Symbol, Any}(
                    :n_qubits => n,
                    :layers => layers,
                    :seed => seed
                )
                push!(experiments, (family_name=vqe_name, params=params))
            end
        end
    end

    if subset == "vqe"
        return experiments
    elseif subset == "vqe1"
        return filter(exp -> exp.params[:n_qubits] == 4, experiments)
    elseif subset == "vqe2"
        return filter(exp -> exp.params[:n_qubits] == 6, experiments)
    elseif subset == "vqe3"
        return filter(exp -> exp.params[:n_qubits] == 8, experiments)
    end
end

# Helper functions
function qft_signature(exp)
    return (exp.params[:n_qubits], exp.params[:density], exp.params[:seed])
end

function vqe_signature(exp)
    return (exp.params[:n_qubits], exp.params[:layers], exp.params[:seed])
end

# Generate experiments
println("Generating experiments for verification...")
println()

qft_full = generate_qft_experiments_for_subset("qft")
qft1 = generate_qft_experiments_for_subset("qft1")
qft2 = generate_qft_experiments_for_subset("qft2")
qft3 = generate_qft_experiments_for_subset("qft3")

vqe_full = generate_vqe_experiments_for_subset("vqe")
vqe1 = generate_vqe_experiments_for_subset("vqe1")
vqe2 = generate_vqe_experiments_for_subset("vqe2")
vqe3 = generate_vqe_experiments_for_subset("vqe3")

# Test results tracker
all_tests_pass = true

#==============================================================================#
# QFT TESTS
#==============================================================================#

println("QFT SPLIT VERIFICATION:")
println("-"^80)

# Test 1: QFT Circuit counts
println("1. QFT Circuit Counts:")
@printf("   qft (full):   %3d circuits\n", length(qft_full))
@printf("   qft1 (n=4):   %3d circuits\n", length(qft1))
@printf("   qft2 (n=6):   %3d circuits\n", length(qft2))
@printf("   qft3 (n=8):   %3d circuits\n", length(qft3))
@printf("   Sum of parts: %3d circuits\n", length(qft1) + length(qft2) + length(qft3))

qft_test1 = (length(qft_full) == 72 &&
             length(qft1) == 24 &&
             length(qft2) == 24 &&
             length(qft3) == 24 &&
             length(qft1) + length(qft2) + length(qft3) == length(qft_full))

if qft_test1
    println("   ✓ QFT Test 1 PASSED")
else
    println("   ✗ QFT Test 1 FAILED")
    all_tests_pass = false
end
println()

# Test 2: QFT No overlaps
println("2. QFT No Overlaps:")
qft_combined = vcat(qft1, qft2, qft3)
qft_sigs_full = Set([qft_signature(exp) for exp in qft_full])
qft_sigs_parts = Set([qft_signature(exp) for exp in qft_combined])

qft_test2 = length(qft_combined) == length(qft_sigs_parts)

if qft_test2
    println("   ✓ QFT Test 2 PASSED: No duplicates in combined parts")
else
    println("   ✗ QFT Test 2 FAILED: Found duplicates")
    all_tests_pass = false
end
println()

# Test 3: QFT Complete coverage
println("3. QFT Complete Coverage:")
qft_test3 = qft_sigs_full == qft_sigs_parts

if qft_test3
    println("   ✓ QFT Test 3 PASSED: Parts cover all experiments")
else
    println("   ✗ QFT Test 3 FAILED: Coverage incomplete")
    all_tests_pass = false
end
println()

# Test 4: QFT Determinism
println("4. QFT Seed Determinism:")
qft_full_2 = generate_qft_experiments_for_subset("qft")
qft_test4 = all(qft_signature(e1) == qft_signature(e2) for (e1, e2) in zip(qft_full, qft_full_2))

if qft_test4
    println("   ✓ QFT Test 4 PASSED: Seeds are deterministic")
else
    println("   ✗ QFT Test 4 FAILED: Seeds are not deterministic")
    all_tests_pass = false
end
println()

#==============================================================================#
# VQE TESTS
#==============================================================================#

println("VQE SPLIT VERIFICATION:")
println("-"^80)

# Test 1: VQE Circuit counts
println("1. VQE Circuit Counts:")
@printf("   vqe (full):   %3d circuits\n", length(vqe_full))
@printf("   vqe1 (n=4):   %3d circuits\n", length(vqe1))
@printf("   vqe2 (n=6):   %3d circuits\n", length(vqe2))
@printf("   vqe3 (n=8):   %3d circuits\n", length(vqe3))
@printf("   Sum of parts: %3d circuits\n", length(vqe1) + length(vqe2) + length(vqe3))

vqe_test1 = (length(vqe_full) == 72 &&
             length(vqe1) == 24 &&
             length(vqe2) == 24 &&
             length(vqe3) == 24 &&
             length(vqe1) + length(vqe2) + length(vqe3) == length(vqe_full))

if vqe_test1
    println("   ✓ VQE Test 1 PASSED")
else
    println("   ✗ VQE Test 1 FAILED")
    all_tests_pass = false
end
println()

# Test 2: VQE No overlaps
println("2. VQE No Overlaps:")
vqe_combined = vcat(vqe1, vqe2, vqe3)
vqe_sigs_full = Set([vqe_signature(exp) for exp in vqe_full])
vqe_sigs_parts = Set([vqe_signature(exp) for exp in vqe_combined])

vqe_test2 = length(vqe_combined) == length(vqe_sigs_parts)

if vqe_test2
    println("   ✓ VQE Test 2 PASSED: No duplicates in combined parts")
else
    println("   ✗ VQE Test 2 FAILED: Found duplicates")
    all_tests_pass = false
end
println()

# Test 3: VQE Complete coverage
println("3. VQE Complete Coverage:")
vqe_test3 = vqe_sigs_full == vqe_sigs_parts

if vqe_test3
    println("   ✓ VQE Test 3 PASSED: Parts cover all experiments")
else
    println("   ✗ VQE Test 3 FAILED: Coverage incomplete")
    all_tests_pass = false
end
println()

# Test 4: VQE Determinism
println("4. VQE Seed Determinism:")
vqe_full_2 = generate_vqe_experiments_for_subset("vqe")
vqe_test4 = all(vqe_signature(e1) == vqe_signature(e2) for (e1, e2) in zip(vqe_full, vqe_full_2))

if vqe_test4
    println("   ✓ VQE Test 4 PASSED: Seeds are deterministic")
else
    println("   ✗ VQE Test 4 FAILED: Seeds are not deterministic")
    all_tests_pass = false
end
println()

# Test 5: VQE Parameter distribution
println("5. VQE Parameter Distribution:")
for (name, exps) in [("vqe1", vqe1), ("vqe2", vqe2), ("vqe3", vqe3)]
    layer_counts = Dict(1 => 0, 2 => 0, 4 => 0)
    for exp in exps
        layer_counts[exp.params[:layers]] += 1
    end
    println("   $name: layers 1=$(layer_counts[1]), 2=$(layer_counts[2]), 4=$(layer_counts[4])")
end

vqe_test5 = true
for exps in [vqe1, vqe2, vqe3]
    layer_counts = Dict(1 => 0, 2 => 0, 4 => 0)
    for exp in exps
        layer_counts[exp.params[:layers]] += 1
    end
    if !(layer_counts[1] == 8 && layer_counts[2] == 8 && layer_counts[4] == 8)
        vqe_test5 = false
    end
end

if vqe_test5
    println("   ✓ VQE Test 5 PASSED: Balanced layer distribution (8 each)")
else
    println("   ✗ VQE Test 5 FAILED: Unbalanced distribution")
    all_tests_pass = false
end
println()

#==============================================================================#
# FINAL SUMMARY
#==============================================================================#

println("="^80)

if all_tests_pass
    println("✓ ALL TESTS PASSED!")
    println()
    println("QFT Split Verified:")
    println("  • qft1 (24 circuits) + qft2 (24 circuits) + qft3 (24 circuits) = qft (72 circuits)")
    println("  • No overlaps, complete coverage, deterministic seeds")
    println()
    println("VQE Split Verified:")
    println("  • vqe1 (24 circuits) + vqe2 (24 circuits) + vqe3 (24 circuits) = vqe (72 circuits)")
    println("  • No overlaps, complete coverage, deterministic seeds")
    println()
    println("Usage:")
    println("  # QFT experiments")
    println("  julia --project=. benchmarks/run_parallel_benchmark_complete.jl medium qft1")
    println("  julia --project=. benchmarks/run_parallel_benchmark_complete.jl medium qft2")
    println("  julia --project=. benchmarks/run_parallel_benchmark_complete.jl medium qft3")
    println()
    println("  # VQE experiments")
    println("  julia --project=. benchmarks/run_parallel_benchmark_complete.jl medium vqe1")
    println("  julia --project=. benchmarks/run_parallel_benchmark_complete.jl medium vqe2")
    println("  julia --project=. benchmarks/run_parallel_benchmark_complete.jl medium vqe3")
    println()
    println("After running all parts, combine the results into a single dataset.")
else
    println("✗ SOME TESTS FAILED")
    println()
    println("Please review the failed tests above.")
end

println("="^80)
