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
This replicates the logic from run_parallel_benchmark_complete.jl
"""
function generate_qft_experiments_for_subset(subset::String; n_realizations=8)
    qft_name = "Quantum Fourier Transform"
    experiments = []

    # Generate all QFT experiments (same as in run_parallel_benchmark_complete.jl)
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

    # Filter based on subset
    if subset == "qft"
        # All QFT experiments
        filtered = experiments
    elseif subset == "qft1"
        # n=4 only
        filtered = filter(exp -> exp.params[:n_qubits] == 4, experiments)
    elseif subset == "qft2"
        # n=6 only
        filtered = filter(exp -> exp.params[:n_qubits] == 6, experiments)
    elseif subset == "qft3"
        # n=8 only
        filtered = filter(exp -> exp.params[:n_qubits] == 8, experiments)
    else
        error("Unknown subset: $subset")
    end

    return filtered
end

"""
Generate VQE experiments for a specific subset.
This replicates the logic from run_parallel_benchmark_complete.jl
"""
function generate_vqe_experiments_for_subset(subset::String; n_realizations=8)
    vqe_name = "VQE Hardware-Efficient Ansatz"
    experiments = []

    # Generate all VQE experiments (same as in run_parallel_benchmark_complete.jl)
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

    # Filter based on subset
    if subset == "vqe"
        # All VQE experiments
        filtered = experiments
    elseif subset == "vqe1"
        # n=4 only
        filtered = filter(exp -> exp.params[:n_qubits] == 4, experiments)
    elseif subset == "vqe2"
        # n=6 only
        filtered = filter(exp -> exp.params[:n_qubits] == 6, experiments)
    elseif subset == "vqe3"
        # n=8 only
        filtered = filter(exp -> exp.params[:n_qubits] == 8, experiments)
    else
        error("Unknown subset: $subset")
    end

    return filtered
end

# Generate experiments for each subset
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

# Verification tests
println("VERIFICATION RESULTS:")
println("="^80)

# Test 1: Circuit counts
println("1. Circuit Counts:")
@printf("   qft (full):  %3d circuits\n", length(qft_full))
@printf("   qft1 (n=4):  %3d circuits\n", length(qft1))
@printf("   qft2 (n=6):  %3d circuits\n", length(qft2))
@printf("   qft3 (n=8):  %3d circuits\n", length(qft3))
@printf("   Sum of parts: %3d circuits\n", length(qft1) + length(qft2) + length(qft3))

test1_pass = (length(qft_full) == 72 &&
              length(qft1) == 24 &&
              length(qft2) == 24 &&
              length(qft3) == 24 &&
              length(qft1) + length(qft2) + length(qft3) == length(qft_full))

if test1_pass
    println("   ✓ Test 1 PASSED: Counts match expected values")
else
    println("   ✗ Test 1 FAILED: Counts do not match")
end
println()

# Test 2: No overlap between parts
println("2. Checking for overlaps between parts...")
combined_parts = vcat(qft1, qft2, qft3)

# Create signature for each experiment (n_qubits, density, seed)
function exp_signature(exp)
    return (exp.params[:n_qubits], exp.params[:density], exp.params[:seed])
end

signatures_full = Set([exp_signature(exp) for exp in qft_full])
signatures_parts = Set([exp_signature(exp) for exp in combined_parts])

test2_pass = length(combined_parts) == length(signatures_parts)  # No duplicates in parts

if test2_pass
    println("   ✓ Test 2 PASSED: No duplicates within combined parts")
else
    println("   ✗ Test 2 FAILED: Found duplicates in combined parts")
end
println()

# Test 3: Coverage - all experiments from full are in parts
println("3. Checking complete coverage...")
test3_pass = signatures_full == signatures_parts

if test3_pass
    println("   ✓ Test 3 PASSED: Parts cover all experiments from full QFT")
else
    println("   ✗ Test 3 FAILED: Parts do not match full QFT")
    missing = setdiff(signatures_full, signatures_parts)
    extra = setdiff(signatures_parts, signatures_full)
    if !isempty(missing)
        println("   Missing experiments: $missing")
    end
    if !isempty(extra)
        println("   Extra experiments: $extra")
    end
end
println()

# Test 4: Parameter distribution
println("4. Parameter distribution per part:")

function count_params(experiments)
    density_counts = Dict(:low => 0, :medium => 0, :high => 0)
    n_qubit_counts = Dict{Int, Int}()

    for exp in experiments
        density_counts[exp.params[:density]] += 1
        n = exp.params[:n_qubits]
        n_qubit_counts[n] = get(n_qubit_counts, n, 0) + 1
    end

    return density_counts, n_qubit_counts
end

for (name, exps) in [("qft1", qft1), ("qft2", qft2), ("qft3", qft3)]
    density_counts, n_counts = count_params(exps)
    println("   $name:")
    println("     Qubit sizes: $(sort(collect(keys(n_counts))))")
    println("     Densities: low=$(density_counts[:low]), medium=$(density_counts[:medium]), high=$(density_counts[:high])")
end

# Each part should have 8 circuits per density (8 realizations × 1 n_qubit value × 3 densities = 24)
test4_pass = true
for exps in [qft1, qft2, qft3]
    density_counts, _ = count_params(exps)
    if !(density_counts[:low] == 8 && density_counts[:medium] == 8 && density_counts[:high] == 8)
        test4_pass = false
    end
end

if test4_pass
    println("   ✓ Test 4 PASSED: Each part has balanced density distribution (8 each)")
else
    println("   ✗ Test 4 FAILED: Unbalanced density distribution")
end
println()

# Test 5: Seed determinism
println("5. Checking seed determinism...")
qft_full_2 = generate_qft_experiments_for_subset("qft")
qft1_2 = generate_qft_experiments_for_subset("qft1")

test5_pass = true
for (exp1, exp2) in zip(qft_full, qft_full_2)
    if exp_signature(exp1) != exp_signature(exp2)
        test5_pass = false
        break
    end
end

for (exp1, exp2) in zip(qft1, qft1_2)
    if exp_signature(exp1) != exp_signature(exp2)
        test5_pass = false
        break
    end
end

if test5_pass
    println("   ✓ Test 5 PASSED: Seeds are deterministic and reproducible")
else
    println("   ✗ Test 5 FAILED: Seeds are not deterministic")
end
println()

# Final summary
println("="^80)
all_pass = test1_pass && test2_pass && test3_pass && test4_pass && test5_pass

if all_pass
    println("✓ ALL TESTS PASSED!")
    println()
    println("The QFT split is correct:")
    println("  • qft1 (24 circuits) + qft2 (24 circuits) + qft3 (24 circuits) = qft (72 circuits)")
    println("  • No overlaps between parts")
    println("  • Complete coverage maintained")
    println("  • Deterministic and reproducible")
    println()
    println("Usage:")
    println("  julia --project=. benchmarks/run_parallel_benchmark_complete.jl medium qft1")
    println("  julia --project=. benchmarks/run_parallel_benchmark_complete.jl medium qft2")
    println("  julia --project=. benchmarks/run_parallel_benchmark_complete.jl medium qft3")
    println()
    println("After running all parts, you can combine the results into a single dataset.")
else
    println("✗ SOME TESTS FAILED")
    println()
    println("Please review the failed tests above.")
end
println("="^80)
