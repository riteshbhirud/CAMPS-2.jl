# CAMPS.jl/benchmarks/gf2_accuracy.jl
# Benchmark: Validate GF(2) bond dimension prediction accuracy
#
# This benchmark verifies that the GF(2) theory accurately predicts
# bond dimension for various circuit structures.

using CAMPS
using Printf
using Statistics

"""
    run_gf2_accuracy_benchmark(; verbose::Bool=true)

Run the GF(2) accuracy benchmark suite.

Tests the accuracy of GF(2) bond dimension prediction across:
1. Independent T gates (rank = t, χ = 1)
2. Entangled T gates (rank < t, χ > 1)
3. Random circuits with varying structure

Returns a summary of prediction accuracy.
"""
function run_gf2_accuracy_benchmark(; verbose::Bool=true)
    results = Dict{String, Any}()

    if verbose
        println("="^60)
        println("GF(2) Bond Dimension Prediction Accuracy Benchmark")
        println("="^60)
        println()
    end

    # Test 1: Independent T gates
    results["independent"] = benchmark_independent_t_gates(verbose)

    # Test 2: Entangled T gates
    results["entangled"] = benchmark_entangled_t_gates(verbose)

    # Test 3: Random circuits
    results["random"] = benchmark_random_circuits(verbose)

    # Test 4: QFT circuits
    results["qft"] = benchmark_qft_circuits(verbose)

    # Summary
    if verbose
        println()
        println("="^60)
        println("Summary")
        println("="^60)
        for (name, res) in results
            @printf("  %s: %.1f%% exact matches\n", name, 100 * res[:exact_match_rate])
        end
    end

    return results
end

"""
Benchmark independent T gates: After H layer, T gates on separate qubits.
GF(2) theory predicts rank = t, so χ = 2^(t-t) = 1.
"""
function benchmark_independent_t_gates(verbose::Bool)
    if verbose
        println("Test 1: Independent T gates")
        println("-"^40)
    end

    results = []

    for n in [4, 6, 8, 10]
        for t in [1, 2, 3, min(n, 4)]
            # Create circuit: H layer + T gates on first t qubits
            circuit = Gate[]
            for q in 1:n
                push!(circuit, HGate(q))
            end
            for q in 1:t
                push!(circuit, TGate(q))
            end

            # Simulate with OFD (should achieve optimal)
            result = simulate_circuit(circuit, n; strategy=OFDStrategy(), verbose=false)

            predicted = result.predicted_bond_dim
            actual = result.final_bond_dim

            push!(results, (n=n, t=t, predicted=predicted, actual=actual))

            if verbose
                match = predicted == actual ? "✓" : "✗"
                @printf("  n=%2d, t=%d: predicted χ=%d, actual χ=%d %s\n",
                        n, t, predicted, actual, match)
            end
        end
    end

    exact_matches = count(r -> r.predicted == r.actual, results)
    total = length(results)

    if verbose
        @printf("  Exact matches: %d/%d (%.1f%%)\n", exact_matches, total, 100*exact_matches/total)
        println()
    end

    return Dict(
        :results => results,
        :exact_match_rate => exact_matches / total,
        :total_tests => total
    )
end

"""
Benchmark entangled T gates: T gates after entangling layer.
GF(2) rank depends on the Pauli structure after Cliffords.
"""
function benchmark_entangled_t_gates(verbose::Bool)
    if verbose
        println("Test 2: Entangled T gates")
        println("-"^40)
    end

    results = []

    for n in [4, 6, 8]
        # Circuit: H layer + CNOT chain + T gates
        circuit = Gate[]

        # H layer
        for q in 1:n
            push!(circuit, HGate(q))
        end

        # CNOT chain: 1→2, 2→3, ..., (n-1)→n
        for q in 1:(n-1)
            push!(circuit, CNOTGate(q, q+1))
        end

        # T gates on alternating qubits
        for q in 1:2:n
            push!(circuit, TGate(q))
        end

        # Simulate
        result = simulate_circuit(circuit, n; strategy=OFDStrategy(), verbose=false)

        predicted = result.predicted_bond_dim
        actual = result.final_bond_dim

        push!(results, (n=n, predicted=predicted, actual=actual))

        if verbose
            match = predicted == actual ? "✓" : "✗"
            @printf("  n=%2d: predicted χ=%d, actual χ=%d %s\n",
                    n, predicted, actual, match)
        end
    end

    # Another test: CZ entangling
    for n in [4, 6]
        circuit = Gate[]
        for q in 1:n
            push!(circuit, HGate(q))
        end
        for q in 1:2:(n-1)
            push!(circuit, CZGate(q, q+1))
        end
        for q in 1:n
            push!(circuit, TGate(q))
        end

        result = simulate_circuit(circuit, n; strategy=OFDStrategy(), verbose=false)
        predicted = result.predicted_bond_dim
        actual = result.final_bond_dim

        push!(results, (n=n, predicted=predicted, actual=actual))

        if verbose
            match = predicted == actual ? "✓" : "✗"
            @printf("  n=%2d (CZ): predicted χ=%d, actual χ=%d %s\n",
                    n, predicted, actual, match)
        end
    end

    exact_matches = count(r -> r.predicted == r.actual, results)
    total = length(results)

    if verbose
        @printf("  Exact matches: %d/%d (%.1f%%)\n", exact_matches, total, 100*exact_matches/total)
        println()
    end

    return Dict(
        :results => results,
        :exact_match_rate => exact_matches / total,
        :total_tests => total
    )
end

"""
Benchmark random circuits with varying T-gate counts.
"""
function benchmark_random_circuits(verbose::Bool)
    if verbose
        println("Test 3: Random Clifford+T circuits")
        println("-"^40)
    end

    results = []

    for n in [4, 6, 8]
        for t in [2, 4, 6]
            for seed in [42, 123, 456]
                circuit = random_clifford_t_circuit(n, t; seed=seed)

                result = simulate_circuit(circuit, n; strategy=OFDStrategy(), verbose=false)
                predicted = result.predicted_bond_dim
                actual = result.final_bond_dim

                push!(results, (n=n, t=t, seed=seed, predicted=predicted, actual=actual))
            end
        end
    end

    exact_matches = count(r -> r.predicted == r.actual, results)
    total = length(results)

    if verbose
        @printf("  Tested %d random circuits\n", total)
        @printf("  Exact matches: %d/%d (%.1f%%)\n", exact_matches, total, 100*exact_matches/total)

        # Show average ratio
        ratios = [r.predicted / max(r.actual, 1) for r in results]
        @printf("  Mean predicted/actual ratio: %.3f\n", mean(ratios))
        println()
    end

    return Dict(
        :results => results,
        :exact_match_rate => exact_matches / total,
        :total_tests => total
    )
end

"""
Benchmark QFT circuits which have structured non-Clifford rotations.
"""
function benchmark_qft_circuits(verbose::Bool)
    if verbose
        println("Test 4: QFT circuits")
        println("-"^40)
    end

    results = []

    for n in [3, 4, 5, 6]
        circuit = qft_circuit(n)

        # Count non-Clifford gates
        analysis = analyze_circuit(circuit, n)

        # Simulate
        result = simulate_circuit(circuit, n; strategy=HybridStrategy(), verbose=false)
        predicted = result.predicted_bond_dim
        actual = result.final_bond_dim

        push!(results, (n=n, predicted=predicted, actual=actual,
                       n_non_clifford=result.n_non_clifford))

        if verbose
            match = predicted == actual ? "✓" : "✗"
            @printf("  QFT n=%d: non-Clifford=%d, predicted χ=%d, actual χ=%d %s\n",
                    n, result.n_non_clifford, predicted, actual, match)
        end
    end

    exact_matches = count(r -> r.predicted == r.actual, results)
    total = length(results)

    if verbose
        @printf("  Exact matches: %d/%d (%.1f%%)\n", exact_matches, total, 100*exact_matches/total)
        println()
    end

    return Dict(
        :results => results,
        :exact_match_rate => exact_matches / total,
        :total_tests => total
    )
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_gf2_accuracy_benchmark()
end
