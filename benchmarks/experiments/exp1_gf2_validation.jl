# CAMPS.jl/benchmarks/experiments/exp1_gf2_validation.jl
# Experiment 1: GF(2) Bond Dimension Prediction Validation
#
# Purpose: Validate that χ = 2^(t - rank) holds across circuit families
# This is a core claim of the CAMPS paper.

# Add parent directory to load path
push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using CAMPS
using Random
using Statistics
using Printf

include(joinpath(@__DIR__, "..", "benchmark_types.jl"))
include(joinpath(@__DIR__, "..", "config.jl"))

"""
    run_gf2_validation_experiment(; config=GF2_QUICK_CONFIG, verbose=true)

Run the GF(2) bond dimension prediction validation experiment.

Tests the formula χ = 2^(t - rank) across multiple circuit families:
1. Independent T gates (H → T, no entangling) - rank = t, χ = 1
2. CNOT chain entangled (H → CNOT chain → T) - rank < t possible
3. CZ pairs entangled - different Pauli structure
4. Random Clifford+T circuits
5. QFT circuits (structured rotations)

# Arguments
- `config::GF2ValidationConfig`: Configuration parameters
- `verbose::Bool`: Whether to print progress

# Returns
- `ExperimentResults`: Complete experiment results with summary
"""
function run_gf2_validation_experiment(;
    config::GF2ValidationConfig = GF2_QUICK_CONFIG,
    verbose::Bool = true
)
    if verbose
        println("="^70)
        println("Experiment 1: GF(2) Bond Dimension Prediction Validation")
        println("="^70)
        println("Testing formula: χ = 2^(t - rank)")
        println()
        println("Configuration:")
        println("  n_qubits: $(config.n_qubits)")
        println("  t_counts: $(config.t_counts)")
        println("  n_seeds: $(config.n_seeds)")
        println("  circuit_types: $(config.circuit_types)")
        println()
    end

    start_time = time()
    results = GF2ValidationResult[]

    # Test each circuit type
    for circuit_type in config.circuit_types
        if verbose
            println("-"^50)
            println("Circuit type: $(circuit_type)")
            println("-"^50)
        end

        type_results = run_circuit_type_validation(
            circuit_type, config; verbose=verbose
        )
        append!(results, type_results)

        if verbose
            n_exact = count(r -> r.exact_match, type_results)
            println("  Exact matches: $(n_exact)/$(length(type_results)) ($(round(100*n_exact/length(type_results), digits=1))%)")
            println()
        end
    end

    wall_time = time() - start_time
    summary = compute_summary(results)

    if verbose
        println("="^70)
        println("Experiment Complete")
        println("="^70)
        println("Total tests: $(length(results))")
        println("Exact matches: $(summary["exact_matches"])/$(summary["n_tests"]) ($(round(100*summary["exact_match_rate"], digits=1))%)")
        println("Mean relative error: $(round(summary["mean_relative_error"], digits=4))")
        println("Wall time: $(round(wall_time, digits=2))s")
        println()
    end

    ExperimentResults(
        "gf2_validation",
        "Validate GF(2) bond dimension prediction: χ = 2^(t - rank)",
        Dict{String,Any}(
            "n_qubits" => config.n_qubits,
            "t_counts" => config.t_counts,
            "n_seeds" => config.n_seeds,
            "circuit_types" => config.circuit_types
        ),
        results,
        summary,
        wall_time
    )
end

"""
    run_circuit_type_validation(circuit_type, config; verbose=true)

Run validation for a specific circuit type.
"""
function run_circuit_type_validation(
    circuit_type::String,
    config::GF2ValidationConfig;
    verbose::Bool = true
)
    results = GF2ValidationResult[]

    for n in config.n_qubits
        for t in config.t_counts
            # Skip invalid combinations
            if t > 2 * n  # More T gates than reasonable
                continue
            end

            for seed in 1:config.n_seeds
                result = validate_single_circuit(
                    circuit_type, n, t, seed; verbose=verbose
                )
                if result !== nothing
                    push!(results, result)
                end
            end
        end
    end

    results
end

"""
    validate_single_circuit(circuit_type, n, t, seed; verbose=false)

Validate GF(2) prediction for a single circuit.
"""
function validate_single_circuit(
    circuit_type::String,
    n::Int,
    t::Int,
    seed::Int;
    verbose::Bool = false
)
    # Generate circuit based on type
    circuit = try
        generate_validation_circuit(circuit_type, n, t, seed)
    catch e
        if verbose
            @warn "Failed to generate circuit" circuit_type n t seed exception=e
        end
        return nothing
    end

    if circuit === nothing
        return nothing
    end

    # Simulate with OFD strategy to test prediction
    try
        result = simulate_circuit(circuit, n;
            strategy = OFDStrategy(),
            verbose = false
        )

        predicted_chi = result.predicted_bond_dim
        actual_chi = result.final_bond_dim

        # Compute GF(2) rank from t and predicted χ
        # χ = 2^(t - rank) => rank = t - log2(χ)
        gf2_rank = predicted_chi > 0 ? t - Int(round(log2(predicted_chi))) : t

        exact_match = (predicted_chi == actual_chi)
        relative_error = abs(predicted_chi - actual_chi) / max(actual_chi, 1)

        if verbose && !exact_match
            @printf("    n=%2d, t=%2d, seed=%d: pred=%d, actual=%d (mismatch)\n",
                    n, t, seed, predicted_chi, actual_chi)
        end

        return GF2ValidationResult(
            n, t, circuit_type, seed,
            predicted_chi, actual_chi, gf2_rank,
            exact_match, relative_error
        )
    catch e
        if verbose
            @warn "Simulation failed" circuit_type n t seed exception=e
        end
        return nothing
    end
end

"""
    generate_validation_circuit(circuit_type, n, t, seed) -> Vector{Gate}

Generate a circuit for GF(2) validation based on type.
"""
function generate_validation_circuit(
    circuit_type::String,
    n::Int,
    t::Int,
    seed::Int
)
    Random.seed!(seed)

    if circuit_type == "independent"
        return generate_independent_t_circuit(n, t)
    elseif circuit_type == "cnot_chain"
        return generate_cnot_chain_circuit(n, t)
    elseif circuit_type == "cz_pairs"
        return generate_cz_pairs_circuit(n, t)
    elseif circuit_type == "random_clifford_t"
        return random_clifford_t_circuit(n, t; seed=seed)
    elseif circuit_type == "qft"
        return generate_qft_approximation(n, t)
    else
        error("Unknown circuit type: $circuit_type")
    end
end

"""
    generate_independent_t_circuit(n, t) -> Vector{Gate}

Generate circuit with independent T gates (H layer, then T on first t qubits).
GF(2) theory predicts: rank = t (all twisted Paulis independent), χ = 1.
"""
function generate_independent_t_circuit(n::Int, t::Int)
    circuit = Gate[]

    # Hadamard layer on all qubits
    for q in 1:n
        push!(circuit, HGate(q))
    end

    # T gates on first t qubits (cycling if t > n)
    for i in 1:t
        q = ((i - 1) % n) + 1
        push!(circuit, TGate(q))
    end

    circuit
end

"""
    generate_cnot_chain_circuit(n, t) -> Vector{Gate}

Generate circuit with CNOT chain before T gates.
This entangles the qubits, potentially reducing GF(2) rank.
"""
function generate_cnot_chain_circuit(n::Int, t::Int)
    circuit = Gate[]

    # Hadamard layer
    for q in 1:n
        push!(circuit, HGate(q))
    end

    # CNOT chain: 1→2, 2→3, ..., (n-1)→n
    for q in 1:(n-1)
        push!(circuit, CNOTGate(q, q+1))
    end

    # T gates on first t qubits
    for i in 1:t
        q = ((i - 1) % n) + 1
        push!(circuit, TGate(q))
    end

    circuit
end

"""
    generate_cz_pairs_circuit(n, t) -> Vector{Gate}

Generate circuit with CZ pairs before T gates.
CZ entangles in Z basis, creating different Pauli structure.
"""
function generate_cz_pairs_circuit(n::Int, t::Int)
    circuit = Gate[]

    # Hadamard layer
    for q in 1:n
        push!(circuit, HGate(q))
    end

    # CZ on pairs: (1,2), (3,4), ...
    for q in 1:2:(n-1)
        push!(circuit, CZGate(q, q+1))
    end

    # T gates
    for i in 1:t
        q = ((i - 1) % n) + 1
        push!(circuit, TGate(q))
    end

    circuit
end

"""
    generate_qft_approximation(n, t) -> Vector{Gate}

Generate approximate QFT with limited T gates.
Uses only the first few non-Clifford rotations.
"""
function generate_qft_approximation(n::Int, t::Int)
    # Get full QFT circuit
    full_qft = qft_circuit(n)

    # Count and limit non-Clifford gates
    circuit = Gate[]
    t_applied = 0

    for gate in full_qft
        if gate isa RotationGate
            # Check if it's a non-Clifford rotation
            if !is_clifford_angle(gate.angle)
                if t_applied < t
                    push!(circuit, gate)
                    t_applied += 1
                end
                # Skip remaining non-Clifford gates
            else
                push!(circuit, gate)
            end
        else
            push!(circuit, gate)
        end
    end

    circuit
end

"""
    is_clifford_angle(angle::Float64) -> Bool

Check if an angle corresponds to a Clifford rotation.
"""
function is_clifford_angle(angle::Float64)
    # Clifford angles are multiples of π/2
    normalized = mod(angle, 2π)
    return any(isapprox(normalized, k * π/2, atol=1e-10) for k in 0:3)
end

# Analysis functions

"""
    analyze_gf2_results(results::Vector{GF2ValidationResult})

Perform detailed analysis of GF(2) validation results.
"""
function analyze_gf2_results(results::Vector{GF2ValidationResult})
    println("\n" * "="^70)
    println("GF(2) Validation Analysis")
    println("="^70)

    # Overall statistics
    n_total = length(results)
    n_exact = count(r -> r.exact_match, results)

    println("\nOverall Statistics:")
    println("  Total tests: $n_total")
    println("  Exact matches: $n_exact ($(round(100*n_exact/n_total, digits=1))%)")

    # By circuit type
    println("\nBy Circuit Type:")
    circuit_types = unique(r.circuit_type for r in results)
    for ct in circuit_types
        ct_results = filter(r -> r.circuit_type == ct, results)
        n_ct = length(ct_results)
        n_ct_exact = count(r -> r.exact_match, ct_results)
        mean_err = mean(r -> r.relative_error, ct_results)

        @printf("  %-20s: %d/%d exact (%.1f%%), mean error: %.4f\n",
                ct, n_ct_exact, n_ct, 100*n_ct_exact/n_ct, mean_err)
    end

    # By qubit count
    println("\nBy Qubit Count:")
    n_qubits = sort(unique(r.n_qubits for r in results))
    for n in n_qubits
        n_results = filter(r -> r.n_qubits == n, results)
        n_tests = length(n_results)
        n_exact_n = count(r -> r.exact_match, n_results)

        @printf("  n=%2d: %d/%d exact (%.1f%%)\n",
                n, n_exact_n, n_tests, 100*n_exact_n/n_tests)
    end

    # By T-count
    println("\nBy T-Count:")
    t_counts = sort(unique(r.t_count for r in results))
    for t in t_counts
        t_results = filter(r -> r.t_count == t, results)
        n_tests = length(t_results)
        n_exact_t = count(r -> r.exact_match, t_results)

        @printf("  t=%2d: %d/%d exact (%.1f%%)\n",
                t, n_exact_t, n_tests, 100*n_exact_t/n_tests)
    end

    # Analyze mismatches
    mismatches = filter(r -> !r.exact_match, results)
    if !isempty(mismatches)
        println("\nMismatch Analysis:")
        println("  Number of mismatches: $(length(mismatches))")

        # Group by circuit type
        for ct in circuit_types
            ct_mismatches = filter(r -> r.circuit_type == ct, mismatches)
            if !isempty(ct_mismatches)
                println("\n  $ct mismatches:")
                for r in ct_mismatches[1:min(5, length(ct_mismatches))]
                    @printf("    n=%d, t=%d: pred=%d, actual=%d\n",
                            r.n_qubits, r.t_count, r.predicted_chi, r.actual_chi)
                end
                if length(ct_mismatches) > 5
                    println("    ... and $(length(ct_mismatches) - 5) more")
                end
            end
        end
    else
        println("\nNo mismatches found - GF(2) prediction is exact!")
    end
end

"""
    get_gf2_scatter_data(results::Vector{GF2ValidationResult})

Extract data for scatter plot of predicted vs actual bond dimension.
"""
function get_gf2_scatter_data(results::Vector{GF2ValidationResult})
    predicted = [r.predicted_chi for r in results]
    actual = [r.actual_chi for r in results]
    circuit_types = [r.circuit_type for r in results]

    (predicted=predicted, actual=actual, circuit_types=circuit_types)
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    results = run_gf2_validation_experiment(config=GF2_QUICK_CONFIG, verbose=true)
    analyze_gf2_results(results.results)
end

export run_gf2_validation_experiment, analyze_gf2_results, get_gf2_scatter_data
