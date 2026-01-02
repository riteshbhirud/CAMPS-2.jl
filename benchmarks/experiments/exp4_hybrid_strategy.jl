# CAMPS.jl/benchmarks/experiments/exp4_hybrid_strategy.jl
# Experiment 4: Hybrid Strategy Benefits
#
# Purpose: Show that Hybrid strategy combines best of OFD and OBD
# Key result: Hybrid achieves OFD performance when possible, falls back to OBD

push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using CAMPS
using Random
using Statistics
using Printf

include(joinpath(@__DIR__, "..", "benchmark_types.jl"))
include(joinpath(@__DIR__, "..", "config.jl"))

"""
    run_hybrid_strategy_experiment(; config=HYBRID_QUICK_CONFIG, verbose=true)

Run the hybrid strategy benefits experiment.

Tests different circuit types to show when each strategy excels:
1. OFD-favorable: H layer → T gates (OFD achieves optimal χ=1)
2. OFD-unfavorable: T gates without H (OFD fails, OBD needed)
3. Mixed: Some qubits with H, some without
4. Hardware-efficient ansatz: Realistic VQE-style circuits

# Arguments
- `config::HybridBenefitConfig`: Configuration parameters
- `verbose::Bool`: Whether to print progress

# Returns
- `ExperimentResults`: Complete experiment results with summary
"""
function run_hybrid_strategy_experiment(;
    config::HybridBenefitConfig = HYBRID_QUICK_CONFIG,
    verbose::Bool = true
)
    if verbose
        println("="^70)
        println("Experiment 4: Hybrid Strategy Benefits")
        println("="^70)
        println()
        println("Configuration:")
        println("  n_qubits: $(config.n_qubits)")
        println("  circuit_types: $(config.circuit_types)")
        println("  n_seeds: $(config.n_seeds)")
        println("  n_timing_samples: $(config.n_timing_samples)")
        println()
    end

    start_time = time()
    all_results = StrategyComparisonResult[]

    strategies = [
        ("OFD", OFDStrategy()),
        ("OBD", OBDStrategy(max_sweeps=3)),
        ("Hybrid", HybridStrategy())
    ]

    for circuit_type in config.circuit_types
        if verbose
            println("-"^50)
            println("Circuit type: $(circuit_type)")
            println("-"^50)
        end

        type_results = benchmark_circuit_type(
            circuit_type, config, strategies; verbose=verbose
        )
        append!(all_results, type_results)
    end

    wall_time = time() - start_time
    summary = compute_hybrid_summary(all_results, config.circuit_types)

    if verbose
        println()
        println("="^70)
        println("Experiment Complete")
        println("="^70)
        println("Total comparisons: $(length(all_results))")
        println("Wall time: $(round(wall_time, digits=2))s")
        print_hybrid_summary(summary)
    end

    ExperimentResults(
        "hybrid_strategy",
        "Analysis of Hybrid strategy benefits across circuit types",
        Dict{String,Any}(
            "n_qubits" => config.n_qubits,
            "circuit_types" => config.circuit_types,
            "n_seeds" => config.n_seeds,
            "n_timing_samples" => config.n_timing_samples
        ),
        all_results,
        summary,
        wall_time
    )
end

"""
    benchmark_circuit_type(circuit_type, config, strategies; verbose=true)

Benchmark all strategies on a specific circuit type.
"""
function benchmark_circuit_type(
    circuit_type::String,
    config::HybridBenefitConfig,
    strategies::Vector{Tuple{String,T}} where T;
    verbose::Bool = true
)
    results = StrategyComparisonResult[]

    for n in config.n_qubits
        for seed in 1:config.n_seeds
            circuit = generate_hybrid_test_circuit(circuit_type, n, seed)

            if circuit === nothing
                continue
            end

            # Count T gates
            t_count = count(g -> g isa RotationGate && abs(g.angle) ≈ π/4, circuit)

            strategy_results = Dict{String,SingleRunResult}()

            for (strat_name, strategy) in strategies
                times = Float64[]
                local result

                for _ in 1:config.n_timing_samples
                    GC.gc()
                    t_start = time()

                    try
                        result = simulate_circuit(circuit, n;
                            strategy = strategy,
                            verbose = false
                        )
                    catch e
                        if verbose
                            @warn "Simulation failed" strat_name circuit_type n seed
                        end
                        break
                    end

                    push!(times, time() - t_start)
                end

                if length(times) < config.n_timing_samples
                    continue
                end

                mean_time = mean(times)

                ofd_success_rate = if result.n_non_clifford > 0
                    result.n_ofd_applied / result.n_non_clifford
                else
                    1.0
                end

                strategy_results[strat_name] = SingleRunResult(
                    circuit_name = circuit_type,
                    circuit_params = Dict{String,Any}("n" => n, "seed" => seed),
                    n_qubits = n,
                    n_gates = result.n_gates,
                    n_clifford = result.n_clifford,
                    n_non_clifford = result.n_non_clifford,
                    t_count = t_count,
                    strategy = strat_name,
                    wall_time_s = mean_time,
                    bond_dim = result.final_bond_dim,
                    predicted_bond_dim = result.predicted_bond_dim,
                    max_entropy = result.final_entropy,
                    n_ofd_applied = result.n_ofd_applied,
                    n_obd_applied = result.n_obd_applied,
                    memory_bytes = 0,
                    gf2_rank = 0,
                    ofd_success_rate = ofd_success_rate
                )
            end

            if !isempty(strategy_results)
                push!(results, StrategyComparisonResult(
                    circuit_type, n, t_count, seed, strategy_results
                ))

                if verbose
                    @printf("  n=%d, seed=%d, t=%d:\n", n, seed, t_count)
                    for (name, sr) in strategy_results
                        @printf("    %-8s: time=%.4fs, χ=%d, OFD=%.0f%%\n",
                                name, sr.wall_time_s, sr.bond_dim,
                                100 * sr.ofd_success_rate)
                    end
                end
            end
        end
    end

    results
end

"""
    generate_hybrid_test_circuit(circuit_type, n, seed) -> Vector{Gate}

Generate circuits that test different strategy advantages.
"""
function generate_hybrid_test_circuit(circuit_type::String, n::Int, seed::Int)
    Random.seed!(seed)
    t = max(2, n ÷ 2)  # Reasonable T count

    if circuit_type == "ofd_favorable"
        return generate_ofd_favorable_circuit(n, t)
    elseif circuit_type == "ofd_unfavorable"
        return generate_ofd_unfavorable_circuit(n, t)
    elseif circuit_type == "mixed"
        return generate_mixed_circuit(n, t)
    elseif circuit_type == "hardware_efficient"
        return generate_hardware_efficient_circuit(n, seed)
    else
        error("Unknown circuit type: $circuit_type")
    end
end

"""
    generate_ofd_favorable_circuit(n, t) -> Vector{Gate}

Circuit where OFD should achieve optimal χ=1.
Pattern: H on all qubits → T on first t qubits.
"""
function generate_ofd_favorable_circuit(n::Int, t::Int)
    circuit = Gate[]

    # Hadamard on all qubits creates X components
    for q in 1:n
        push!(circuit, HGate(q))
    end

    # T gates - with H preceding, OFD should find free qubits
    for q in 1:min(t, n)
        push!(circuit, TGate(q))
    end

    circuit
end

"""
    generate_ofd_unfavorable_circuit(n, t) -> Vector{Gate}

Circuit where OFD cannot succeed - needs OBD.
Pattern: T gates applied directly without H (Z stays Z).
"""
function generate_ofd_unfavorable_circuit(n::Int, t::Int)
    circuit = Gate[]

    # No H gates - qubits stay in computational basis
    # T|0⟩ = |0⟩ (global phase), but for non-trivial states...

    # Apply X to create |1⟩ states
    for q in 1:(n÷2)
        push!(circuit, XGate(q))
    end

    # Apply T - twisted Pauli is Z, no X/Y for OFD
    for q in 1:min(t, n)
        push!(circuit, TGate(q))
    end

    circuit
end

"""
    generate_mixed_circuit(n, t) -> Vector{Gate}

Circuit with both OFD-favorable and unfavorable qubits.
Some qubits get H (OFD possible), others don't (OBD needed).
"""
function generate_mixed_circuit(n::Int, t::Int)
    circuit = Gate[]

    # H only on odd qubits
    for q in 1:2:n
        push!(circuit, HGate(q))
    end

    # Entangling layer
    for q in 1:(n-1)
        push!(circuit, CNOTGate(q, q+1))
    end

    # T on all qubits - some will use OFD, others OBD
    for q in 1:min(t, n)
        push!(circuit, TGate(q))
    end

    circuit
end

"""
    generate_hardware_efficient_circuit(n, seed) -> Vector{Gate}

Hardware-efficient ansatz circuit typical of VQE.
Alternating rotation and entangling layers.
"""
function generate_hardware_efficient_circuit(n::Int, seed::Int)
    Random.seed!(seed)

    circuit = Gate[]
    layers = 2

    for layer in 1:layers
        # Rotation layer: Ry + Rz
        for q in 1:n
            # Ry - always non-Clifford
            θy = rand() * 2π
            push!(circuit, RotationGate(q, :Y, θy))

            # Rz - mix of Clifford and non-Clifford
            θz = rand() * 2π
            push!(circuit, RotationGate(q, :Z, θz))
        end

        # Entangling layer: CNOT ladder
        for q in 1:(n-1)
            push!(circuit, CNOTGate(q, q+1))
        end
    end

    circuit
end

"""
    compute_hybrid_summary(results, circuit_types) -> Dict

Compute summary statistics for hybrid experiment.
"""
function compute_hybrid_summary(
    results::Vector{StrategyComparisonResult},
    circuit_types::Vector{String}
)
    summary = Dict{String,Any}()

    for ct in circuit_types
        ct_results = filter(r -> r.circuit_name == ct, results)

        if isempty(ct_results)
            continue
        end

        ct_summary = Dict{String,Any}()

        for strat in ["OFD", "OBD", "Hybrid"]
            strat_data = [r.results[strat] for r in ct_results
                          if haskey(r.results, strat)]

            if !isempty(strat_data)
                ct_summary["$(strat)_mean_time"] = mean(d -> d.wall_time_s, strat_data)
                ct_summary["$(strat)_mean_chi"] = mean(d -> d.bond_dim, strat_data)
                ct_summary["$(strat)_ofd_rate"] = mean(d -> d.ofd_success_rate, strat_data)
            end
        end

        # Compute relative performance
        if haskey(ct_summary, "OFD_mean_time") && haskey(ct_summary, "OBD_mean_time")
            ct_summary["ofd_vs_obd_speedup"] = ct_summary["OBD_mean_time"] / ct_summary["OFD_mean_time"]
        end

        if haskey(ct_summary, "Hybrid_mean_time") && haskey(ct_summary, "OBD_mean_time")
            ct_summary["hybrid_vs_obd_speedup"] = ct_summary["OBD_mean_time"] / ct_summary["Hybrid_mean_time"]
        end

        summary[ct] = ct_summary
    end

    # Overall OFD success analysis
    all_ofd_rates = Float64[]
    for r in results
        if haskey(r.results, "OFD")
            push!(all_ofd_rates, r.results["OFD"].ofd_success_rate)
        end
    end

    if !isempty(all_ofd_rates)
        summary["overall_ofd_success_rate"] = mean(all_ofd_rates)
    end

    summary
end

"""
    print_hybrid_summary(summary::Dict)

Print formatted summary of hybrid experiment.
"""
function print_hybrid_summary(summary::Dict)
    println("\nHybrid Strategy Analysis:")
    println("-"^60)

    # Header
    @printf("%-20s %10s %10s %10s %12s\n",
            "Circuit Type", "OFD Time", "Hybrid", "OBD Time", "Hybrid χ")
    println("-"^60)

    for (ct, ct_summary) in summary
        if ct == "overall_ofd_success_rate"
            continue
        end

        ofd_time = get(ct_summary, "OFD_mean_time", NaN)
        hybrid_time = get(ct_summary, "Hybrid_mean_time", NaN)
        obd_time = get(ct_summary, "OBD_mean_time", NaN)
        hybrid_chi = get(ct_summary, "Hybrid_mean_chi", NaN)

        @printf("%-20s %10.4f %10.4f %10.4f %12.1f\n",
                ct, ofd_time, hybrid_time, obd_time, hybrid_chi)
    end

    println("-"^60)

    # Key insights
    println("\nKey Findings:")

    if haskey(summary, "ofd_favorable")
        ofd_fav = summary["ofd_favorable"]
        if haskey(ofd_fav, "ofd_vs_obd_speedup")
            @printf("  OFD-favorable: OFD is %.1fx faster than OBD\n",
                    ofd_fav["ofd_vs_obd_speedup"])
        end
    end

    if haskey(summary, "ofd_unfavorable")
        ofd_unfav = summary["ofd_unfavorable"]
        if haskey(ofd_unfav, "OFD_ofd_rate")
            @printf("  OFD-unfavorable: OFD success rate = %.1f%%\n",
                    100 * ofd_unfav["OFD_ofd_rate"])
        end
    end

    if haskey(summary, "mixed")
        mixed = summary["mixed"]
        if haskey(mixed, "Hybrid_ofd_rate")
            @printf("  Mixed circuits: Hybrid achieves %.1f%% OFD success\n",
                    100 * mixed["Hybrid_ofd_rate"])
        end
    end

    if haskey(summary, "overall_ofd_success_rate")
        @printf("\n  Overall OFD applicability: %.1f%%\n",
                100 * summary["overall_ofd_success_rate"])
    end

    println("\nRecommendation: Use Hybrid as default strategy")
    println("  - Gets OFD speed when possible")
    println("  - Falls back to OBD when needed")
end

"""
    get_hybrid_comparison_data(results::Vector{StrategyComparisonResult})

Extract data for comparison plots.
"""
function get_hybrid_comparison_data(results::Vector{StrategyComparisonResult})
    circuit_types = unique(r.circuit_name for r in results)

    data = Dict{String,Any}()

    for ct in circuit_types
        ct_results = filter(r -> r.circuit_name == ct, results)

        ct_data = Dict{String,Vector{Float64}}()

        for strat in ["OFD", "OBD", "Hybrid"]
            times = Float64[]
            chis = Float64[]
            ofd_rates = Float64[]

            for r in ct_results
                if haskey(r.results, strat)
                    push!(times, r.results[strat].wall_time_s)
                    push!(chis, Float64(r.results[strat].bond_dim))
                    push!(ofd_rates, r.results[strat].ofd_success_rate)
                end
            end

            ct_data["$(strat)_times"] = times
            ct_data["$(strat)_chis"] = chis
            ct_data["$(strat)_ofd_rates"] = ofd_rates
        end

        data[ct] = ct_data
    end

    data
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    results = run_hybrid_strategy_experiment(config=HYBRID_QUICK_CONFIG, verbose=true)
end

export run_hybrid_strategy_experiment
export get_hybrid_comparison_data
