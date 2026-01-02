# CAMPS.jl/benchmarks/experiments/exp2_ofd_vs_obd.jl
# Experiment 2: OFD vs OBD Strategy Comparison
#
# Purpose: Systematic head-to-head comparison of disentangling strategies
# Key result: OFD is 10-100x faster when applicable

push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using CAMPS
using Random
using Statistics
using Printf

include(joinpath(@__DIR__, "..", "benchmark_types.jl"))
include(joinpath(@__DIR__, "..", "config.jl"))

"""
    run_ofd_vs_obd_experiment(; config=STRATEGY_QUICK_CONFIG, verbose=true)

Run the OFD vs OBD strategy comparison experiment.

Compares strategies on:
1. Wall-clock time vs T-count (fixed n)
2. Wall-clock time vs n (fixed t)
3. Final bond dimension achieved
4. OFD success rate

# Arguments
- `config::StrategyComparisonConfig`: Configuration parameters
- `verbose::Bool`: Whether to print progress

# Returns
- `ExperimentResults`: Complete experiment results with summary
"""
function run_ofd_vs_obd_experiment(;
    config::StrategyComparisonConfig = STRATEGY_QUICK_CONFIG,
    verbose::Bool = true
)
    if verbose
        println("="^70)
        println("Experiment 2: OFD vs OBD Strategy Comparison")
        println("="^70)
        println()
        println("Configuration:")
        println("  n_qubits: $(config.n_qubits)")
        println("  t_counts: $(config.t_counts)")
        println("  n_seeds: $(config.n_seeds)")
        println("  n_timing_samples: $(config.n_timing_samples)")
        println("  strategies: $([s[1] for s in config.strategies])")
        println()
    end

    start_time = time()
    results = StrategyComparisonResult[]

    total_tests = length(config.n_qubits) * length(config.t_counts) * config.n_seeds
    test_count = 0

    for n in config.n_qubits
        for t in config.t_counts
            # Skip invalid combinations
            if t > 2 * n
                continue
            end

            for seed in 1:config.n_seeds
                test_count += 1

                if verbose
                    @printf("Test %d/%d: n=%d, t=%d, seed=%d\n",
                            test_count, total_tests, n, t, seed)
                end

                result = compare_strategies_single_circuit(
                    n, t, seed, config; verbose=verbose
                )

                if result !== nothing
                    push!(results, result)
                end
            end
        end
    end

    wall_time = time() - start_time
    summary = compute_summary(results)

    if verbose
        println()
        println("="^70)
        println("Experiment Complete")
        println("="^70)
        println("Total comparisons: $(length(results))")
        println("Wall time: $(round(wall_time, digits=2))s")
        print_strategy_summary(results)
    end

    ExperimentResults(
        "ofd_vs_obd",
        "Systematic comparison of OFD vs OBD disentangling strategies",
        Dict{String,Any}(
            "n_qubits" => config.n_qubits,
            "t_counts" => config.t_counts,
            "n_seeds" => config.n_seeds,
            "n_timing_samples" => config.n_timing_samples,
            "strategies" => [s[1] for s in config.strategies]
        ),
        results,
        summary,
        wall_time
    )
end

"""
    compare_strategies_single_circuit(n, t, seed, config; verbose=false)

Compare all strategies on a single circuit configuration.
"""
function compare_strategies_single_circuit(
    n::Int,
    t::Int,
    seed::Int,
    config::StrategyComparisonConfig;
    verbose::Bool = false
)
    # Generate circuit
    Random.seed!(seed)
    circuit = generate_benchmark_circuit(n, t, seed)

    if circuit === nothing
        return nothing
    end

    strategy_results = Dict{String,SingleRunResult}()

    for (strat_name, strat_param) in config.strategies
        strategy = create_strategy(strat_name, strat_param)

        # Time multiple runs
        times = Float64[]
        local result

        for i in 1:config.n_timing_samples
            GC.gc()  # Clean up before timing
            t_start = time()

            try
                result = simulate_circuit(circuit, n;
                    strategy = strategy,
                    verbose = false
                )
            catch e
                if verbose
                    @warn "Simulation failed" strat_name n t seed exception=e
                end
                break
            end

            push!(times, time() - t_start)
        end

        if length(times) < config.n_timing_samples
            continue  # Simulation failed
        end

        mean_time = mean(times)

        # Compute OFD success rate
        ofd_success_rate = if result.n_non_clifford > 0
            result.n_ofd_applied / result.n_non_clifford
        else
            1.0
        end

        strategy_results[strat_name] = SingleRunResult(
            circuit_name = "random_clifford_t",
            circuit_params = Dict{String,Any}("n" => n, "t" => t, "seed" => seed),
            n_qubits = n,
            n_gates = result.n_gates,
            n_clifford = result.n_clifford,
            n_non_clifford = result.n_non_clifford,
            t_count = t,
            strategy = strat_name,
            wall_time_s = mean_time,
            bond_dim = result.final_bond_dim,
            predicted_bond_dim = result.predicted_bond_dim,
            max_entropy = result.final_entropy,
            n_ofd_applied = result.n_ofd_applied,
            n_obd_applied = result.n_obd_applied,
            memory_bytes = 0,  # TODO: Add memory tracking
            gf2_rank = 0,
            ofd_success_rate = ofd_success_rate
        )

        if verbose
            @printf("  %-8s: time=%.4fs, χ=%d, OFD=%d/%d\n",
                    strat_name, mean_time, result.final_bond_dim,
                    result.n_ofd_applied, result.n_non_clifford)
        end
    end

    if isempty(strategy_results)
        return nothing
    end

    StrategyComparisonResult(
        "random_clifford_t",
        n, t, seed,
        strategy_results
    )
end

"""
    create_strategy(name::String, param) -> DisentanglingStrategy

Create a disentangling strategy from name and parameter.
"""
function create_strategy(name::String, param)
    if name == "OFD"
        return OFDStrategy()
    elseif startswith(name, "OBD")
        sweeps = param isa Int ? param : 3
        return OBDStrategy(max_sweeps=sweeps)
    elseif name == "Hybrid"
        return HybridStrategy()
    elseif name == "None"
        return NoDisentangling()
    else
        error("Unknown strategy: $name")
    end
end

"""
    generate_benchmark_circuit(n, t, seed) -> Vector{Gate}

Generate a standard benchmark circuit for strategy comparison.
Uses H layer + random Cliffords + T gates pattern.
"""
function generate_benchmark_circuit(n::Int, t::Int, seed::Int)
    Random.seed!(seed)
    random_clifford_t_circuit(n, t; seed=seed)
end

# Analysis functions

"""
    print_strategy_summary(results::Vector{StrategyComparisonResult})

Print a summary table of strategy comparison results.
"""
function print_strategy_summary(results::Vector{StrategyComparisonResult})
    if isempty(results)
        println("No results to summarize")
        return
    end

    strategies = collect(keys(first(results).results))
    sort!(strategies)

    println("\nStrategy Comparison Summary:")
    println("-"^80)

    # Header
    header = @sprintf("%-8s", "Strategy")
    header *= @sprintf(" %12s", "Mean Time(s)")
    header *= @sprintf(" %10s", "Mean χ")
    header *= @sprintf(" %12s", "OFD Success")
    header *= @sprintf(" %10s", "Count")
    println(header)
    println("-"^80)

    for strat in strategies
        strat_data = [r.results[strat] for r in results if haskey(r.results, strat)]

        if isempty(strat_data)
            continue
        end

        mean_time = mean(d -> d.wall_time_s, strat_data)
        mean_chi = mean(d -> d.bond_dim, strat_data)
        mean_ofd_rate = mean(d -> d.ofd_success_rate, strat_data)
        count = length(strat_data)

        row = @sprintf("%-8s", strat)
        row *= @sprintf(" %12.4f", mean_time)
        row *= @sprintf(" %10.1f", mean_chi)
        row *= @sprintf(" %11.1f%%", 100 * mean_ofd_rate)
        row *= @sprintf(" %10d", count)
        println(row)
    end

    println("-"^80)

    # Speedup analysis
    if "OFD" in strategies && any(startswith(s, "OBD") for s in strategies)
        ofd_data = [r.results["OFD"] for r in results if haskey(r.results, "OFD")]
        obd_strat = first(s for s in strategies if startswith(s, "OBD"))
        obd_data = [r.results[obd_strat] for r in results if haskey(r.results, obd_strat)]

        if !isempty(ofd_data) && !isempty(obd_data)
            ofd_mean = mean(d -> d.wall_time_s, ofd_data)
            obd_mean = mean(d -> d.wall_time_s, obd_data)

            speedup = obd_mean / ofd_mean
            println("\nOFD vs $(obd_strat) speedup: $(round(speedup, digits=1))x")
        end
    end
end

"""
    analyze_timing_vs_t(results::Vector{StrategyComparisonResult})

Analyze how timing scales with T-gate count for each strategy.
"""
function analyze_timing_vs_t(results::Vector{StrategyComparisonResult})
    println("\n" * "="^70)
    println("Timing vs T-Count Analysis")
    println("="^70)

    if isempty(results)
        return
    end

    strategies = collect(keys(first(results).results))
    t_counts = sort(unique(r.t_count for r in results))

    for strat in strategies
        println("\nStrategy: $strat")
        println("-"^40)

        for t in t_counts
            t_data = [r.results[strat] for r in results
                      if r.t_count == t && haskey(r.results, strat)]

            if !isempty(t_data)
                mean_time = mean(d -> d.wall_time_s, t_data)
                std_time = std([d.wall_time_s for d in t_data])
                mean_chi = mean(d -> d.bond_dim, t_data)

                @printf("  t=%2d: time=%.4f±%.4fs, χ=%.1f (n=%d samples)\n",
                        t, mean_time, std_time, mean_chi, length(t_data))
            end
        end
    end
end

"""
    analyze_timing_vs_n(results::Vector{StrategyComparisonResult})

Analyze how timing scales with system size for each strategy.
"""
function analyze_timing_vs_n(results::Vector{StrategyComparisonResult})
    println("\n" * "="^70)
    println("Timing vs System Size Analysis")
    println("="^70)

    if isempty(results)
        return
    end

    strategies = collect(keys(first(results).results))
    n_qubits = sort(unique(r.n_qubits for r in results))

    for strat in strategies
        println("\nStrategy: $strat")
        println("-"^40)

        for n in n_qubits
            n_data = [r.results[strat] for r in results
                      if r.n_qubits == n && haskey(r.results, strat)]

            if !isempty(n_data)
                mean_time = mean(d -> d.wall_time_s, n_data)
                std_time = std([d.wall_time_s for d in n_data])
                mean_chi = mean(d -> d.bond_dim, n_data)

                @printf("  n=%2d: time=%.4f±%.4fs, χ=%.1f (n=%d samples)\n",
                        n, mean_time, std_time, mean_chi, length(n_data))
            end
        end
    end
end

"""
    get_timing_data(results::Vector{StrategyComparisonResult}, strategy::String)

Extract timing data for plotting.
Returns NamedTuple with n_qubits, t_counts, times arrays.
"""
function get_timing_data(results::Vector{StrategyComparisonResult}, strategy::String)
    n_qubits = Int[]
    t_counts = Int[]
    times = Float64[]
    bond_dims = Int[]

    for r in results
        if haskey(r.results, strategy)
            push!(n_qubits, r.n_qubits)
            push!(t_counts, r.t_count)
            push!(times, r.results[strategy].wall_time_s)
            push!(bond_dims, r.results[strategy].bond_dim)
        end
    end

    (n_qubits=n_qubits, t_counts=t_counts, times=times, bond_dims=bond_dims)
end

"""
    compute_speedup_matrix(results::Vector{StrategyComparisonResult})

Compute pairwise speedup matrix between strategies.
"""
function compute_speedup_matrix(results::Vector{StrategyComparisonResult})
    if isempty(results)
        return nothing
    end

    strategies = sort(collect(keys(first(results).results)))
    n_strat = length(strategies)
    speedup = zeros(n_strat, n_strat)

    for (i, strat1) in enumerate(strategies)
        for (j, strat2) in enumerate(strategies)
            times1 = [r.results[strat1].wall_time_s for r in results
                      if haskey(r.results, strat1) && haskey(r.results, strat2)]
            times2 = [r.results[strat2].wall_time_s for r in results
                      if haskey(r.results, strat1) && haskey(r.results, strat2)]

            if !isempty(times1) && !isempty(times2)
                speedup[i, j] = mean(times2) / mean(times1)
            else
                speedup[i, j] = NaN
            end
        end
    end

    (strategies=strategies, speedup=speedup)
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    results = run_ofd_vs_obd_experiment(config=STRATEGY_QUICK_CONFIG, verbose=true)
    analyze_timing_vs_t(results.results)
    analyze_timing_vs_n(results.results)
end

export run_ofd_vs_obd_experiment
export analyze_timing_vs_t, analyze_timing_vs_n
export get_timing_data, compute_speedup_matrix
