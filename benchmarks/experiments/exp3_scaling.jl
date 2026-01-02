# CAMPS.jl/benchmarks/experiments/exp3_scaling.jl
# Experiment 3: Scaling Analysis
#
# Purpose: Demonstrate CAMPS scaling advantages
# Key results:
#   - Clifford operations scale as O(n²)
#   - With OFD, total time scales polynomially
#   - Bond dimension controlled by GF(2) structure

push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using CAMPS
using Random
using Statistics
using Printf

include(joinpath(@__DIR__, "..", "benchmark_types.jl"))
include(joinpath(@__DIR__, "..", "config.jl"))

"""
    run_scaling_experiment(; config=SCALING_QUICK_CONFIG, verbose=true)

Run the scaling analysis experiment.

Studies how CAMPS scales with:
1. System size (n qubits) for Clifford-only circuits
2. System size for Clifford+T circuits with OFD
3. T-gate count at fixed system size
4. Bond dimension growth patterns

# Arguments
- `config::ScalingConfig`: Configuration parameters
- `verbose::Bool`: Whether to print progress

# Returns
- `ExperimentResults`: Complete experiment results with summary
"""
function run_scaling_experiment(;
    config::ScalingConfig = SCALING_QUICK_CONFIG,
    verbose::Bool = true
)
    if verbose
        println("="^70)
        println("Experiment 3: Scaling Analysis")
        println("="^70)
        println()
        println("Configuration:")
        println("  n_qubits_range: $(config.n_qubits_range)")
        println("  t_counts_range: $(config.t_counts_range)")
        println("  fixed_n: $(config.fixed_n)")
        println("  fixed_t: $(config.fixed_t)")
        println("  n_timing_samples: $(config.n_timing_samples)")
        println("  n_seeds: $(config.n_seeds)")
        println()
    end

    start_time = time()
    all_results = ScalingDataPoint[]

    # Part 1: Clifford-only scaling with system size
    if verbose
        println("-"^50)
        println("Part 1: Clifford-only scaling (O(n²) expected)")
        println("-"^50)
    end
    clifford_results = benchmark_clifford_scaling(config; verbose=verbose)
    append!(all_results, clifford_results)

    # Part 2: Clifford+T scaling with system size (OFD)
    if verbose
        println()
        println("-"^50)
        println("Part 2: Clifford+T scaling with OFD")
        println("-"^50)
    end
    ofd_results = benchmark_ofd_scaling(config; verbose=verbose)
    append!(all_results, ofd_results)

    # Part 3: T-count scaling at fixed n
    if verbose
        println()
        println("-"^50)
        println("Part 3: T-count scaling (fixed n=$(config.fixed_n))")
        println("-"^50)
    end
    tcount_results = benchmark_tcount_scaling(config; verbose=verbose)
    append!(all_results, tcount_results)

    # Part 4: Bond dimension growth
    if verbose
        println()
        println("-"^50)
        println("Part 4: Bond dimension growth patterns")
        println("-"^50)
    end
    bonddim_results = benchmark_bonddim_growth(config; verbose=verbose)
    append!(all_results, bonddim_results)

    wall_time = time() - start_time
    summary = compute_scaling_summary(all_results)

    if verbose
        println()
        println("="^70)
        println("Experiment Complete")
        println("="^70)
        println("Total data points: $(length(all_results))")
        println("Wall time: $(round(wall_time, digits=2))s")
        print_scaling_summary(summary)
    end

    ExperimentResults(
        "scaling",
        "Scaling analysis of CAMPS with system size and T-count",
        Dict{String,Any}(
            "n_qubits_range" => config.n_qubits_range,
            "t_counts_range" => config.t_counts_range,
            "fixed_n" => config.fixed_n,
            "fixed_t" => config.fixed_t,
            "n_timing_samples" => config.n_timing_samples,
            "n_seeds" => config.n_seeds
        ),
        all_results,
        summary,
        wall_time
    )
end

"""
    benchmark_clifford_scaling(config; verbose=true)

Benchmark Clifford-only circuit scaling with system size.
Expected: O(n²) from Clifford tableau operations.
"""
function benchmark_clifford_scaling(config::ScalingConfig; verbose::Bool = true)
    results = ScalingDataPoint[]

    for n in config.n_qubits_range
        # Create Clifford-only circuit: H layer + CNOT chain + S layer
        circuit = Gate[]
        for q in 1:n
            push!(circuit, HGate(q))
        end
        for q in 1:(n-1)
            push!(circuit, CNOTGate(q, q+1))
        end
        for q in 1:n
            push!(circuit, SGate(q))
        end

        # Time multiple runs
        times = Float64[]
        bond_dims = Float64[]

        for seed in 1:config.n_seeds
            for _ in 1:config.n_timing_samples
                GC.gc()
                t_start = time()

                result = simulate_circuit(circuit, n;
                    strategy = OFDStrategy(),
                    verbose = false
                )

                push!(times, time() - t_start)
                push!(bond_dims, Float64(result.final_bond_dim))
            end
        end

        mean_time = mean(times)
        std_time = std(times)
        mean_chi = mean(bond_dims)
        std_chi = std(bond_dims)

        push!(results, ScalingDataPoint(
            "n_clifford_only",
            n,
            Dict{String,Any}("circuit_type" => "clifford_only"),
            mean_time, std_time,
            mean_chi, std_chi,
            length(times)
        ))

        if verbose
            @printf("  n=%2d: time=%.4f±%.4fs, χ=%.1f\n",
                    n, mean_time, std_time, mean_chi)
        end
    end

    results
end

"""
    benchmark_ofd_scaling(config; verbose=true)

Benchmark Clifford+T circuit scaling with OFD strategy.
"""
function benchmark_ofd_scaling(config::ScalingConfig; verbose::Bool = true)
    results = ScalingDataPoint[]
    t = config.fixed_t

    for n in config.n_qubits_range
        if t > n  # Skip if more T gates than qubits
            continue
        end

        times = Float64[]
        bond_dims = Float64[]

        for seed in 1:config.n_seeds
            # Create circuit: H layer + T gates
            circuit = Gate[]
            for q in 1:n
                push!(circuit, HGate(q))
            end
            for q in 1:t
                push!(circuit, TGate(q))
            end

            for _ in 1:config.n_timing_samples
                GC.gc()
                t_start = time()

                result = simulate_circuit(circuit, n;
                    strategy = OFDStrategy(),
                    verbose = false
                )

                push!(times, time() - t_start)
                push!(bond_dims, Float64(result.final_bond_dim))
            end
        end

        mean_time = mean(times)
        std_time = std(times)
        mean_chi = mean(bond_dims)
        std_chi = std(bond_dims)

        push!(results, ScalingDataPoint(
            "n_ofd",
            n,
            Dict{String,Any}("t" => t, "strategy" => "OFD"),
            mean_time, std_time,
            mean_chi, std_chi,
            length(times)
        ))

        if verbose
            @printf("  n=%2d, t=%d: time=%.4f±%.4fs, χ=%.1f\n",
                    n, t, mean_time, std_time, mean_chi)
        end
    end

    results
end

"""
    benchmark_tcount_scaling(config; verbose=true)

Benchmark scaling with T-gate count at fixed system size.
"""
function benchmark_tcount_scaling(config::ScalingConfig; verbose::Bool = true)
    results = ScalingDataPoint[]
    n = config.fixed_n

    for t in config.t_counts_range
        if t > 2 * n  # Skip unreasonable T counts
            continue
        end

        times_ofd = Float64[]
        times_none = Float64[]
        bond_dims_ofd = Float64[]
        bond_dims_none = Float64[]

        for seed in 1:config.n_seeds
            # Create circuit
            circuit = Gate[]
            for q in 1:n
                push!(circuit, HGate(q))
            end
            for i in 1:t
                q = ((i - 1) % n) + 1
                push!(circuit, TGate(q))
            end

            # OFD strategy
            for _ in 1:config.n_timing_samples
                GC.gc()
                t_start = time()
                result = simulate_circuit(circuit, n;
                    strategy = OFDStrategy(),
                    verbose = false
                )
                push!(times_ofd, time() - t_start)
                push!(bond_dims_ofd, Float64(result.final_bond_dim))
            end

            # No disentangling (only for small t to avoid explosion)
            if t <= 6
                for _ in 1:config.n_timing_samples
                    GC.gc()
                    t_start = time()
                    result = simulate_circuit(circuit, n;
                        strategy = NoDisentangling(),
                        verbose = false
                    )
                    push!(times_none, time() - t_start)
                    push!(bond_dims_none, Float64(result.final_bond_dim))
                end
            end
        end

        # OFD results
        push!(results, ScalingDataPoint(
            "t_ofd",
            t,
            Dict{String,Any}("n" => n, "strategy" => "OFD"),
            mean(times_ofd), std(times_ofd),
            mean(bond_dims_ofd), std(bond_dims_ofd),
            length(times_ofd)
        ))

        if verbose
            @printf("  t=%2d (OFD):  time=%.4f±%.4fs, χ=%.1f\n",
                    t, mean(times_ofd), std(times_ofd), mean(bond_dims_ofd))
        end

        # No disentangling results
        if !isempty(times_none)
            push!(results, ScalingDataPoint(
                "t_none",
                t,
                Dict{String,Any}("n" => n, "strategy" => "None"),
                mean(times_none), std(times_none),
                mean(bond_dims_none), std(bond_dims_none),
                length(times_none)
            ))

            if verbose
                @printf("  t=%2d (None): time=%.4f±%.4fs, χ=%.1f\n",
                        t, mean(times_none), std(times_none), mean(bond_dims_none))
            end
        end
    end

    results
end

"""
    benchmark_bonddim_growth(config; verbose=true)

Analyze bond dimension growth patterns under different conditions.
"""
function benchmark_bonddim_growth(config::ScalingConfig; verbose::Bool = true)
    results = ScalingDataPoint[]
    n = config.fixed_n

    # Case 1: Independent T gates (expect χ = 1)
    if verbose
        println("  Case A: Independent T gates (χ = 1 expected)")
    end

    for t in filter(x -> x <= n, config.t_counts_range)
        circuit = Gate[]
        for q in 1:n
            push!(circuit, HGate(q))
        end
        for q in 1:t
            push!(circuit, TGate(q))
        end

        result = simulate_circuit(circuit, n;
            strategy = OFDStrategy(),
            verbose = false
        )

        push!(results, ScalingDataPoint(
            "chi_independent",
            t,
            Dict{String,Any}("n" => n, "case" => "independent"),
            0.0, 0.0,  # No timing for this
            Float64(result.final_bond_dim), 0.0,
            1
        ))

        if verbose
            @printf("    t=%2d: χ=%d (expected: 1)\n", t, result.final_bond_dim)
        end
    end

    # Case 2: CNOT-entangled T gates
    if verbose
        println("  Case B: CNOT-entangled T gates")
    end

    for t in filter(x -> x <= n, config.t_counts_range)
        circuit = Gate[]
        for q in 1:n
            push!(circuit, HGate(q))
        end
        for q in 1:(n-1)
            push!(circuit, CNOTGate(q, q+1))
        end
        for q in 1:t
            push!(circuit, TGate(q))
        end

        result = simulate_circuit(circuit, n;
            strategy = OFDStrategy(),
            verbose = false
        )

        push!(results, ScalingDataPoint(
            "chi_entangled",
            t,
            Dict{String,Any}("n" => n, "case" => "cnot_chain"),
            0.0, 0.0,
            Float64(result.final_bond_dim), 0.0,
            1
        ))

        if verbose
            @printf("    t=%2d: χ=%d (predicted: %d)\n",
                    t, result.final_bond_dim, result.predicted_bond_dim)
        end
    end

    results
end

"""
    compute_scaling_summary(results::Vector{ScalingDataPoint}) -> Dict

Compute summary statistics including scaling exponents.
"""
function compute_scaling_summary(results::Vector{ScalingDataPoint})
    summary = Dict{String,Any}()

    # Group by parameter name
    param_names = unique(r.parameter_name for r in results)

    for pname in param_names
        pdata = filter(r -> r.parameter_name == pname, results)

        if isempty(pdata)
            continue
        end

        # Compute scaling exponent for time
        x = Float64[r.parameter_value for r in pdata]
        y = Float64[r.mean_time for r in pdata]

        valid = (x .> 0) .& (y .> 0) .& isfinite.(y)

        if sum(valid) >= 3
            log_x = log.(x[valid])
            log_y = log.(y[valid])
            x_mean = mean(log_x)
            y_mean = mean(log_y)
            num = sum((log_x .- x_mean) .* (log_y .- y_mean))
            den = sum((log_x .- x_mean).^2)
            exponent = den > 0 ? num / den : NaN
        else
            exponent = NaN
        end

        summary["$(pname)_time_exponent"] = exponent
        summary["$(pname)_n_points"] = length(pdata)

        # For bond dimension growth
        if startswith(pname, "chi_")
            chi_values = [r.mean_bond_dim for r in pdata]
            summary["$(pname)_max_chi"] = maximum(chi_values)
            summary["$(pname)_min_chi"] = minimum(chi_values)
        end
    end

    summary
end

"""
    print_scaling_summary(summary::Dict)

Print formatted scaling summary.
"""
function print_scaling_summary(summary::Dict)
    println("\nScaling Summary:")
    println("-"^50)

    for (key, value) in sort(collect(summary), by=x->x[1])
        if endswith(key, "_time_exponent")
            name = replace(key, "_time_exponent" => "")
            if !isnan(value)
                @printf("  %-20s: time ~ n^%.2f\n", name, value)
            end
        end
    end

    println()
    println("Expected scaling:")
    println("  Clifford-only: O(n²) from tableau operations")
    println("  Clifford+T with OFD: O(n² + t·n²) = O(n²·(1+t))")
    println("  Without disentangling: O(2^t) exponential in T-count")
end

"""
    get_scaling_plot_data(results::Vector{ScalingDataPoint}, param_name::String)

Extract data for scaling plots.
"""
function get_scaling_plot_data(results::Vector{ScalingDataPoint}, param_name::String)
    pdata = filter(r -> r.parameter_name == param_name, results)

    x = [r.parameter_value for r in pdata]
    y_time = [r.mean_time for r in pdata]
    y_err = [r.std_time for r in pdata]
    y_chi = [r.mean_bond_dim for r in pdata]

    (x=x, time=y_time, time_err=y_err, bond_dim=y_chi)
end

"""
    fit_power_law(x::Vector, y::Vector) -> (a, b)

Fit y = a * x^b using log-log linear regression.
Returns (a, b) where b is the scaling exponent.
"""
function fit_power_law(x::Vector{<:Real}, y::Vector{<:Real})
    valid = (x .> 0) .& (y .> 0) .& isfinite.(x) .& isfinite.(y)

    if sum(valid) < 2
        return (NaN, NaN)
    end

    log_x = log.(x[valid])
    log_y = log.(y[valid])

    x_mean = mean(log_x)
    y_mean = mean(log_y)

    num = sum((log_x .- x_mean) .* (log_y .- y_mean))
    den = sum((log_x .- x_mean).^2)

    b = den > 0 ? num / den : NaN
    a = exp(y_mean - b * x_mean)

    (a, b)
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    results = run_scaling_experiment(config=SCALING_QUICK_CONFIG, verbose=true)
end

export run_scaling_experiment
export get_scaling_plot_data, fit_power_law
