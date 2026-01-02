# CAMPS.jl/benchmarks/benchmark_types.jl
# Data structures for publication-quality benchmarking
#
# These types provide structured storage for benchmark results,
# enabling reproducibility and systematic analysis.

using Dates

"""
    BenchmarkMetadata

Metadata about the benchmark run for reproducibility.

# Fields
- `timestamp::DateTime`: When the benchmark was run
- `julia_version::String`: Julia version string
- `camps_version::String`: CAMPS.jl version
- `hardware_info::Dict{String,Any}`: CPU, memory, etc.
- `git_commit::String`: Git commit hash (if available)
- `hostname::String`: Machine hostname
- `notes::String`: Any additional notes
"""
struct BenchmarkMetadata
    timestamp::DateTime
    julia_version::String
    camps_version::String
    hardware_info::Dict{String,Any}
    git_commit::String
    hostname::String
    notes::String
end

"""
    BenchmarkMetadata()

Construct metadata with automatically detected system information.
"""
function BenchmarkMetadata(; notes::String = "")
    hardware_info = Dict{String,Any}(
        "cpu_threads" => Sys.CPU_THREADS,
        "total_memory_gb" => round(Sys.total_memory() / 2^30, digits=2),
        "word_size" => Sys.WORD_SIZE,
        "kernel" => String(Sys.KERNEL),
        "machine" => String(Sys.MACHINE)
    )

    # Try to get git commit
    git_commit = try
        strip(read(`git rev-parse HEAD`, String))
    catch
        "unknown"
    end

    hostname = try
        strip(read(`hostname`, String))
    catch
        gethostname()
    end

    BenchmarkMetadata(
        now(),
        string(VERSION),
        "0.1.0",  # TODO: Get from Project.toml
        hardware_info,
        git_commit,
        hostname,
        notes
    )
end

"""
    SingleRunResult

Result from a single circuit simulation run.

# Fields
- `circuit_name::String`: Identifier for the circuit type
- `circuit_params::Dict{String,Any}`: Parameters used to generate circuit
- `n_qubits::Int`: Number of qubits
- `n_gates::Int`: Total number of gates
- `n_clifford::Int`: Number of Clifford gates
- `n_non_clifford::Int`: Number of non-Clifford gates
- `t_count::Int`: Number of T gates specifically
- `strategy::String`: Disentangling strategy used
- `wall_time_s::Float64`: Total wall-clock time in seconds
- `bond_dim::Int`: Final bond dimension
- `predicted_bond_dim::Int`: GF(2) predicted bond dimension
- `max_entropy::Float64`: Maximum entanglement entropy
- `n_ofd_applied::Int`: Number of successful OFD applications
- `n_obd_applied::Int`: Number of OBD applications
- `memory_bytes::Int`: Approximate memory usage
- `gf2_rank::Int`: Rank of the GF(2) matrix
- `ofd_success_rate::Float64`: Fraction of T-gates where OFD succeeded
"""
struct SingleRunResult
    circuit_name::String
    circuit_params::Dict{String,Any}
    n_qubits::Int
    n_gates::Int
    n_clifford::Int
    n_non_clifford::Int
    t_count::Int
    strategy::String
    wall_time_s::Float64
    bond_dim::Int
    predicted_bond_dim::Int
    max_entropy::Float64
    n_ofd_applied::Int
    n_obd_applied::Int
    memory_bytes::Int
    gf2_rank::Int
    ofd_success_rate::Float64
end

"""
    SingleRunResult(; kwargs...)

Construct a SingleRunResult with keyword arguments.
"""
function SingleRunResult(;
    circuit_name::String,
    circuit_params::Dict{String,Any} = Dict{String,Any}(),
    n_qubits::Int,
    n_gates::Int = 0,
    n_clifford::Int = 0,
    n_non_clifford::Int = 0,
    t_count::Int = 0,
    strategy::String,
    wall_time_s::Float64,
    bond_dim::Int,
    predicted_bond_dim::Int = 0,
    max_entropy::Float64 = 0.0,
    n_ofd_applied::Int = 0,
    n_obd_applied::Int = 0,
    memory_bytes::Int = 0,
    gf2_rank::Int = 0,
    ofd_success_rate::Float64 = 0.0
)
    SingleRunResult(
        circuit_name, circuit_params, n_qubits, n_gates, n_clifford,
        n_non_clifford, t_count, strategy, wall_time_s, bond_dim,
        predicted_bond_dim, max_entropy, n_ofd_applied, n_obd_applied,
        memory_bytes, gf2_rank, ofd_success_rate
    )
end

"""
    TimingBreakdown

Detailed timing information for a simulation.

# Fields
- `total_s::Float64`: Total wall-clock time
- `clifford_s::Float64`: Time spent on Clifford gates
- `non_clifford_s::Float64`: Time on non-Clifford gates
- `ofd_s::Float64`: Time spent in OFD
- `obd_s::Float64`: Time spent in OBD
- `mps_operations_s::Float64`: Time in MPS operations (SVD, contraction)
- `overhead_s::Float64`: Other overhead
"""
struct TimingBreakdown
    total_s::Float64
    clifford_s::Float64
    non_clifford_s::Float64
    ofd_s::Float64
    obd_s::Float64
    mps_operations_s::Float64
    overhead_s::Float64
end

"""
    GF2ValidationResult

Result from GF(2) prediction validation.

# Fields
- `n_qubits::Int`: System size
- `t_count::Int`: Number of T gates
- `circuit_type::String`: Type of circuit
- `predicted_chi::Int`: GF(2) predicted bond dimension
- `actual_chi::Int`: Actual achieved bond dimension
- `gf2_rank::Int`: Rank of GF(2) matrix
- `exact_match::Bool`: Whether prediction == actual
- `relative_error::Float64`: |predicted - actual| / max(actual, 1)
"""
struct GF2ValidationResult
    n_qubits::Int
    t_count::Int
    circuit_type::String
    seed::Int
    predicted_chi::Int
    actual_chi::Int
    gf2_rank::Int
    exact_match::Bool
    relative_error::Float64
end

"""
    StrategyComparisonResult

Result comparing different disentangling strategies on same circuit.

# Fields
- `circuit_name::String`: Circuit identifier
- `n_qubits::Int`: Number of qubits
- `t_count::Int`: Number of T gates
- `seed::Int`: Random seed used
- `results::Dict{String,SingleRunResult}`: Results keyed by strategy name
"""
struct StrategyComparisonResult
    circuit_name::String
    n_qubits::Int
    t_count::Int
    seed::Int
    results::Dict{String,SingleRunResult}
end

"""
    ScalingDataPoint

Single data point for scaling analysis.

# Fields
- `parameter_name::String`: What's being varied ("n_qubits", "t_count", etc.)
- `parameter_value::Int`: Value of the parameter
- `other_params::Dict{String,Any}`: Other fixed parameters
- `mean_time::Float64`: Mean wall-clock time
- `std_time::Float64`: Standard deviation of time
- `mean_bond_dim::Float64`: Mean bond dimension
- `std_bond_dim::Float64`: Standard deviation of bond dimension
- `n_samples::Int`: Number of samples averaged
"""
struct ScalingDataPoint
    parameter_name::String
    parameter_value::Int
    other_params::Dict{String,Any}
    mean_time::Float64
    std_time::Float64
    mean_bond_dim::Float64
    std_bond_dim::Float64
    n_samples::Int
end

"""
    CorrectnessResult

Result from correctness verification test.

# Fields
- `test_name::String`: Name of the test
- `n_qubits::Int`: System size
- `circuit_description::String`: What circuit was tested
- `fidelity::Float64`: Fidelity with reference state
- `max_amplitude_error::Float64`: Maximum amplitude error
- `passed::Bool`: Whether test passed threshold
- `threshold::Float64`: Threshold used for pass/fail
"""
struct CorrectnessResult
    test_name::String
    n_qubits::Int
    circuit_description::String
    fidelity::Float64
    max_amplitude_error::Float64
    passed::Bool
    threshold::Float64
end

"""
    ExperimentResults

Container for results from a single experiment.

# Fields
- `experiment_name::String`: Name of the experiment
- `description::String`: What the experiment tests
- `parameters::Dict{String,Any}`: Parameters used
- `results::Vector`: Vector of result structs (type varies by experiment)
- `summary::Dict{String,Any}`: Summary statistics
- `wall_time_s::Float64`: Total time for experiment
"""
struct ExperimentResults
    experiment_name::String
    description::String
    parameters::Dict{String,Any}
    results::Vector
    summary::Dict{String,Any}
    wall_time_s::Float64
end

"""
    BenchmarkSuite

Complete benchmark suite with all experiments.

# Fields
- `metadata::BenchmarkMetadata`: Run metadata
- `experiments::Dict{String,ExperimentResults}`: All experiment results
- `total_time_s::Float64`: Total benchmark time
"""
struct BenchmarkSuite
    metadata::BenchmarkMetadata
    experiments::Dict{String,ExperimentResults}
    total_time_s::Float64
end

# Convenience constructors and methods

"""
    compute_summary(results::Vector{GF2ValidationResult}) -> Dict{String,Any}

Compute summary statistics for GF(2) validation results.
"""
function compute_summary(results::Vector{GF2ValidationResult})
    n = length(results)
    if n == 0
        return Dict{String,Any}()
    end

    exact_matches = count(r -> r.exact_match, results)

    Dict{String,Any}(
        "n_tests" => n,
        "exact_matches" => exact_matches,
        "exact_match_rate" => exact_matches / n,
        "mean_relative_error" => sum(r -> r.relative_error, results) / n,
        "max_relative_error" => maximum(r -> r.relative_error, results),
        "by_circuit_type" => summarize_by_field(results, :circuit_type)
    )
end

"""
    compute_summary(results::Vector{StrategyComparisonResult}) -> Dict{String,Any}

Compute summary statistics for strategy comparison results.
"""
function compute_summary(results::Vector{StrategyComparisonResult})
    n = length(results)
    if n == 0
        return Dict{String,Any}()
    end

    strategies = collect(keys(first(results).results))

    summary = Dict{String,Any}(
        "n_tests" => n,
        "strategies" => strategies
    )

    for strat in strategies
        times = [r.results[strat].wall_time_s for r in results if haskey(r.results, strat)]
        bond_dims = [r.results[strat].bond_dim for r in results if haskey(r.results, strat)]

        summary["$(strat)_mean_time"] = isempty(times) ? NaN : sum(times) / length(times)
        summary["$(strat)_mean_bond_dim"] = isempty(bond_dims) ? NaN : sum(bond_dims) / length(bond_dims)
    end

    summary
end

"""
    compute_summary(results::Vector{ScalingDataPoint}) -> Dict{String,Any}

Compute summary statistics for scaling results.
"""
function compute_summary(results::Vector{ScalingDataPoint})
    if isempty(results)
        return Dict{String,Any}()
    end

    # Fit scaling exponent using log-log regression
    param_name = first(results).parameter_name
    x = Float64[r.parameter_value for r in results]
    y = Float64[r.mean_time for r in results]

    # Filter valid values
    valid = (x .> 0) .& (y .> 0) .& isfinite.(y)

    scaling_exponent = if sum(valid) >= 2
        log_x = log.(x[valid])
        log_y = log.(y[valid])
        x_mean = sum(log_x) / length(log_x)
        y_mean = sum(log_y) / length(log_y)
        num = sum((log_x .- x_mean) .* (log_y .- y_mean))
        den = sum((log_x .- x_mean).^2)
        den > 0 ? num / den : NaN
    else
        NaN
    end

    Dict{String,Any}(
        "parameter_name" => param_name,
        "n_points" => length(results),
        "scaling_exponent" => scaling_exponent,
        "min_value" => minimum(r -> r.parameter_value, results),
        "max_value" => maximum(r -> r.parameter_value, results)
    )
end

"""
    compute_summary(results::Vector{CorrectnessResult}) -> Dict{String,Any}

Compute summary statistics for correctness results.
"""
function compute_summary(results::Vector{CorrectnessResult})
    n = length(results)
    if n == 0
        return Dict{String,Any}()
    end

    passed = count(r -> r.passed, results)
    fidelities = [r.fidelity for r in results]

    Dict{String,Any}(
        "n_tests" => n,
        "passed" => passed,
        "pass_rate" => passed / n,
        "mean_fidelity" => sum(fidelities) / n,
        "min_fidelity" => minimum(fidelities),
        "all_passed" => passed == n
    )
end

"""
    summarize_by_field(results::Vector, field::Symbol) -> Dict

Group results by a field and compute statistics for each group.
"""
function summarize_by_field(results::Vector{GF2ValidationResult}, field::Symbol)
    groups = Dict{Any,Vector{GF2ValidationResult}}()

    for r in results
        key = getfield(r, field)
        if !haskey(groups, key)
            groups[key] = GF2ValidationResult[]
        end
        push!(groups[key], r)
    end

    Dict(k => Dict(
        "n" => length(v),
        "exact_match_rate" => count(r -> r.exact_match, v) / length(v)
    ) for (k, v) in groups)
end

# Export all types
export BenchmarkMetadata, SingleRunResult, TimingBreakdown
export GF2ValidationResult, StrategyComparisonResult, ScalingDataPoint
export CorrectnessResult, ExperimentResults, BenchmarkSuite
export compute_summary
