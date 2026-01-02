# CAMPS.jl/benchmarks/data_io.jl
# Data serialization and I/O utilities for benchmark results
#
# Provides JSON and CSV export/import for reproducibility and analysis.

include("benchmark_types.jl")

using Dates
using Printf

# JSON-like serialization without external dependencies
# (We'll use simple string formatting for portability)

"""
    to_dict(x) -> Dict

Convert a struct to a dictionary for serialization.
"""
function to_dict(x::BenchmarkMetadata)
    Dict{String,Any}(
        "timestamp" => Dates.format(x.timestamp, "yyyy-mm-ddTHH:MM:SS"),
        "julia_version" => x.julia_version,
        "camps_version" => x.camps_version,
        "hardware_info" => x.hardware_info,
        "git_commit" => x.git_commit,
        "hostname" => x.hostname,
        "notes" => x.notes
    )
end

function to_dict(x::SingleRunResult)
    Dict{String,Any}(
        "circuit_name" => x.circuit_name,
        "circuit_params" => x.circuit_params,
        "n_qubits" => x.n_qubits,
        "n_gates" => x.n_gates,
        "n_clifford" => x.n_clifford,
        "n_non_clifford" => x.n_non_clifford,
        "t_count" => x.t_count,
        "strategy" => x.strategy,
        "wall_time_s" => x.wall_time_s,
        "bond_dim" => x.bond_dim,
        "predicted_bond_dim" => x.predicted_bond_dim,
        "max_entropy" => x.max_entropy,
        "n_ofd_applied" => x.n_ofd_applied,
        "n_obd_applied" => x.n_obd_applied,
        "memory_bytes" => x.memory_bytes,
        "gf2_rank" => x.gf2_rank,
        "ofd_success_rate" => x.ofd_success_rate
    )
end

function to_dict(x::GF2ValidationResult)
    Dict{String,Any}(
        "n_qubits" => x.n_qubits,
        "t_count" => x.t_count,
        "circuit_type" => x.circuit_type,
        "seed" => x.seed,
        "predicted_chi" => x.predicted_chi,
        "actual_chi" => x.actual_chi,
        "gf2_rank" => x.gf2_rank,
        "exact_match" => x.exact_match,
        "relative_error" => x.relative_error
    )
end

function to_dict(x::StrategyComparisonResult)
    Dict{String,Any}(
        "circuit_name" => x.circuit_name,
        "n_qubits" => x.n_qubits,
        "t_count" => x.t_count,
        "seed" => x.seed,
        "results" => Dict(k => to_dict(v) for (k, v) in x.results)
    )
end

function to_dict(x::ScalingDataPoint)
    Dict{String,Any}(
        "parameter_name" => x.parameter_name,
        "parameter_value" => x.parameter_value,
        "other_params" => x.other_params,
        "mean_time" => x.mean_time,
        "std_time" => x.std_time,
        "mean_bond_dim" => x.mean_bond_dim,
        "std_bond_dim" => x.std_bond_dim,
        "n_samples" => x.n_samples
    )
end

function to_dict(x::CorrectnessResult)
    Dict{String,Any}(
        "test_name" => x.test_name,
        "n_qubits" => x.n_qubits,
        "circuit_description" => x.circuit_description,
        "fidelity" => x.fidelity,
        "max_amplitude_error" => x.max_amplitude_error,
        "passed" => x.passed,
        "threshold" => x.threshold
    )
end

function to_dict(x::ExperimentResults)
    Dict{String,Any}(
        "experiment_name" => x.experiment_name,
        "description" => x.description,
        "parameters" => x.parameters,
        "results" => [to_dict(r) for r in x.results],
        "summary" => x.summary,
        "wall_time_s" => x.wall_time_s
    )
end

function to_dict(x::BenchmarkSuite)
    Dict{String,Any}(
        "metadata" => to_dict(x.metadata),
        "experiments" => Dict(k => to_dict(v) for (k, v) in x.experiments),
        "total_time_s" => x.total_time_s
    )
end

"""
    dict_to_json(d::Dict; indent::Int=2) -> String

Convert a dictionary to a JSON string.
Simple implementation without external dependencies.
"""
function dict_to_json(d::Dict; indent::Int=2, level::Int=0)
    if isempty(d)
        return "{}"
    end

    prefix = " "^(indent * level)
    inner_prefix = " "^(indent * (level + 1))

    lines = String[]
    push!(lines, "{")

    items = collect(d)
    for (i, (k, v)) in enumerate(items)
        comma = i < length(items) ? "," : ""
        value_str = value_to_json(v; indent=indent, level=level+1)
        push!(lines, """$(inner_prefix)"$(k)": $(value_str)$(comma)""")
    end

    push!(lines, "$(prefix)}")
    join(lines, "\n")
end

function value_to_json(v::String; kwargs...)
    # Escape special characters
    escaped = replace(v, "\\" => "\\\\", "\"" => "\\\"", "\n" => "\\n", "\t" => "\\t")
    "\"$(escaped)\""
end

function value_to_json(v::Number; kwargs...)
    if isnan(v) || isinf(v)
        "null"
    else
        string(v)
    end
end

function value_to_json(v::Bool; kwargs...)
    v ? "true" : "false"
end

function value_to_json(v::Nothing; kwargs...)
    "null"
end

function value_to_json(v::Dict; indent::Int=2, level::Int=0)
    dict_to_json(v; indent=indent, level=level)
end

function value_to_json(v::Vector; indent::Int=2, level::Int=0)
    if isempty(v)
        return "[]"
    end

    prefix = " "^(indent * level)
    inner_prefix = " "^(indent * (level + 1))

    lines = String[]
    push!(lines, "[")

    for (i, item) in enumerate(v)
        comma = i < length(v) ? "," : ""
        item_str = value_to_json(item; indent=indent, level=level+1)
        push!(lines, "$(inner_prefix)$(item_str)$(comma)")
    end

    push!(lines, "$(prefix)]")
    join(lines, "\n")
end

function value_to_json(v::Tuple; kwargs...)
    value_to_json(collect(v); kwargs...)
end

function value_to_json(v::Symbol; kwargs...)
    value_to_json(string(v); kwargs...)
end

function value_to_json(v; kwargs...)
    # Fallback: convert to string
    value_to_json(string(v); kwargs...)
end

"""
    save_json(filepath::String, data::Dict)

Save a dictionary as a JSON file.
"""
function save_json(filepath::String, data::Dict)
    json_str = dict_to_json(data)
    open(filepath, "w") do f
        write(f, json_str)
    end
    println("Saved: $(filepath)")
end

"""
    save_results(suite::BenchmarkSuite, filepath::String)

Save benchmark suite to JSON file.
"""
function save_results(suite::BenchmarkSuite, filepath::String)
    save_json(filepath, to_dict(suite))
end

"""
    save_experiment(results::ExperimentResults, filepath::String)

Save single experiment results to JSON file.
"""
function save_experiment(results::ExperimentResults, filepath::String)
    save_json(filepath, to_dict(results))
end

# CSV export for specific result types

"""
    export_gf2_results_csv(results::Vector{GF2ValidationResult}, filepath::String)

Export GF(2) validation results to CSV.
"""
function export_gf2_results_csv(results::Vector{GF2ValidationResult}, filepath::String)
    open(filepath, "w") do f
        # Header
        println(f, "n_qubits,t_count,circuit_type,seed,predicted_chi,actual_chi,gf2_rank,exact_match,relative_error")

        # Data rows
        for r in results
            println(f, join([
                r.n_qubits,
                r.t_count,
                r.circuit_type,
                r.seed,
                r.predicted_chi,
                r.actual_chi,
                r.gf2_rank,
                r.exact_match ? 1 : 0,
                @sprintf("%.6f", r.relative_error)
            ], ","))
        end
    end
    println("Saved: $(filepath)")
end

"""
    export_strategy_results_csv(results::Vector{StrategyComparisonResult}, filepath::String)

Export strategy comparison results to CSV (one row per strategy per circuit).
"""
function export_strategy_results_csv(results::Vector{StrategyComparisonResult}, filepath::String)
    open(filepath, "w") do f
        # Header
        println(f, "circuit_name,n_qubits,t_count,seed,strategy,wall_time_s,bond_dim,predicted_bond_dim,max_entropy,n_ofd_applied,n_obd_applied")

        # Data rows
        for r in results
            for (strat, sr) in r.results
                println(f, join([
                    r.circuit_name,
                    r.n_qubits,
                    r.t_count,
                    r.seed,
                    strat,
                    @sprintf("%.6f", sr.wall_time_s),
                    sr.bond_dim,
                    sr.predicted_bond_dim,
                    @sprintf("%.6f", sr.max_entropy),
                    sr.n_ofd_applied,
                    sr.n_obd_applied
                ], ","))
            end
        end
    end
    println("Saved: $(filepath)")
end

"""
    export_scaling_results_csv(results::Vector{ScalingDataPoint}, filepath::String)

Export scaling results to CSV.
"""
function export_scaling_results_csv(results::Vector{ScalingDataPoint}, filepath::String)
    open(filepath, "w") do f
        # Header
        println(f, "parameter_name,parameter_value,mean_time,std_time,mean_bond_dim,std_bond_dim,n_samples")

        # Data rows
        for r in results
            println(f, join([
                r.parameter_name,
                r.parameter_value,
                @sprintf("%.6f", r.mean_time),
                @sprintf("%.6f", r.std_time),
                @sprintf("%.2f", r.mean_bond_dim),
                @sprintf("%.2f", r.std_bond_dim),
                r.n_samples
            ], ","))
        end
    end
    println("Saved: $(filepath)")
end

"""
    export_correctness_results_csv(results::Vector{CorrectnessResult}, filepath::String)

Export correctness results to CSV.
"""
function export_correctness_results_csv(results::Vector{CorrectnessResult}, filepath::String)
    open(filepath, "w") do f
        # Header
        println(f, "test_name,n_qubits,circuit_description,fidelity,max_amplitude_error,passed,threshold")

        # Data rows
        for r in results
            println(f, join([
                r.test_name,
                r.n_qubits,
                "\"$(r.circuit_description)\"",
                @sprintf("%.10f", r.fidelity),
                @sprintf("%.2e", r.max_amplitude_error),
                r.passed ? 1 : 0,
                @sprintf("%.2e", r.threshold)
            ], ","))
        end
    end
    println("Saved: $(filepath)")
end

"""
    create_output_directory(base_dir::String; timestamp::Bool=true) -> String

Create a timestamped output directory for benchmark results.
Returns the path to the created directory.
"""
function create_output_directory(base_dir::String; timestamp::Bool=true)
    if timestamp
        ts = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
        dir = joinpath(base_dir, ts)
    else
        dir = base_dir
    end

    mkpath(dir)
    mkpath(joinpath(dir, "figures"))
    mkpath(joinpath(dir, "csv"))

    println("Created output directory: $(dir)")
    return dir
end

"""
    save_metadata(metadata::BenchmarkMetadata, filepath::String)

Save benchmark metadata to JSON file.
"""
function save_metadata(metadata::BenchmarkMetadata, filepath::String)
    save_json(filepath, to_dict(metadata))
end

"""
    print_summary(results::ExperimentResults)

Print a formatted summary of experiment results.
"""
function print_summary(results::ExperimentResults)
    println()
    println("="^60)
    println("Experiment: $(results.experiment_name)")
    println("="^60)
    println("Description: $(results.description)")
    println("Wall time: $(@sprintf("%.2f", results.wall_time_s)) seconds")
    println()
    println("Summary:")
    for (k, v) in results.summary
        if v isa Float64
            println("  $(k): $(@sprintf("%.4f", v))")
        else
            println("  $(k): $(v)")
        end
    end
    println()
end

"""
    print_suite_summary(suite::BenchmarkSuite)

Print a formatted summary of the entire benchmark suite.
"""
function print_suite_summary(suite::BenchmarkSuite)
    println()
    println("="^70)
    println("CAMPS.jl Benchmark Suite Results")
    println("="^70)
    println()
    println("Metadata:")
    println("  Timestamp: $(suite.metadata.timestamp)")
    println("  Julia version: $(suite.metadata.julia_version)")
    println("  CAMPS version: $(suite.metadata.camps_version)")
    println("  Hostname: $(suite.metadata.hostname)")
    println("  Git commit: $(suite.metadata.git_commit)")
    println()
    println("Total runtime: $(@sprintf("%.2f", suite.total_time_s)) seconds")
    println()
    println("Experiments run: $(length(suite.experiments))")

    for (name, exp) in suite.experiments
        println()
        println("-"^50)
        println("$(name): $(exp.description)")
        println("  Results: $(length(exp.results)) data points")
        println("  Time: $(@sprintf("%.2f", exp.wall_time_s))s")

        # Print key summary stats
        if haskey(exp.summary, "exact_match_rate")
            println("  Exact match rate: $(@sprintf("%.1f", 100*exp.summary["exact_match_rate"]))%")
        end
        if haskey(exp.summary, "pass_rate")
            println("  Pass rate: $(@sprintf("%.1f", 100*exp.summary["pass_rate"]))%")
        end
        if haskey(exp.summary, "scaling_exponent")
            se = exp.summary["scaling_exponent"]
            if !isnan(se)
                println("  Scaling exponent: $(@sprintf("%.2f", se))")
            end
        end
    end

    println()
    println("="^70)
end

export to_dict, dict_to_json, save_json, save_results, save_experiment
export export_gf2_results_csv, export_strategy_results_csv
export export_scaling_results_csv, export_correctness_results_csv
export create_output_directory, save_metadata
export print_summary, print_suite_summary
