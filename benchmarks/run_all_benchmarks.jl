# CAMPS.jl/benchmarks/run_all_benchmarks.jl
# Master script to run all publication benchmarks
#
# Usage:
#   julia --project=. benchmarks/run_all_benchmarks.jl [--quick] [--full] [--minimal]
#
# This script orchestrates all experiments and generates figures.

using Dates
using Printf

# Add CAMPS to load path
push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using CAMPS

# Include benchmark modules
include("benchmark_types.jl")
include("config.jl")
include("data_io.jl")
include("plotting.jl")

# Include experiments
include("experiments/exp1_gf2_validation.jl")
include("experiments/exp2_ofd_vs_obd.jl")
include("experiments/exp3_scaling.jl")
include("experiments/exp4_hybrid_strategy.jl")
include("experiments/exp5_correctness.jl")

"""
    run_publication_benchmarks(;
        mode::Symbol = :quick,
        output_dir::String = nothing,
        generate_figures::Bool = true,
        save_data::Bool = true,
        verbose::Bool = true
    )

Run the complete benchmark suite for publication.

# Arguments
- `mode::Symbol`: :quick, :full, or :minimal
- `output_dir::String`: Output directory (auto-generated if nothing)
- `generate_figures::Bool`: Whether to generate PDF figures
- `save_data::Bool`: Whether to save raw data as JSON
- `verbose::Bool`: Whether to print progress

# Returns
- `BenchmarkSuite`: Complete benchmark results
"""
function run_publication_benchmarks(;
    mode::Symbol = :quick,
    output_dir::Union{String,Nothing} = nothing,
    generate_figures::Bool = true,
    save_data::Bool = true,
    verbose::Bool = true
)
    start_time = time()

    # Create output directory
    if output_dir === nothing
        ts = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
        output_dir = joinpath(@__DIR__, "..", "results", ts)
    end

    mkpath(output_dir)
    mkpath(joinpath(output_dir, "figures"))
    mkpath(joinpath(output_dir, "csv"))

    if verbose
        println()
        println("╔" * "═"^68 * "╗")
        println("║" * " "^20 * "CAMPS.jl Benchmark Suite" * " "^24 * "║")
        println("╚" * "═"^68 * "╝")
        println()
        println("Mode: $mode")
        println("Output: $output_dir")
        println("Started: $(now())")
        println()
    end

    # Create metadata
    metadata = BenchmarkMetadata(notes = "Mode: $mode")

    if save_data
        save_metadata(metadata, joinpath(output_dir, "metadata.json"))
    end

    # Select configurations based on mode
    configs = get_configs_for_mode(mode)

    # Run experiments
    experiments = Dict{String,ExperimentResults}()

    # Experiment 1: GF(2) Validation
    if verbose
        println()
        println("┌" * "─"^68 * "┐")
        println("│ Experiment 1: GF(2) Prediction Validation" * " "^25 * "│")
        println("└" * "─"^68 * "┘")
    end

    exp1_results = run_gf2_validation_experiment(
        config = configs[:gf2],
        verbose = verbose
    )
    experiments["gf2_validation"] = exp1_results

    if save_data
        save_experiment(exp1_results, joinpath(output_dir, "exp1_gf2_validation.json"))
        export_gf2_results_csv(exp1_results.results,
                               joinpath(output_dir, "csv", "gf2_validation.csv"))
    end

    # Experiment 2: OFD vs OBD
    if verbose
        println()
        println("┌" * "─"^68 * "┐")
        println("│ Experiment 2: OFD vs OBD Strategy Comparison" * " "^22 * "│")
        println("└" * "─"^68 * "┘")
    end

    exp2_results = run_ofd_vs_obd_experiment(
        config = configs[:strategy],
        verbose = verbose
    )
    experiments["ofd_vs_obd"] = exp2_results

    if save_data
        save_experiment(exp2_results, joinpath(output_dir, "exp2_ofd_vs_obd.json"))
        export_strategy_results_csv(exp2_results.results,
                                    joinpath(output_dir, "csv", "strategy_comparison.csv"))
    end

    # Experiment 3: Scaling Analysis
    if verbose
        println()
        println("┌" * "─"^68 * "┐")
        println("│ Experiment 3: Scaling Analysis" * " "^36 * "│")
        println("└" * "─"^68 * "┘")
    end

    exp3_results = run_scaling_experiment(
        config = configs[:scaling],
        verbose = verbose
    )
    experiments["scaling"] = exp3_results

    if save_data
        save_experiment(exp3_results, joinpath(output_dir, "exp3_scaling.json"))
        export_scaling_results_csv(exp3_results.results,
                                   joinpath(output_dir, "csv", "scaling.csv"))
    end

    # Experiment 4: Hybrid Strategy Benefits
    if verbose
        println()
        println("┌" * "─"^68 * "┐")
        println("│ Experiment 4: Hybrid Strategy Benefits" * " "^28 * "│")
        println("└" * "─"^68 * "┘")
    end

    exp4_results = run_hybrid_strategy_experiment(
        config = configs[:hybrid],
        verbose = verbose
    )
    experiments["hybrid_strategy"] = exp4_results

    if save_data
        save_experiment(exp4_results, joinpath(output_dir, "exp4_hybrid_strategy.json"))
    end

    # Experiment 5: Correctness Verification
    if verbose
        println()
        println("┌" * "─"^68 * "┐")
        println("│ Experiment 5: Correctness Verification" * " "^28 * "│")
        println("└" * "─"^68 * "┘")
    end

    exp5_results = run_correctness_experiment(
        config = configs[:correctness],
        verbose = verbose
    )
    experiments["correctness"] = exp5_results

    if save_data
        save_experiment(exp5_results, joinpath(output_dir, "exp5_correctness.json"))
        export_correctness_results_csv(exp5_results.results,
                                       joinpath(output_dir, "csv", "correctness.csv"))
    end

    # Create benchmark suite
    total_time = time() - start_time
    suite = BenchmarkSuite(metadata, experiments, total_time)

    if save_data
        save_results(suite, joinpath(output_dir, "benchmark_suite.json"))
    end

    # Generate figures
    if generate_figures
        if verbose
            println()
            println("┌" * "─"^68 * "┐")
            println("│ Generating Publication Figures" * " "^36 * "│")
            println("└" * "─"^68 * "┘")
        end

        figures_dir = joinpath(output_dir, "figures")
        generate_all_figures(suite, figures_dir)
    end

    # Print final summary
    if verbose
        print_suite_summary(suite)

        println()
        println("╔" * "═"^68 * "╗")
        println("║ Benchmark Complete!" * " "^49 * "║")
        println("╠" * "═"^68 * "╣")
        @printf("║ Total time: %.2f seconds%s║\n", total_time, " "^(44 - length(@sprintf("%.2f", total_time))))
        println("║ Output directory: $(output_dir[1:min(49, length(output_dir))])" * " "^max(0, 49 - length(output_dir)) * "║")
        println("╚" * "═"^68 * "╝")
    end

    suite
end

"""
    get_configs_for_mode(mode::Symbol) -> Dict

Get experiment configurations for the specified mode.
"""
function get_configs_for_mode(mode::Symbol)
    if mode == :full
        Dict(
            :gf2 => GF2_FULL_CONFIG,
            :strategy => STRATEGY_FULL_CONFIG,
            :scaling => SCALING_FULL_CONFIG,
            :hybrid => HYBRID_FULL_CONFIG,
            :correctness => CORRECTNESS_FULL_CONFIG
        )
    elseif mode == :quick
        Dict(
            :gf2 => GF2_QUICK_CONFIG,
            :strategy => STRATEGY_QUICK_CONFIG,
            :scaling => SCALING_QUICK_CONFIG,
            :hybrid => HYBRID_QUICK_CONFIG,
            :correctness => CORRECTNESS_QUICK_CONFIG
        )
    elseif mode == :minimal
        # Minimal configs for smoke testing
        Dict(
            :gf2 => GF2ValidationConfig([4, 6], [2, 4], 2, ["independent", "random_clifford_t"]),
            :strategy => StrategyComparisonConfig([6], [4], 1, 2, [("OFD", nothing), ("Hybrid", nothing)]),
            :scaling => ScalingConfig([4, 6, 8], [0, 2, 4], 6, 2, 2, 1),
            :hybrid => HybridBenefitConfig([6], ["ofd_favorable", "mixed"], 2, 2),
            :correctness => CorrectnessConfig(4, ["initial_state", "ghz"], 1.0 - 1e-8, 1e-8)
        )
    else
        error("Unknown mode: $mode. Use :full, :quick, or :minimal")
    end
end

"""
    reproduce_paper_figures(data_dir::String; output_dir::String = nothing)

Regenerate figures from saved benchmark data.
"""
function reproduce_paper_figures(data_dir::String; output_dir::Union{String,Nothing} = nothing)
    if output_dir === nothing
        output_dir = joinpath(data_dir, "figures_regenerated")
    end
    mkpath(output_dir)

    println("Loading benchmark data from: $data_dir")

    # Load individual experiment results and regenerate figures
    # This allows regenerating figures without re-running benchmarks

    # For full implementation, we'd need JSON parsing
    # For now, provide instructions
    println("To regenerate figures:")
    println("  1. Load the benchmark data: suite = load_results(\"$data_dir/benchmark_suite.json\")")
    println("  2. Generate figures: generate_all_figures(suite, \"$output_dir\")")
end

# Command line interface
function main()
    mode = :quick

    for arg in ARGS
        if arg == "--quick"
            mode = :quick
        elseif arg == "--full"
            mode = :full
        elseif arg == "--minimal"
            mode = :minimal
        elseif arg == "--help" || arg == "-h"
            println("CAMPS.jl Benchmark Suite")
            println()
            println("Usage: julia --project=. benchmarks/run_all_benchmarks.jl [OPTIONS]")
            println()
            println("Options:")
            println("  --minimal    Run minimal smoke tests (~5 minutes)")
            println("  --quick      Run quick benchmarks (~30 minutes) [default]")
            println("  --full       Run full publication benchmarks (~12 hours)")
            println("  --help, -h   Show this help message")
            println()
            println("Output is saved to results/<timestamp>/")
            return
        else
            @warn "Unknown argument: $arg"
        end
    end

    run_publication_benchmarks(mode = mode)
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

export run_publication_benchmarks, reproduce_paper_figures
