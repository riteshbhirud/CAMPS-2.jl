# CAMPS.jl/benchmarks/config.jl
# Configuration parameters for benchmark experiments
#
# Provides full and quick parameter sets for publication and testing.

"""
    BenchmarkConfig

Configuration for benchmark experiments.

# Fields
- `name::String`: Configuration name ("full" or "quick")
- `n_qubits::Vector{Int}`: System sizes to test
- `t_counts::Vector{Int}`: T-gate counts to test
- `n_seeds::Int`: Number of random seeds per configuration
- `n_timing_samples::Int`: Number of timing repetitions
- `strategies::Vector{String}`: Strategies to compare
- `max_bond_dim::Int`: Maximum bond dimension for MPS
- `cutoff::Float64`: SVD cutoff for MPS truncation
"""
struct BenchmarkConfig
    name::String
    n_qubits::Vector{Int}
    t_counts::Vector{Int}
    n_seeds::Int
    n_timing_samples::Int
    strategies::Vector{String}
    max_bond_dim::Int
    cutoff::Float64
end

"""
    FULL_CONFIG

Full configuration for publication-quality benchmarks.
Expected runtime: ~12 hours on modern hardware.
"""
const FULL_CONFIG = BenchmarkConfig(
    "full",
    [4, 6, 8, 10, 12, 14, 16],        # n_qubits
    [2, 4, 6, 8, 10, 12, 14, 16],     # t_counts
    10,                                # n_seeds
    5,                                 # n_timing_samples
    ["OFD", "OBD_1", "OBD_3", "OBD_5", "Hybrid", "None"],  # strategies
    512,                               # max_bond_dim
    1e-12                              # cutoff
)

"""
    QUICK_CONFIG

Quick configuration for testing and development.
Expected runtime: ~30 minutes on modern hardware.
"""
const QUICK_CONFIG = BenchmarkConfig(
    "quick",
    [4, 6, 8, 10],                    # n_qubits
    [2, 4, 6, 8],                     # t_counts
    3,                                 # n_seeds
    3,                                 # n_timing_samples
    ["OFD", "OBD_3", "Hybrid"],       # strategies
    256,                               # max_bond_dim
    1e-10                              # cutoff
)

"""
    MINIMAL_CONFIG

Minimal configuration for smoke testing.
Expected runtime: ~5 minutes.
"""
const MINIMAL_CONFIG = BenchmarkConfig(
    "minimal",
    [4, 6],                           # n_qubits
    [2, 4],                           # t_counts
    2,                                 # n_seeds
    2,                                 # n_timing_samples
    ["OFD", "Hybrid"],                # strategies
    128,                               # max_bond_dim
    1e-8                               # cutoff
)

# Experiment-specific configurations

"""
    GF2ValidationConfig

Configuration for GF(2) validation experiment.
"""
struct GF2ValidationConfig
    n_qubits::Vector{Int}
    t_counts::Vector{Int}
    n_seeds::Int
    circuit_types::Vector{String}
end

const GF2_FULL_CONFIG = GF2ValidationConfig(
    [4, 6, 8, 10, 12, 14, 16],
    [2, 4, 6, 8, 10, 12, 14, 16],
    10,
    ["independent", "cnot_chain", "cz_pairs", "random_clifford_t", "qft"]
)

const GF2_QUICK_CONFIG = GF2ValidationConfig(
    [4, 6, 8, 10],
    [2, 4, 6, 8],
    3,
    ["independent", "cnot_chain", "random_clifford_t"]
)

"""
    StrategyComparisonConfig

Configuration for strategy comparison experiment.
"""
struct StrategyComparisonConfig
    n_qubits::Vector{Int}
    t_counts::Vector{Int}
    n_seeds::Int
    n_timing_samples::Int
    strategies::Vector{Tuple{String,Any}}  # (name, strategy_object)
end

# Strategy objects will be created at runtime using CAMPS types

const STRATEGY_FULL_CONFIG = StrategyComparisonConfig(
    [6, 8, 10, 12, 14],
    [4, 8, 12, 16, 20],
    3,
    5,
    [
        ("OFD", nothing),      # Will be OFDStrategy()
        ("OBD_1", 1),          # Will be OBDStrategy(max_sweeps=1)
        ("OBD_3", 3),          # Will be OBDStrategy(max_sweeps=3)
        ("OBD_5", 5),          # Will be OBDStrategy(max_sweeps=5)
        ("Hybrid", nothing),   # Will be HybridStrategy()
        ("None", nothing)      # Will be NoDisentangling()
    ]
)

const STRATEGY_QUICK_CONFIG = StrategyComparisonConfig(
    [6, 8, 10],
    [4, 8, 12],
    2,
    3,
    [
        ("OFD", nothing),
        ("OBD_3", 3),
        ("Hybrid", nothing)
    ]
)

"""
    ScalingConfig

Configuration for scaling analysis experiment.
"""
struct ScalingConfig
    n_qubits_range::Vector{Int}     # For system size scaling
    t_counts_range::Vector{Int}      # For T-count scaling
    fixed_n::Int                     # Fixed n for T-count experiments
    fixed_t::Int                     # Fixed t for system size experiments
    n_timing_samples::Int
    n_seeds::Int
end

const SCALING_FULL_CONFIG = ScalingConfig(
    [4, 6, 8, 10, 12, 14, 16, 18, 20],  # n range
    [0, 2, 4, 6, 8, 10, 12, 14, 16],    # t range
    10,                                  # fixed_n
    8,                                   # fixed_t
    5,                                   # timing samples
    3                                    # seeds
)

const SCALING_QUICK_CONFIG = ScalingConfig(
    [4, 6, 8, 10, 12],
    [0, 2, 4, 6, 8],
    8,
    4,
    3,
    2
)

"""
    HybridBenefitConfig

Configuration for hybrid strategy benefit experiment.
"""
struct HybridBenefitConfig
    n_qubits::Vector{Int}
    circuit_types::Vector{String}
    n_seeds::Int
    n_timing_samples::Int
end

const HYBRID_FULL_CONFIG = HybridBenefitConfig(
    [6, 8, 10, 12],
    ["ofd_favorable", "ofd_unfavorable", "mixed", "hardware_efficient"],
    5,
    5
)

const HYBRID_QUICK_CONFIG = HybridBenefitConfig(
    [6, 8],
    ["ofd_favorable", "ofd_unfavorable", "mixed"],
    3,
    3
)

"""
    CorrectnessConfig

Configuration for correctness verification experiment.
"""
struct CorrectnessConfig
    max_n_qubits::Int               # Max for exact state vector comparison
    test_cases::Vector{String}      # Which tests to run
    fidelity_threshold::Float64
    amplitude_threshold::Float64
end

const CORRECTNESS_FULL_CONFIG = CorrectnessConfig(
    6,
    ["initial_state", "single_h", "ghz", "bell", "qft_small", "random_clifford_t", "t_gate_phases"],
    1.0 - 1e-10,
    1e-10
)

const CORRECTNESS_QUICK_CONFIG = CorrectnessConfig(
    5,
    ["initial_state", "ghz", "random_clifford_t"],
    1.0 - 1e-8,
    1e-8
)

# Helper functions

"""
    get_config(mode::Symbol) -> BenchmarkConfig

Get benchmark configuration by mode (:full, :quick, or :minimal).
"""
function get_config(mode::Symbol)
    if mode == :full
        return FULL_CONFIG
    elseif mode == :quick
        return QUICK_CONFIG
    elseif mode == :minimal
        return MINIMAL_CONFIG
    else
        error("Unknown config mode: $mode. Use :full, :quick, or :minimal")
    end
end

"""
    estimate_runtime(config::BenchmarkConfig) -> Float64

Estimate benchmark runtime in hours based on configuration.
"""
function estimate_runtime(config::BenchmarkConfig)
    # Rough estimates based on typical operation times
    n_circuits = length(config.n_qubits) * length(config.t_counts) * config.n_seeds
    n_strategy_runs = n_circuits * length(config.strategies) * config.n_timing_samples

    # Estimate ~0.1 seconds per simple run, scales with n^2 and t
    avg_n = sum(config.n_qubits) / length(config.n_qubits)
    avg_t = sum(config.t_counts) / length(config.t_counts)

    time_per_run_s = 0.1 * (avg_n / 8)^2 * (1 + avg_t / 8)
    total_s = n_strategy_runs * time_per_run_s

    return total_s / 3600  # Convert to hours
end

"""
    print_config(config::BenchmarkConfig)

Print configuration details.
"""
function print_config(config::BenchmarkConfig)
    println("Benchmark Configuration: $(config.name)")
    println("="^50)
    println("  n_qubits: $(config.n_qubits)")
    println("  t_counts: $(config.t_counts)")
    println("  n_seeds: $(config.n_seeds)")
    println("  n_timing_samples: $(config.n_timing_samples)")
    println("  strategies: $(config.strategies)")
    println("  max_bond_dim: $(config.max_bond_dim)")
    println("  cutoff: $(config.cutoff)")
    println()
    println("  Estimated runtime: $(round(estimate_runtime(config), digits=2)) hours")
end

export BenchmarkConfig, FULL_CONFIG, QUICK_CONFIG, MINIMAL_CONFIG
export GF2ValidationConfig, GF2_FULL_CONFIG, GF2_QUICK_CONFIG
export StrategyComparisonConfig, STRATEGY_FULL_CONFIG, STRATEGY_QUICK_CONFIG
export ScalingConfig, SCALING_FULL_CONFIG, SCALING_QUICK_CONFIG
export HybridBenefitConfig, HYBRID_FULL_CONFIG, HYBRID_QUICK_CONFIG
export CorrectnessConfig, CORRECTNESS_FULL_CONFIG, CORRECTNESS_QUICK_CONFIG
export get_config, estimate_runtime, print_config
