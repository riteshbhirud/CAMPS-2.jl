# CAMPS.jl Benchmark Suite

Publication-quality benchmarking suite for the CAMPS.jl quantum simulation package.

## Overview

This benchmark suite generates the data and figures for the CAMPS.jl paper, demonstrating:

1. **GF(2) Bond Dimension Prediction**: Validates that χ = 2^(t - rank) holds in practice
2. **OFD vs OBD Comparison**: Shows OFD is 10-100× faster when applicable
3. **Scaling Analysis**: Demonstrates polynomial (not exponential) scaling with system size
4. **Hybrid Strategy Benefits**: Combines OFD optimality with OBD fallback

## Quick Start

### Running Benchmarks

```bash
# Quick mode (~30 minutes) - for testing
julia --project=. benchmarks/run_all_benchmarks.jl --quick

# Full mode (~12 hours) - for publication
julia --project=. benchmarks/run_all_benchmarks.jl --full

# Minimal mode (~5 minutes) - smoke testing
julia --project=. benchmarks/run_all_benchmarks.jl --minimal
```

### From Julia REPL

```julia
using CAMPS
include("benchmarks/run_all_benchmarks.jl")

# Run quick benchmarks
suite = run_publication_benchmarks(mode=:quick)

# Run full benchmarks with custom output
suite = run_publication_benchmarks(
    mode=:full,
    output_dir="my_results",
    generate_figures=true,
    save_data=true
)
```

## Experiments

### Experiment 1: GF(2) Prediction Validation

**File**: `experiments/exp1_gf2_validation.jl`

Tests the core theoretical prediction χ = 2^(t - rank) across circuit families:
- Independent T gates (rank = t, χ = 1)
- CNOT-chain entangled circuits
- CZ-pairs entangled circuits
- Random Clifford+T circuits
- QFT circuits

**Key metrics**: Exact match rate, mean relative error

### Experiment 2: OFD vs OBD Comparison

**File**: `experiments/exp2_ofd_vs_obd.jl`

Systematic head-to-head comparison of disentangling strategies:
- Wall-clock time vs T-count
- Wall-clock time vs system size
- Final bond dimension achieved
- OFD success rate

**Strategies tested**: OFD, OBD (1/3/5 sweeps), Hybrid, None

### Experiment 3: Scaling Analysis

**File**: `experiments/exp3_scaling.jl`

Analyzes scaling with:
- System size (n qubits) for Clifford-only circuits
- System size for Clifford+T with OFD
- T-gate count at fixed n
- Bond dimension growth patterns

**Expected scaling**: O(n²) for Cliffords, polynomial for Clifford+T with OFD

### Experiment 4: Hybrid Strategy Benefits

**File**: `experiments/exp4_hybrid_strategy.jl`

Shows when each strategy excels:
- OFD-favorable: H layer → T gates
- OFD-unfavorable: T gates without H
- Mixed: Some qubits with H, some without
- Hardware-efficient ansatz circuits

### Experiment 5: Correctness Verification

**File**: `experiments/exp5_correctness.jl`

Verifies CAMPS produces correct quantum states:
- Initial state |0⟩^n
- Single Hadamard gate
- GHZ state preparation
- Bell state preparation
- Small QFT circuits
- T gate phase verification

## Output Structure

```
results/
└── 2025-01-01_120000/           # Timestamped run
    ├── metadata.json            # Run metadata
    ├── benchmark_suite.json     # Complete results
    ├── exp1_gf2_validation.json
    ├── exp2_ofd_vs_obd.json
    ├── exp3_scaling.json
    ├── exp4_hybrid_strategy.json
    ├── exp5_correctness.json
    ├── csv/
    │   ├── gf2_validation.csv
    │   ├── strategy_comparison.csv
    │   ├── scaling.csv
    │   └── correctness.csv
    └── figures/
        ├── fig1_gf2_validation.pdf
        ├── fig2_ofd_vs_obd.pdf
        ├── fig3_scaling.pdf
        └── fig4_hybrid_benefits.pdf
```

## Configuration

Configurations are defined in `config.jl`:

| Mode | n_qubits | t_counts | Seeds | Est. Time |
|------|----------|----------|-------|-----------|
| minimal | [4, 6] | [2, 4] | 2 | ~5 min |
| quick | [4-10] | [2-8] | 3 | ~30 min |
| full | [4-16] | [2-16] | 10 | ~12 hours |

## Figure Generation

Figures are generated using CairoMakie for publication-quality vector graphics.

```julia
# Generate figures from saved data
include("benchmarks/plotting.jl")
include("benchmarks/data_io.jl")

# Load results (if you have JSON parsing)
# suite = load_results("results/2025-01-01_120000/benchmark_suite.json")

# Generate all figures
generate_all_figures(suite, "output/figures")
```

### Figure Specifications

| Figure | Size | Format | Content |
|--------|------|--------|---------|
| Fig 1 | Single column | PDF | GF(2) prediction scatter |
| Fig 2 | Double column | PDF | OFD vs OBD (4 panels) |
| Fig 3 | Double column | PDF | Scaling analysis (4 panels) |
| Fig 4 | Single column | PDF | Hybrid strategy comparison |

## Dependencies

Required packages (added to Project.toml):
- `CairoMakie` - Publication-quality plotting
- `Statistics` - Statistical functions
- `Dates` - Timestamping
- Core CAMPS dependencies (ITensors, QuantumClifford)

Install with:
```julia
] add CairoMakie
```

## Reproducibility

Each benchmark run includes:
- Timestamp
- Julia version
- CAMPS version
- Git commit hash
- Hardware information (CPU, memory)
- Hostname

To reproduce figures from saved data:
```julia
reproduce_paper_figures("results/2025-01-01_120000/")
```

## Hardware Requirements

| Mode | RAM | CPU Cores | Recommended |
|------|-----|-----------|-------------|
| minimal | 4 GB | 1 | Any modern laptop |
| quick | 8 GB | 2-4 | Modern laptop |
| full | 16 GB | 4-8 | Workstation or cluster |

## Troubleshooting

### CairoMakie not installed
```
WARNING: CairoMakie not available. Install with: ] add CairoMakie
```
The benchmark will still run and save data, but won't generate figures.

### Out of memory
Reduce `max_bond_dim` in config or use fewer qubits.

### Slow performance
- Ensure Julia is started with multiple threads: `julia -t auto`
- Use `--quick` mode for development

## File Structure

```
benchmarks/
├── README.md                    # This file
├── benchmark_types.jl           # Data structures for results
├── config.jl                    # Parameter configurations
├── data_io.jl                   # JSON/CSV I/O functions
├── plotting.jl                  # CairoMakie figure generation
├── run_all_benchmarks.jl        # Master runner script
├── experiments/
│   ├── exp1_gf2_validation.jl   # GF(2) prediction accuracy
│   ├── exp2_ofd_vs_obd.jl       # Strategy comparison
│   ├── exp3_scaling.jl          # Scaling analysis
│   ├── exp4_hybrid_strategy.jl  # Hybrid benefits
│   └── exp5_correctness.jl      # Correctness verification
├── gf2_accuracy.jl              # Legacy benchmark
├── strategy_comparison.jl       # Legacy benchmark
└── scaling.jl                   # Legacy benchmark
```

## Citation

If you use this benchmark suite, please cite:
```bibtex
@article{camps2025,
  title={CAMPS.jl: Clifford-Augmented Matrix Product States for Quantum Simulation},
  author={...},
  journal={Quantum},
  year={2025}
}
```

## License

MIT License - see main repository.
