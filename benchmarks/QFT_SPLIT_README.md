# QFT Experiment Split Documentation

## Overview

The Quantum Fourier Transform (QFT) experiments can now be split into 3 independent parts to manage long-running benchmarks. This allows you to run the 72 QFT circuits in smaller batches without losing any experimental rigor or data.

## Split Strategy

The 72 QFT circuits are split by **qubit size** into 3 equal parts:

| Part | Subset | n_qubits | Densities | Realizations | Total Circuits |
|------|--------|----------|-----------|--------------|----------------|
| 1    | `qft1` | 4        | 3 (low, medium, high) | 8 | **24** |
| 2    | `qft2` | 6        | 3 (low, medium, high) | 8 | **24** |
| 3    | `qft3` | 8        | 3 (low, medium, high) | 8 | **24** |
| **Total** | `qft` | 4, 6, 8 | 3 | 8 | **72** |

### Why This Split?

- **Equal distribution**: Each part has exactly 24 circuits
- **Complete coverage**: Each part covers all 3 density levels (low, medium, high) with 8 realizations each
- **Logical grouping**: Splitting by qubit size keeps related experiments together
- **Deterministic**: All seeds are identical whether you run `qft` or `qft1 + qft2 + qft3`

## Usage

### Running Individual Parts

Run each part separately with the `medium` mode:

```bash
# Part 1: n=4 qubits (24 circuits)
julia --project=. benchmarks/run_parallel_benchmark_complete.jl medium qft1

# Part 2: n=6 qubits (24 circuits)
julia --project=. benchmarks/run_parallel_benchmark_complete.jl medium qft2

# Part 3: n=8 qubits (24 circuits)
julia --project=. benchmarks/run_parallel_benchmark_complete.jl medium qft3
```

### Running All QFT at Once

For comparison, you can still run all 72 circuits together:

```bash
# All QFT circuits (72 circuits)
julia --project=. benchmarks/run_parallel_benchmark_complete.jl medium qft
```

## Results

Each part will generate its own results file with a timestamp:

```
results/complete_benchmark/
├── results_20260116_123456.csv  # qft1 results (24 circuits)
├── results_20260116_134567.csv  # qft2 results (24 circuits)
└── results_20260116_145678.csv  # qft3 results (24 circuits)
```

### Combining Results

After running all parts, you can combine the CSV files into a single dataset:

```bash
# Option 1: Manual concatenation (preserving headers)
head -n 1 results_qft1.csv > qft_combined.csv
tail -n +2 results_qft1.csv >> qft_combined.csv
tail -n +2 results_qft2.csv >> qft_combined.csv
tail -n +2 results_qft3.csv >> qft_combined.csv

# Option 2: Using a combine script (if available)
julia combine_results.jl results_qft1.csv results_qft2.csv results_qft3.csv qft_combined.csv
```

## Verification

A verification script is provided to ensure the split is correct:

```bash
julia --project=. benchmarks/verify_qft_split.jl
```

This script verifies:
1. ✓ Circuit counts (24 + 24 + 24 = 72)
2. ✓ No overlaps between parts
3. ✓ Complete coverage of all experiments
4. ✓ Balanced parameter distribution
5. ✓ Deterministic seed generation

## Guarantees

### Experimental Rigor Maintained

- **Same experiments**: Running `qft1 + qft2 + qft3` produces identical experiments as `qft`
- **Same seeds**: All random seeds are deterministic and identical across splits
- **Same parameters**: Each configuration (n_qubits, density, realization) appears exactly once
- **No data loss**: Complete coverage of all 72 circuits

### Parameter Distribution

Each part maintains balanced coverage:

- **qft1**: 4 qubits × 3 densities × 8 realizations = 24 circuits
  - Low density: 8 circuits
  - Medium density: 8 circuits
  - High density: 8 circuits

- **qft2**: 6 qubits × 3 densities × 8 realizations = 24 circuits
  - Low density: 8 circuits
  - Medium density: 8 circuits
  - High density: 8 circuits

- **qft3**: 8 qubits × 3 densities × 8 realizations = 24 circuits
  - Low density: 8 circuits
  - Medium density: 8 circuits
  - High density: 8 circuits

## Implementation Details

### Code Changes

The split is implemented in `run_parallel_benchmark_complete.jl`:

1. **New subset options**: Added `qft1`, `qft2`, `qft3` alongside existing `qft`
2. **Filtering logic**: Filters experiments by `n_qubits` value after generation
3. **Documentation**: Updated help text and usage examples

### Seed Generation

Seeds are generated using a deterministic hash function:

```julia
seed = Int(hash(("QFT", n, density, real)) % UInt32)
```

This ensures that:
- Same parameters → Same seed
- Independent of whether you use `qft` or `qft1/qft2/qft3`
- Reproducible across runs

### Filter Implementation

```julia
elseif subset == "qft1"
    phase2_experiments = filter(exp ->
        exp.family_name == "Quantum Fourier Transform" &&
        exp.params[:n_qubits] == 4,
        phase2_experiments)
```

## Performance Considerations

### Estimated Runtime (per part)

Assuming `medium` mode with 8 realizations:

- **qft1** (n=4): ~2-4 hours
- **qft2** (n=6): ~4-6 hours
- **qft3** (n=8): ~6-10 hours

**Total**: ~12-20 hours (split across 3 runs)

Compare to running `qft` all at once: ~12-20 hours (single run)

### Advantages of Splitting

1. **Checkpoint progress**: Each part completes independently
2. **Resource management**: Can run parts at different times
3. **Failure recovery**: If one part fails, only rerun that part
4. **Parallel execution**: Can run parts on different machines
5. **Easier scheduling**: Fits into shorter time windows

## Example Workflow

### Sequential Execution

```bash
# Run parts sequentially
julia --project=. benchmarks/run_parallel_benchmark_complete.jl medium qft1
# Wait for completion...

julia --project=. benchmarks/run_parallel_benchmark_complete.jl medium qft2
# Wait for completion...

julia --project=. benchmarks/run_parallel_benchmark_complete.jl medium qft3
# Wait for completion...

# Combine results
cat results_qft1.csv > qft_combined.csv
tail -n +2 results_qft2.csv >> qft_combined.csv
tail -n +2 results_qft3.csv >> qft_combined.csv
```

### Parallel Execution (Multiple Machines)

```bash
# Machine 1:
julia --project=. benchmarks/run_parallel_benchmark_complete.jl medium qft1

# Machine 2:
julia --project=. benchmarks/run_parallel_benchmark_complete.jl medium qft2

# Machine 3:
julia --project=. benchmarks/run_parallel_benchmark_complete.jl medium qft3

# Later: Combine results on one machine
```

## Troubleshooting

### Q: Do I get the same results from split vs. full?

**A**: Yes! The experiments are identical. The only difference is how they're distributed across runs.

### Q: Can I split other families the same way?

**A**: Currently, only QFT is split. However, the same approach can be applied to other families if needed. Contact the developers if you need splits for Grover, VQE, or other families.

### Q: What if I only want to run one part?

**A**: That's fine! Each part is independent. However, for ML training, you should eventually run all parts to ensure complete dataset coverage.

### Q: Can I use different modes for different parts?

**A**: Not recommended. Always use the same mode (e.g., `medium`) for all parts to ensure consistent realization counts.

## Summary

The QFT split feature allows you to:
- ✅ Split 72 QFT circuits into 3 parts of 24 circuits each
- ✅ Maintain exact same experimental rigor and coverage
- ✅ Run experiments in smaller, manageable batches
- ✅ Combine results into a single dataset
- ✅ Verify correctness with automated testing

**Recommended Usage**:
```bash
julia --project=. benchmarks/run_parallel_benchmark_complete.jl medium qft1
julia --project=. benchmarks/run_parallel_benchmark_complete.jl medium qft2
julia --project=. benchmarks/run_parallel_benchmark_complete.jl medium qft3
```

This gives you the same 72 experiments as:
```bash
julia --project=. benchmarks/run_parallel_benchmark_complete.jl medium qft
```

But with better control over execution timing and resource management!
