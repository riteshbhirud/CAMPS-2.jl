# Split Experiments Documentation (QFT & VQE)

## Overview

Both the **Quantum Fourier Transform (QFT)** and **VQE Hardware-Efficient** experiments can now be split into 3 independent parts to manage long-running benchmarks. This allows you to run 72 circuits in smaller batches of 24 circuits each, without losing any experimental rigor or data.

## Split Strategy

Both families use the same split strategy: **split by qubit size** into 3 equal parts.

### QFT Split (72 → 3×24 circuits)

| Part | Subset | n_qubits | Densities | Realizations | Total Circuits |
|------|--------|----------|-----------|--------------|----------------|
| 1    | `qft1` | 4        | 3 (low, medium, high) | 8 | **24** |
| 2    | `qft2` | 6        | 3 (low, medium, high) | 8 | **24** |
| 3    | `qft3` | 8        | 3 (low, medium, high) | 8 | **24** |
| **Total** | `qft` | 4, 6, 8 | 3 | 8 | **72** |

### VQE Split (72 → 3×24 circuits)

| Part | Subset | n_qubits | Layers | Realizations | Total Circuits |
|------|--------|----------|--------|--------------|----------------|
| 1    | `vqe1` | 4        | 3 (1, 2, 4) | 8 | **24** |
| 2    | `vqe2` | 6        | 3 (1, 2, 4) | 8 | **24** |
| 3    | `vqe3` | 8        | 3 (1, 2, 4) | 8 | **24** |
| **Total** | `vqe` | 4, 6, 8 | 3 | 8 | **72** |

## Why This Split?

✅ **Equal distribution**: Each part has exactly 24 circuits
✅ **Complete coverage**: Each part covers all parameter variations with 8 realizations
✅ **Logical grouping**: Splitting by qubit size keeps related experiments together
✅ **Deterministic**: All seeds are identical whether you run full or split
✅ **No data loss**: Complete coverage of all experiments

## Usage

### Running QFT Split Experiments

```bash
# Part 1: n=4 qubits (24 circuits)
julia --project=. benchmarks/run_parallel_benchmark_complete.jl medium qft1

# Part 2: n=6 qubits (24 circuits)
julia --project=. benchmarks/run_parallel_benchmark_complete.jl medium qft2

# Part 3: n=8 qubits (24 circuits)
julia --project=. benchmarks/run_parallel_benchmark_complete.jl medium qft3

# Or run all QFT at once (72 circuits)
julia --project=. benchmarks/run_parallel_benchmark_complete.jl medium qft
```

### Running VQE Split Experiments

```bash
# Part 1: n=4 qubits (24 circuits)
julia --project=. benchmarks/run_parallel_benchmark_complete.jl medium vqe1

# Part 2: n=6 qubits (24 circuits)
julia --project=. benchmarks/run_parallel_benchmark_complete.jl medium vqe2

# Part 3: n=8 qubits (24 circuits)
julia --project=. benchmarks/run_parallel_benchmark_complete.jl medium vqe3

# Or run all VQE at once (72 circuits)
julia --project=. benchmarks/run_parallel_benchmark_complete.jl medium vqe
```

## Results Management

Each part generates its own results file with a timestamp:

```
results/complete_benchmark/
├── results_20260117_100000.csv  # qft1 results (24 circuits)
├── results_20260117_110000.csv  # qft2 results (24 circuits)
├── results_20260117_120000.csv  # qft3 results (24 circuits)
├── results_20260117_130000.csv  # vqe1 results (24 circuits)
├── results_20260117_140000.csv  # vqe2 results (24 circuits)
└── results_20260117_150000.csv  # vqe3 results (24 circuits)
```

### Combining Results

After running all parts, combine the CSV files:

```bash
# Combine QFT parts
head -n 1 results_qft1.csv > qft_combined.csv
tail -n +2 results_qft1.csv >> qft_combined.csv
tail -n +2 results_qft2.csv >> qft_combined.csv
tail -n +2 results_qft3.csv >> qft_combined.csv

# Combine VQE parts
head -n 1 results_vqe1.csv > vqe_combined.csv
tail -n +2 results_vqe1.csv >> vqe_combined.csv
tail -n +2 results_vqe2.csv >> vqe_combined.csv
tail -n +2 results_vqe3.csv >> vqe_combined.csv
```

## Verification

A comprehensive verification script tests both QFT and VQE splits:

```bash
julia --project=. benchmarks/verify_splits.jl
```

### Verification Tests

✅ **Circuit counts**: 24 + 24 + 24 = 72 for both families
✅ **No overlaps**: Each experiment appears exactly once
✅ **Complete coverage**: All experiments from full run are in parts
✅ **Parameter distribution**: Balanced across all dimensions
✅ **Deterministic seeds**: Same seeds in split vs. full runs

## Guarantees

### QFT Guarantees

- **Same experiments**: `qft1 + qft2 + qft3` ≡ `qft`
- **Same seeds**: Deterministic hash-based generation
- **Same parameters**: Each (n_qubits, density, realization) appears exactly once
- **Complete coverage**: All 72 circuits accounted for

**Parameter Distribution per Part:**
- Each part: 1 qubit size × 3 densities × 8 realizations = 24 circuits
  - Low density: 8 circuits
  - Medium density: 8 circuits
  - High density: 8 circuits

### VQE Guarantees

- **Same experiments**: `vqe1 + vqe2 + vqe3` ≡ `vqe`
- **Same seeds**: Deterministic hash-based generation
- **Same parameters**: Each (n_qubits, layers, realization) appears exactly once
- **Complete coverage**: All 72 circuits accounted for

**Parameter Distribution per Part:**
- Each part: 1 qubit size × 3 layer counts × 8 realizations = 24 circuits
  - Layer 1: 8 circuits
  - Layer 2: 8 circuits
  - Layer 4: 8 circuits

## Performance Considerations

### Estimated Runtime per Part (Medium Mode)

**QFT:**
- `qft1` (n=4): ~2-4 hours
- `qft2` (n=6): ~4-6 hours
- `qft3` (n=8): ~6-10 hours
- **Total**: ~12-20 hours (split across 3 runs)

**VQE:**
- `vqe1` (n=4): ~2-4 hours
- `vqe2` (n=6): ~4-6 hours
- `vqe3` (n=8): ~6-10 hours
- **Total**: ~12-20 hours (split across 3 runs)

### Advantages of Splitting

1. **Checkpoint progress**: Each part completes independently
2. **Resource management**: Run parts at different times
3. **Failure recovery**: Only rerun failed parts
4. **Parallel execution**: Run parts on different machines
5. **Easier scheduling**: Fits into shorter time windows

## Example Workflows

### Sequential Execution (Single Machine)

```bash
# Run QFT parts sequentially
julia --project=. benchmarks/run_parallel_benchmark_complete.jl medium qft1
julia --project=. benchmarks/run_parallel_benchmark_complete.jl medium qft2
julia --project=. benchmarks/run_parallel_benchmark_complete.jl medium qft3

# Run VQE parts sequentially
julia --project=. benchmarks/run_parallel_benchmark_complete.jl medium vqe1
julia --project=. benchmarks/run_parallel_benchmark_complete.jl medium vqe2
julia --project=. benchmarks/run_parallel_benchmark_complete.jl medium vqe3
```

### Parallel Execution (Multiple Machines)

```bash
# Machine 1: QFT Part 1 + VQE Part 1
julia --project=. benchmarks/run_parallel_benchmark_complete.jl medium qft1
julia --project=. benchmarks/run_parallel_benchmark_complete.jl medium vqe1

# Machine 2: QFT Part 2 + VQE Part 2
julia --project=. benchmarks/run_parallel_benchmark_complete.jl medium qft2
julia --project=. benchmarks/run_parallel_benchmark_complete.jl medium vqe2

# Machine 3: QFT Part 3 + VQE Part 3
julia --project=. benchmarks/run_parallel_benchmark_complete.jl medium qft3
julia --project=. benchmarks/run_parallel_benchmark_complete.jl medium vqe3

# Later: Combine all results
```

## Implementation Details

### Seed Generation

Both families use deterministic hash-based seed generation:

```julia
# QFT
seed = Int(hash(("QFT", n, density, real)) % UInt32)

# VQE
seed = Int(hash(("VQE", n, layers, real)) % UInt32)
```

This ensures:
- Same parameters → Same seed
- Independent of split vs. full run
- Reproducible across runs

### Filter Implementation

```julia
# QFT filtering
elseif subset == "qft1"
    phase2_experiments = filter(exp ->
        exp.family_name == "Quantum Fourier Transform" &&
        exp.params[:n_qubits] == 4,
        phase2_experiments)

# VQE filtering
elseif subset == "vqe1"
    phase2_experiments = filter(exp ->
        exp.family_name == "VQE Hardware-Efficient Ansatz" &&
        exp.params[:n_qubits] == 4,
        phase2_experiments)
```

## Troubleshooting

### Q: Do I get the same results from split vs. full?

**A**: Yes! The experiments are identical. The only difference is how they're distributed across runs.

### Q: Can I run only some parts?

**A**: Yes, each part is independent. However, for complete ML training datasets, you should eventually run all parts.

### Q: Can I use different modes for different parts?

**A**: Not recommended. Always use the same mode (e.g., `medium`) for all parts to ensure consistent realization counts.

### Q: What if one part fails?

**A**: Simply rerun that part. The other parts' results are still valid.

### Q: Can I split other families?

**A**: Currently, only QFT and VQE have built-in splits. Contact developers if you need splits for other families (Grover, QAOA, Surface Code, etc.).

## Summary Table

| Family | Full Subset | Part 1 | Part 2 | Part 3 | Total Circuits | Split By |
|--------|-------------|--------|--------|--------|----------------|----------|
| QFT    | `qft`       | `qft1` (n=4) | `qft2` (n=6) | `qft3` (n=8) | 72 | Qubit size |
| VQE    | `vqe`       | `vqe1` (n=4) | `vqe2` (n=6) | `vqe3` (n=8) | 72 | Qubit size |

## Quick Reference

### Run All Parts

```bash
# QFT
julia --project=. benchmarks/run_parallel_benchmark_complete.jl medium qft1
julia --project=. benchmarks/run_parallel_benchmark_complete.jl medium qft2
julia --project=. benchmarks/run_parallel_benchmark_complete.jl medium qft3

# VQE
julia --project=. benchmarks/run_parallel_benchmark_complete.jl medium vqe1
julia --project=. benchmarks/run_parallel_benchmark_complete.jl medium vqe2
julia --project=. benchmarks/run_parallel_benchmark_complete.jl medium vqe3
```

### Verify Correctness

```bash
julia --project=. benchmarks/verify_splits.jl
```

### Combine Results

```bash
# QFT
head -n 1 qft1.csv > qft_combined.csv
tail -n +2 -q qft1.csv qft2.csv qft3.csv >> qft_combined.csv

# VQE
head -n 1 vqe1.csv > vqe_combined.csv
tail -n +2 -q vqe1.csv vqe2.csv vqe3.csv >> vqe_combined.csv
```

---

**Key Takeaway**: Running split experiments (`qft1 + qft2 + qft3` or `vqe1 + vqe2 + vqe3`) produces **exactly the same** 72 experiments as running the full experiments (`qft` or `vqe`), just distributed across multiple runs for better resource management!
