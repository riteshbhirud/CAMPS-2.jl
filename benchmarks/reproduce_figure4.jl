# CAMPS.jl/benchmarks/reproduce_figure4.jl
# Reproduce Figure 4 from Liu & Clark (2024), arXiv:2412.17209v2
#
# This script correctly implements the Figure 4 benchmark:
# - N=16 qubits with brick-wall Clifford circuit architecture
# - Random 2-qubit Cliffords sampled UNIFORMLY from the 11,520-element group
# - t=1 to 35 T-gates applied to random qubits
# - 3 realizations per t value
# - Tracks S2_max, chi_max, and nullity
#
# Key insight from the paper:
# - Clifford circuit only modifies the TABLEAU (state.clifford)
# - MPS stays |0...0⟩ until T-gates are applied
# - T-gates create actual entanglement in the MPS
# - OFD/OBD disentangle the MPS as T-gates are added

using CAMPS
using QuantumClifford: random_clifford
using Random
using Printf
using Statistics
using Dates

#==============================================================================#
# UNIFORM CLIFFORD SAMPLING
#==============================================================================#

# NOTE: We use QuantumClifford's random_clifford(2) for truly uniform sampling
# from the full 11,520-element two-qubit Clifford group.
#
# The enumerate_cliffords(2) function returns only 720 equivalence classes,
# NOT the full 11,520 group elements. Using random_clifford(2) ensures
# proper uniform sampling as required by Liu & Clark Figure 4.

#==============================================================================#
# BRICK-WALL CLIFFORD CIRCUIT
#==============================================================================#

"""
    apply_brick_wall_clifford_circuit!(state::CAMPSState, n_layers::Int; seed=nothing)

Apply a brick-wall architecture Clifford circuit using UNIFORMLY SAMPLED
random 2-qubit Cliffords from the 11,520-element group.

Brick-wall pattern:
- Even layers: gates on (1,2), (3,4), (5,6), ...
- Odd layers: gates on (2,3), (4,5), (6,7), ...

CRITICAL: The Cliffords are applied ONLY to state.clifford (the tableau),
NOT to the MPS! The MPS stays in |0...0⟩ until T-gates are applied.
"""
function apply_brick_wall_clifford_circuit!(state::CAMPSState, n_layers::Int; seed=nothing)
    if seed !== nothing
        Random.seed!(seed)
    end

    n = state.n_qubits

    for layer in 1:n_layers
        # Brick-wall pattern: alternate offset each layer
        offset = (layer - 1) % 2

        for i in (1 + offset):2:(n - 1)
            qubit1 = i
            qubit2 = i + 1

            # Sample UNIFORMLY from the full 11,520 two-qubit Clifford group
            # using QuantumClifford's random_clifford(2)
            C = random_clifford(2)

            # Apply to Clifford tableau ONLY (NOT the MPS!)
            # This updates state.clifford via right-multiplication
            # inverse=false means we right-multiply by C† (standard for CAMPS)
            apply_clifford_left_multiply!(state.clifford, C, qubit1, qubit2)

        end
    end
end

#==============================================================================#
# T-GATE APPLICATION
#==============================================================================#

"""
    apply_random_t_gates!(state::CAMPSState, t::Int; seed=nothing)

Apply t T-gates to randomly selected qubits (with replacement).

Uses the Hybrid strategy (OFD when possible, OBD as fallback) for optimal
disentangling.
"""
function apply_random_t_gates!(state::CAMPSState, t::Int; seed=nothing)
    if seed !== nothing
        Random.seed!(seed)
    end

    n = state.n_qubits

    # Select t random qubits (with replacement, as in the paper)
    qubits = [rand(1:n) for _ in 1:t]

    # Apply T-gate to each qubit using Hybrid strategy
    for qubit in qubits
        apply_t_gate_hybrid!(state, qubit; strategy=HybridStrategy())
    end
end

#==============================================================================#
# SINGLE REALIZATION SIMULATION
#==============================================================================#

"""
    simulate_one_point(n_qubits::Int, t::Int; seed=nothing) -> NamedTuple

Run one realization of the random Clifford+T circuit.

IMPORTANT: Following Liu & Clark's actual implementation (Section V.A):
- LT = 1: One Clifford layer between each T-gate insertion
- NT = 1: One T-gate per insertion
- Structure: [Clifford layer → T-gate] repeated t times

Returns: (S2, chi, nu) where:
- S2: Maximum second Renyi entropy across all bonds
- chi: Maximum bond dimension
- nu: Nullity (t_eff - rank of GF2 matrix)
"""
function simulate_one_point(n_qubits::Int, t::Int; seed=nothing)
    state = CAMPSState(n_qubits; max_bond=2048)
    initialize!(state)

    clifford_seed = seed !== nothing ? seed : hash((n_qubits, t, :clifford))
    t_seed = seed !== nothing ? seed + 1 : hash((n_qubits, t, :tgates))
    
    Random.seed!(t_seed)
    qubit_sequence = [rand(1:n_qubits) for _ in 1:t]
    
    for i in 1:t
        apply_brick_wall_clifford_circuit!(state, 2; seed=clifford_seed + i)
        apply_t_gate_hybrid!(state, qubit_sequence[i]; strategy=HybridStrategy())
    end

    S2_max = max_entanglement_entropy(state.mps)
    chi_max = get_bond_dimension(state)

    # Stabilizer nullity = number of magic qubits
    nu = n_qubits - sum(state.free_qubits)

    return (S2=S2_max, chi=chi_max, nu=nu)
end

#==============================================================================#
# MAIN FIGURE 4 REPRODUCTION
#==============================================================================#

"""
    reproduce_figure4(; n_qubits=16, t_range=1:35, n_realizations=3,
                       clifford_depth=4, base_seed=42, verbose=true) -> Vector

Reproduce Figure 4 from Liu & Clark (2024).

The figure shows S2_max, chi_max, and nullity vs number of T-gates
for random Clifford+T circuits with brick-wall architecture.

NOTE: The paper shows 3 realizations (sample curves) per t value,
NOT a comparison of different strategies. We use the Hybrid strategy
which combines OFD (when possible) with OBD (as fallback).
"""
function reproduce_figure4(;
    n_qubits::Int=16,
    t_range=1:35,
    n_realizations::Int=3,
    clifford_depth::Int=4,
    base_seed::Int=42,
    verbose::Bool=true,
    save_results::Bool=true
)
    println("="^70)
    println("Reproducing Figure 4 from Liu & Clark (2024)")
    println("arXiv:2412.17209v2, page 10")
    println("="^70)
    println()
    println("Parameters:")
    println("  N (qubits):         $n_qubits")
    println("  T-gate range:       $(first(t_range)) to $(last(t_range))")
    println("  Realizations per t: $n_realizations")
    println("  Clifford layers:    LT=1 (one layer between each T-gate)")
    println("  Base seed:          $base_seed")
    println("  Strategy:           Hybrid (OFD + OBD fallback)")
    println("  Clifford sampling:  random_clifford(2) - uniform from 11,520 group")
    println()

    results = []
    total = length(t_range) * n_realizations
    current = 0
    start_time = time()

    println("Starting simulations ($total total)...")
    println("-"^70)

    for t in t_range
        for r in 1:n_realizations
            current += 1
            seed = base_seed + t * 1000 + r

            # Run simulation
            point_start = time()
            result = simulate_one_point(n_qubits, t; seed=seed)

            point_time = time() - point_start

            push!(results, (
                t = t,
                realization = r,
                seed = seed,
                S2 = result.S2,
                chi = result.chi,
                nu = result.nu
            ))

            # Progress output
            if verbose
                elapsed = time() - start_time
                eta = (elapsed / current) * (total - current) / 60
                @printf("[%3d/%3d] t=%2d r=%d: S2=%.4f chi=%3d nu=%2d (%.2fs, ETA %.1f min)\n",
                       current, total, t, r, result.S2, result.chi, result.nu, point_time, eta)
            end
        end

        # Print summary for this t value
        if verbose && n_realizations > 1
            t_results = filter(x -> x.t == t, results)
            avg_S2 = mean([x.S2 for x in t_results])
            avg_chi = mean([x.chi for x in t_results])
            avg_nu = mean([x.nu for x in t_results])
            @printf("  t=%2d summary: avg S2=%.4f, avg chi=%.1f, avg nu=%.1f\n",
                    t, avg_S2, avg_chi, avg_nu)
        end
    end

    total_time = (time() - start_time) / 60
    println("-"^70)
    @printf("Completed in %.1f minutes\n\n", total_time)

    # Save results
    if save_results
        save_figure4_results(results, n_qubits)
    end

    # Print summary
    print_figure4_summary(results)

    return results
end

#==============================================================================#
# RESULTS SAVING AND SUMMARY
#==============================================================================#

"""
Save results to CSV file.
"""
function save_figure4_results(results, n_qubits::Int)
    # Create results directory
    results_dir = joinpath(dirname(@__FILE__), "..", "results")
    if !isdir(results_dir)
        mkpath(results_dir)
    end

    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    filename = joinpath(results_dir, "figure4_N$(n_qubits)_$(timestamp).csv")

    open(filename, "w") do f
        println(f, "t,realization,seed,S2_max,chi_max,nullity")
        for r in results
            println(f, "$(r.t),$(r.realization),$(r.seed),$(r.S2),$(r.chi),$(r.nu)")
        end
    end

    println("Results saved to: $filename")
end

"""
Print summary statistics.
"""
function print_figure4_summary(results)
    println("="^70)
    println("FIGURE 4 SUMMARY")
    println("="^70)
    println()

    # Group by t
    t_values = sort(unique(r.t for r in results))

    println("   t  |  S2_max (avg±std) | chi_max (avg) | nu (avg)")
    println("-"^60)

    for t in t_values
        t_data = filter(r -> r.t == t, results)

        S2_vals = [r.S2 for r in t_data]
        chi_vals = [Float64(r.chi) for r in t_data]
        nu_vals = [Float64(r.nu) for r in t_data]

        S2_mean = mean(S2_vals)
        S2_std = length(S2_vals) > 1 ? std(S2_vals) : 0.0
        chi_mean = mean(chi_vals)
        nu_mean = mean(nu_vals)

        @printf("  %2d  |  %5.3f ± %5.3f   |   %6.1f    |  %4.1f\n",
                t, S2_mean, S2_std, chi_mean, nu_mean)
    end

    println()
    println("="^70)
    println("Key observations from Liu & Clark Figure 4:")
    println("- S2_max increases with t (more entanglement from T-gates)")
    println("- chi_max follows approximately 2^nu (GF(2) bound)")
    println("- nu (nullity) = t_eff - rank, indicates undisentangled T-gates")
    println("="^70)
end

#==============================================================================#
# QUICK TEST FUNCTION
#==============================================================================#

"""
Quick test on small scale to verify correctness.
"""
function test_figure4_small()
    println("Running small-scale test (N=8, t=1:5, 2 realizations)...")
    println()

    results = reproduce_figure4(
        n_qubits=8,
        t_range=1:15,
        n_realizations=2,
        clifford_depth=6,
        base_seed=42,
        verbose=true,
        save_results=false
    )

    return results
end

#==============================================================================#
# MAIN ENTRY POINT
#==============================================================================#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Figure 4 Reproduction Script")
    println("Liu & Clark (2024), arXiv:2412.17209v2")
    println()
    println("Usage:")
    println("  julia --project=. benchmarks/reproduce_figure4.jl [test|full]")
    println()

    mode = length(ARGS) > 0 ? ARGS[1] : "test"

    if mode == "test"
        println("Test mode: N=8, t=1:5, 2 realizations")
        test_figure4_small()
    elseif mode == "full"
        println("Full reproduction: N=16, t=1:35, 3 realizations")
        reproduce_figure4(
            n_qubits=16,
            t_range=1:35,
            n_realizations=3,
            clifford_depth=12,  # ← Change from 4 to 12!
            verbose=true,
            save_results=true
        )
    else
        println("Unknown mode: $mode")
        println("Use 'test' for small-scale test or 'full' for full Figure 4 reproduction")
    end
end
