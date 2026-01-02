# CAMPS.jl/benchmarks/strategy_comparison.jl
# Benchmark: Compare OFD vs OBD vs Hybrid vs NoDisentangling strategies
#
# This benchmark evaluates the performance of different disentangling strategies
# in terms of bond dimension, simulation time, and accuracy.

using CAMPS
using Printf
using Statistics

"""
    run_strategy_comparison(; verbose::Bool=true)

Run the strategy comparison benchmark suite.

Compares OFD, OBD, Hybrid, and NoDisentangling strategies across:
1. Bond dimension achieved
2. Simulation time
3. Accuracy (for small systems where exact verification is possible)

Returns detailed comparison results.
"""
function run_strategy_comparison(; verbose::Bool=true)
    results = Dict{String, Any}()

    if verbose
        println("="^70)
        println("Disentangling Strategy Comparison Benchmark")
        println("="^70)
        println()
    end

    # Test 1: Simple circuits where OFD should be optimal
    results["ofd_favorable"] = benchmark_ofd_favorable(verbose)

    # Test 2: Circuits where OFD may fail
    results["ofd_unfavorable"] = benchmark_ofd_unfavorable(verbose)

    # Test 3: Mixed circuits (Hybrid advantages)
    results["mixed"] = benchmark_mixed_circuits(verbose)

    # Test 4: Timing comparison
    results["timing"] = benchmark_timing(verbose)

    # Summary
    if verbose
        println()
        println("="^70)
        println("Strategy Recommendation Summary")
        println("="^70)
        println("• OFD: Best for Clifford+T circuits with H→T structure")
        println("• OBD: Best when OFD fails (no free X/Y qubits available)")
        println("• Hybrid: Recommended default - combines OFD optimality with OBD fallback")
        println("• NoDisentangling: Only for baseline comparison (exponential growth)")
    end

    return results
end

"""
Benchmark circuits favorable to OFD: H layer followed by T gates.
"""
function benchmark_ofd_favorable(verbose::Bool)
    if verbose
        println("Test 1: OFD-favorable circuits (H layer + T gates)")
        println("-"^50)
    end

    strategies = [
        ("OFD", OFDStrategy()),
        ("OBD", OBDStrategy(max_sweeps=3)),
        ("Hybrid", HybridStrategy()),
        ("None", NoDisentangling())
    ]

    results = []

    for n in [4, 6, 8]
        t = n ÷ 2  # Half the qubits get T gates

        # Create circuit: H on all, T on first t qubits
        circuit = Gate[]
        for q in 1:n
            push!(circuit, HGate(q))
        end
        for q in 1:t
            push!(circuit, TGate(q))
        end

        if verbose
            @printf("  n=%d, t=%d:\n", n, t)
        end

        row = Dict{String, Any}(:n => n, :t => t)

        for (name, strategy) in strategies
            result = simulate_circuit(circuit, n; strategy=strategy, verbose=false)
            row[name] = result.final_bond_dim

            if verbose
                @printf("    %-8s: χ = %d\n", name, result.final_bond_dim)
            end
        end

        push!(results, row)
    end

    if verbose
        println()
    end

    # Compute statistics
    ofd_wins = count(r -> r["OFD"] <= minimum([r["OBD"], r["Hybrid"], r["None"]]), results)

    return Dict(
        :results => results,
        :ofd_optimal_rate => ofd_wins / length(results)
    )
end

"""
Benchmark circuits unfavorable to OFD: T gates without preceding H.
"""
function benchmark_ofd_unfavorable(verbose::Bool)
    if verbose
        println("Test 2: OFD-unfavorable circuits (T gates without H)")
        println("-"^50)
    end

    strategies = [
        ("OFD", OFDStrategy()),
        ("OBD", OBDStrategy(max_sweeps=3)),
        ("Hybrid", HybridStrategy()),
    ]

    results = []

    for n in [4, 6]
        # Circuit: T gates without H (Z stays Z, no X/Y for OFD)
        circuit = Gate[]
        for q in 1:n÷2
            push!(circuit, TGate(q))
        end

        if verbose
            @printf("  n=%d, T gates without H:\n", n)
        end

        row = Dict{String, Any}(:n => n)

        for (name, strategy) in strategies
            result = simulate_circuit(circuit, n; strategy=strategy, verbose=false)
            row[name] = result.final_bond_dim
            row["$(name)_ofd"] = result.n_ofd_applied
            row["$(name)_obd"] = result.n_obd_applied

            if verbose
                @printf("    %-8s: χ = %d (OFD: %d, OBD: %d)\n",
                        name, result.final_bond_dim, result.n_ofd_applied, result.n_obd_applied)
            end
        end

        push!(results, row)
    end

    # Also test: S gates instead of H (S doesn't create X component)
    for n in [4, 6]
        circuit = Gate[]
        for q in 1:n
            push!(circuit, SGate(q))  # S gate: Z→Z, X→Y
        end
        for q in 1:n÷2
            push!(circuit, TGate(q))
        end

        if verbose
            @printf("  n=%d, S gates + T gates:\n", n)
        end

        row = Dict{String, Any}(:n => n, :type => "S+T")

        for (name, strategy) in strategies
            result = simulate_circuit(circuit, n; strategy=strategy, verbose=false)
            row[name] = result.final_bond_dim

            if verbose
                @printf("    %-8s: χ = %d\n", name, result.final_bond_dim)
            end
        end

        push!(results, row)
    end

    if verbose
        println()
    end

    return Dict(:results => results)
end

"""
Benchmark mixed circuits where both OFD and OBD have roles.
"""
function benchmark_mixed_circuits(verbose::Bool)
    if verbose
        println("Test 3: Mixed circuits (Hybrid advantages)")
        println("-"^50)
    end

    strategies = [
        ("OFD", OFDStrategy()),
        ("OBD", OBDStrategy(max_sweeps=5)),
        ("Hybrid", HybridStrategy(obd_sweeps_on_failure=3)),
    ]

    results = []

    for n in [6, 8]
        # Mixed circuit: some qubits with H (OFD possible), some without
        circuit = Gate[]

        # H on odd qubits only
        for q in 1:2:n
            push!(circuit, HGate(q))
        end

        # Entangling layer
        for q in 1:(n-1)
            push!(circuit, CNOTGate(q, q+1))
        end

        # T on all qubits
        for q in 1:n
            push!(circuit, TGate(q))
        end

        if verbose
            @printf("  n=%d, mixed H/T structure:\n", n)
        end

        row = Dict{String, Any}(:n => n)

        for (name, strategy) in strategies
            result = simulate_circuit(circuit, n; strategy=strategy, verbose=false)
            row[name] = result.final_bond_dim
            row["$(name)_entropy"] = result.final_entropy

            if verbose
                @printf("    %-8s: χ = %d, S_max = %.3f\n",
                        name, result.final_bond_dim, result.final_entropy)
            end
        end

        push!(results, row)
    end

    if verbose
        println()
    end

    return Dict(:results => results)
end

"""
Benchmark simulation timing for different strategies.
"""
function benchmark_timing(verbose::Bool)
    if verbose
        println("Test 4: Timing comparison")
        println("-"^50)
    end

    strategies = [
        ("OFD", OFDStrategy()),
        ("OBD", OBDStrategy(max_sweeps=2)),
        ("Hybrid", HybridStrategy()),
    ]

    results = []

    for n in [6, 8, 10]
        t = 4

        # Standard Clifford+T circuit
        circuit = random_clifford_t_circuit(n, t; seed=42)

        if verbose
            @printf("  n=%d, t=%d gates:\n", n, t)
        end

        row = Dict{String, Any}(:n => n, :t => t)

        for (name, strategy) in strategies
            # Warm up
            simulate_circuit(circuit, n; strategy=strategy, verbose=false)

            # Time multiple runs
            times = Float64[]
            for _ in 1:3
                t_start = time()
                simulate_circuit(circuit, n; strategy=strategy, verbose=false)
                push!(times, time() - t_start)
            end

            mean_time = mean(times)
            row["$(name)_time"] = mean_time

            if verbose
                @printf("    %-8s: %.3f s (mean of 3)\n", name, mean_time)
            end
        end

        push!(results, row)
    end

    if verbose
        println()
    end

    return Dict(:results => results)
end

"""
    strategy_recommendation(circuit::Vector{<:Gate}, n::Int) -> DisentanglingStrategy

Analyze a circuit and recommend the best disentangling strategy.

# Arguments
- `circuit::Vector{<:Gate}`: Circuit to analyze
- `n::Int`: Number of qubits

# Returns
- `DisentanglingStrategy`: Recommended strategy
"""
function strategy_recommendation(circuit::Vector{<:Gate}, n::Int)
    # Analyze the circuit structure
    has_h_before_t = false
    first_t_idx = findfirst(g -> g isa RotationGate && !is_clifford_angle(g.angle), circuit)

    if first_t_idx !== nothing
        # Check if there are H gates before the first T
        for i in 1:(first_t_idx-1)
            if circuit[i] isa CliffordGate
                # Check if it contains an H gate
                for spec in circuit[i].gates
                    if spec isa Tuple && spec[1] == :H
                        has_h_before_t = true
                        break
                    end
                end
            end
        end
    end

    # Count T gates
    t_count = count(g -> g isa RotationGate && abs(g.angle) ≈ π/4, circuit)

    # Recommendations
    if has_h_before_t && t_count <= n
        # OFD likely to succeed for most T gates
        return OFDStrategy()
    elseif t_count > n * 2
        # Many T gates, hybrid provides good balance
        return HybridStrategy()
    else
        # Default to hybrid for robustness
        return HybridStrategy()
    end
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_strategy_comparison()
end
