# CAMPS.jl/benchmarks/scaling.jl
# Benchmark: Analyze scaling with system size and T-gate count
#
# This benchmark studies how CAMPS performance scales with:
# 1. Number of qubits (n)
# 2. Number of T gates (t)
# 3. Circuit depth

using CAMPS
using Printf
using Statistics

"""
    run_scaling_benchmark(; verbose::Bool=true, max_qubits::Int=12)

Run the scaling analysis benchmark suite.

Studies how CAMPS scales with:
1. System size (n qubits)
2. T-gate count
3. T-gate depth
4. Bond dimension growth

# Arguments
- `verbose::Bool`: Print detailed output
- `max_qubits::Int`: Maximum number of qubits to test

Returns comprehensive scaling analysis results.
"""
function run_scaling_benchmark(; verbose::Bool=true, max_qubits::Int=12)
    results = Dict{String, Any}()

    if verbose
        println("="^70)
        println("CAMPS Scaling Analysis Benchmark")
        println("="^70)
        println()
    end

    # Test 1: Scaling with system size
    results["system_size"] = benchmark_system_size_scaling(verbose, max_qubits)

    # Test 2: Scaling with T-gate count
    results["t_count"] = benchmark_t_count_scaling(verbose)

    # Test 3: Bond dimension analysis
    results["bond_dimension"] = benchmark_bond_dimension_growth(verbose)

    # Test 4: Memory and time scaling
    results["resources"] = benchmark_resource_scaling(verbose, max_qubits)

    # Summary
    if verbose
        println()
        println("="^70)
        println("Scaling Summary")
        println("="^70)
        println("• System size: Clifford operations scale as O(n²), MPS as O(n·χ³)")
        println("• T-count: With OFD, χ = 2^(t - rank) where rank ≤ min(t, n)")
        println("• For ideal Clifford+T circuits: χ remains O(1) when rank = t")
    end

    return results
end

"""
Benchmark how simulation scales with system size (number of qubits).
"""
function benchmark_system_size_scaling(verbose::Bool, max_qubits::Int)
    if verbose
        println("Test 1: System size scaling")
        println("-"^50)
    end

    results = []

    qubit_counts = [4, 6, 8, 10, min(12, max_qubits)]
    filter!(n -> n <= max_qubits, qubit_counts)

    for n in qubit_counts
        t = min(4, n)  # Fixed T-count

        # Create standard circuit
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

        # Time the simulation
        times = Float64[]
        for _ in 1:3
            t_start = time()
            result = simulate_circuit(circuit, n; strategy=OFDStrategy(), verbose=false)
            push!(times, time() - t_start)
        end

        result = simulate_circuit(circuit, n; strategy=OFDStrategy(), verbose=false)
        mean_time = mean(times)

        row = (
            n = n,
            t = t,
            time = mean_time,
            bond_dim = result.final_bond_dim,
            entropy = result.final_entropy
        )
        push!(results, row)

        if verbose
            @printf("  n=%2d: time=%.4fs, χ=%d, S=%.3f\n",
                    n, mean_time, result.final_bond_dim, result.final_entropy)
        end
    end

    if verbose
        println()
    end

    return Dict(:results => results)
end

"""
Benchmark how simulation scales with T-gate count.
"""
function benchmark_t_count_scaling(verbose::Bool)
    if verbose
        println("Test 2: T-gate count scaling")
        println("-"^50)
    end

    results = []
    n = 8  # Fixed system size

    t_counts = [2, 4, 6, 8, 10, 12]

    for t in t_counts
        # Circuit with t T-gates distributed randomly
        circuit = Gate[]

        # H layer
        for q in 1:n
            push!(circuit, HGate(q))
        end

        # T gates on cycling qubits
        for i in 1:t
            q = ((i - 1) % n) + 1
            push!(circuit, TGate(q))
        end

        # Simulate with different strategies
        result_ofd = simulate_circuit(circuit, n; strategy=OFDStrategy(), verbose=false)
        result_hybrid = simulate_circuit(circuit, n; strategy=HybridStrategy(), verbose=false)

        row = (
            t = t,
            chi_ofd = result_ofd.final_bond_dim,
            chi_hybrid = result_hybrid.final_bond_dim,
            predicted = result_ofd.predicted_bond_dim,
            rank = t - Int(round(log2(max(result_ofd.predicted_bond_dim, 1))))
        )
        push!(results, row)

        if verbose
            @printf("  t=%2d: χ_OFD=%d, χ_Hybrid=%d, χ_pred=%d, rank=%d\n",
                    t, row.chi_ofd, row.chi_hybrid, row.predicted, row.rank)
        end
    end

    if verbose
        println()
    end

    return Dict(:results => results, :n => n)
end

"""
Benchmark bond dimension growth patterns.
"""
function benchmark_bond_dimension_growth(verbose::Bool)
    if verbose
        println("Test 3: Bond dimension growth analysis")
        println("-"^50)
    end

    results = []

    # Test 1: Ideal case - independent T gates
    if verbose
        println("  Case A: Independent T gates (H + T, no entangling)")
    end

    n = 8
    for t in [2, 4, 6, 8]
        circuit = Gate[]
        for q in 1:n
            push!(circuit, HGate(q))
        end
        for q in 1:t
            push!(circuit, TGate(q))
        end

        result = simulate_circuit(circuit, n; strategy=OFDStrategy(), verbose=false)

        push!(results, (
            case = "independent",
            n = n,
            t = t,
            chi = result.final_bond_dim,
            expected_chi = 1  # rank = t for independent
        ))

        if verbose
            @printf("    t=%d: χ=%d (expected: 1)\n", t, result.final_bond_dim)
        end
    end

    # Test 2: Worst case - highly entangled
    if verbose
        println("  Case B: Entangled T gates (H + CNOT chain + T)")
    end

    for t in [2, 4, 6]
        circuit = Gate[]
        for q in 1:n
            push!(circuit, HGate(q))
        end
        # Dense entangling
        for _ in 1:2
            for q in 1:(n-1)
                push!(circuit, CNOTGate(q, q+1))
            end
        end
        for q in 1:t
            push!(circuit, TGate(q))
        end

        result = simulate_circuit(circuit, n; strategy=OFDStrategy(), verbose=false)

        push!(results, (
            case = "entangled",
            n = n,
            t = t,
            chi = result.final_bond_dim,
            predicted = result.predicted_bond_dim
        ))

        if verbose
            @printf("    t=%d: χ=%d (predicted: %d)\n", t, result.final_bond_dim, result.predicted_bond_dim)
        end
    end

    # Test 3: No disentangling baseline
    if verbose
        println("  Case C: No disentangling baseline (exponential growth)")
    end

    for t in [1, 2, 3, 4]
        circuit = Gate[]
        for q in 1:4
            push!(circuit, HGate(q))
        end
        for q in 1:t
            push!(circuit, TGate(q))
        end

        result = simulate_circuit(circuit, 4; strategy=NoDisentangling(), verbose=false)

        push!(results, (
            case = "no_disentangle",
            n = 4,
            t = t,
            chi = result.final_bond_dim,
            expected_chi = 2^t
        ))

        if verbose
            @printf("    t=%d: χ=%d (expected: %d)\n", t, result.final_bond_dim, 2^t)
        end
    end

    if verbose
        println()
    end

    return Dict(:results => results)
end

"""
Benchmark resource (time and memory) scaling.
"""
function benchmark_resource_scaling(verbose::Bool, max_qubits::Int)
    if verbose
        println("Test 4: Resource scaling")
        println("-"^50)
    end

    results = []

    for n in [4, 6, 8, min(10, max_qubits)]
        if n > max_qubits
            continue
        end

        t = min(4, n)

        # Create circuit
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

        # Measure time
        times = Float64[]
        for _ in 1:5
            GC.gc()  # Clean up before timing
            t_start = time()
            result = simulate_circuit(circuit, n; strategy=OFDStrategy(), verbose=false)
            push!(times, time() - t_start)
        end

        result = simulate_circuit(circuit, n; strategy=OFDStrategy(), verbose=false)

        row = (
            n = n,
            t = t,
            mean_time = mean(times),
            std_time = std(times),
            bond_dim = result.final_bond_dim,
            n_gates = result.n_gates
        )
        push!(results, row)

        if verbose
            @printf("  n=%2d: time=%.4f±%.4fs, χ=%d, gates=%d\n",
                    n, row.mean_time, row.std_time, row.bond_dim, row.n_gates)
        end
    end

    if verbose
        println()
    end

    return Dict(:results => results)
end

"""
    analyze_scaling_exponent(data::Vector, x_key::Symbol, y_key::Symbol)

Fit a power law y = a * x^b to the data and return the exponent b.

Uses log-log linear regression.
"""
function analyze_scaling_exponent(data::Vector, x_key::Symbol, y_key::Symbol)
    x = Float64[getfield(d, x_key) for d in data]
    y = Float64[getfield(d, y_key) for d in data]

    # Filter positive values for log
    valid = (x .> 0) .& (y .> 0)
    x = x[valid]
    y = y[valid]

    if length(x) < 2
        return NaN
    end

    # Log-log linear regression
    log_x = log.(x)
    log_y = log.(y)

    # Simple least squares: b = Σ(x-x̄)(y-ȳ) / Σ(x-x̄)²
    x_mean = mean(log_x)
    y_mean = mean(log_y)

    num = sum((log_x .- x_mean) .* (log_y .- y_mean))
    den = sum((log_x .- x_mean).^2)

    return den > 0 ? num / den : NaN
end

"""
    print_scaling_report(results::Dict)

Print a formatted scaling analysis report.
"""
function print_scaling_report(results::Dict)
    println()
    println("="^70)
    println("CAMPS Scaling Analysis Report")
    println("="^70)
    println()

    # System size scaling
    if haskey(results, "system_size")
        data = results["system_size"][:results]
        println("System Size Scaling (n qubits):")
        println("  Time scaling exponent: ", round(analyze_scaling_exponent(data, :n, :time), digits=2))
        println()
    end

    # T-count scaling
    if haskey(results, "t_count")
        data = results["t_count"][:results]
        println("T-Count Scaling:")
        for d in data
            @printf("  t=%2d: χ=%d (rank estimate: %d)\n", d.t, d.chi_ofd, d.rank)
        end
        println()
    end

    println("Key Findings:")
    println("• CAMPS achieves χ=1 for independent T gates (optimal)")
    println("• Bond dimension growth depends on GF(2) rank structure")
    println("• Without disentangling: χ grows as 2^t (exponential)")
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    results = run_scaling_benchmark()
    print_scaling_report(results)
end
