# CAMPS.jl/benchmarks/run_minimal_benchmark.jl
#
# Minimal working benchmark for 9 exact circuit families
# Uses only QuantumClifford's public API

using CAMPS
using QuantumClifford
using Random
using Statistics
using DataFrames
using CSV
using Dates
using Printf

include("circuit_families.jl")

#==============================================================================#
# GATE APPLICATION
#==============================================================================#

"""
Apply a symbolic gate to a CAMPS state.
Uses CAMPS's gate constructors and apply_gate! function.
"""
function apply_symbolic_gate!(state::CAMPSState, gate_type::Symbol, qubits::Vector{Int})
    if gate_type == :H
        # Hadamard gate
        gate = HGate(qubits[1])
        apply_gate!(state, gate)
    elseif gate_type == :CNOT
        # CNOT gate
        gate = CNOTGate(qubits[1], qubits[2])
        apply_gate!(state, gate)
    elseif gate_type == :X
        # Pauli X gate
        gate = XGate(qubits[1])
        apply_gate!(state, gate)
    elseif gate_type == :Z
        # Pauli Z gate
        gate = ZGate(qubits[1])
        apply_gate!(state, gate)
    elseif gate_type == :S
        # S gate (Phase)
        gate = SGate(qubits[1])
        apply_gate!(state, gate)
    elseif gate_type == :random2q
        # Random 2-qubit Clifford using SparseGate
        # This provides TRUE uniform sampling from the 11,520-element Cl_2 group
        q1, q2 = qubits[1], qubits[2]
        
        cliff = random_clifford(2)
        sparse = SparseGate(cliff, [q1, q2])
        apply!(state.clifford, sparse)
        
    elseif gate_type == :T
        # T-gate - shouldn't be called from here
        error("T-gates should be handled separately!")
    end
end

#==============================================================================#
# BENCHMARK EXECUTION
#==============================================================================#

"""
Run a single circuit instance and collect metrics.
"""
function run_single_circuit(circuit::CircuitInstance, family_name::String; verbose=false)
    start_time = time()
    
    # Initialize CAMPS
    state = CAMPSState(circuit.n_qubits; max_bond=2048)
    initialize!(state)
    
    ofd_success = 0
    ofd_fail = 0
    
    # Apply gates
    for (gate_idx, (gate_type, qubits)) in enumerate(circuit.gates)
        if gate_type == :T
            # This is a T-gate position
            qubit = qubits[1]
            
            # Try OFD - it returns (success, state)
            success, _ = apply_t_gate_ofd!(state, qubit)
            
            if success
                ofd_success += 1
            else
                # OFD failed, use OBD via regular TGate application with OBD strategy
                apply_gate!(state, TGate(qubit), strategy=OBDStrategy())
                ofd_fail += 1
            end
        else
            # Regular Clifford gate
            apply_symbolic_gate!(state, gate_type, qubits)
        end
    end
    
    # Final metrics
    final_chi = get_bond_dimension(state)
    final_nu = circuit.n_qubits - sum(state.free_qubits)
    final_S2 = max_entanglement_entropy(state.mps)
    runtime = time() - start_time
    
    ofd_rate = ofd_success / max(1, ofd_success + ofd_fail)
    
    if verbose
        @printf("  %-35s (n=%2d, t=%2d): OFD=%5.1f%%, χ=%4d, ν=%2d, %.2fs\n",
                family_name, circuit.n_qubits, length(circuit.t_gate_positions),
                ofd_rate * 100, final_chi, final_nu, runtime)
    end
    
    return (
        family = family_name,
        n_qubits = circuit.n_qubits,
        n_t_gates = length(circuit.t_gate_positions),
        ofd_success = ofd_success,
        ofd_fail = ofd_fail,
        ofd_rate = ofd_rate,
        final_chi = final_chi,
        final_nu = final_nu,
        final_S2 = final_S2,
        runtime = runtime
    )
end

#==============================================================================#
# MAIN BENCHMARK
#==============================================================================#

"""
Run comprehensive benchmark on all 9 families.
"""
function run_minimal_benchmark(;
    n_range = [8, 12, 16],
    t_fraction_range = [0.5, 1.0, 1.5, 2.0],
    n_realizations = 10,
    output_dir = "results/minimal_benchmark",
    verbose = true)
    
    println("="^70)
    println("MINIMAL BENCHMARK - 9 EXACT CIRCUIT FAMILIES")
    println("="^70)
    println()
    
    families = get_rigorous_circuit_families()
    mkpath(output_dir)
    
    # Generate all parameter combinations
    experiments = []
    for family in families
        family_name = get_name(family)
        for n in n_range
            for t_frac in t_fraction_range
                n_t = Int(round(n * t_frac))
                for real in 1:n_realizations
                    seed = hash((family_name, n, n_t, real))
                    
                    params = Dict{Symbol, Any}(
                        :n_qubits => n,
                        :n_t_gates => n_t,
                        :seed => seed
                    )
                    
                    # Family-specific parameters
                    if family isa RandomBrickwallCliffordT
                        params[:clifford_depth] = 2
                    elseif family isa RandomAllToAllCliffordT
                        params[:clifford_layers] = 2 * n
                    elseif family isa SimonCircuit && n % 2 != 0
                        continue
                    elseif family isa DeutschJozsaCircuit
                        params[:function_type] = rand([:constant, :balanced])
                    elseif family isa GraphStateCircuit
                        params[:edge_probability] = 0.3
                    end
                    
                    push!(experiments, (family=family, family_name=family_name, params=params))
                end
            end
        end
    end
    
    n_total = length(experiments)
    println("Total experiments: ", n_total)
    println("Circuit families: 9 (all exact)")
    println()
    println("Starting benchmarks...")
    println("-"^70)
    
    results = []
    start_time = time()
    
    for (idx, (family, family_name, params)) in enumerate(experiments)
        if verbose && idx % 20 == 0
            elapsed = time() - start_time
            eta = (elapsed / idx) * (n_total - idx) / 60
            @printf("[%5d/%5d] ETA: %.1f min\n", idx, n_total, eta)
        end
        
        try
            circuit = generate_circuit(family, params)
            result = run_single_circuit(circuit, family_name; verbose=verbose)
            push!(results, result)
        catch e
            println("  ERROR in ", family_name, ": ", e)
            if verbose
                println(stacktrace(catch_backtrace()))
            end
        end
    end
    
    total_time = (time() - start_time) / 60
    println("-"^70)
    @printf("Completed %d/%d experiments in %.1f minutes\n", 
            length(results), n_total, total_time)
    println()
    
    # Check if we have any results
    if isempty(results)
        println("ERROR: All experiments failed!")
        println("Check the error messages above.")
        return DataFrame()
    end
    
    # Save results
    df = DataFrame(results)
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    
    csv_file = joinpath(output_dir, "results_$(timestamp).csv")
    CSV.write(csv_file, df)
    println("Results saved to: ", csv_file)
    
    # Aggregate
    agg_df = combine(groupby(df, [:family, :n_qubits, :n_t_gates]),
        :ofd_rate => mean => :mean_ofd_rate,
        :ofd_rate => std => :std_ofd_rate,
        :final_chi => mean => :mean_chi,
        :final_nu => mean => :mean_nu,
        nrow => :n_samples
    )
    
    agg_file = joinpath(output_dir, "aggregated_$(timestamp).csv")
    CSV.write(agg_file, agg_df)
    
    # Summary
    println()
    println("="^70)
    println("SUMMARY BY FAMILY")
    println("="^70)
    family_stats = combine(groupby(agg_df, :family),
        :mean_ofd_rate => mean => :overall_ofd_rate
    )
    sort!(family_stats, :overall_ofd_rate, rev=true)
    
    for row in eachrow(family_stats)
        @printf("%-35s | %5.1f%%\n", row.family, row.overall_ofd_rate * 100)
    end
    println("="^70)
    
    return df
end

#==============================================================================#
# QUICK TEST
#==============================================================================#

"""
Test all 9 families work correctly.
"""
function quick_test_minimal_families()
    println("Testing all 9 circuit families...")
    println()
    
    families = get_rigorous_circuit_families()
    
    for family in families
        name = get_name(family)
        try
            params = Dict{Symbol, Any}(
                :n_qubits => 8,
                :n_t_gates => 8,
                :seed => 42
            )
            
            if family isa RandomBrickwallCliffordT
                params[:clifford_depth] = 2
            elseif family isa RandomAllToAllCliffordT
                params[:clifford_layers] = 16
            elseif family isa GraphStateCircuit
                params[:edge_probability] = 0.3
            end
            
            circuit = generate_circuit(family, params)
            
            n_clifford = count(g -> g[1] != :T, circuit.gates)
            n_t = length(circuit.t_gate_positions)
            
            @printf("✓ %-40s - %3d Cliffords, %2d T-gates\n",
                    name, n_clifford, n_t)
        catch e
            println("✗ ", name, " - ERROR: ", e)
        end
    end
    
    println()
    println("All 9 families tested successfully!")
end

#==============================================================================#
# ENTRY POINT
#==============================================================================#

function main(mode::String="standard")
    config = if mode == "quick"
        (n_range=[8, 12], t_fraction_range=[0.5, 1.0], n_realizations=3)
    elseif mode == "standard"
        (n_range=[8, 12, 16], t_fraction_range=[0.5, 1.0, 1.5, 2.0], n_realizations=10)
    else
        error("Unknown mode: $mode")
    end
    
    println("Mode: ", mode)
    println()
    
    quick_test_minimal_families()
    println()
    
    run_minimal_benchmark(;
        n_range=config.n_range,
        t_fraction_range=config.t_fraction_range,
        n_realizations=config.n_realizations,
        verbose=true
    )
end

export run_minimal_benchmark, quick_test_minimal_families, main

if abspath(PROGRAM_FILE) == @__FILE__
    mode = length(ARGS) >= 1 ? ARGS[1] : "standard"
    main(mode)
end
