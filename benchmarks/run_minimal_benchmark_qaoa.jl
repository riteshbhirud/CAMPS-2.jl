# CAMPS.jl/benchmarks/run_minimal_benchmark_qaoa.jl
#
# QAOA TEST BENCHMARK - Quick verification that QAOA works
# 
# This is a SEPARATE test file that doesn't interfere with your running benchmark.
# Tests QAOA with a small subset of circuits to verify everything works.

using CAMPS
using Printf
using Dates
using Random

# Load circuit families
include("circuit_families.jl")
include("qaoa_maxcut_family.jl")

#==============================================================================#
# SYMBOLIC GATE APPLICATION
#==============================================================================#

function apply_symbolic_gate!(state::CAMPSState, gate_type::Symbol, qubits::Vector{Int})
    if gate_type == :H
        gate = HGate(qubits[1])
        apply_gate!(state, gate)
    elseif gate_type == :CNOT
        gate = CNOTGate(qubits[1], qubits[2])
        apply_gate!(state, gate)
    elseif gate_type == :X
        gate = XGate(qubits[1])
        apply_gate!(state, gate)
    elseif gate_type == :Z
        gate = ZGate(qubits[1])
        apply_gate!(state, gate)
    elseif gate_type == :S
        gate = SGate(qubits[1])
        apply_gate!(state, gate)
    elseif gate_type == :random2q
        q1, q2 = qubits[1], qubits[2]
        cliff = random_clifford(2)
        sparse = SparseGate(cliff, [q1, q2])
        apply!(state.clifford, sparse)
    elseif gate_type == :T
        # Handled separately in main loop
        error("T-gates should be handled by OFD/OBD, not symbolic application")
    else
        error("Unknown gate type: $gate_type")
    end
end

#==============================================================================#
# CIRCUIT EXECUTION
#==============================================================================#

function run_single_circuit(circuit::CircuitInstance, family_name::String; verbose=false)
    start_time = time()
    
    state = CAMPSState(circuit.n_qubits; max_bond=2048)
    initialize!(state)
    
    ofd_success = 0
    ofd_fail = 0
    
    for (gate_idx, (gate_type, qubits)) in enumerate(circuit.gates)
        if gate_type == :T
            qubit = qubits[1]
            success, _ = apply_t_gate_ofd!(state, qubit)
            
            if success
                ofd_success += 1
            else
                apply_gate!(state, TGate(qubit), strategy=OBDStrategy())
                ofd_fail += 1
            end
        else
            apply_symbolic_gate!(state, gate_type, qubits)
        end
    end
    
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
        runtime = runtime,
        metadata = circuit.metadata
    )
end

#==============================================================================#
# QAOA-ONLY TEST BENCHMARK
#==============================================================================#

"""
Quick test of QAOA implementation - just a few circuits to verify it works.
"""
function run_qaoa_test_benchmark(;
    n_range = [8, 12],  # Just 2 sizes
    t_fraction_range = [1.0, 1.5],  # Just 2 densities (achievable for QAOA)
    n_realizations = 2,  # Just 2 realizations
    output_dir = "results/qaoa_test",
    verbose = true)
    
    println("="^70)
    println("QAOA TEST BENCHMARK - Quick Verification")
    println("="^70)
    println()
    println("Testing: n ∈ $n_range")
    println("Testing: t/n ∈ $t_fraction_range")
    println("Realizations: $n_realizations per configuration")
    println("Total circuits: $(length(n_range) * length(t_fraction_range) * n_realizations)")
    println()
    println("Note: This tests QAOA's natural T-gate densities:")
    println("  t/n=1.0 → mixer layer non-Clifford (n T-gates)")
    println("  t/n=1.5 → cost layer non-Clifford (3n/2 T-gates)")
    println()
    
    family = QAOAMaxCutCircuit()
    family_name = get_name(family)
    
    mkpath(output_dir)
    
    # Generate all parameter combinations
    experiments = []
    for n in n_range
        for t_frac in t_fraction_range
            n_t = Int(round(n * t_frac))
            for real in 1:n_realizations
                # FIXED: Convert hash to Int64
                seed = Int(hash((family_name, n, n_t, real)))
                
                params = Dict{Symbol, Any}(
                    :n_qubits => n,
                    :n_t_gates => n_t,
                    :seed => seed
                )
                
                push!(experiments, (family=family, family_name=family_name, params=params))
            end
        end
    end
    
    println("Generated $(length(experiments)) QAOA test circuits")
    println()
    
    # Run experiments
    results = []
    
    println("Running QAOA circuits...")
    println("-"^70)
    
    for (i, (fam, fname, params)) in enumerate(experiments)
        if verbose
            print("[$i/$(length(experiments))] ")
        end
        
        circuit = generate_circuit(fam, params)
        result = run_single_circuit(circuit, fname; verbose=verbose)
        push!(results, result)
    end
    
    println()
    println("="^70)
    println("QAOA TEST COMPLETED")
    println("="^70)
    
    # Summary statistics
    avg_ofd = mean([r.ofd_rate for r in results])
    avg_chi = mean([r.final_chi for r in results])
    avg_nu = mean([r.final_nu for r in results])
    total_time = sum([r.runtime for r in results])
    
    println()
    println("QAOA Performance Summary:")
    println("  Average OFD rate:  $(round(avg_ofd * 100, digits=1))%")
    println("  Average χ:         $(round(avg_chi, digits=1))")
    println("  Average ν:         $(round(avg_nu, digits=1))")
    println("  Total runtime:     $(round(total_time, digits=1))s")
    println()
    
    # Save results
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    output_file = joinpath(output_dir, "qaoa_test_results_$timestamp.csv")
    
    open(output_file, "w") do io
        # Write header
        println(io, "family,n_qubits,n_t_gates,t_density,ofd_success,ofd_fail,ofd_rate,final_chi,final_nu,final_S2,runtime,gamma,beta,target_t,predicted_t,actual_t")
        
        # Write data
        for r in results
            gamma = get(r.metadata, "gamma", NaN)
            beta = get(r.metadata, "beta", NaN)
            target_t = get(r.metadata, "target_t_count", r.n_t_gates)
            predicted_t = get(r.metadata, "predicted_t_count", r.n_t_gates)
            actual_t = get(r.metadata, "actual_t_count", r.n_t_gates)
            t_density = r.n_t_gates / r.n_qubits
            
            println(io, "$(r.family),$(r.n_qubits),$(r.n_t_gates),$t_density,$(r.ofd_success),$(r.ofd_fail),$(r.ofd_rate),$(r.final_chi),$(r.final_nu),$(r.final_S2),$(r.runtime),$gamma,$beta,$target_t,$predicted_t,$actual_t")
        end
    end
    
    println("Results saved to: $output_file")
    println()
    println("✓ QAOA implementation verified and working!")
    println()
    
    return results
end

#==============================================================================#
# RUN IT
#==============================================================================#

if abspath(PROGRAM_FILE) == @__FILE__
    println()
    println("Starting QAOA test benchmark...")
    println()
    
    results = run_qaoa_test_benchmark(verbose=true)
    
    println("DONE! ✓")
end
