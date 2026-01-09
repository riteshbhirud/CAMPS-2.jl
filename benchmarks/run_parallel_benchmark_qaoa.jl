# CAMPS.jl/benchmarks/run_parallel_benchmark_qaoa.jl
#
# QAOA TEST BENCHMARK - Parallel version
# 
# Quick parallel test of QAOA with a small subset of circuits.
# Safe to run alongside your main benchmark.

using Distributed
using CAMPS
using Printf
using Dates
using Statistics

# Check if workers are already added
if nworkers() == 1
    println("Adding 4 parallel workers...")
    addprocs(4)
    println("Workers added: ", nworkers())
end

# Load packages on all workers
@everywhere using CAMPS
@everywhere using Printf
@everywhere using Random

# Load circuit families on all workers
@everywhere include("circuit_families.jl")
@everywhere include("qaoa_maxcut_family.jl")

#==============================================================================#
# SYMBOLIC GATE APPLICATION (on all workers)
#==============================================================================#

@everywhere function apply_symbolic_gate!(state::CAMPSState, gate_type::Symbol, qubits::Vector{Int})
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
        error("T-gates should be handled by OFD/OBD, not symbolic application")
    else
        error("Unknown gate type: $gate_type")
    end
end

#==============================================================================#
# CIRCUIT EXECUTION (on all workers)
#==============================================================================#

@everywhere function run_single_circuit(circuit::CircuitInstance, family_name::String)
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
# PARALLEL QAOA TEST
#==============================================================================#

function run_qaoa_parallel_test(;
    n_range = [8, 12],
    t_fraction_range = [1.0, 1.5],
    n_realizations = 2,
    output_dir = "results/qaoa_parallel_test")
    
    println("="^70)
    println("QAOA PARALLEL TEST - Quick Verification")
    println("="^70)
    println()
    println("Workers: $(nworkers())")
    println("Testing: n ∈ $n_range")
    println("Testing: t/n ∈ $t_fraction_range")
    println("Realizations: $n_realizations")
    println("Total: $(length(n_range) * length(t_fraction_range) * n_realizations) circuits")
    println()
    
    family = QAOAMaxCutCircuit()
    family_name = get_name(family)
    
    mkpath(output_dir)
    
    # Generate experiments
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
    
    # Run in parallel
    start_time = time()
    println("Running QAOA circuits in parallel...")
    println("-"^70)
    
    results = pmap(experiments) do (fam, fname, params)
        circuit = generate_circuit(fam, params)
        result = run_single_circuit(circuit, fname)
        
        # Progress indicator
        @printf("✓ n=%d, t=%d (%.1f%% OFD)\n", 
                result.n_qubits, result.n_t_gates, result.ofd_rate * 100)
        
        return result
    end
    
    total_time = time() - start_time
    
    println()
    println("="^70)
    println("QAOA PARALLEL TEST COMPLETED")
    println("="^70)
    
    # Statistics
    avg_ofd = mean([r.ofd_rate for r in results])
    avg_chi = mean([r.final_chi for r in results])
    avg_nu = mean([r.final_nu for r in results])
    
    println()
    println("Performance Summary:")
    println("  Average OFD rate:  $(round(avg_ofd * 100, digits=1))%")
    println("  Average χ:         $(round(avg_chi, digits=1))")
    println("  Average ν:         $(round(avg_nu, digits=1))")
    println("  Total time:        $(round(total_time, digits=1))s")
    println("  Circuits/second:   $(round(length(results)/total_time, digits=2))")
    println()
    
    # Save results
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    output_file = joinpath(output_dir, "qaoa_parallel_test_$timestamp.csv")
    
    open(output_file, "w") do io
        println(io, "family,n_qubits,n_t_gates,t_density,ofd_success,ofd_fail,ofd_rate,final_chi,final_nu,final_S2,runtime,gamma,beta,target_t,predicted_t,actual_t")
        
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
    println("✓ QAOA parallel implementation verified!")
    println()
    
    return results
end

#==============================================================================#
# RUN IT
#==============================================================================#

if abspath(PROGRAM_FILE) == @__FILE__
    println()
    results = run_qaoa_parallel_test()
    println("DONE! ✓")
end
