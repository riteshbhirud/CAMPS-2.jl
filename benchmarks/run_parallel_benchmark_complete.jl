# CAMPS.jl/benchmarks/run_parallel_benchmark_complete.jl
#
# CPU-PARALLEL BENCHMARK - ALL 14 CIRCUIT FAMILIES
# 
# Runs comprehensive benchmark across all Phase 1 (9 families) and Phase 2 (5 families)
# with appropriate parameter sweeps for ML training.
#
# Total circuits: 1,008
# - Phase 1: 648 circuits (9 families × 3 sizes × 3 t_fractions × 8 realizations)
# - Phase 2: 360 circuits (5 families × 72 circuits each)
#
# Key guarantees:
# - Deterministic seeds → reproducible results
# - No data races
# - Progress tracking
# - Incremental saves
# - Handles both symbolic gates (Phase 1) and CAMPS gates (Phase 2)

using Distributed
using Printf

# Activate CAMPS.jl project environment FIRST (before adding workers)
camps_dir = dirname(dirname(@__FILE__))  # Go up from benchmarks/ to CAMPS.jl/
println("Activating CAMPS.jl environment: $camps_dir")
using Pkg
Pkg.activate(camps_dir)

# Determine number of workers
const N_PHYSICAL_CORES = Sys.CPU_THREADS ÷ 2

# Check for environment variable override (useful for SLURM)
const N_WORKERS = if haskey(ENV, "JULIA_NUM_WORKERS")
    parse(Int, ENV["JULIA_NUM_WORKERS"])
else
    max(1, N_PHYSICAL_CORES - 1)
end

println("="^80)
println("CPU-PARALLEL BENCHMARK - ALL 14 CIRCUIT FAMILIES")
println("="^80)
println("Detected: $(Sys.CPU_THREADS) logical cores, $N_PHYSICAL_CORES physical cores")
println("Using: $N_WORKERS worker processes")
println()

# Add worker processes
if nprocs() == 1
    println("Starting $N_WORKERS workers...")
    # Pass --project flag so workers use same environment
    addprocs(N_WORKERS; exeflags=`--project=$(camps_dir)`)
    println("Workers started: ", workers())
else
    println("Workers already running: ", workers())
end
println()

# Load dependencies on ALL workers
println("Loading dependencies on all workers...")

@everywhere begin
    # Workers should already have the right project, but double-check
    using Pkg
    if !isnothing(Base.active_project())
        println("Worker $(myid()): Using project $(Base.active_project())")
    end
    
    # Load packages
    using CAMPS
    using QuantumClifford
    using QuantumClifford.ECC
    using Random
    using Statistics
    using DataFrames
    using CSV
    using Dates
    using Printf
    
    # Load unified circuit families
    include("circuit_families_complete.jl")
    
    # Helper to recreate family from name (avoids serialization issues)
    function get_family_from_name(family_name::String)
        if family_name == "Random Clifford+T (Brick-wall)"
            return RandomBrickwallCliffordT()
        elseif family_name == "Random Clifford+T (All-to-all)"
            return RandomAllToAllCliffordT()
        elseif family_name == "Bernstein-Vazirani"
            return BernsteinVaziraniCircuit()
        elseif family_name == "Simon's Algorithm"
            return SimonCircuit()
        elseif family_name == "Deutsch-Jozsa"
            return DeutschJozsaCircuit()
        elseif family_name == "GHZ State"
            return GHZStateCircuit()
        elseif family_name == "Bell State / EPR Pairs"
            return BellStateCircuit()
        elseif family_name == "Graph State"
            return GraphStateCircuit()
        elseif family_name == "Cluster State (1D)"
            return ClusterStateCircuit()
        elseif family_name == "QAOA MaxCut (p=1, 3-regular)"
            return QAOAMaxCutCircuit()
        elseif family_name == "Surface Code"
            return SurfaceCodeFamily()
        elseif family_name == "Quantum Fourier Transform"
            return QFTFamily()
        elseif family_name == "Grover Search"
            return GroverFamily()
        elseif family_name == "VQE Hardware-Efficient Ansatz"
            return VQEFamily()
        else
            error("Unknown family: $family_name")
        end
    end
end

println("✓ Dependencies loaded on all workers")
println()

#==============================================================================#
# CIRCUIT EXECUTION - HANDLES BOTH PHASES
#==============================================================================#

#=
Execute single circuit with comprehensive error handling and debugging.
=#
@everywhere function run_single_circuit_parallel(
    family_name::String,
    params::Dict
)
    try
        # Recreate family from name on worker (avoids serialization issues)
        family = get_family_from_name(family_name)
        # Debug: Show we started
        println("Worker $(myid()): Starting $(family_name) with $(params[:n_qubits]) qubits")
        
        # Set random seed for reproducibility
        Random.seed!(params[:seed])
        
        # Generate circuit - handles both Phase 1 and Phase 2 syntax
        circuit_result = try
            if family isa Union{QFTFamily, GroverFamily, VQEFamily, SurfaceCodeFamily}
                # Phase 2: keyword arguments
                if family isa QFTFamily
                    generate_circuit(family; n_qubits=params[:n_qubits], 
                                   density=params[:density], seed=params[:seed])
                elseif family isa GroverFamily
                    generate_circuit(family; n_qubits=params[:n_qubits],
                                   density=params[:density], seed=params[:seed])
                elseif family isa VQEFamily
                    generate_circuit(family; n_qubits=params[:n_qubits],
                                   layers=params[:layers], seed=params[:seed])
                elseif family isa SurfaceCodeFamily
                    generate_circuit(family; n_qubits=params[:n_qubits],
                                   n_t_gates=params[:n_t_gates], seed=params[:seed])
                end
            else
                # Phase 1 or QAOA: Dict parameters
                generate_circuit(family, params)
            end
        catch e
            return (
                success = false,
                family = family_name,
                n_qubits = get(params, :n_qubits, 0),
                n_t_gates = 0,
                error = "Circuit generation failed: $(string(e))",
                seed = params[:seed]
            )
        end
        
        # Extract circuit components
        if circuit_result isa CircuitInstance
            n_qubits = circuit_result.n_qubits
            gates = circuit_result.gates
            t_positions = circuit_result.t_gate_positions
            metadata = circuit_result.metadata
        else
            # Named tuple from Phase 2
            n_qubits = circuit_result.n_qubits
            gates = circuit_result.gates
            t_positions = circuit_result.t_positions
            metadata = circuit_result.metadata
        end
        
        # Initialize CAMPS state
        state = try
            s = CAMPSState(n_qubits; max_bond=2048)
            initialize!(s)
            s
        catch e
            return (
                success = false,
                family = family_name,
                n_qubits = n_qubits,
                n_t_gates = length(t_positions),
                error = "CAMPS initialization failed: $(string(e))",
                seed = params[:seed]
            )
        end
        
        # Track metrics
        start_time = time()
        ofd_success = 0
        ofd_fail = 0
        
        # Apply gates with error handling
        try
            for (idx, gate) in enumerate(gates)
                # Check if symbolic gate (Phase 1) or CAMPS Gate (Phase 2)
                if gate isa Tuple
                    # Phase 1: Symbolic gate
                    gate_type, qubits = gate
                    
                    if gate_type == :T
                        qubit = qubits[1]
                        success, _ = apply_t_gate_ofd!(state, qubit)
                        if success
                            ofd_success += 1
                        else
                            apply_gate!(state, RotationGate(qubit, :Z, π/4), strategy=OBDStrategy())
                            ofd_fail += 1
                        end
                    elseif gate_type == :H
                        apply_gate!(state, CliffordGate([(:H, qubits[1])], [qubits[1]]))
                    elseif gate_type == :CNOT
                        apply_gate!(state, CliffordGate([(:CNOT, qubits[1], qubits[2])], [qubits[1], qubits[2]]))
                    elseif gate_type == :X
                        apply_gate!(state, CliffordGate([(:X, qubits[1])], [qubits[1]]))
                    elseif gate_type == :Z
                        apply_gate!(state, CliffordGate([(:Z, qubits[1])], [qubits[1]]))
                    elseif gate_type == :S
                        apply_gate!(state, CliffordGate([(:S, qubits[1])], [qubits[1]]))
                    elseif gate_type == :random2q
                        q1, q2 = qubits[1], qubits[2]
                        cliff = random_clifford(2)
                        sparse = SparseGate(cliff, [q1, q2])
                        apply!(state.clifford, sparse)
                    end
                else
                    # Phase 2: CAMPS Gate object - just apply directly
                    if idx in t_positions
                        # This is a T-gate - try OFD
                        if gate isa RotationGate && gate.axis == :Z && abs(gate.angle - π/4) < 1e-10
                            success, _ = apply_t_gate_ofd!(state, gate.qubit)
                            if success
                                ofd_success += 1
                            else
                                apply_gate!(state, gate, strategy=OBDStrategy())
                                ofd_fail += 1
                            end
                        else
                            apply_gate!(state, gate)
                        end
                    else
                        # Regular gate (Clifford)
                        apply_gate!(state, gate)
                    end
                end
            end
        catch e
            return (
                success = false,
                family = family_name,
                n_qubits = n_qubits,
                n_t_gates = length(t_positions),
                n_total_gates = length(gates),
                error = "Gate application failed: $(string(e))",
                seed = params[:seed]
            )
        end
        
        # Final metrics
        final_chi = get_bond_dimension(state)
        final_nu = n_qubits - sum(state.free_qubits)
        final_S2 = max_entanglement_entropy(state.mps)
        runtime = time() - start_time
        
        ofd_rate = ofd_success / max(1, ofd_success + ofd_fail)
        
        # Extract family-specific metadata
        extra_metadata = Dict{String, Any}()
        if haskey(metadata, "density")
            extra_metadata["density"] = metadata["density"]
        end
        if haskey(metadata, "n_layers")
            extra_metadata["n_layers"] = metadata["n_layers"]
        end
        if haskey(metadata, "ansatz")
            extra_metadata["ansatz"] = metadata["ansatz"]
        end
        if haskey(metadata, "code_distance")
            extra_metadata["code_distance"] = metadata["code_distance"]
        end
        if haskey(metadata, "graph_type")
            extra_metadata["graph_type"] = metadata["graph_type"]
        end
        
        return (
            success = true,
            family = family_name,
            n_qubits = n_qubits,
            n_t_gates = length(t_positions),
            n_total_gates = length(gates),
            ofd_success = ofd_success,
            ofd_fail = ofd_fail,
            ofd_rate = ofd_rate,
            final_chi = final_chi,
            final_nu = final_nu,
            final_S2 = final_S2,
            runtime = runtime,
            seed = params[:seed],
            extra_metadata = extra_metadata
        )
        
    catch e
        # Catch-all for any unexpected errors
        return (
            success = false,
            family = family_name,
            n_qubits = get(params, :n_qubits, 0),
            n_t_gates = 0,
            error = "Unexpected error: $(string(e))\n$(sprint(showerror, e, catch_backtrace()))",
            seed = params[:seed]
        )
    end
end

#==============================================================================#
# EXPERIMENT GENERATION
#==============================================================================#

"""
Generate ULTRA-FAST test experiments (14 circuits, one per family).
Uses minimal parameters for fastest execution.
"""
function generate_ultrafast_phase1()
    experiments = []
    seed_base = 1000
    
    # 1 circuit per Phase 1 family with minimal parameters
    families_and_params = [
        (RandomBrickwallCliffordT(), Dict(:n_qubits => 8, :n_t_gates => 4, :clifford_depth => 2, :seed => seed_base)),
        (RandomAllToAllCliffordT(), Dict(:n_qubits => 8, :n_t_gates => 4, :clifford_layers => 16, :seed => seed_base+1)),
        (BernsteinVaziraniCircuit(), Dict(:n_qubits => 8, :n_t_gates => 4, :seed => seed_base+2)),
        (SimonCircuit(), Dict(:n_qubits => 8, :n_t_gates => 4, :seed => seed_base+3)),
        (DeutschJozsaCircuit(), Dict(:n_qubits => 8, :n_t_gates => 4, :function_type => :balanced, :seed => seed_base+4)),
        (GHZStateCircuit(), Dict(:n_qubits => 8, :n_t_gates => 4, :seed => seed_base+5)),
        (BellStateCircuit(), Dict(:n_qubits => 8, :n_t_gates => 4, :seed => seed_base+6)),
        (GraphStateCircuit(), Dict(:n_qubits => 8, :n_t_gates => 4, :edge_probability => 0.3, :seed => seed_base+7)),
        (ClusterStateCircuit(), Dict(:n_qubits => 8, :n_t_gates => 4, :seed => seed_base+8))
    ]
    
    for (family, params) in families_and_params
        family_name = get_name(family)
        push!(experiments, (family_name=family_name, params=params))
    end
    
    return experiments
end

function generate_ultrafast_phase2()
    experiments = []
    seed_base = 2000
    
    # 1 circuit per Phase 2 family with MINIMAL parameters (fast!)
    phase2_configs = [
        (get_name(QAOAMaxCutCircuit()), Dict(:n_qubits => 8, :n_t_gates => 4, :seed => seed_base)),
        (get_name(SurfaceCodeFamily()), Dict(:n_qubits => 8, :n_t_gates => 4, :seed => seed_base+1)),
        (get_name(QFTFamily()), Dict(:n_qubits => 4, :density => :low, :seed => seed_base+2)),
        (get_name(GroverFamily()), Dict(:n_qubits => 4, :density => :quarter, :seed => seed_base+3)),  # CRITICAL: quarter density, not full!
        (get_name(VQEFamily()), Dict(:n_qubits => 4, :layers => 1, :seed => seed_base+4))
    ]
    
    for (family_name, params) in phase2_configs
        push!(experiments, (family_name=family_name, params=params))
    end
    
    return experiments
end

"""
Generate experiment specifications for Phase 1 families (9 families).

Parameters:
- n_qubits: [8, 12, 16]
- t_fraction: [0.5, 1.0, 1.5]
- realizations: 8 per config

Total: 9 × 3 × 3 × 8 = 648 circuits
"""
function generate_phase1_experiments(n_realizations=8)
    experiments = []
    
    families = get_phase1_families()
    n_range = [8, 12, 16]
    t_fraction_range = [0.5, 1.0, 1.5]
    
    for family in families
        family_name = get_name(family)
        
        for n in n_range
            for t_frac in t_fraction_range
                n_t = Int(round(n * t_frac))
                
                for real in 1:n_realizations
                    # Deterministic seed
                    seed = hash((family_name, n, n_t, real)) % UInt32
                    
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
                        continue  # Skip odd n for Simon
                    elseif family isa DeutschJozsaCircuit
                        rng_temp = Random.MersenneTwister(seed)
                        params[:function_type] = rand(rng_temp, [:constant, :balanced])
                    elseif family isa GraphStateCircuit
                        params[:edge_probability] = 0.3
                    end
                    
                    push!(experiments, (family_name=family_name, params=params))
                end
            end
        end
    end
    
    return experiments
end

"""
Generate experiment specifications for Phase 2 families (5 families).

QAOA MaxCut: 72 circuits (3 sizes × 3 t_fractions × 8 realizations)
Surface Code: 72 circuits (3 sizes × 3 t_levels × 8 realizations)
QFT: 72 circuits (3 sizes × 3 densities × 8 realizations)
Grover: 72 circuits (3 sizes × 3 densities × 8 realizations)
VQE: 72 circuits (3 sizes × 3 layers × 8 realizations)

Total: 5 × 72 = 360 circuits
"""
function generate_phase2_experiments(n_realizations=8)
    experiments = []
    
    # QAOA MaxCut (72 circuits)
    qaoa_name = get_name(QAOAMaxCutCircuit())
    for n in [8, 12, 16]  # Must be even for 3-regular
        for t_frac in [0.5, 1.0, 1.5]
            n_t = Int(round(n * t_frac))
            for real in 1:n_realizations
                seed = Int(hash(("QAOA", n, n_t, real)) % UInt32)
                params = Dict{Symbol, Any}(
                    :n_qubits => n,
                    :n_t_gates => n_t,
                    :seed => seed
                )
                push!(experiments, (family_name=qaoa_name, params=params))
            end
        end
    end
    
    # Surface Code (72 circuits)
    surface_name = get_name(SurfaceCodeFamily())
    for n_target in [8, 12, 16]
        for n_t in [4, 8, 16]  # Low, medium, high T-gate density
            for real in 1:n_realizations
                seed = Int(hash(("Surface", n_target, n_t, real)) % UInt32)
                params = Dict{Symbol, Any}(
                    :n_qubits => n_target,
                    :n_t_gates => n_t,
                    :seed => seed
                )
                push!(experiments, (family_name=surface_name, params=params))
            end
        end
    end
    
    # QFT (72 circuits)
    # Can be split into 3 parts: part=1 (n=4), part=2 (n=6), part=3 (n=8)
    qft_name = get_name(QFTFamily())
    qft_n_range = [4, 6, 8]
    for n in qft_n_range
        for density in [:low, :medium, :high]
            for real in 1:n_realizations
                seed = Int(hash(("QFT", n, density, real)) % UInt32)
                params = Dict{Symbol, Any}(
                    :n_qubits => n,
                    :density => density,
                    :seed => seed
                )
                push!(experiments, (family_name=qft_name, params=params))
            end
        end
    end
    
    # Grover Search (72 circuits)
    grover_name = get_name(GroverFamily())
    for n in [4, 6, 8]
        for density in [:full, :half, :quarter]
            for real in 1:n_realizations
                seed = Int(hash(("Grover", n, density, real)) % UInt32)
                params = Dict{Symbol, Any}(
                    :n_qubits => n,
                    :density => density,
                    :seed => seed
                )
                push!(experiments, (family_name=grover_name, params=params))
            end
        end
    end
    
    # VQE Hardware-Efficient (72 circuits)
    # Can be split into 3 parts: part=1 (n=4), part=2 (n=6), part=3 (n=8)
    vqe_name = get_name(VQEFamily())
    vqe_n_range = [4, 6, 8]
    for n in vqe_n_range
        for layers in [1, 2, 4]
            for real in 1:n_realizations
                seed = Int(hash(("VQE", n, layers, real)) % UInt32)
                params = Dict{Symbol, Any}(
                    :n_qubits => n,
                    :layers => layers,
                    :seed => seed
                )
                push!(experiments, (family_name=vqe_name, params=params))
            end
        end
    end
    
    return experiments
end

#==============================================================================#
# MAIN PARALLEL BENCHMARK
#==============================================================================#

"""
Run complete parallel benchmark across all 14 families.

Modes:
- "test": Quick test (126 circuits)
- "quick": Fast run (378 circuits)
- "medium": Standard run (1,008 circuits) ← RECOMMENDED
- "full": Extended run with more realizations

Subsets (allows splitting into multiple runs for time limits):
- "all": All 14 families (1,008 circuits) - default
- "phase1": Phase 1 only - 9 basic families (648 circuits)
- "qaoa": QAOA MaxCut only (72 circuits)
- "surface": Surface Code only (72 circuits)
- "qft": Quantum Fourier Transform only (72 circuits)
- "qft1": QFT Part 1 - n=4 qubits only (24 circuits)
- "qft2": QFT Part 2 - n=6 qubits only (24 circuits)
- "qft3": QFT Part 3 - n=8 qubits only (24 circuits)
- "grover": Grover Search only (72 circuits)
- "vqe": VQE Hardware-Efficient only (72 circuits)
- "vqe1": VQE Part 1 - n=4 qubits only (24 circuits)
- "vqe2": VQE Part 2 - n=6 qubits only (24 circuits)
- "vqe3": VQE Part 3 - n=8 qubits only (24 circuits)

All subsets maintain exact same experiment parameters and can be combined
into a single dataset for ML training.
"""
function run_parallel_benchmark_complete(;
    mode = "medium",
    subset = "all",
    output_dir = "results/complete_benchmark",
    verbose = true)
    
    println("="^80)
    if subset == "all"
        println("PARALLEL BENCHMARK - ALL 14 CIRCUIT FAMILIES")
    elseif subset == "phase1"
        println("PARALLEL BENCHMARK - PHASE 1 ONLY (9 Basic Families)")
    elseif subset == "qaoa"
        println("PARALLEL BENCHMARK - QAOA MAXCUT ONLY")
    elseif subset == "surface"
        println("PARALLEL BENCHMARK - SURFACE CODE ONLY")
    elseif subset == "qft"
        println("PARALLEL BENCHMARK - QUANTUM FOURIER TRANSFORM ONLY")
    elseif subset == "qft1"
        println("PARALLEL BENCHMARK - QFT PART 1/3 (n=4 qubits)")
    elseif subset == "qft2"
        println("PARALLEL BENCHMARK - QFT PART 2/3 (n=6 qubits)")
    elseif subset == "qft3"
        println("PARALLEL BENCHMARK - QFT PART 3/3 (n=8 qubits)")
    elseif subset == "grover"
        println("PARALLEL BENCHMARK - GROVER SEARCH ONLY")
    elseif subset == "vqe"
        println("PARALLEL BENCHMARK - VQE HARDWARE-EFFICIENT ONLY")
    elseif subset == "vqe1"
        println("PARALLEL BENCHMARK - VQE PART 1/3 (n=4 qubits)")
    elseif subset == "vqe2"
        println("PARALLEL BENCHMARK - VQE PART 2/3 (n=6 qubits)")
    elseif subset == "vqe3"
        println("PARALLEL BENCHMARK - VQE PART 3/3 (n=8 qubits)")
    end
    
    if mode == "savetest"
        println("*** SAVE TEST MODE - 1 CIRCUIT ONLY FOR CSV VERIFICATION ***")
    end
    
    println("="^80)
    println("Mode: ", mode)
    println("Subset: ", subset)
    println()
    
    # Set parameters based on mode
    n_realizations = if mode == "savetest"
        # Save test: 1 circuit per subset to verify CSV saving works
        "savetest"  # Special marker
    elseif mode == "ultrafast"
        # Ultra-fast: 1 circuit per family only (14 total)
        "ultrafast"  # Special marker
    elseif mode == "test"
        1  # Quick test
    elseif mode == "quick"
        3
    elseif mode == "medium"
        8  # Standard: 1,008 circuits
    elseif mode == "full"
        10  # Extended: 1,260 circuits
    else
        error("Unknown mode: $mode. Use 'savetest', 'ultrafast', 'test', 'quick', 'medium', or 'full'")
    end
    
    mkpath(output_dir)
    
    # Generate experiments
    println("Generating experiment specifications...")
    
    if n_realizations == "savetest"
        # Save test mode: 1 circuit only for quick CSV verification
        if subset == "phase1"
            # Just one Phase 1 family, one circuit
            experiments = [(family_name="Random Clifford+T (Brick-wall)",
                          params=Dict(:n_qubits => 8, :n_t_gates => 4, :clifford_depth => 2, :seed => 1000))]
        elseif subset == "qaoa"
            experiments = [(family_name="QAOA MaxCut (p=1, 3-regular)",
                          params=Dict(:n_qubits => 8, :n_t_gates => 4, :seed => 2000))]
        elseif subset == "surface"
            experiments = [(family_name="Surface Code",
                          params=Dict(:n_qubits => 8, :n_t_gates => 4, :seed => 2001))]
        elseif subset == "qft" || subset == "qft1"
            experiments = [(family_name="Quantum Fourier Transform",
                          params=Dict(:n_qubits => 4, :density => :low, :seed => 2002))]
        elseif subset == "qft2"
            experiments = [(family_name="Quantum Fourier Transform",
                          params=Dict(:n_qubits => 6, :density => :low, :seed => 2002))]
        elseif subset == "qft3"
            experiments = [(family_name="Quantum Fourier Transform",
                          params=Dict(:n_qubits => 8, :density => :low, :seed => 2002))]
        elseif subset == "grover"
            experiments = [(family_name="Grover Search",
                          params=Dict(:n_qubits => 4, :density => :quarter, :seed => 2003))]
        elseif subset == "vqe" || subset == "vqe1"
            experiments = [(family_name="VQE Hardware-Efficient Ansatz",
                          params=Dict(:n_qubits => 4, :layers => 1, :seed => 2004))]
        elseif subset == "vqe2"
            experiments = [(family_name="VQE Hardware-Efficient Ansatz",
                          params=Dict(:n_qubits => 6, :layers => 1, :seed => 2004))]
        elseif subset == "vqe3"
            experiments = [(family_name="VQE Hardware-Efficient Ansatz",
                          params=Dict(:n_qubits => 8, :layers => 1, :seed => 2004))]
        else  # all
            experiments = [(family_name="Random Clifford+T (Brick-wall)",
                          params=Dict(:n_qubits => 8, :n_t_gates => 4, :clifford_depth => 2, :seed => 1000))]
        end
        phase1_experiments = subset == "phase1" ? experiments : []
        phase2_experiments = subset != "phase1" ? experiments : []
    elseif n_realizations == "ultrafast"
        # Ultra-fast mode: 1 circuit per family, minimal parameters
        phase1_experiments = generate_ultrafast_phase1()
        phase2_experiments = generate_ultrafast_phase2()
    else
        phase1_experiments = generate_phase1_experiments(n_realizations)
        phase2_experiments = generate_phase2_experiments(n_realizations)
    end
    
    # Filter experiments based on subset
    if subset == "all"
        # No filtering - include everything
        println("Subset: 'all' - Including all 14 families (1,008 circuits)")
    elseif subset == "phase1"
        # Phase 1 only - all 9 basic families
        phase2_experiments = []  # Exclude all Phase 2
        println("Subset: 'phase1' - Including Phase 1 only (648 circuits)")
    elseif subset == "qaoa"
        # QAOA only
        phase1_experiments = []
        phase2_experiments = filter(exp -> exp.family_name == "QAOA MaxCut (p=1, 3-regular)", phase2_experiments)
        println("Subset: 'qaoa' - Including QAOA MaxCut only (72 circuits)")
    elseif subset == "surface"
        # Surface Code only
        phase1_experiments = []
        phase2_experiments = filter(exp -> exp.family_name == "Surface Code", phase2_experiments)
        println("Subset: 'surface' - Including Surface Code only (72 circuits)")
    elseif subset == "qft"
        # QFT only - all 72 circuits
        phase1_experiments = []
        phase2_experiments = filter(exp -> exp.family_name == "Quantum Fourier Transform", phase2_experiments)
        println("Subset: 'qft' - Including Quantum Fourier Transform only (72 circuits)")
    elseif subset == "qft1"
        # QFT Part 1 - n=4 qubits only (24 circuits: 3 densities × 8 realizations)
        phase1_experiments = []
        phase2_experiments = filter(exp -> exp.family_name == "Quantum Fourier Transform" && exp.params[:n_qubits] == 4, phase2_experiments)
        println("Subset: 'qft1' - Including QFT Part 1/3 (n=4, 24 circuits)")
    elseif subset == "qft2"
        # QFT Part 2 - n=6 qubits only (24 circuits: 3 densities × 8 realizations)
        phase1_experiments = []
        phase2_experiments = filter(exp -> exp.family_name == "Quantum Fourier Transform" && exp.params[:n_qubits] == 6, phase2_experiments)
        println("Subset: 'qft2' - Including QFT Part 2/3 (n=6, 24 circuits)")
    elseif subset == "qft3"
        # QFT Part 3 - n=8 qubits only (24 circuits: 3 densities × 8 realizations)
        phase1_experiments = []
        phase2_experiments = filter(exp -> exp.family_name == "Quantum Fourier Transform" && exp.params[:n_qubits] == 8, phase2_experiments)
        println("Subset: 'qft3' - Including QFT Part 3/3 (n=8, 24 circuits)")
    elseif subset == "grover"
        # Grover only
        phase1_experiments = []
        phase2_experiments = filter(exp -> exp.family_name == "Grover Search", phase2_experiments)
        println("Subset: 'grover' - Including Grover Search only (72 circuits)")
    elseif subset == "vqe"
        # VQE only - all 72 circuits
        phase1_experiments = []
        phase2_experiments = filter(exp -> exp.family_name == "VQE Hardware-Efficient Ansatz", phase2_experiments)
        println("Subset: 'vqe' - Including VQE Hardware-Efficient Ansatz only (72 circuits)")
    elseif subset == "vqe1"
        # VQE Part 1 - n=4 qubits only (24 circuits: 3 layers × 8 realizations)
        phase1_experiments = []
        phase2_experiments = filter(exp -> exp.family_name == "VQE Hardware-Efficient Ansatz" && exp.params[:n_qubits] == 4, phase2_experiments)
        println("Subset: 'vqe1' - Including VQE Part 1/3 (n=4, 24 circuits)")
    elseif subset == "vqe2"
        # VQE Part 2 - n=6 qubits only (24 circuits: 3 layers × 8 realizations)
        phase1_experiments = []
        phase2_experiments = filter(exp -> exp.family_name == "VQE Hardware-Efficient Ansatz" && exp.params[:n_qubits] == 6, phase2_experiments)
        println("Subset: 'vqe2' - Including VQE Part 2/3 (n=6, 24 circuits)")
    elseif subset == "vqe3"
        # VQE Part 3 - n=8 qubits only (24 circuits: 3 layers × 8 realizations)
        phase1_experiments = []
        phase2_experiments = filter(exp -> exp.family_name == "VQE Hardware-Efficient Ansatz" && exp.params[:n_qubits] == 8, phase2_experiments)
        println("Subset: 'vqe3' - Including VQE Part 3/3 (n=8, 24 circuits)")
    else
        error("Unknown subset: $subset. Use 'all', 'phase1', 'qaoa', 'surface', 'qft', 'qft1', 'qft2', 'qft3', 'grover', 'vqe', 'vqe1', 'vqe2', or 'vqe3'")
    end
    
    experiments = vcat(phase1_experiments, phase2_experiments)
    
    n_total = length(experiments)
    n_phase1 = length(phase1_experiments)
    n_phase2 = length(phase2_experiments)
    
    println("Phase 1 experiments: ", n_phase1)
    println("Phase 2 experiments: ", n_phase2)
    println("Total experiments: ", n_total)
    println("Workers: ", nworkers())
    println()
    println("Starting parallel execution...")
    println("-"^80)
    
    # Parallel execution with progress tracking
    start_time = time()
    batch_size = max(1, n_total ÷ (nworkers() * 4))
    
    println("Batch size: $batch_size experiments per worker task")
    println()
    
    # Progress tracking state
    progress_lock = ReentrantLock()
    completed = Ref(0)
    last_update = Ref(time())
    
    # Progress callback
    function update_progress(result)
        lock(progress_lock) do
            completed[] += 1
            
            # Update every 20 completions or every 60 seconds
            if completed[] % 20 == 0 || (time() - last_update[]) > 60
                elapsed = time() - start_time
                eta = (elapsed / completed[]) * (n_total - completed[]) / 60
                
                @printf("[%5d/%5d] %.1f%% complete, ETA: %.1f min\n",
                        completed[], n_total, 
                        100 * completed[] / n_total,
                        eta)
                
                last_update[] = time()
            end
        end
    end
    
    # Run experiments in parallel
    results_raw = pmap(experiments, batch_size=batch_size) do (family_name, params)
        result = run_single_circuit_parallel(family_name, params)
        
        # Update progress (thread-safe)
        update_progress(result)
        
        return result
    end
    
    total_time = (time() - start_time) / 60
    println("-"^80)
    println("✓ Parallel execution complete!")
    @printf("Completed %d experiments in %.1f minutes (%.2f hours)\n", 
            n_total, total_time, total_time/60)
    println()
    
    # Filter successful results
    results = filter(r -> get(r, :success, true), results_raw)
    n_failed = n_total - length(results)
    
    if n_failed > 0
        println("⚠️  Warning: $n_failed experiments failed")
        
        # Print sample errors from failed circuits
        failed_results = filter(r -> !get(r, :success, true), results_raw)
        
        if length(failed_results) > 0
            println()
            println("Sample errors from failed circuits:")
            println("-"^80)
            
            # Group by family to see patterns
            families_with_errors = unique([r.family for r in failed_results])
            
            for family in families_with_errors
                family_failures = filter(r -> r.family == family, failed_results)
                n_family_failures = length(family_failures)
                
                println("$family: $n_family_failures failures")
                
                # Show first error for this family
                if haskey(family_failures[1], :error)
                    error_msg = family_failures[1].error
                    # Truncate to first 200 chars
                    truncated = length(error_msg) > 200 ? error_msg[1:200]*"..." : error_msg
                    println("  First error: ", truncated)
                end
                println()
            end
            println("-"^80)
            println()
        end
    end
    
    if isempty(results)
        println("ERROR: All experiments failed!")
        return DataFrame()
    end
    
    # Save results
    df = DataFrame(results)
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    
    # Flatten extra_metadata into separate columns
    if "extra_metadata" in names(df)
        # First pass: collect all metadata keys that actually exist
        metadata_keys = Set{String}()
        for row in eachrow(df)
            if !ismissing(row.extra_metadata) && !isnothing(row.extra_metadata)
                union!(metadata_keys, keys(row.extra_metadata))
            end
        end
        
        # Create columns with proper defaults
        for key in metadata_keys
            df[!, Symbol(key)] = fill("", nrow(df))  # Initialize with empty strings
        end
        
        # Second pass: fill in actual values
        for (i, row) in enumerate(eachrow(df))
            if !ismissing(row.extra_metadata) && !isnothing(row.extra_metadata)
                for (k, v) in row.extra_metadata
                    df[i, Symbol(k)] = string(v)  # Convert to string for CSV
                end
            end
        end
    end
    
    csv_file = joinpath(output_dir, "results_$(timestamp).csv")
    CSV.write(csv_file, df)
    println("✓ Results saved to: ", csv_file)
    
    # Aggregate statistics
    agg_df = combine(groupby(df, [:family, :n_qubits, :n_t_gates]),
        :ofd_rate => mean => :mean_ofd_rate,
        :ofd_rate => std => :std_ofd_rate,
        :final_chi => mean => :mean_chi,
        :final_nu => mean => :mean_nu,
        :final_S2 => mean => :mean_S2,
        :runtime => mean => :mean_runtime,
        nrow => :n_samples
    )
    
    agg_file = joinpath(output_dir, "aggregated_$(timestamp).csv")
    CSV.write(agg_file, agg_df)
    println("✓ Aggregated stats saved to: ", agg_file)
    
    # Summary statistics
    println()
    println("="^80)
    println("SUMMARY BY FAMILY")
    println("="^80)
    family_stats = combine(groupby(df, :family),
        :ofd_rate => mean => :mean_ofd_rate,
        :final_chi => mean => :mean_chi,
        nrow => :n_circuits
    )
    sort!(family_stats, :mean_ofd_rate, rev=true)
    
    println()
    @printf("%-40s | %8s | %8s | %8s\n", "Family", "OFD Rate", "Avg χ", "Circuits")
    println("-"^80)
    for row in eachrow(family_stats)
        @printf("%-40s | %7.1f%% | %8.1f | %8d\n", 
                row.family, row.mean_ofd_rate * 100, row.mean_chi, row.n_circuits)
    end
    println("="^80)
    
    # Phase-wise summary
    println()
    println("PHASE-WISE SUMMARY")
    println("-"^80)
    phase1_families = [get_name(f) for f in get_phase1_families()]
    phase2_families = [get_name(f) for f in get_phase2_families()]
    
    phase1_results = filter(r -> r.family in phase1_families, df)
    phase2_results = filter(r -> r.family in phase2_families, df)
    
    @printf("Phase 1 (9 families):  %4d circuits, OFD rate: %.1f%%, Avg χ: %.1f\n",
            nrow(phase1_results), 
            mean(phase1_results.ofd_rate) * 100,
            mean(phase1_results.final_chi))
    
    @printf("Phase 2 (5 families):  %4d circuits, OFD rate: %.1f%%, Avg χ: %.1f\n",
            nrow(phase2_results),
            mean(phase2_results.ofd_rate) * 100,
            mean(phase2_results.final_chi))
    println("-"^80)
    
    # Performance statistics
    println()
    println("PERFORMANCE STATISTICS")
    println("-"^80)
    @printf("Total experiments:    %d\n", n_total)
    @printf("Successful:           %d\n", length(results))
    @printf("Failed:               %d\n", n_failed)
    @printf("Total time:           %.1f minutes (%.2f hours)\n", total_time, total_time/60)
    @printf("Avg time per circuit: %.2f seconds\n", (total_time * 60) / n_total)
    @printf("Workers used:         %d\n", nworkers())
    
    theoretical_speedup = nworkers()
    avg_runtime = mean(df.runtime)
    actual_speedup = (total_time * 60) / avg_runtime
    efficiency = 100 * actual_speedup / theoretical_speedup
    
    @printf("Theoretical speedup:  %.1fx\n", theoretical_speedup)
    @printf("Actual speedup:       %.1fx (%.0f%% efficiency)\n", 
            actual_speedup, efficiency)
    println("="^80)
    
    # ML readiness check
    println()
    println("ML READINESS CHECK")
    println("-"^80)
    @printf("Total circuits:       %d\n", nrow(df))
    @printf("Families:             %d\n", length(unique(df.family)))
    @printf("Qubit range:          %d - %d\n", minimum(df.n_qubits), maximum(df.n_qubits))
    @printf("T-gate range:         %d - %d\n", minimum(df.n_t_gates), maximum(df.n_t_gates))
    @printf("OFD rate range:       %.1f%% - %.1f%%\n", 
            minimum(df.ofd_rate)*100, maximum(df.ofd_rate)*100)
    
    # Check feature completeness
    required_features = ["n_qubits", "n_t_gates", "n_total_gates", "ofd_rate", 
                        "final_chi", "final_nu", "final_S2"]
    all_present = all(f in names(df) for f in required_features)
    
    if all_present
        println("✓ All required ML features present")
    else
        missing = [f for f in required_features if !(f in names(df))]
        println("✗ Missing features: ", join(missing, ", "))
    end
    
    println("-"^80)
    println()
    println("✓ Benchmark complete! Ready for ML training.")
    println("  Results file: ", csv_file)
    println("  Aggregated file: ", agg_file)
    println("="^80)
    
    return df
end

#==============================================================================#
# ENTRY POINT
#==============================================================================#

function main(mode::String="medium", subset::String="all")
    println("CAMPS.jl Complete Benchmark Suite")
    println("Mode: $mode")
    println("Subset: $subset")
    println()
    
    if !(mode in ["savetest", "ultrafast", "test", "quick", "medium", "full"])
        println("ERROR: Unknown mode '$mode'")
        println("Available modes:")
        println("  savetest  - Save test (1 circuit, ~30 sec) ← TEST CSV SAVING")
        println("  ultrafast - Ultra-fast test (14 circuits, ~5 min) ← DEBUG/VERIFY")
        println("  test   - Quick test (126 circuits, ~30 hours with Grover)")
        println("  quick  - Fast run (126 circuits, ~15 minutes)")
        println("  medium - Standard run (1,008 circuits, ~15-20 hours) ← RECOMMENDED")
        println("  full   - Extended run (1,260 circuits, ~20-25 hours)")
        return
    end
    
    if !(subset in ["all", "phase1", "qaoa", "surface", "qft", "qft1", "qft2", "qft3", "grover", "vqe", "vqe1", "vqe2", "vqe3"])
        println("ERROR: Unknown subset '$subset'")
        println()
        println("Available subsets:")
        println("  all     - All 14 families (1,008 circuits) ← DEFAULT")
        println("  phase1  - Phase 1 only: 9 basic families (648 circuits)")
        println("  qaoa    - QAOA MaxCut only (72 circuits)")
        println("  surface - Surface Code only (72 circuits)")
        println("  qft     - Quantum Fourier Transform only (72 circuits)")
        println("  qft1    - QFT Part 1/3: n=4 qubits (24 circuits)")
        println("  qft2    - QFT Part 2/3: n=6 qubits (24 circuits)")
        println("  qft3    - QFT Part 3/3: n=8 qubits (24 circuits)")
        println("  grover  - Grover Search only (72 circuits)")
        println("  vqe     - VQE Hardware-Efficient only (72 circuits)")
        println("  vqe1    - VQE Part 1/3: n=4 qubits (24 circuits)")
        println("  vqe2    - VQE Part 2/3: n=6 qubits (24 circuits)")
        println("  vqe3    - VQE Part 3/3: n=8 qubits (24 circuits)")
        println()
        println("USAGE:")
        println("  julia benchmarks/run_parallel_benchmark_complete.jl medium phase1")
        println("  julia benchmarks/run_parallel_benchmark_complete.jl medium qaoa")
        println("  julia benchmarks/run_parallel_benchmark_complete.jl medium surface")
        println("  julia benchmarks/run_parallel_benchmark_complete.jl medium qft")
        println("  julia benchmarks/run_parallel_benchmark_complete.jl medium qft1")
        println("  julia benchmarks/run_parallel_benchmark_complete.jl medium qft2")
        println("  julia benchmarks/run_parallel_benchmark_complete.jl medium qft3")
        println("  julia benchmarks/run_parallel_benchmark_complete.jl medium grover")
        println("  julia benchmarks/run_parallel_benchmark_complete.jl medium vqe")
        println("  julia benchmarks/run_parallel_benchmark_complete.jl medium vqe1")
        println("  julia benchmarks/run_parallel_benchmark_complete.jl medium vqe2")
        println("  julia benchmarks/run_parallel_benchmark_complete.jl medium vqe3")
        println()
        println("COMBINING RESULTS:")
        println("  For QFT split experiments, combine parts 1-3:")
        println("    julia combine_results.jl qft1.csv qft2.csv qft3.csv qft_combined.csv")
        println("  For VQE split experiments, combine parts 1-3:")
        println("    julia combine_results.jl vqe1.csv vqe2.csv vqe3.csv vqe_combined.csv")
        println("  For all subsets, combine all CSV files:")
        println("    julia combine_results.jl phase1.csv qaoa.csv surface.csv qft.csv grover.csv vqe.csv combined.csv")
        return
    end
    
    run_parallel_benchmark_complete(mode=mode, subset=subset, verbose=true)
end

# Auto-run if executed as script
if abspath(PROGRAM_FILE) == @__FILE__
    mode = length(ARGS) >= 1 ? ARGS[1] : "medium"
    subset = length(ARGS) >= 2 ? ARGS[2] : "all"
    main(mode, subset)
end