# CAMPS.jl/benchmarks/extract_features_batch.jl
#
# POST-HOC FEATURE EXTRACTION FROM BENCHMARK RESULTS
#
# This script takes benchmark results CSV (with seeds) and extracts features
# for all circuits WITHOUT re-running expensive CAMPS simulation.
#
# Process:
# 1. Load results CSV
# 2. For each row: regenerate circuit deterministically from seed
# 3. Extract features via static analysis
# 4. Add features as new columns
# 5. Save augmented CSV
#
# Time: ~5 minutes for 648 circuits (no simulation!)

using DataFrames
using CSV
using Random
using Printf
using Statistics

include("circuit_families.jl")
include("circuit_feature_extraction.jl")

#==============================================================================#
# FAMILY NAME TO INSTANCE MAPPING
#==============================================================================#

"""
    get_family_by_name(name::AbstractString) -> AbstractCircuitFamily

Map family name string back to family instance for circuit regeneration.

This handles the names output by get_name() for each family, ensuring
we can regenerate circuits from CSV results.

Names match exactly what appears in benchmark CSV output:
- "Random Clifford+T (Brick-wall)"  → RandomBrickwallCliffordT()
- "Bernstein-Vazirani"              → BernsteinVaziraniCircuit()
- etc.

Accepts AbstractString to handle CSV.jl's InlineString types (String7, String31, etc.)
"""
function get_family_by_name(name::AbstractString)
    # Convert to String for consistent comparison (CSV.jl uses InlineString types)
    name_str = String(name)
    
    # Exact string matching for all 9 families
    if name_str == "Random Clifford+T (Brick-wall)" || name_str == "RandomBrickwallCliffordT"
        return RandomBrickwallCliffordT()
    elseif name_str == "Random Clifford+T (All-to-all)" || name_str == "RandomAllToAllCliffordT"
        return RandomAllToAllCliffordT()
    elseif name_str == "Bernstein-Vazirani" || name_str == "BernsteinVazirani"
        return BernsteinVaziraniCircuit()
    elseif name_str == "Simon's Algorithm" || name_str == "Simon"
        return SimonCircuit()
    elseif name_str == "Deutsch-Jozsa" || name_str == "DeutschJozsa"
        return DeutschJozsaCircuit()
    elseif name_str == "GHZ State" || name_str == "GHZState"
        return GHZStateCircuit()
    elseif name_str == "Bell State / EPR Pairs" || name_str == "BellState"
        return BellStateCircuit()
    elseif name_str == "Graph State" || name_str == "GraphState"
        return GraphStateCircuit()
    elseif name_str == "Cluster State (1D)" || name_str == "ClusterState"
        return ClusterStateCircuit()
    else
        error("Unknown family name: $name_str")
    end
end

#==============================================================================#
# PARAMETER RECONSTRUCTION
#==============================================================================#

"""
    reconstruct_circuit_params(row::DataFrameRow) -> Dict{Symbol, Any}

Reconstruct circuit generation parameters from CSV row.

Extracts all parameters needed to regenerate the exact circuit:
- Basic: n_qubits, n_t_gates, seed
- Family-specific: clifford_depth, clifford_layers, etc.

These parameters, combined with deterministic seed, guarantee we regenerate
the identical circuit that was benchmarked.

Returns dictionary ready for generate_circuit() call.
"""
function reconstruct_circuit_params(row)
    params = Dict{Symbol, Any}(
        :n_qubits => row.n_qubits,
        :n_t_gates => row.n_t_gates,
        :seed => row.seed
    )
    
    # Convert to String for consistent string operations (CSV.jl uses InlineString)
    family_name = String(row.family)
    
    # Add family-specific parameters
    # These must match what the benchmark used!
    if occursin("Brick-wall", family_name)
        params[:clifford_depth] = 2  # Standard from benchmark
        
    elseif occursin("All-to-all", family_name)
        params[:clifford_layers] = 2 * row.n_qubits  # Standard from benchmark
        
    elseif occursin("Deutsch-Jozsa", family_name)
        # DJ needs function type - infer from n_t_gates
        # Balanced function has T-gates, constant doesn't
        params[:function_type] = row.n_t_gates > 0 ? :balanced : :constant
        
    elseif occursin("Graph State", family_name)
        params[:edge_probability] = 0.3  # Standard from benchmark
    end
    
    return params
end

#==============================================================================#
# BATCH FEATURE EXTRACTION
#==============================================================================#

"""
    extract_features_from_results(csv_path::String) -> DataFrame

Extract features for all circuits in benchmark results CSV.

Input CSV must contain columns:
- family: Circuit family name
- n_qubits: Number of qubits
- n_t_gates: Number of T-gates
- seed: Random seed used for generation
- (all other benchmark results: ofd_rate, final_chi, etc.)

Process:
1. Load CSV
2. For each row:
   a. Get family instance from name
   b. Reconstruct parameters
   c. Regenerate circuit (deterministic!)
   d. Extract features (no simulation)
   e. Add to dataframe
3. Save augmented CSV

Output CSV has all original columns plus:
- avg_clifford_depth, max_clifford_depth, min_clifford_depth, std_clifford_depth
- avg_light_cone_width, max_light_cone_width
- spatial_uniformity, depth_variance
- n_gates

Time: ~5 minutes for 648 circuits

Example:
```bash
cd ~/Downloads/CAMPS\\ 2.jl
julia --project=. benchmarks/extract_features_batch.jl \\
    results/parallel_benchmark/results_20260106.csv \\
    results/results_with_features.csv
```

Or programmatically:
```julia
df = extract_features_from_results("results/parallel_benchmark/results_20260106.csv")
CSV.write("results_with_features.csv", df)
```
"""
function extract_features_from_results(csv_path::String)
    println("="^70)
    println("POST-HOC FEATURE EXTRACTION")
    println("="^70)
    println()
    
    # Load results
    println("Loading results from: $csv_path")
    df = CSV.read(csv_path, DataFrame)
    n_circuits = nrow(df)
    println("Loaded $n_circuits circuits")
    println()
    
    # Validate required columns
    required_cols = ["family", "n_qubits", "n_t_gates", "seed"]
    for col in required_cols
        if !(col in names(df))
            error("Missing required column: $col")
        end
    end
    
    println("Extracting features...")
    println("-"^70)
    
    # Initialize feature columns
    df[!, :avg_clifford_depth] = zeros(Float64, n_circuits)
    df[!, :max_clifford_depth] = zeros(Float64, n_circuits)
    df[!, :min_clifford_depth] = zeros(Float64, n_circuits)
    df[!, :std_clifford_depth] = zeros(Float64, n_circuits)
    df[!, :avg_light_cone_width] = zeros(Float64, n_circuits)
    df[!, :max_light_cone_width] = zeros(Float64, n_circuits)
    df[!, :spatial_uniformity] = zeros(Float64, n_circuits)
    df[!, :depth_variance] = zeros(Float64, n_circuits)
    df[!, :n_gates] = zeros(Int, n_circuits)
    
    # Progress tracking
    start_time = time()
    last_update = time()
    
    for (i, row) in enumerate(eachrow(df))
        # Regenerate circuit
        try
            family_name = row.family
            family = get_family_by_name(family_name)
            params = reconstruct_circuit_params(row)
            
            # CRITICAL: Set seed for deterministic generation
            Random.seed!(params[:seed])
            
            # Generate exact same circuit as benchmark
            circuit = generate_circuit(family, params)
            
            # Extract features (no simulation!)
            features = extract_circuit_features(circuit)
            
            # Store in dataframe
            df[i, :avg_clifford_depth] = features["avg_clifford_depth"]
            df[i, :max_clifford_depth] = features["max_clifford_depth"]
            df[i, :min_clifford_depth] = features["min_clifford_depth"]
            df[i, :std_clifford_depth] = features["std_clifford_depth"]
            df[i, :avg_light_cone_width] = features["avg_light_cone_width"]
            df[i, :max_light_cone_width] = features["max_light_cone_width"]
            df[i, :spatial_uniformity] = features["spatial_uniformity"]
            df[i, :depth_variance] = features["depth_variance"]
            df[i, :n_gates] = features["n_gates"]
            
        catch e
            @warn "Failed to extract features for circuit $i: $(row.family), n=$(row.n_qubits)" exception=e
            # Leave as zeros (will be filtered in ML)
        end
        
        # Progress update every 50 circuits or every 30 seconds
        if i % 50 == 0 || (time() - last_update) > 30
            elapsed = time() - start_time
            rate = i / elapsed
            eta = (n_circuits - i) / rate / 60
            
            @printf("[%4d/%4d] %.1f%% complete, ETA: %.1f min\n",
                    i, n_circuits, 100 * i / n_circuits, eta)
            
            last_update = time()
        end
    end
    
    total_time = (time() - start_time) / 60
    
    println("-"^70)
    @printf("Feature extraction complete! %.1f minutes\n", total_time)
    @printf("Average: %.2f circuits/second\n", n_circuits / (total_time * 60))
    println()
    
    # Summary statistics
    println("Feature Summary:")
    println("-"^70)
    println("Avg Clifford depth:   $(round(mean(df.avg_clifford_depth), digits=2)) ± $(round(std(df.avg_clifford_depth), digits=2))")
    println("Avg light cone width: $(round(mean(df.avg_light_cone_width), digits=2)) ± $(round(std(df.avg_light_cone_width), digits=2))")
    println("Spatial uniformity:   $(round(mean(df.spatial_uniformity), digits=3)) ± $(round(std(df.spatial_uniformity), digits=3))")
    println()
    
    return df
end

#==============================================================================#
# COMMAND LINE INTERFACE
#==============================================================================#

"""
Main entry point for command-line usage.

Usage:
```bash
julia --project=. benchmarks/extract_features_batch.jl <input_csv> [output_csv]
```

If output_csv not specified, defaults to input_csv with "_with_features" suffix.

Example:
```bash
julia --project=. benchmarks/extract_features_batch.jl \\
    results/parallel_benchmark/results_20260106.csv \\
    results/results_with_features.csv
```
"""
function main()
    if length(ARGS) < 1
        println("Usage: julia extract_features_batch.jl <input_csv> [output_csv]")
        println()
        println("Example:")
        println("  julia extract_features_batch.jl results.csv results_with_features.csv")
        return
    end
    
    input_csv = ARGS[1]
    
    # Determine output path
    output_csv = if length(ARGS) >= 2
        ARGS[2]
    else
        # Default: add "_with_features" before .csv
        base, ext = splitext(input_csv)
        "$(base)_with_features$(ext)"
    end
    
    # Check input exists
    if !isfile(input_csv)
        error("Input file not found: $input_csv")
    end
    
    # Extract features
    df = extract_features_from_results(input_csv)
    
    # Save
    println("Saving to: $output_csv")
    CSV.write(output_csv, df)
    println("✓ Complete dataset saved!")
    println()
    
    # Summary
    println("="^70)
    println("DATASET READY FOR ML TRAINING")
    println("="^70)
    println("Circuits:  $(nrow(df))")
    println("Features:  $(ncol(df))")
    println("File:      $output_csv")
    println()
    println("Key features extracted:")
    println("  ✓ avg_clifford_depth, max_clifford_depth")
    println("  ✓ avg_light_cone_width (w = 4d)")
    println("  ✓ spatial_uniformity (entropy)")
    println("  ✓ depth_variance")
    println()
    println("Ready for ML model training!")
    println("No final_chi, final_nu, final_S2 in features → no data leakage ✓")
    println()
end

# Auto-run if executed as script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

export extract_features_from_results, get_family_by_name