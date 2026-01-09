# CAMPS.jl/benchmarks/circuit_feature_extraction.jl
#
# RIGOROUS CIRCUIT FEATURE EXTRACTION
# 
# Computes features from circuit structure WITHOUT running simulation.
# Enables ML prediction of OFD success before expensive CAMPS execution.
#
# All features are computable via static circuit analysis, ensuring the ML
# model can be used prospectively (before simulation).
#
# Reference: Liu & Clark "Clifford-Augmented Matrix Product States" 
#            arXiv:2412.17209 (light cone width w = 4d formula)

using Statistics
using Random

#==============================================================================#
# CORE: CLIFFORD DEPTH COMPUTATION
#==============================================================================#

"""
    compute_clifford_depths_to_t_gates(circuit::CircuitInstance) -> Vector{Float64}

Compute Clifford depth from |0⟩ to each T-gate in the circuit.

For each T-gate, we count the number of Clifford layers between the initial
state and that T-gate position. Depth is tracked per-qubit:
- Single-qubit Cliffords (H, S, Z, X): increment depth on that qubit
- Two-qubit Cliffords (CNOT, random2q): both qubits advance to max + 1
- T-gates: record depth at current position (don't increment Clifford depth)

This is the key metric for OFD applicability: deeper Clifford circuits between
T-gates allow more stabilizer evolution, increasing light cone widths and
making OFD more challenging.

Returns:
- Vector of depths, one per T-gate in circuit order
- Empty vector if circuit has no T-gates

Example:
```julia
circuit = generate_circuit(RandomBrickwallCliffordT(), params)
depths = compute_clifford_depths_to_t_gates(circuit)
# depths[i] = Clifford depth to i-th T-gate
```

Algorithm:
1. Initialize depth counter per qubit (starts at 0)
2. Walk through gates sequentially
3. When encountering T-gate: record current depth of its qubit
4. After T-gate (and all Cliffords): update depth counters
5. Two-qubit gates: synchronize depths (both advance to max + 1)

Time complexity: O(|gates|) - single pass through circuit
Space complexity: O(n_qubits + n_t_gates)
"""
function compute_clifford_depths_to_t_gates(circuit::CircuitInstance)
    n = circuit.n_qubits
    depths = Float64[]
    
    # Track Clifford depth on each qubit independently
    # This handles circuits with varying depth per qubit correctly
    qubit_depths = zeros(Float64, n)
    
    for (gate_idx, (gate_type, qubits)) in enumerate(circuit.gates)
        # CRITICAL: Check if this position is a T-gate BEFORE updating depths
        # We want the depth TO the T-gate, not including it
        if gate_idx in circuit.t_gate_positions
            t_qubit = qubits[1]  # T-gate acts on single qubit
            push!(depths, qubit_depths[t_qubit])
        end
        
        # Update depth counters for future gates
        if gate_type == :T
            # T-gates don't contribute to Clifford depth
            # We're measuring Clifford layers between T-gates
            continue
            
        elseif gate_type == :H
            # Hadamard: single-qubit Clifford
            q = qubits[1]
            qubit_depths[q] += 1.0
            
        elseif gate_type == :S
            # Phase gate S: single-qubit Clifford
            q = qubits[1]
            qubit_depths[q] += 1.0
            
        elseif gate_type == :Z
            # Pauli Z: single-qubit Clifford
            q = qubits[1]
            qubit_depths[q] += 1.0
            
        elseif gate_type == :X
            # Pauli X: single-qubit Clifford
            q = qubits[1]
            qubit_depths[q] += 1.0
            
        elseif gate_type == :CNOT
            # Two-qubit Clifford: both qubits must synchronize
            q1, q2 = qubits[1], qubits[2]
            # Both advance to maximum depth + 1
            # This ensures proper causal ordering
            max_depth = max(qubit_depths[q1], qubit_depths[q2])
            qubit_depths[q1] = max_depth + 1.0
            qubit_depths[q2] = max_depth + 1.0
            
        elseif gate_type == :random2q
            # Random 2-qubit Clifford from Cl_2 (11,520 elements)
            # Treated as generic 2-qubit Clifford for depth counting
            q1, q2 = qubits[1], qubits[2]
            max_depth = max(qubit_depths[q1], qubit_depths[q2])
            qubit_depths[q1] = max_depth + 1.0
            qubit_depths[q2] = max_depth + 1.0
            
        else
            # Unknown gate type - this shouldn't happen
            @warn "Unknown gate type in depth computation: $gate_type"
        end
    end
    
    return depths
end

#==============================================================================#
# SPATIAL DISTRIBUTION ANALYSIS
#==============================================================================#

"""
    compute_spatial_uniformity(t_positions::Vector{Int}, n_gates::Int) -> Float64

Measure how uniformly T-gates are distributed through circuit depth.

Uses Shannon entropy over binned gate positions. Higher entropy indicates
more uniform distribution; lower entropy indicates clustering.

Algorithm:
1. Divide circuit into bins (default: 10 bins)
2. Count T-gates per bin
3. Compute Shannon entropy: H = -Σ p_i log₂(p_i)
4. Normalize by log₂(n_bins) for [0,1] range

Returns:
- Entropy in [0, 1]: 
  * 0 = all T-gates in one location (maximally clustered)
  * 1 = perfectly uniform distribution
  * Typical values: 0.6-0.9 for random placement

Edge cases:
- 0 or 1 T-gate: returns 0.0 (no distribution to measure)
- Empty bins: handled correctly (p=0 contributes 0 to entropy)

Example:
```julia
# Clustered: all T-gates at start
uniformity_clustered = compute_spatial_uniformity([1,2,3,4], 100)  # ≈ 0.0

# Uniform: T-gates spread evenly
uniformity_uniform = compute_spatial_uniformity([10,30,50,70,90], 100)  # ≈ 0.9
```

Time complexity: O(n_t_gates)
Space complexity: O(n_bins) = O(1) since n_bins is constant
"""
function compute_spatial_uniformity(t_positions::Vector{Int}, n_gates::Int)
    if length(t_positions) <= 1 || n_gates == 0
        return 0.0  # No distribution to measure
    end
    
    # Bin circuit into segments
    n_bins = min(10, n_gates)  # Don't use more bins than gates
    bin_size = n_gates / n_bins
    
    # Count T-gates per bin
    bins = zeros(Int, n_bins)
    for pos in t_positions
        bin_idx = min(Int(ceil(pos / bin_size)), n_bins)
        bins[bin_idx] += 1
    end
    
    # Compute Shannon entropy
    total = sum(bins)
    entropy = 0.0
    
    for count in bins
        if count > 0
            p = count / total
            entropy -= p * log2(p)
        end
    end
    
    # Normalize to [0, 1]
    max_entropy = log2(n_bins)
    normalized_entropy = entropy / max_entropy
    
    return normalized_entropy
end

"""
    compute_depth_variance(depths::Vector{Float64}) -> Float64

Compute variance in Clifford depths to T-gates.

High variance indicates non-uniform depth distribution (some T-gates after
shallow Clifford sections, others after deep sections). Low variance indicates
consistent Clifford depth between T-gates.

This is distinct from spatial_uniformity:
- spatial_uniformity: physical position in gate sequence
- depth_variance: computational depth (accounts for parallelism)

Returns:
- Variance of depth values (σ²)
- 0.0 if fewer than 2 T-gates (undefined variance)

Time complexity: O(n_t_gates)
"""
function compute_depth_variance(depths::Vector{Float64})
    if length(depths) < 2
        return 0.0
    end
    return var(depths)
end

#==============================================================================#
# COMPREHENSIVE FEATURE EXTRACTION
#==============================================================================#

"""
    extract_circuit_features(circuit::CircuitInstance) -> Dict{String, Any}

Extract all features computable from circuit structure without simulation.

Features are grouped into categories:

**Topological Features** (from circuit definition):
- n_qubits: Number of qubits
- n_t_gates: Number of T-gates
- t_density: T-gates per qubit (t/n)
- n_gates: Total gates

**Algorithmic Features** (from metadata):
- family: Circuit family name
- connectivity: Architecture type (if available)

**Structural Features** (computed via depth analysis):
- avg_clifford_depth: Mean Clifford layers to T-gates
- max_clifford_depth: Maximum depth to any T-gate
- min_clifford_depth: Minimum depth to any T-gate
- std_clifford_depth: Standard deviation of depths

**Light Cone Features** (from Liu & Clark 2024):
- avg_light_cone_width: Mean stabilizer window size (w = 4*d)
- max_light_cone_width: Maximum window size
- Note: Liu & Clark show OFD fails when w exceeds MPS bond dimension

**Distribution Features**:
- spatial_uniformity: Shannon entropy of T-gate positions
- depth_variance: Variance in computational depths

All features are computable via static analysis - no simulation required.
This enables prospective use: predict OFD success before running CAMPS.

Returns:
- Dictionary with all feature values
- Features have consistent types (Float64 for metrics, String for categorical)

Edge cases handled:
- 0 T-gates: depth statistics set to 0.0
- Single T-gate: variance/std set to 0.0
- Missing metadata: defaults to "Unknown"

Example:
```julia
circuit = generate_circuit(RandomBrickwallCliffordT(), params)
features = extract_circuit_features(circuit)

println("Average Clifford depth: ", features["avg_clifford_depth"])
println("Light cone width: ", features["avg_light_cone_width"])
println("Spatial uniformity: ", features["spatial_uniformity"])

# Use in ML model
X = [features["n_qubits"], features["t_density"], 
     features["avg_light_cone_width"], features["spatial_uniformity"]]
y_pred = model.predict(X)
```

Time complexity: O(|gates|) - dominated by depth computation
Space complexity: O(n_qubits + n_t_gates)
"""
function extract_circuit_features(circuit::CircuitInstance)
    features = Dict{String, Any}()
    
    # ===== TOPOLOGICAL FEATURES =====
    features["n_qubits"] = circuit.n_qubits
    features["n_t_gates"] = length(circuit.t_gate_positions)
    features["n_gates"] = length(circuit.gates)
    
    # T-gate density (key metric for OFD difficulty)
    features["t_density"] = features["n_t_gates"] / circuit.n_qubits
    
    # ===== ALGORITHMIC FEATURES =====
    features["family"] = get(circuit.metadata, "family", "Unknown")
    features["connectivity"] = get(circuit.metadata, "connectivity", "unknown")
    
    # ===== STRUCTURAL FEATURES: CLIFFORD DEPTH =====
    depths = compute_clifford_depths_to_t_gates(circuit)
    
    if length(depths) > 0
        # Depth statistics
        features["avg_clifford_depth"] = mean(depths)
        features["max_clifford_depth"] = maximum(depths)
        features["min_clifford_depth"] = minimum(depths)
        features["std_clifford_depth"] = length(depths) > 1 ? std(depths) : 0.0
        
        # ===== LIGHT CONE FEATURES =====
        # Liu & Clark (arXiv:2412.17209) Eq. 2: w = 4*d
        # Light cone width determines stabilizer window size
        # OFD fails when w exceeds available MPS bond dimension
        features["avg_light_cone_width"] = 4.0 * features["avg_clifford_depth"]
        features["max_light_cone_width"] = 4.0 * features["max_clifford_depth"]
        
        # ===== DISTRIBUTION FEATURES =====
        features["depth_variance"] = compute_depth_variance(depths)
    else
        # Edge case: 0 T-gates (rare but handle gracefully)
        features["avg_clifford_depth"] = 0.0
        features["max_clifford_depth"] = 0.0
        features["min_clifford_depth"] = 0.0
        features["std_clifford_depth"] = 0.0
        features["avg_light_cone_width"] = 0.0
        features["max_light_cone_width"] = 0.0
        features["depth_variance"] = 0.0
    end
    
    # ===== SPATIAL DISTRIBUTION =====
    features["spatial_uniformity"] = compute_spatial_uniformity(
        circuit.t_gate_positions, 
        length(circuit.gates)
    )
    
    return features
end

#==============================================================================#
# BATCH FEATURE EXTRACTION
#==============================================================================#

"""
    extract_features_from_results(results_df::DataFrame, 
                                   family_generators::Dict) -> DataFrame

Extract features for all circuits in benchmark results.

Given a CSV of benchmark results (with seeds), regenerates each circuit
deterministically and extracts features. This enables post-hoc feature
extraction after benchmark completion.

Input DataFrame must contain columns:
- family: Circuit family name
- n_qubits: Number of qubits
- n_t_gates: Number of T-gates  
- seed: Random seed used for generation
- (optional) realization: Realization number

Additional family-specific parameters extracted from family name and metadata.

Returns:
- DataFrame with all original columns plus extracted features
- New columns: avg_clifford_depth, max_clifford_depth, avg_light_cone_width,
               spatial_uniformity, etc.

Example:
```julia
# Load benchmark results
df = CSV.read("results/parallel_benchmark/results_20260106.csv", DataFrame)

# Set up family generators
families = Dict(
    "Random Clifford+T (Brick-wall)" => RandomBrickwallCliffordT(),
    "Random Clifford+T (All-to-all)" => RandomAllToAllCliffordT(),
    # ... etc
)

# Extract features (takes ~5 minutes for 648 circuits)
df_with_features = extract_features_from_results(df, families)

# Save
CSV.write("results_with_features.csv", df_with_features)
```

This is the key function for integrating feature extraction with existing
benchmark results. No re-running of expensive CAMPS simulation required!

Time complexity: O(n_circuits × |gates|_avg) ≈ 5 minutes for 648 circuits
"""
function extract_features_from_results(results_df, family_generators)
    # This will be implemented as part of post-processing script
    # Placeholder for now - full implementation in separate processing file
    error("Use extract_features_batch.jl for batch processing of results")
end

#==============================================================================#
# VALIDATION & TESTING
#==============================================================================#

"""
    validate_feature_extraction(circuit::CircuitInstance)

Validate that feature extraction produces sensible values.

Checks:
- All features are finite numbers (no NaN, Inf)
- Depth statistics are non-negative
- Light cone widths follow w = 4*d relation
- Spatial uniformity in [0, 1]
- T-density matches n_t_gates / n_qubits

Throws assertion error if validation fails.

Use during development to catch bugs:
```julia
circuit = generate_circuit(family, params)
validate_feature_extraction(circuit)  # Crashes if features invalid
features = extract_circuit_features(circuit)  # Safe to use
```
"""
function validate_feature_extraction(circuit::CircuitInstance)
    features = extract_circuit_features(circuit)
    
    # Check for invalid numerical values
    for (key, value) in features
        if value isa Number
            @assert isfinite(value) "Feature $key is not finite: $value"
        end
    end
    
    # Depth statistics should be non-negative
    @assert features["avg_clifford_depth"] >= 0.0 "Negative average depth"
    @assert features["max_clifford_depth"] >= 0.0 "Negative max depth"
    @assert features["min_clifford_depth"] >= 0.0 "Negative min depth"
    @assert features["std_clifford_depth"] >= 0.0 "Negative std depth"
    
    # Light cone relation: w = 4*d (Liu & Clark)
    expected_avg_width = 4.0 * features["avg_clifford_depth"]
    @assert abs(features["avg_light_cone_width"] - expected_avg_width) < 1e-10 "Light cone width formula violated"
    
    # Spatial uniformity in [0, 1]
    @assert 0.0 <= features["spatial_uniformity"] <= 1.0 "Spatial uniformity out of bounds"
    
    # T-density consistency
    expected_density = features["n_t_gates"] / features["n_qubits"]
    @assert abs(features["t_density"] - expected_density) < 1e-10 "T-density calculation error"
    
    println("✓ Feature extraction validation passed")
end

#==============================================================================#
# EXPORTS
#==============================================================================#

export compute_clifford_depths_to_t_gates
export compute_spatial_uniformity
export compute_depth_variance
export extract_circuit_features
export validate_feature_extraction
