# CAMPS.jl/benchmarks/test_feature_extraction.jl
#
# COMPREHENSIVE TEST SUITE FOR FEATURE EXTRACTION
#
# Validates that feature extraction:
# 1. Works on all 9 circuit families
# 2. Produces sensible values
# 3. Handles edge cases correctly
# 4. Is deterministic (same circuit → same features)
# 5. Satisfies mathematical relations (w = 4*d)

using Test
using Statistics
using Random

include("circuit_families.jl")
include("circuit_feature_extraction.jl")

println("="^70)
println("FEATURE EXTRACTION TEST SUITE")
println("="^70)
println()

#==============================================================================#
# TEST 1: DEPTH COMPUTATION ON SIMPLE CIRCUITS
#==============================================================================#

@testset "Clifford Depth Computation - Simple Cases" begin
    println("Test 1: Depth Computation on Simple Circuits")
    println("-"^70)
    
    # Case 1: Single T-gate at start (depth = 0)
    gates_1 = [
        (:T, [1])
    ]
    circuit_1 = CircuitInstance(2, gates_1, [1], Dict("family" => "Test"))
    depths_1 = compute_clifford_depths_to_t_gates(circuit_1)
    
    @test length(depths_1) == 1
    @test depths_1[1] == 0.0  # No Clifford gates before T
    println("  ✓ Case 1: T-gate at start has depth 0")
    
    # Case 2: H then T (depth = 1)
    gates_2 = [
        (:H, [1]),
        (:T, [1])
    ]
    circuit_2 = CircuitInstance(2, gates_2, [2], Dict("family" => "Test"))
    depths_2 = compute_clifford_depths_to_t_gates(circuit_2)
    
    @test length(depths_2) == 1
    @test depths_2[1] == 1.0  # One H before T
    println("  ✓ Case 2: H → T has depth 1")
    
    # Case 3: Multiple Cliffords on same qubit
    gates_3 = [
        (:H, [1]),
        (:S, [1]),
        (:Z, [1]),
        (:T, [1])
    ]
    circuit_3 = CircuitInstance(2, gates_3, [4], Dict("family" => "Test"))
    depths_3 = compute_clifford_depths_to_t_gates(circuit_3)
    
    @test length(depths_3) == 1
    @test depths_3[1] == 3.0  # H + S + Z
    println("  ✓ Case 3: H → S → Z → T has depth 3")
    
    # Case 4: CNOT increases depth on both qubits
    gates_4 = [
        (:H, [1]),
        (:H, [2]),
        (:CNOT, [1, 2]),
        (:T, [1]),
        (:T, [2])
    ]
    circuit_4 = CircuitInstance(2, gates_4, [4, 5], Dict("family" => "Test"))
    depths_4 = compute_clifford_depths_to_t_gates(circuit_4)
    
    @test length(depths_4) == 2
    @test depths_4[1] == 2.0  # H + CNOT on qubit 1
    @test depths_4[2] == 2.0  # H + CNOT on qubit 2
    println("  ✓ Case 4: CNOT synchronizes depths correctly")
    
    # Case 5: Two T-gates with Cliffords between
    gates_5 = [
        (:H, [1]),
        (:T, [1]),
        (:H, [1]),
        (:H, [1]),
        (:T, [1])
    ]
    circuit_5 = CircuitInstance(2, gates_5, [2, 5], Dict("family" => "Test"))
    depths_5 = compute_clifford_depths_to_t_gates(circuit_5)
    
    @test length(depths_5) == 2
    @test depths_5[1] == 1.0  # First T after 1 H
    @test depths_5[2] == 3.0  # Second T after 1 H + (T) + 2 H = depth 3
    println("  ✓ Case 5: Multiple T-gates track correctly")
    
    println()
end

#==============================================================================#
# TEST 2: DEPTH COMPUTATION ON ALL 9 FAMILIES
#==============================================================================#

@testset "Depth Computation - All Families" begin
    println("Test 2: Depth Computation on All 9 Families")
    println("-"^70)
    
    families = get_rigorous_circuit_families()
    
    for family in families
        family_name = get_name(family)
        
        # Generate test circuit
        params = Dict{Symbol, Any}(
            :n_qubits => 8,
            :n_t_gates => 4,
            :seed => 12345
        )
        
        # Add family-specific parameters
        if family isa RandomBrickwallCliffordT
            params[:clifford_depth] = 2
        elseif family isa RandomAllToAllCliffordT
            params[:clifford_layers] = 16
        elseif family isa DeutschJozsaCircuit
            params[:function_type] = :balanced
        elseif family isa GraphStateCircuit
            params[:edge_probability] = 0.3
        end
        
        circuit = generate_circuit(family, params)
        depths = compute_clifford_depths_to_t_gates(circuit)
        
        # Validate
        @test length(depths) == params[:n_t_gates]
        @test all(d >= 0.0 for d in depths)
        @test all(isfinite(d) for d in depths)
        
        println("  ✓ $(rpad(family_name, 40)): $(length(depths)) depths, avg=$(round(mean(depths), digits=2))")
    end
    
    println()
end

#==============================================================================#
# TEST 3: SPATIAL UNIFORMITY
#==============================================================================#

@testset "Spatial Uniformity Computation" begin
    println("Test 3: Spatial Uniformity Computation")
    println("-"^70)
    
    # Case 1: Perfectly clustered (all T-gates at start)
    uniformity_clustered = compute_spatial_uniformity([1, 2, 3, 4], 100)
    @test 0.0 <= uniformity_clustered <= 0.3  # Should be low
    println("  ✓ Clustered T-gates: uniformity = $(round(uniformity_clustered, digits=3))")
    
    # Case 2: Well-distributed (evenly spaced)
    # With 5 T-gates in 10 bins, max achievable entropy is ~0.7
    # (can't fill all bins with equal counts)
    uniformity_uniform = compute_spatial_uniformity([10, 30, 50, 70, 90], 100)
    @test 0.65 <= uniformity_uniform <= 1.0  # Should be high
    println("  ✓ Uniform T-gates: uniformity = $(round(uniformity_uniform, digits=3))")
    
    # Case 2b: Better uniformity with more T-gates (can achieve higher values)
    # 10 T-gates spread evenly → each bin gets 1 T-gate → maximum entropy
    uniformity_perfect = compute_spatial_uniformity([5, 15, 25, 35, 45, 55, 65, 75, 85, 95], 100)
    @test 0.95 <= uniformity_perfect <= 1.0  # Nearly perfect
    println("  ✓ Perfect distribution: uniformity = $(round(uniformity_perfect, digits=3))")
    
    # Case 3: Edge case - single T-gate
    uniformity_single = compute_spatial_uniformity([50], 100)
    @test uniformity_single == 0.0  # No distribution to measure
    println("  ✓ Single T-gate: uniformity = $(round(uniformity_single, digits=3))")
    
    # Case 4: Edge case - no T-gates
    uniformity_none = compute_spatial_uniformity(Int[], 100)
    @test uniformity_none == 0.0
    println("  ✓ No T-gates: uniformity = $(round(uniformity_none, digits=3))")
    
    println()
end

#==============================================================================#
# TEST 4: COMPLETE FEATURE EXTRACTION
#==============================================================================#

@testset "Complete Feature Extraction" begin
    println("Test 4: Complete Feature Extraction on All Families")
    println("-"^70)
    
    families = get_rigorous_circuit_families()
    
    for family in families
        family_name = get_name(family)
        
        params = Dict{Symbol, Any}(
            :n_qubits => 12,
            :n_t_gates => 8,
            :seed => 42
        )
        
        if family isa RandomBrickwallCliffordT
            params[:clifford_depth] = 2
        elseif family isa RandomAllToAllCliffordT
            params[:clifford_layers] = 24
        elseif family isa DeutschJozsaCircuit
            params[:function_type] = :balanced
        elseif family isa GraphStateCircuit
            params[:edge_probability] = 0.3
        end
        
        circuit = generate_circuit(family, params)
        features = extract_circuit_features(circuit)
        
        # Validate all features exist
        required_features = [
            "n_qubits", "n_t_gates", "t_density",
            "avg_clifford_depth", "max_clifford_depth",
            "avg_light_cone_width", "max_light_cone_width",
            "spatial_uniformity", "family"
        ]
        
        for feat in required_features
            @test haskey(features, feat)
        end
        
        # Validate feature values
        @test features["n_qubits"] == 12
        @test features["n_t_gates"] == 8
        @test abs(features["t_density"] - 8/12) < 1e-10
        
        # Light cone relation: w = 4*d
        @test abs(features["avg_light_cone_width"] - 4 * features["avg_clifford_depth"]) < 1e-10
        @test abs(features["max_light_cone_width"] - 4 * features["max_clifford_depth"]) < 1e-10
        
        # Spatial uniformity in [0, 1]
        @test 0.0 <= features["spatial_uniformity"] <= 1.0
        
        # All numerical features are finite
        for (key, value) in features
            if value isa Number
                if !isfinite(value)
                    error("Feature $key is not finite for $family_name: $value")
                end
                @test isfinite(value)
            end
        end
        
        println("  ✓ $(rpad(family_name, 40)): avg_depth=$(round(features["avg_clifford_depth"], digits=2)), width=$(round(features["avg_light_cone_width"], digits=2))")
    end
    
    println()
end

#==============================================================================#
# TEST 5: DETERMINISM
#==============================================================================#

@testset "Deterministic Feature Extraction" begin
    println("Test 5: Deterministic Feature Extraction")
    println("-"^70)
    
    # Same parameters → same features
    params = Dict{Symbol, Any}(
        :n_qubits => 8,
        :n_t_gates => 4,
        :seed => 99999,
        :clifford_depth => 2
    )
    
    family = RandomBrickwallCliffordT()
    
    # Generate same circuit twice
    circuit1 = generate_circuit(family, params)
    circuit2 = generate_circuit(family, params)
    
    # Extract features twice
    features1 = extract_circuit_features(circuit1)
    features2 = extract_circuit_features(circuit2)
    
    # All numerical features should match exactly
    for key in ["avg_clifford_depth", "max_clifford_depth", "avg_light_cone_width", 
                "spatial_uniformity", "t_density"]
        diff = abs(features1[key] - features2[key])
        if diff >= 1e-10
            error("Feature $key not deterministic: diff = $diff")
        end
        @test diff < 1e-10
    end
    
    println("  ✓ Same seed → identical features (deterministic ✓)")
    println()
end

#==============================================================================#
# TEST 6: MATHEMATICAL RELATIONS
#==============================================================================#

@testset "Mathematical Relations" begin
    println("Test 6: Mathematical Relations")
    println("-"^70)
    
    families = get_rigorous_circuit_families()
    
    for family in families[1:3]  # Test on subset
        params = Dict{Symbol, Any}(
            :n_qubits => 8,
            :n_t_gates => 6,
            :seed => 777
        )
        
        # Add family-specific parameters
        if family isa RandomBrickwallCliffordT
            params[:clifford_depth] = 3
        elseif family isa RandomAllToAllCliffordT
            params[:clifford_layers] = 16
        elseif family isa DeutschJozsaCircuit
            params[:function_type] = :balanced
        elseif family isa GraphStateCircuit
            params[:edge_probability] = 0.3
        end
        
        circuit = generate_circuit(family, params)
        features = extract_circuit_features(circuit)
        
        # Test 1: Light cone width = 4 × depth (Liu & Clark)
        @test abs(features["avg_light_cone_width"] - 4 * features["avg_clifford_depth"]) < 1e-10
        @test abs(features["max_light_cone_width"] - 4 * features["max_clifford_depth"]) < 1e-10
        
        # Test 2: T-density = n_t_gates / n_qubits
        expected_density = features["n_t_gates"] / features["n_qubits"]
        @test abs(features["t_density"] - expected_density) < 1e-10
        
        # Test 3: Max depth >= Avg depth >= Min depth
        @test features["max_clifford_depth"] >= features["avg_clifford_depth"]
        @test features["avg_clifford_depth"] >= features["min_clifford_depth"]
    end
    
    println("  ✓ All mathematical relations satisfied")
    println()
end

#==============================================================================#
# TEST 7: EDGE CASES
#==============================================================================#

@testset "Edge Cases" begin
    println("Test 7: Edge Cases")
    println("-"^70)
    
    # Case 1: Circuit with 0 T-gates (rare but should handle)
    gates_empty = [
        (:H, [1]),
        (:CNOT, [1, 2]),
        (:H, [2])
    ]
    circuit_empty = CircuitInstance(2, gates_empty, Int[], Dict("family" => "Test"))
    features_empty = extract_circuit_features(circuit_empty)
    
    @test features_empty["n_t_gates"] == 0
    @test features_empty["avg_clifford_depth"] == 0.0
    @test features_empty["spatial_uniformity"] == 0.0
    println("  ✓ Circuit with 0 T-gates handled correctly")
    
    # Case 2: Circuit with 1 T-gate
    gates_one = [
        (:H, [1]),
        (:T, [1])
    ]
    circuit_one = CircuitInstance(2, gates_one, [2], Dict("family" => "Test"))
    features_one = extract_circuit_features(circuit_one)
    
    @test features_one["n_t_gates"] == 1
    @test features_one["std_clifford_depth"] == 0.0  # Undefined for n=1
    @test features_one["depth_variance"] == 0.0
    println("  ✓ Circuit with 1 T-gate handled correctly")
    
    # Case 3: Very deep circuit
    gates_deep = [(:H, [1]) for _ in 1:100]
    push!(gates_deep, (:T, [1]))
    circuit_deep = CircuitInstance(2, gates_deep, [101], Dict("family" => "Test"))
    features_deep = extract_circuit_features(circuit_deep)
    
    @test features_deep["avg_clifford_depth"] == 100.0
    @test features_deep["avg_light_cone_width"] == 400.0
    println("  ✓ Very deep circuit (depth=100) handled correctly")
    
    println()
end

#==============================================================================#
# TEST 8: REALISTIC CIRCUITS
#==============================================================================#

@testset "Realistic Circuit Analysis" begin
    println("Test 8: Realistic Circuit Analysis")
    println("-"^70)
    
    # Brick-wall vs All-to-all comparison
    params_brick = Dict{Symbol, Any}(
        :n_qubits => 16,
        :n_t_gates => 12,
        :seed => 42,
        :clifford_depth => 3
    )
    
    params_all = Dict{Symbol, Any}(
        :n_qubits => 16,
        :n_t_gates => 12,
        :seed => 42,
        :clifford_layers => 32
    )
    
    circuit_brick = generate_circuit(RandomBrickwallCliffordT(), params_brick)
    circuit_all = generate_circuit(RandomAllToAllCliffordT(), params_all)
    
    features_brick = extract_circuit_features(circuit_brick)
    features_all = extract_circuit_features(circuit_all)
    
    println("  Brick-wall Architecture:")
    println("    Avg Clifford depth:  $(round(features_brick["avg_clifford_depth"], digits=2))")
    println("    Light cone width:    $(round(features_brick["avg_light_cone_width"], digits=2))")
    println("    Spatial uniformity:  $(round(features_brick["spatial_uniformity"], digits=3))")
    
    println("  All-to-all Architecture:")
    println("    Avg Clifford depth:  $(round(features_all["avg_clifford_depth"], digits=2))")
    println("    Light cone width:    $(round(features_all["avg_light_cone_width"], digits=2))")
    println("    Spatial uniformity:  $(round(features_all["spatial_uniformity"], digits=3))")
    
    # Both should have valid features
    @test features_brick["avg_clifford_depth"] > 0
    @test features_all["avg_clifford_depth"] > 0
    
    println()
end

#==============================================================================#
# SUMMARY
#==============================================================================#

println("="^70)
println("ALL TESTS PASSED ✓")
println("="^70)
println()
println("Feature extraction is:")
println("  ✓ Mathematically correct (w = 4d relation holds)")
println("  ✓ Works on all 9 families")
println("  ✓ Handles edge cases (0 T-gates, 1 T-gate, deep circuits)")
println("  ✓ Deterministic (same seed → same features)")
println("  ✓ Produces sensible values (all finite, proper ranges)")
println()
println("Ready for production use!")
println()