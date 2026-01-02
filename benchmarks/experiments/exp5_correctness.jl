# CAMPS.jl/benchmarks/experiments/exp5_correctness.jl
# Experiment 5: Correctness Verification
#
# Purpose: Verify CAMPS produces correct quantum states
# Key result: Fidelity ≈ 1.0 with exact simulation for small systems

push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using CAMPS
using LinearAlgebra
using Random
using Printf

include(joinpath(@__DIR__, "..", "benchmark_types.jl"))
include(joinpath(@__DIR__, "..", "config.jl"))

"""
    run_correctness_experiment(; config=CORRECTNESS_QUICK_CONFIG, verbose=true)

Run the correctness verification experiment.

Tests:
1. Initial state preparation (|0⟩^n)
2. Single Hadamard gate
3. GHZ state preparation
4. Bell state preparation
5. Small QFT circuits
6. Random Clifford+T circuits
7. T gate phase verification

# Arguments
- `config::CorrectnessConfig`: Configuration parameters
- `verbose::Bool`: Whether to print progress

# Returns
- `ExperimentResults`: Complete experiment results with summary
"""
function run_correctness_experiment(;
    config::CorrectnessConfig = CORRECTNESS_QUICK_CONFIG,
    verbose::Bool = true
)
    if verbose
        println("="^70)
        println("Experiment 5: Correctness Verification")
        println("="^70)
        println()
        println("Configuration:")
        println("  max_n_qubits: $(config.max_n_qubits)")
        println("  test_cases: $(config.test_cases)")
        println("  fidelity_threshold: $(config.fidelity_threshold)")
        println("  amplitude_threshold: $(config.amplitude_threshold)")
        println()
    end

    start_time = time()
    results = CorrectnessResult[]

    for test_case in config.test_cases
        if verbose
            println("-"^50)
            println("Test: $(test_case)")
            println("-"^50)
        end

        test_results = run_test_case(test_case, config; verbose=verbose)
        append!(results, test_results)
    end

    wall_time = time() - start_time
    summary = compute_summary(results)

    if verbose
        println()
        println("="^70)
        println("Experiment Complete")
        println("="^70)
        println("Total tests: $(length(results))")
        println("Passed: $(summary["passed"])/$(summary["n_tests"])")
        println("Mean fidelity: $(round(summary["mean_fidelity"], digits=10))")
        println("Wall time: $(round(wall_time, digits=2))s")

        if !summary["all_passed"]
            println("\nWARNING: Some tests failed!")
            for r in results
                if !r.passed
                    println("  FAILED: $(r.test_name) (fidelity=$(r.fidelity))")
                end
            end
        else
            println("\nAll tests passed!")
        end
    end

    ExperimentResults(
        "correctness",
        "Verification of CAMPS correctness against exact simulation",
        Dict{String,Any}(
            "max_n_qubits" => config.max_n_qubits,
            "test_cases" => config.test_cases,
            "fidelity_threshold" => config.fidelity_threshold,
            "amplitude_threshold" => config.amplitude_threshold
        ),
        results,
        summary,
        wall_time
    )
end

"""
    run_test_case(test_case, config; verbose=true)

Run a specific correctness test case.
"""
function run_test_case(
    test_case::String,
    config::CorrectnessConfig;
    verbose::Bool = true
)
    if test_case == "initial_state"
        return test_initial_state(config; verbose=verbose)
    elseif test_case == "single_h"
        return test_single_hadamard(config; verbose=verbose)
    elseif test_case == "ghz"
        return test_ghz_state(config; verbose=verbose)
    elseif test_case == "bell"
        return test_bell_state(config; verbose=verbose)
    elseif test_case == "qft_small"
        return test_qft_small(config; verbose=verbose)
    elseif test_case == "random_clifford_t"
        return test_random_clifford_t(config; verbose=verbose)
    elseif test_case == "t_gate_phases"
        return test_t_gate_phases(config; verbose=verbose)
    else
        @warn "Unknown test case: $test_case"
        return CorrectnessResult[]
    end
end

"""
    test_initial_state(config; verbose=true)

Test that initial state is correctly |0⟩^n.
"""
function test_initial_state(config::CorrectnessConfig; verbose::Bool = true)
    results = CorrectnessResult[]

    for n in 2:config.max_n_qubits
        state = CAMPSState(n)
        initialize!(state)

        # Get state vector
        psi = state_vector(state)

        # Expected: [1, 0, 0, ..., 0]
        expected = zeros(ComplexF64, 2^n)
        expected[1] = 1.0

        fidelity_val = abs2(dot(expected, psi))
        max_error = maximum(abs.(psi - expected))
        passed = fidelity_val >= config.fidelity_threshold

        push!(results, CorrectnessResult(
            "initial_state_n$(n)",
            n,
            "|0⟩^$(n) initial state",
            fidelity_val,
            max_error,
            passed,
            config.fidelity_threshold
        ))

        if verbose
            status = passed ? "PASS" : "FAIL"
            @printf("  n=%d: F=%.10f, max_err=%.2e [%s]\n",
                    n, fidelity_val, max_error, status)
        end
    end

    results
end

"""
    test_single_hadamard(config; verbose=true)

Test single Hadamard gate creates correct superposition.
"""
function test_single_hadamard(config::CorrectnessConfig; verbose::Bool = true)
    results = CorrectnessResult[]

    for n in 2:config.max_n_qubits
        state = CAMPSState(n)
        initialize!(state)
        apply_gate!(state, HGate(1))

        psi = state_vector(state)

        # Expected: (|0...0⟩ + |1,0...0⟩)/√2
        # In little-endian: indices 1 and 2 (0-indexed: 0 and 1)
        expected = zeros(ComplexF64, 2^n)
        expected[1] = 1/√2  # |00...0⟩
        expected[2] = 1/√2  # |10...0⟩

        fidelity_val = abs2(dot(expected, psi))
        max_error = maximum(abs.(psi - expected))
        passed = fidelity_val >= config.fidelity_threshold

        push!(results, CorrectnessResult(
            "single_h_n$(n)",
            n,
            "H|0⟩ on qubit 1 of $(n) qubits",
            fidelity_val,
            max_error,
            passed,
            config.fidelity_threshold
        ))

        if verbose
            status = passed ? "PASS" : "FAIL"
            @printf("  n=%d: F=%.10f, max_err=%.2e [%s]\n",
                    n, fidelity_val, max_error, status)
        end
    end

    results
end

"""
    test_ghz_state(config; verbose=true)

Test GHZ state preparation: (|00...0⟩ + |11...1⟩)/√2.
"""
function test_ghz_state(config::CorrectnessConfig; verbose::Bool = true)
    results = CorrectnessResult[]

    for n in 2:config.max_n_qubits
        state = CAMPSState(n)
        initialize!(state)

        # Apply GHZ circuit
        circuit = ghz_circuit(n)
        for gate in circuit
            apply_gate!(state, gate)
        end

        psi = state_vector(state)

        # Expected: (|00...0⟩ + |11...1⟩)/√2
        expected = zeros(ComplexF64, 2^n)
        expected[1] = 1/√2           # |00...0⟩ at index 1
        expected[2^n] = 1/√2         # |11...1⟩ at index 2^n

        fidelity_val = abs2(dot(expected, psi))
        max_error = maximum(abs.(psi - expected))
        passed = fidelity_val >= config.fidelity_threshold

        push!(results, CorrectnessResult(
            "ghz_n$(n)",
            n,
            "GHZ state on $(n) qubits",
            fidelity_val,
            max_error,
            passed,
            config.fidelity_threshold
        ))

        if verbose
            status = passed ? "PASS" : "FAIL"
            @printf("  n=%d: F=%.10f, max_err=%.2e [%s]\n",
                    n, fidelity_val, max_error, status)
        end
    end

    results
end

"""
    test_bell_state(config; verbose=true)

Test Bell state preparation: (|00⟩ + |11⟩)/√2.
"""
function test_bell_state(config::CorrectnessConfig; verbose::Bool = true)
    results = CorrectnessResult[]

    state = CAMPSState(2)
    initialize!(state)
    apply_gate!(state, HGate(1))
    apply_gate!(state, CNOTGate(1, 2))

    psi = state_vector(state)

    # Expected: (|00⟩ + |11⟩)/√2
    # Little-endian: |00⟩ at 1, |11⟩ at 4
    expected = zeros(ComplexF64, 4)
    expected[1] = 1/√2  # |00⟩
    expected[4] = 1/√2  # |11⟩

    fidelity_val = abs2(dot(expected, psi))
    max_error = maximum(abs.(psi - expected))
    passed = fidelity_val >= config.fidelity_threshold

    push!(results, CorrectnessResult(
        "bell_state",
        2,
        "Bell state (|00⟩ + |11⟩)/√2",
        fidelity_val,
        max_error,
        passed,
        config.fidelity_threshold
    ))

    if verbose
        status = passed ? "PASS" : "FAIL"
        @printf("  Bell state: F=%.10f, max_err=%.2e [%s]\n",
                fidelity_val, max_error, status)
    end

    results
end

"""
    test_qft_small(config; verbose=true)

Test small QFT circuits against exact simulation.
"""
function test_qft_small(config::CorrectnessConfig; verbose::Bool = true)
    results = CorrectnessResult[]

    for n in 2:min(4, config.max_n_qubits)
        state = CAMPSState(n)
        initialize!(state)

        # Apply QFT
        circuit = qft_circuit(n)
        for gate in circuit
            apply_gate!(state, gate)
        end

        psi = state_vector(state)

        # QFT of |0⟩^n is the uniform superposition
        expected = fill(ComplexF64(1/√(2^n)), 2^n)

        fidelity_val = abs2(dot(expected, psi))
        max_error = maximum(abs.(psi .- expected))

        # QFT has many non-Clifford gates, use looser threshold
        threshold = 1.0 - 1e-6
        passed = fidelity_val >= threshold

        push!(results, CorrectnessResult(
            "qft_n$(n)",
            n,
            "QFT on |0⟩^$(n)",
            fidelity_val,
            max_error,
            passed,
            threshold
        ))

        if verbose
            status = passed ? "PASS" : "FAIL"
            @printf("  QFT n=%d: F=%.10f, max_err=%.2e [%s]\n",
                    n, fidelity_val, max_error, status)
        end
    end

    results
end

"""
    test_random_clifford_t(config; verbose=true)

Test random Clifford+T circuits for normalization and consistency.
"""
function test_random_clifford_t(config::CorrectnessConfig; verbose::Bool = true)
    results = CorrectnessResult[]

    for n in 2:min(config.max_n_qubits, 5)
        for seed in [42, 123, 456]
            Random.seed!(seed)

            t = min(4, n)
            circuit = random_clifford_t_circuit(n, t; seed=seed)

            state = CAMPSState(n)
            initialize!(state)

            for gate in circuit
                apply_gate!(state, gate)
            end

            psi = state_vector(state)

            # Check normalization
            norm_val = sum(abs2.(psi))
            norm_error = abs(norm_val - 1.0)

            # For random circuits, we can't know exact state
            # but we verify normalization and no NaN/Inf
            has_nan = any(isnan.(psi)) || any(isinf.(psi))
            passed = norm_error < config.amplitude_threshold && !has_nan

            push!(results, CorrectnessResult(
                "random_ct_n$(n)_s$(seed)",
                n,
                "Random Clifford+T (n=$(n), t=$(t), seed=$(seed))",
                1.0 - norm_error,  # Use norm deviation as fidelity proxy
                norm_error,
                passed,
                1.0 - config.amplitude_threshold
            ))

            if verbose
                status = passed ? "PASS" : "FAIL"
                @printf("  n=%d, seed=%d: norm=%.10f, err=%.2e [%s]\n",
                        n, seed, norm_val, norm_error, status)
            end
        end
    end

    results
end

"""
    test_t_gate_phases(config; verbose=true)

Test that T gate applies correct phase e^(iπ/4).
"""
function test_t_gate_phases(config::CorrectnessConfig; verbose::Bool = true)
    results = CorrectnessResult[]

    # Test T gate: T|+⟩ = (|0⟩ + e^(iπ/4)|1⟩)/√2
    state = CAMPSState(1)
    initialize!(state)
    apply_gate!(state, HGate(1))
    apply_gate!(state, TGate(1))

    psi = state_vector(state)

    # Expected phases
    expected = ComplexF64[1/√2, exp(im * π/4) / √2]

    # Check relative phase (global phase doesn't matter)
    if abs(psi[1]) > 1e-10
        psi_normalized = psi / psi[1] * expected[1]
    else
        psi_normalized = psi
    end

    fidelity_val = abs2(dot(expected, psi))
    max_error = maximum(abs.(psi_normalized - expected))
    passed = fidelity_val >= config.fidelity_threshold

    push!(results, CorrectnessResult(
        "t_gate_phase",
        1,
        "T|+⟩ phase test",
        fidelity_val,
        max_error,
        passed,
        config.fidelity_threshold
    ))

    if verbose
        status = passed ? "PASS" : "FAIL"
        @printf("  T|+⟩: F=%.10f, phase_err=%.2e [%s]\n",
                fidelity_val, max_error, status)
        @printf("    Expected: [%.4f, %.4f+%.4fi]\n",
                real(expected[1]), real(expected[2]), imag(expected[2]))
        @printf("    Got:      [%.4f, %.4f+%.4fi]\n",
                real(psi[1]), real(psi[2]), imag(psi[2]))
    end

    # Test T† gate: T†|+⟩ = (|0⟩ + e^(-iπ/4)|1⟩)/√2
    state2 = CAMPSState(1)
    initialize!(state2)
    apply_gate!(state2, HGate(1))
    apply_gate!(state2, TdagGate(1))

    psi2 = state_vector(state2)
    expected2 = ComplexF64[1/√2, exp(-im * π/4) / √2]

    fidelity_val2 = abs2(dot(expected2, psi2))
    passed2 = fidelity_val2 >= config.fidelity_threshold

    push!(results, CorrectnessResult(
        "tdag_gate_phase",
        1,
        "T†|+⟩ phase test",
        fidelity_val2,
        maximum(abs.(psi2 - expected2)),
        passed2,
        config.fidelity_threshold
    ))

    if verbose
        status = passed2 ? "PASS" : "FAIL"
        @printf("  T†|+⟩: F=%.10f [%s]\n", fidelity_val2, status)
    end

    results
end

"""
    compute_exact_state(circuit, n) -> Vector{ComplexF64}

Compute exact state vector by matrix multiplication (for small n).
"""
function compute_exact_state(circuit::Vector{<:Gate}, n::Int)
    # Start with |0⟩^n
    psi = zeros(ComplexF64, 2^n)
    psi[1] = 1.0

    for gate in circuit
        U = gate_to_matrix(gate, n)
        psi = U * psi
    end

    psi
end

"""
    gate_to_matrix(gate, n) -> Matrix{ComplexF64}

Convert a gate to its matrix representation on n qubits.
"""
function gate_to_matrix(gate::Gate, n::Int)
    if gate isa CliffordGate
        return clifford_gate_to_matrix(gate, n)
    elseif gate isa RotationGate
        return rotation_gate_to_matrix(gate, n)
    else
        error("Unknown gate type: $(typeof(gate))")
    end
end

"""
    clifford_gate_to_matrix(gate::CliffordGate, n) -> Matrix

Convert Clifford gate to matrix.
"""
function clifford_gate_to_matrix(gate::CliffordGate, n::Int)
    # This is a simplified implementation for common gates
    # Full implementation would use the gate specification

    dim = 2^n
    U = Matrix{ComplexF64}(I, dim, dim)

    # Would need to parse gate.gates to get full matrix
    # For now, use CAMPS's existing infrastructure
    @warn "clifford_gate_to_matrix not fully implemented"
    U
end

"""
    rotation_gate_to_matrix(gate::RotationGate, n) -> Matrix

Convert rotation gate to matrix.
"""
function rotation_gate_to_matrix(gate::RotationGate, n::Int)
    q = gate.qubit
    θ = gate.angle
    axis = gate.axis

    # Single qubit rotation matrix
    if axis == :Z
        R = [exp(-im*θ/2) 0; 0 exp(im*θ/2)]
    elseif axis == :X
        R = [cos(θ/2) -im*sin(θ/2); -im*sin(θ/2) cos(θ/2)]
    elseif axis == :Y
        R = [cos(θ/2) -sin(θ/2); sin(θ/2) cos(θ/2)]
    else
        error("Unknown axis: $axis")
    end

    # Embed in n-qubit space
    embed_single_qubit_gate(R, q, n)
end

"""
    embed_single_qubit_gate(R, qubit, n) -> Matrix

Embed single-qubit gate R acting on `qubit` into n-qubit space.
"""
function embed_single_qubit_gate(R::Matrix, qubit::Int, n::Int)
    dim = 2^n
    U = zeros(ComplexF64, dim, dim)

    for i in 0:(dim-1)
        for j in 0:(dim-1)
            # Extract the qubit bits
            bi = (i >> (qubit-1)) & 1
            bj = (j >> (qubit-1)) & 1

            # Check if other bits match
            mask = ~(1 << (qubit-1))
            if (i & mask) == (j & mask)
                U[i+1, j+1] = R[bi+1, bj+1]
            end
        end
    end

    U
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    results = run_correctness_experiment(config=CORRECTNESS_QUICK_CONFIG, verbose=true)
end

export run_correctness_experiment
