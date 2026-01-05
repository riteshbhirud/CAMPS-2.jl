# CAMPS.jl/src/CAMPS.jl
# Main module for Clifford-Augmented Matrix Product States
#
# CAMPS represents quantum states as |ψ⟩ = C · |mps⟩ where:
# - C is a Clifford operator (tracked symbolically via QuantumClifford.jl)
# - |mps⟩ is a Matrix Product State (stored via ITensorMPS.jl)
#
# This representation enables efficient simulation of Clifford+T circuits
# by keeping Clifford gates "free" and only modifying the MPS for T-gates.

module CAMPS

#==============================================================================#
# EXTERNAL DEPENDENCIES
#==============================================================================#

# QuantumClifford.jl - Clifford algebra operations
using QuantumClifford
using QuantumClifford: PauliOperator, Stabilizer, Destabilizer, MixedDestabilizer, CliffordOperator
using QuantumClifford: single_x, single_y, single_z
using QuantumClifford: nqubits, xbit, zbit
using QuantumClifford: apply!, inv
using QuantumClifford: sHadamard, sPhase, sInvPhase, sX, sY, sZ
using QuantumClifford: sCNOT, sCPHASE, sSWAP
using QuantumClifford: enumerate_cliffords, random_clifford
using QuantumClifford: gf2_gausselim!

# ITensors and ITensorMPS - tensor network operations
using ITensors
using ITensors: ITensor, Index, dag, array, prime, noprime
using ITensors: commonind, uniqueinds, hasind, inds, dim
using ITensors: svd

using ITensorMPS
using ITensorMPS: MPS, siteinds, linkind, maxlinkdim
using ITensorMPS: orthogonalize, truncate!
using ITensorMPS: inner, sample, add

# Standard library
using LinearAlgebra
using Random

#==============================================================================#
# INCLUDE SOURCE FILES - ORDER MATTERS
#==============================================================================#

# Phase 1: Foundation (types and utilities)
include("types.jl")
include("utils.jl")

# Phase 2: Library interfaces (depend on types.jl)
include("clifford_interface.jl")
include("mps_interface.jl")
include("gf2.jl")

# Phase 3: Core algorithms
include("two_qubit_cliffords.jl")
include("ofd.jl")
include("obd.jl")

# Phase 4: Simulation interface
include("simulation.jl")

#==============================================================================#
# STATE INITIALIZATION
# Connects types.jl placeholders with actual QuantumClifford/ITensorMPS objects
#==============================================================================#

"""
    initialize!(state::CAMPSState) -> CAMPSState

Initialize the Clifford and MPS components of a CAMPSState.

This replaces the placeholder values with actual QuantumClifford.Destabilizer
and ITensorMPS.MPS objects.

# Arguments
- `state::CAMPSState`: State with placeholder values

# Returns
- `CAMPSState`: State with properly initialized clifford and mps

# Example
```julia
state = CAMPSState(5)  # Creates with placeholders
initialize!(state)      # Initializes actual Clifford and MPS
```
"""
function initialize!(state::CAMPSState)::CAMPSState
    n = state.n_qubits
    
    # Initialize Clifford using QuantumClifford
    state.clifford = initialize_clifford(n)
    
    # Initialize MPS using ITensorMPS
    mps, sites = initialize_mps(n)
    state.mps = mps
    state.sites = Vector{Any}(sites)  # Convert to Any for storage
    
    # Reset twisted paulis vector with proper type hint
    state.twisted_paulis = PauliOperator[]
    
    return state
end

"""
    is_initialized(state::CAMPSState) -> Bool

Check if a CAMPSState has been initialized.

# Arguments
- `state::CAMPSState`: CAMPS state

# Returns
- `Bool`: true if clifford and mps are not nothing
"""
function is_initialized(state::CAMPSState)::Bool
    return state.clifford !== nothing && state.mps !== nothing
end

"""
    ensure_initialized!(state::CAMPSState) -> CAMPSState

Ensure a CAMPSState is initialized, initializing if needed.

# Arguments
- `state::CAMPSState`: CAMPS state

# Returns
- `CAMPSState`: Initialized state
"""
function ensure_initialized!(state::CAMPSState)::CAMPSState
    if !is_initialized(state)
        initialize!(state)
    end
    return state
end

#==============================================================================#
# HIGH-LEVEL STATE OPERATIONS
# Convenience functions that combine Phase 2 interfaces
#==============================================================================#

"""
    get_bond_dimension(state::CAMPSState) -> Int

Get the current maximum bond dimension of the CAMPS state's MPS.

# Arguments
- `state::CAMPSState`: CAMPS state

# Returns
- `Int`: Maximum bond dimension
"""
function get_bond_dimension(state::CAMPSState)::Int
    ensure_initialized!(state)
    return get_mps_bond_dimension(state.mps)
end

"""
    get_predicted_bond_dimension(state::CAMPSState) -> Int

Get the predicted bond dimension based on GF(2) theory.

Uses the twisted Paulis recorded in the state to predict χ = 2^(t - rank).

# Arguments
- `state::CAMPSState`: CAMPS state

# Returns
- `Int`: Predicted bond dimension
"""
function get_predicted_bond_dimension(state::CAMPSState)::Int
    if isempty(state.twisted_paulis)
        return 1
    end
    # Convert Any[] to PauliOperator[] if needed
    paulis = PauliOperator[p for p in state.twisted_paulis]
    return predict_bond_dimension(paulis)
end

"""
    add_twisted_pauli!(state::CAMPSState, P::PauliOperator) -> CAMPSState

Record a twisted Pauli in the state's history.

# Arguments
- `state::CAMPSState`: CAMPS state
- `P::PauliOperator`: Twisted Pauli to record

# Returns
- `CAMPSState`: Modified state
"""
function add_twisted_pauli!(state::CAMPSState, P::PauliOperator)::CAMPSState
    push!(state.twisted_paulis, P)
    return state
end

"""
    compute_twisted_pauli(state::CAMPSState, axis::Symbol, qubit::Int) -> PauliOperator

Compute the twisted Pauli for a rotation on the given axis and qubit.

# Arguments
- `state::CAMPSState`: CAMPS state
- `axis::Symbol`: Rotation axis (:X, :Y, or :Z)
- `qubit::Int`: Target qubit

# Returns
- `PauliOperator`: Twisted Pauli P̃ = C† · P · C
"""
function compute_twisted_pauli(state::CAMPSState, axis::Symbol, qubit::Int)::PauliOperator
    ensure_initialized!(state)
    n = state.n_qubits
    P = axis_to_pauli(axis, qubit, n)
    return commute_pauli_through_clifford(P, state.clifford)
end

#==============================================================================#
# RE-EXPORT QUANTUMCLIFFORD TYPES (for user convenience)
#==============================================================================#

export PauliOperator, Stabilizer, Destabilizer, MixedDestabilizer, CliffordOperator

#==============================================================================#
# EXPORTS - Phase 1
#==============================================================================#

# Strategy types
export DisentanglingStrategy
export OFDStrategy, OBDStrategy, HybridStrategy, NoDisentangling

# Gate types
export Gate, CliffordGate, RotationGate

# Non-Clifford gate constructors
export TGate, TdagGate
export RzGate, RxGate, RyGate

# Clifford gate constructors
export HGate, SGate, SdagGate
export XGate, YGate, ZGate
export CNOTGate, CZGate, SWAPGate
export iSWAPGate, SqrtXGate, SqrtXdagGate, SqrtYGate, SqrtYdagGate

# State type and accessors
export CAMPSState
export n_qubits, num_free_qubits, num_magic_qubits, num_twisted_paulis
export is_free, is_magic, mark_as_magic!
export get_free_qubit_indices, get_magic_qubit_indices

# Utility functions - Pauli conversion
export xz_to_symbol, symbol_to_xz
export phase_to_complex, complex_to_phase
export pauli_matrix, rotation_matrix
export rotation_coefficients

# Utility functions - angles
export is_clifford_angle, normalize_angle

# Utility functions - bit manipulation
export int_to_bits, bits_to_int
export bitstring_to_vector, vector_to_bitstring

# Utility functions - circuit analysis
export count_gates_by_type, gate_depth
export validate_qubit_index, validate_circuit

# Utility functions - numerical
export isapprox_zero, isapprox_one, safe_log

#==============================================================================#
# EXPORTS - Phase 2: State Initialization
#==============================================================================#

export initialize!, is_initialized, ensure_initialized!
export get_bond_dimension, get_predicted_bond_dimension
export add_twisted_pauli!, compute_twisted_pauli

#==============================================================================#
# EXPORTS - Phase 2: Clifford Interface
#==============================================================================#

export initialize_clifford
export commute_pauli_through_clifford
export axis_to_pauli
export apply_clifford_gate!, apply_clifford_gates!, apply_inverse_gates!
export resolve_symbolic_gate, resolve_clifford_gate
export apply_clifford_gate_to_state!
export get_pauli_at, get_pauli_phase
export pauli_weight, pauli_support
export has_x_or_y, get_xbit_vector, get_zbit_vector
export clifford_nqubits
export create_pauli_string, create_pauli_string_with_phase, pauli_to_string

#==============================================================================#
# EXPORTS - Phase 2: MPS Interface
#==============================================================================#

export initialize_mps
export get_mps_bond_dimension, get_mps_norm, normalize_mps!
export pauli_to_itensor, rotation_to_itensor, identity_itensor
export apply_single_site_gate!, apply_local_rotation!
export apply_pauli_to_mps!, apply_pauli_string!, apply_pauli_string_to_copy
export apply_twisted_rotation!, apply_twisted_rotation_to_copy
export truncate_mps!
export entanglement_entropy, entanglement_entropy_all_bonds, max_entanglement_entropy
export extract_singular_values, bond_dimension_at, all_bond_dimensions
export sample_mps, sample_mps_multiple
export mps_overlap, mps_probability, mps_amplitude
export apply_two_qubit_gate!, matrix_to_two_qubit_itensor

#==============================================================================#
# EXPORTS - Phase 2: GF(2)
#==============================================================================#

export build_gf2_matrix, build_gf2_matrix_from_xbits
export gf2_rank, gf2_rank!
export predict_bond_dimension
export count_disentanglable, can_disentangle, find_disentangling_qubit
export analyze_gf2_structure, find_independent_rows, find_independent_rows_with_basis
export gf2_null_space, incremental_rank_update

#==============================================================================#
# EXPORTS - Phase 3: Two-Qubit Cliffords
#==============================================================================#

export generate_single_qubit_clifford, resolve_single_qubit_clifford
export get_all_two_qubit_cliffords, sample_two_qubit_cliffords
export clifford_to_matrix, clifford_on_basis_state
export stabilizer_to_statevector, basis_state_amplitude, pauli_eigenvalue_on_basis
export clifford_to_itensor, make_clifford_gate_tensors
export TwoQubitCliffordCache, build_clifford_cache
export get_clifford_matrix, get_clifford_inverse_matrix
export get_cnot_class_representatives, get_expanded_representatives
export compute_renyi2_entropy, compute_von_neumann_entropy
export extract_two_site_rdm, transform_rdm, partial_trace_4x4

#==============================================================================#
# EXPORTS - Phase 3: OFD
#==============================================================================#

export build_disentangler_gates, build_controlled_pauli_gate, flatten_gate_sequence
export apply_ofd!, try_apply_ofd!
export apply_t_gate_ofd!, apply_tdag_gate_ofd!
export OFDSResult, apply_ofds!
export update_twisted_pauli_after_ofd
export analyze_ofd_applicability, count_ofd_applicable
export generate_ofd_circuit
export t_gate_magic_state, magic_state_vector

#==============================================================================#
# EXPORTS - Phase 3: OBD
#==============================================================================#

export find_optimal_clifford_for_bond
export apply_clifford_to_mps!, apply_clifford_index_to_mps!
export OBDSweepResult, obd_sweep!
export apply_clifford_to_destabilizer!, decompose_two_qubit_clifford
export OBDResult, obd!
export apply_rotation_with_obd!
export apply_rotation_hybrid!, apply_t_gate_hybrid!, apply_tdag_gate_hybrid!
export get_entropy_profile, get_bond_dimension_profile, estimate_obd_improvement

# Equation 19 (Liu & Clark) tensor contraction approach
export Renyi2BaseTensor, Renyi2ContractionKernel
export precompute_renyi2_base_tensor, compute_renyi2_kernel
export get_or_build_renyi2_kernel_cache
export evaluate_renyi2_equation19, find_optimal_clifford_equation19

# Pauli-basis fast evaluation
export PauliExpectations, CliffordPauliMap
export precompute_pauli_expectations, build_clifford_pauli_map
export get_or_build_clifford_pauli_cache
export evaluate_renyi2_from_pauli, find_optimal_clifford_fast

#==============================================================================#
# EXPORTS - Phase 4: Simulation
#==============================================================================#

# Gate application
export apply_gate!, rotation_to_clifford

# Circuit simulation
export SimulationResult, simulate_circuit

# Observable extraction
export sample_state, amplitude, probability
export expectation_value, expectation_value_x, expectation_value_y, expectation_value_z

# State vector and fidelity (for small systems)
export state_vector, state_vector_sparse
export fidelity, overlap, fidelity_sampling, fidelity_with_target
export normalize!

# Clifford action on basis states
export apply_clifford_to_basis_state, compute_clifford_action_on_basis_state
export pauli_eigenvalue_on_computational_basis, compute_clifford_phase_on_basis

# Standard circuit generators
export qft_circuit, inverse_qft_circuit
export ghz_circuit, w_state_circuit
export random_clifford_t_circuit, random_brickwork_circuit
export random_t_depth_circuit, hardware_efficient_ansatz

# Circuit analysis
export analyze_circuit, predict_bond_dimension_for_circuit
export apply_clifford_left_multiply!

#==============================================================================#
# VERSION INFO
#==============================================================================#

"""
    CAMPS.version() -> VersionNumber

Return the version of CAMPS.jl.
"""
version() = v"0.4.0"

"""
    CAMPS.info()

Print information about CAMPS.jl.
"""
function info()
    println("CAMPS.jl v$(version())")
    println("Clifford-Augmented Matrix Product States for quantum circuit simulation")
    println()
    println("Based on:")
    println("  • Liu & Clark, arXiv:2412.17209 (CAMPS method)")
    println("  • Qian et al., arXiv:2405.09217 (Ground state preparation)")
    println()
    println("Implementation Status:")
    println("  ✓ Phase 1: Foundation (types, utilities)")
    println("  ✓ Phase 2: Library interfaces (QuantumClifford, ITensorMPS, GF(2))")
    println("  ✓ Phase 3: Core algorithms (OFD, OBD)")
    println("  ✓ Phase 4: Simulation interface & benchmarks")
    println()
    println("Key features:")
    println("  • OFD: Optimization-Free Disentangling for efficient T-gate handling")
    println("  • OBD: Optimization-Based Disentangling as fallback strategy")
    println("  • Hybrid strategy: Best of OFD and OBD combined")
    println("  • GF(2) bond dimension prediction")
    println("  • Full Clifford+T circuit simulation")
    println("  • State vector extraction and fidelity computation")
    println("  • Standard circuit generators (QFT, GHZ, W-state)")
    println("  • Comprehensive benchmarking suite")
end

end # module CAMPS
