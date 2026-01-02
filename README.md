# CAMPS.jl

**Clifford-Augmented Matrix Product States** for efficient simulation of quantum circuits.

## Overview

CAMPS.jl implements the CAMPS representation from [Liu & Clark (arXiv:2412.17209)](https://arxiv.org/abs/2412.17209), which represents quantum states as:

```
|ψ⟩ = C · |ψ_MPS⟩
```

where:
- `C` is a Clifford operator (tracked symbolically via [QuantumClifford.jl](https://github.com/QuantumSavory/QuantumClifford.jl))
- `|ψ_MPS⟩` is a Matrix Product State (stored via [ITensorMPS.jl](https://github.com/ITensor/ITensorMPS.jl))

This representation enables efficient simulation of **Clifford+T circuits** by keeping Clifford gates "free" (only updating `C`) and only modifying the MPS for non-Clifford gates like T-gates.

## Features

### Implemented (Phase 1 & 2)
- ✅ **Type System**: DisentanglingStrategy types (OFD, OBD, Hybrid, NoDisentangling)
- ✅ **Gate Types**: CliffordGate and RotationGate with convenience constructors
- ✅ **CAMPSState**: Full state representation with qubit tracking
- ✅ **QuantumClifford Integration**: Twisted Pauli computation, Clifford gate application
- ✅ **ITensorMPS Integration**: MPS initialization, Pauli/rotation application, entropy calculation
- ✅ **GF(2) Analysis**: Bond dimension prediction, disentanglability analysis

### Coming Soon (Phase 3 & 4)
- 🔲 Optimization-Free Disentangling (OFD)
- 🔲 Optimization-Based Disentangling (OBD)
- 🔲 Full circuit simulation
- 🔲 Benchmarking suite

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/yourusername/CAMPS.jl")
```

Or in development mode:

```julia
using Pkg
Pkg.develop(path="/path/to/CAMPS.jl")
```

## Quick Start

```julia
using CAMPS
using QuantumClifford

# Create and initialize a 5-qubit state
state = CAMPSState(5)
initialize!(state)

# Apply Clifford gates (only updates Clifford, MPS unchanged)
apply_clifford_gate!(state.clifford, sHadamard(1))
apply_clifford_gate!(state.clifford, sCNOT(1, 2))

# Compute twisted Pauli for a T-gate
P_twisted = compute_twisted_pauli(state, :Z, 1)
println("Twisted Pauli: ", pauli_to_string(P_twisted))

# Check if OFD disentangling is possible
if can_disentangle(P_twisted, state.free_qubits)
    qubit = find_disentangling_qubit(P_twisted, state.free_qubits)
    println("Can disentangle using qubit $qubit")
end

# Record twisted Pauli and predict bond dimension
add_twisted_pauli!(state, P_twisted)
χ = get_predicted_bond_dimension(state)
println("Predicted bond dimension: $χ")
```

## Implementation Status

| Phase | Component | Status |
|-------|-----------|--------|
| 1 | Types & Utilities | ✅ Complete |
| 2 | QuantumClifford Interface | ✅ Complete |
| 2 | ITensorMPS Interface | ✅ Complete |
| 2 | GF(2) Analysis | ✅ Complete |
| 3 | OFD Algorithm | 🔲 Planned |
| 3 | OBD Algorithm | 🔲 Planned |
| 4 | Circuit Simulation | 🔲 Planned |
| 4 | Benchmarking | 🔲 Planned |

## Disentangling Strategies

CAMPS.jl supports multiple disentangling strategies for handling non-Clifford gates:

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `OFDStrategy()` | Optimization-Free Disentangling | Default for Clifford+T circuits |
| `OBDStrategy()` | Optimization-Based Disentangling | When OFD unavailable |
| `HybridStrategy()` | OFD + OBD fallback | Production use (recommended) |
| `NoDisentangling()` | Direct application | Baseline comparison |

## Gate Types

### Non-Clifford Gates (applied to MPS)

```julia
TGate(qubit)        # T gate: Rz(π/4)
TdagGate(qubit)     # T†: Rz(-π/4)
RzGate(qubit, θ)    # Rz(θ)
RxGate(qubit, θ)    # Rx(θ)
RyGate(qubit, θ)    # Ry(θ)
```

### Clifford Gates (tracked symbolically)

```julia
HGate(qubit)           # Hadamard
SGate(qubit)           # S gate (√Z)
SdagGate(qubit)        # S†
XGate(qubit)           # Pauli X
YGate(qubit)           # Pauli Y
ZGate(qubit)           # Pauli Z
CNOTGate(ctrl, tgt)    # CNOT
CZGate(q1, q2)         # CZ
SWAPGate(q1, q2)       # SWAP
```

## Key Concepts

### Twisted Paulis

When a rotation gate R_P(θ) is applied after accumulated Clifford C, we compute:
```
P̃ = C† · P · C
```
This "twisted Pauli" determines how the rotation affects the MPS.

### GF(2) Bond Dimension Prediction

For t non-Clifford gates with twisted Paulis P[1], ..., P[t], construct the GF(2) matrix:
```
M[k, j] = 1 if P[k] has X or Y at qubit j
```
Then the predicted bond dimension is:
```
χ = 2^(t - rank(M))
```

### OFD Disentangling (Phase 3)

A twisted Pauli P can be disentangled using OFD if there exists a "free" qubit j where:
- The qubit is in |0⟩ state (not yet used for OFD)
- P has X or Y at position j (xbit[j] = true)

## API Reference

### State Management

```julia
CAMPSState(n; max_bond=1024, cutoff=1e-15)  # Create state
initialize!(state)                           # Initialize Clifford and MPS
is_initialized(state)                        # Check if initialized
ensure_initialized!(state)                   # Initialize if needed
```

### Clifford Operations

```julia
initialize_clifford(n)                              # Create identity Clifford
commute_pauli_through_clifford(P, C)               # Compute C†·P·C
apply_clifford_gate!(C, gate)                      # Apply gate to Clifford
compute_twisted_pauli(state, axis, qubit)          # Compute twisted Pauli
```

### MPS Operations

```julia
initialize_mps(n)                                   # Create |0⟩^⊗n MPS
get_mps_bond_dimension(mps)                        # Get max bond dim
apply_pauli_string!(mps, P, sites)                 # Apply Pauli to MPS
apply_twisted_rotation!(mps, sites, P, θ)          # Apply rotation
entanglement_entropy(mps, bond)                    # Compute entropy
sample_mps(mps)                                    # Sample bitstring
```

### GF(2) Analysis

```julia
build_gf2_matrix(twisted_paulis)                   # Build GF(2) matrix
gf2_rank(M)                                        # Compute rank over GF(2)
predict_bond_dimension(twisted_paulis)             # Predict χ = 2^(t-rank)
can_disentangle(P, free_qubits)                   # Check OFD possibility
find_disentangling_qubit(P, free_qubits)          # Find control qubit
```

## Running Tests

```julia
using Pkg
Pkg.test("CAMPS")
```

## References

- Liu & Clark, "Optimization-Free Disentangling for Clifford+T Circuits via Matrix Product States" ([arXiv:2412.17209](https://arxiv.org/abs/2412.17209))
- Qian et al., "CAMPS for Ground State Preparation" ([arXiv:2405.09217](https://arxiv.org/abs/2405.09217))

## License

MIT License - see LICENSE file for details.
