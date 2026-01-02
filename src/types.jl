# CAMPS.jl/src/types.jl
# Core type definitions for CAMPS (Clifford-Augmented Matrix Product States)

#==============================================================================#
# DISENTANGLING STRATEGIES
#==============================================================================#

"""
    DisentanglingStrategy

Abstract type for disentangling strategies used during non-Clifford gate application.

Concrete subtypes:
- `OFDStrategy`: Optimization-Free Disentangling (algebraic, optimal when applicable)
- `OBDStrategy`: Optimization-Based Disentangling (search-based, always applicable)
- `HybridStrategy`: OFD when possible, OBD fallback
- `NoDisentangling`: Direct application without disentangling (baseline)
"""
abstract type DisentanglingStrategy end

"""
    OFDStrategy <: DisentanglingStrategy

Optimization-Free Disentangling strategy.

Uses the algebraic construction from Liu & Clark (arXiv:2412.17209) to build
optimal disentanglers when a "free qubit" (qubit in |0⟩ state with X or Y in
the twisted Pauli) is available.

# Properties
- Deterministic: always produces the same result
- Optimal: achieves minimum possible bond dimension when applicable
- Fast: O(n²) per gate, no optimization loop

# When to use
- Default choice for Clifford+T circuits
- When GF(2) prediction is important (preserves exact rank structure)
"""
struct OFDStrategy <: DisentanglingStrategy end

"""
    OBDStrategy <: DisentanglingStrategy

Optimization-Based Disentangling strategy.

Searches over all two-qubit Clifford gates (720 up to single-qubit equivalence,
11,520 total from QuantumClifford.enumerate_cliffords(2)) to find the one that
minimizes entanglement entropy at each bond.

# Fields
- `max_sweeps::Int`: Maximum number of sweeps through the chain (default: 10)
- `improvement_threshold::Float64`: Stop if entropy improvement < this (default: 1e-10)

# Properties
- Non-deterministic: may find different local minima
- Always applicable: works even when OFD cannot find a disentangler
- Slower: O(720 × χ³) per bond per sweep

# When to use
- When OFD is not applicable (no free qubit with X/Y)
- For comparison benchmarks
- When exploring non-Clifford+T circuits
"""
struct OBDStrategy <: DisentanglingStrategy
    max_sweeps::Int
    improvement_threshold::Float64
    
    function OBDStrategy(; max_sweeps::Int=10, improvement_threshold::Float64=1e-10)
        max_sweeps > 0 || throw(ArgumentError("max_sweeps must be positive"))
        improvement_threshold >= 0 || throw(ArgumentError("improvement_threshold must be non-negative"))
        new(max_sweeps, improvement_threshold)
    end
end

"""
    HybridStrategy <: DisentanglingStrategy

Hybrid disentangling strategy: use OFD when possible, fall back to OBD.

This combines the optimality of OFD (when a free qubit is available) with
the universality of OBD (when OFD cannot be applied).

# Fields
- `obd_sweeps_on_failure::Int`: Number of OBD sweeps when OFD unavailable (default: 2)
- `obd_improvement_threshold::Float64`: OBD convergence threshold (default: 1e-10)

# Properties
- Best of both worlds: optimal when OFD works, graceful fallback otherwise
- Recommended default strategy for general use

# When to use
- Default choice for production use
- When circuit structure is unknown a priori
"""
struct HybridStrategy <: DisentanglingStrategy
    obd_sweeps_on_failure::Int
    obd_improvement_threshold::Float64
    
    function HybridStrategy(; obd_sweeps_on_failure::Int=2, 
                            obd_improvement_threshold::Float64=1e-10)
        obd_sweeps_on_failure > 0 || throw(ArgumentError("obd_sweeps_on_failure must be positive"))
        obd_improvement_threshold >= 0 || throw(ArgumentError("obd_improvement_threshold must be non-negative"))
        new(obd_sweeps_on_failure, obd_improvement_threshold)
    end
end

"""
    NoDisentangling <: DisentanglingStrategy

No disentangling: directly apply twisted rotations to MPS without any disentangling.

This strategy serves as a baseline for comparison. Without disentangling,
bond dimension grows as 2^t where t is the number of non-Clifford gates.

# When to use
- Baseline for benchmarking
- Debugging and testing
- When bond dimension growth is acceptable
"""
struct NoDisentangling <: DisentanglingStrategy end

#==============================================================================#
# GATE TYPES
#==============================================================================#

"""
    Gate

Abstract type for quantum gates in CAMPS simulation.

Concrete subtypes:
- `CliffordGate`: Clifford gates (H, S, CNOT, CZ, etc.) - tracked symbolically
- `RotationGate`: Non-Clifford rotations (T, Rz, Rx, Ry) - applied to MPS
"""
abstract type Gate end

"""
    CliffordGate <: Gate

Wrapper for Clifford gates from QuantumClifford.jl.

Clifford gates are "free" in CAMPS: they only update the accumulated Clifford
operator C without modifying the coefficient MPS.

# Fields
- `gates::Vector{Any}`: Vector of QuantumClifford symbolic gates (e.g., sHadamard, sCNOT)
- `qubits::Vector{Int}`: Qubits the gate acts on (1-indexed)

# Examples
```julia
# Single-qubit Hadamard on qubit 3
h = CliffordGate([sHadamard(3)], [3])

# CNOT with control=1, target=2
cnot = CliffordGate([sCNOT(1, 2)], [1, 2])

# Composite: S followed by H on qubit 1
sh = CliffordGate([sPhase(1), sHadamard(1)], [1])
```

# Note
Use the convenience constructors (HGate, SGate, CNOTGate, etc.) instead of
constructing CliffordGate directly.
"""
struct CliffordGate <: Gate
    gates::Vector{Any}  # QuantumClifford symbolic gates
    qubits::Vector{Int}
    
    function CliffordGate(gates::Vector, qubits::Vector{Int})
        isempty(gates) && throw(ArgumentError("gates vector cannot be empty"))
        isempty(qubits) && throw(ArgumentError("qubits vector cannot be empty"))
        all(q -> q > 0, qubits) || throw(ArgumentError("qubit indices must be positive"))
        new(collect(Any, gates), qubits)
    end
end

"""
    RotationGate <: Gate

Non-Clifford rotation gate: R_axis(θ) = exp(-i θ σ_axis / 2)

These gates introduce "magic" into the state and require MPS manipulation.
The rotation is applied as (cos(θ/2)I - i sin(θ/2)P) where P is the
twisted Pauli obtained by conjugating the axis through the accumulated Clifford.

# Fields
- `qubit::Int`: Target qubit (1-indexed)
- `axis::Symbol`: Rotation axis (:X, :Y, or :Z)
- `angle::Float64`: Rotation angle in radians

# Special cases
- T gate: Rz(π/4)
- T† gate: Rz(-π/4)  
- S gate: Rz(π/2) - but this is Clifford, use SGate instead
- General Rz(θ) where θ/π is irrational: non-Clifford

# Examples
```julia
# T gate on qubit 2
t = RotationGate(2, :Z, π/4)

# Rx(π/3) on qubit 1
rx = RotationGate(1, :X, π/3)
```

# Note
Use the convenience constructors (TGate, TdagGate, RzGate, etc.) for common cases.
"""
struct RotationGate <: Gate
    qubit::Int
    axis::Symbol
    angle::Float64
    
    function RotationGate(qubit::Int, axis::Symbol, angle::Float64)
        qubit > 0 || throw(ArgumentError("qubit index must be positive"))
        axis in (:X, :Y, :Z) || throw(ArgumentError("axis must be :X, :Y, or :Z"))
        new(qubit, axis, angle)
    end
end

#==============================================================================#
# GATE CONVENIENCE CONSTRUCTORS
#==============================================================================#

# Note: These return Gate objects. The actual QuantumClifford symbolic gates
# (sHadamard, sPhase, etc.) will be created in Phase 2 when we have the
# QuantumClifford import. For now, we store placeholder symbols that will
# be resolved when the full module is loaded.

"""
    TGate(qubit::Int) -> RotationGate

Create a T gate (π/8 gate) on the specified qubit.

T = Rz(π/4) = diag(1, exp(iπ/4))

This is the standard non-Clifford gate for fault-tolerant quantum computing.
"""
TGate(qubit::Int) = RotationGate(qubit, :Z, π/4)

"""
    TdagGate(qubit::Int) -> RotationGate

Create a T† (T-dagger) gate on the specified qubit.

T† = Rz(-π/4) = diag(1, exp(-iπ/4))
"""
TdagGate(qubit::Int) = RotationGate(qubit, :Z, -π/4)

"""
    RzGate(qubit::Int, θ::Real) -> RotationGate

Create an Rz(θ) rotation gate on the specified qubit.

Rz(θ) = exp(-iθZ/2) = diag(exp(-iθ/2), exp(iθ/2))
"""
RzGate(qubit::Int, θ::Real) = RotationGate(qubit, :Z, Float64(θ))

"""
    RxGate(qubit::Int, θ::Real) -> RotationGate

Create an Rx(θ) rotation gate on the specified qubit.

Rx(θ) = exp(-iθX/2) = cos(θ/2)I - i sin(θ/2)X
"""
RxGate(qubit::Int, θ::Real) = RotationGate(qubit, :X, Float64(θ))

"""
    RyGate(qubit::Int, θ::Real) -> RotationGate

Create an Ry(θ) rotation gate on the specified qubit.

Ry(θ) = exp(-iθY/2) = cos(θ/2)I - i sin(θ/2)Y
"""
RyGate(qubit::Int, θ::Real) = RotationGate(qubit, :Y, Float64(θ))

#==============================================================================#
# CLIFFORD GATE CONSTRUCTORS
# These create CliffordGate objects with tuple specifications.
# The tuples are resolved to actual QuantumClifford symbolic gates
# via resolve_symbolic_gate() in clifford_interface.jl when applied.
#
# This design allows types.jl to be loaded before QuantumClifford symbols
# are available, while still providing a clean interface.
#==============================================================================#

"""
    HGate(qubit::Int) -> CliffordGate

Create a Hadamard gate on the specified qubit.

H = (X + Z) / √2

Transforms: X ↔ Z, Y → -Y

# Example
```julia
h = HGate(1)  # Hadamard on qubit 1
```
"""
HGate(qubit::Int) = CliffordGate([(:H, qubit)], [qubit])

"""
    SGate(qubit::Int) -> CliffordGate

Create an S gate (phase gate, √Z) on the specified qubit.

S = diag(1, i) = Rz(π/2)

Transforms: X → Y, Y → -X, Z → Z

# Example
```julia
s = SGate(2)  # S gate on qubit 2
```
"""
SGate(qubit::Int) = CliffordGate([(:S, qubit)], [qubit])

"""
    SdagGate(qubit::Int) -> CliffordGate

Create an S† (S-dagger) gate on the specified qubit.

S† = diag(1, -i) = Rz(-π/2)

Transforms: X → -Y, Y → X, Z → Z

# Example
```julia
sd = SdagGate(3)  # S† gate on qubit 3
```
"""
SdagGate(qubit::Int) = CliffordGate([(:Sdag, qubit)], [qubit])

"""
    XGate(qubit::Int) -> CliffordGate

Create a Pauli X gate on the specified qubit.

Transforms: X → X, Y → -Y, Z → -Z

# Example
```julia
x = XGate(1)  # Pauli X on qubit 1
```
"""
XGate(qubit::Int) = CliffordGate([(:X, qubit)], [qubit])

"""
    YGate(qubit::Int) -> CliffordGate

Create a Pauli Y gate on the specified qubit.

Transforms: X → -X, Y → Y, Z → -Z

# Example
```julia
y = YGate(2)  # Pauli Y on qubit 2
```
"""
YGate(qubit::Int) = CliffordGate([(:Y, qubit)], [qubit])

"""
    ZGate(qubit::Int) -> CliffordGate

Create a Pauli Z gate on the specified qubit.

Transforms: X → -X, Y → -Y, Z → Z

# Example
```julia
z = ZGate(3)  # Pauli Z on qubit 3
```
"""
ZGate(qubit::Int) = CliffordGate([(:Z, qubit)], [qubit])

"""
    CNOTGate(control::Int, target::Int) -> CliffordGate

Create a CNOT (controlled-X) gate.

CNOT|c,t⟩ = |c, c⊕t⟩

Transforms:
- X_c → X_c X_t
- Z_c → Z_c
- X_t → X_t
- Z_t → Z_c Z_t

# Example
```julia
cnot = CNOTGate(1, 2)  # CNOT with control=1, target=2
```
"""
function CNOTGate(control::Int, target::Int)
    control != target || throw(ArgumentError("control and target must be different qubits"))
    CliffordGate([(:CNOT, control, target)], [control, target])
end

"""
    CZGate(qubit1::Int, qubit2::Int) -> CliffordGate

Create a CZ (controlled-Z) gate.

CZ is symmetric: CZ(q1, q2) = CZ(q2, q1)

Transforms:
- X_1 → X_1 Z_2
- Z_1 → Z_1
- X_2 → Z_1 X_2
- Z_2 → Z_2

# Example
```julia
cz = CZGate(1, 2)  # CZ on qubits 1 and 2
```
"""
function CZGate(qubit1::Int, qubit2::Int)
    qubit1 != qubit2 || throw(ArgumentError("qubits must be different"))
    CliffordGate([(:CZ, qubit1, qubit2)], [qubit1, qubit2])
end

"""
    SWAPGate(qubit1::Int, qubit2::Int) -> CliffordGate

Create a SWAP gate.

SWAP|q1, q2⟩ = |q2, q1⟩

Transforms: Exchanges all Paulis between the two qubits.

# Example
```julia
swap = SWAPGate(1, 3)  # SWAP qubits 1 and 3
```
"""
function SWAPGate(qubit1::Int, qubit2::Int)
    qubit1 != qubit2 || throw(ArgumentError("qubits must be different"))
    CliffordGate([(:SWAP, qubit1, qubit2)], [qubit1, qubit2])
end

"""
    iSWAPGate(qubit1::Int, qubit2::Int) -> CliffordGate

Create an iSWAP gate.

iSWAP = SWAP · (CZ) · (S ⊗ S)

# Example
```julia
iswap = iSWAPGate(1, 2)  # iSWAP on qubits 1 and 2
```
"""
function iSWAPGate(qubit1::Int, qubit2::Int)
    qubit1 != qubit2 || throw(ArgumentError("qubits must be different"))
    # iSWAP can be decomposed as: S₁ · S₂ · CNOT(1,2) · CNOT(2,1) · S₁ · S₂
    # Or more simply as a sequence of gates
    CliffordGate([
        (:S, qubit1),
        (:S, qubit2),
        (:CNOT, qubit1, qubit2),
        (:CNOT, qubit2, qubit1),
        (:S, qubit1),
        (:S, qubit2)
    ], [qubit1, qubit2])
end

"""
    SqrtXGate(qubit::Int) -> CliffordGate

Create a √X gate (Rx(π/2)) on the specified qubit.

√X = (1/√2)(I - iX) = H · S · H

# Example
```julia
sqrtx = SqrtXGate(1)  # √X on qubit 1
```
"""
function SqrtXGate(qubit::Int)
    CliffordGate([(:H, qubit), (:S, qubit), (:H, qubit)], [qubit])
end

"""
    SqrtXdagGate(qubit::Int) -> CliffordGate

Create a √X† gate (Rx(-π/2)) on the specified qubit.

√X† = (1/√2)(I + iX) = H · S† · H

# Example
```julia
sqrtxdag = SqrtXdagGate(1)  # √X† on qubit 1
```
"""
function SqrtXdagGate(qubit::Int)
    CliffordGate([(:H, qubit), (:Sdag, qubit), (:H, qubit)], [qubit])
end

"""
    SqrtYGate(qubit::Int) -> CliffordGate

Create a √Y gate (Ry(π/2)) on the specified qubit.

√Y = (1/√2)(I - iY) = S · H · S

# Example
```julia
sqrty = SqrtYGate(1)  # √Y on qubit 1
```
"""
function SqrtYGate(qubit::Int)
    CliffordGate([(:S, qubit), (:H, qubit), (:S, qubit)], [qubit])
end

"""
    SqrtYdagGate(qubit::Int) -> CliffordGate

Create a √Y† gate (Ry(-π/2)) on the specified qubit.

√Y† = (1/√2)(I + iY) = S† · H · S†

# Example
```julia
sqrtydag = SqrtYdagGate(1)  # √Y† on qubit 1
```
"""
function SqrtYdagGate(qubit::Int)
    CliffordGate([(:Sdag, qubit), (:H, qubit), (:Sdag, qubit)], [qubit])
end

#==============================================================================#
# CAMPS STATE
#==============================================================================#

# Forward declaration of MPS and related types
# These will be properly typed when ITensorMPS is loaded in Phase 2

"""
    CAMPSState

The Clifford-Augmented Matrix Product State.

Represents a quantum state as: |ψ⟩ = C · |mps⟩

where C is a Clifford operator (tracked via QuantumClifford.jl's Destabilizer)
and |mps⟩ is the coefficient MPS (stored via ITensorMPS.jl's MPS).

We use Destabilizer (not MixedDestabilizer) because:
1. We're tracking a Clifford unitary, which is always full-rank
2. Destabilizer can be converted to CliffordOperator for Pauli conjugation
3. MixedDestabilizer is meant for mixed states, not unitary tracking

# Fields
- `clifford::Any`: Accumulated Clifford operator (Destabilizer from QuantumClifford)
- `mps::Any`: Coefficient MPS (MPS from ITensorMPS)
- `sites::Vector{Any}`: ITensor site indices for the MPS
- `n_qubits::Int`: Number of qubits
- `free_qubits::BitVector`: true if qubit is in |0⟩ state (available for OFD control)
- `magic_qubits::BitVector`: true if qubit has been used as OFD control
- `twisted_paulis::Vector{Any}`: History of twisted Paulis (for GF(2) analysis)
- `max_bond::Int`: Maximum bond dimension for MPS truncation
- `cutoff::Float64`: Singular value cutoff for MPS truncation

# Invariants
- `length(free_qubits) == length(magic_qubits) == n_qubits`
- `free_qubits[j] && magic_qubits[j]` is always false (mutually exclusive)
- Initially: all qubits are free, none are magic

# State evolution
- Clifford gates: Update only `clifford`, MPS unchanged
- Non-Clifford gates: May update `mps`, may convert free→magic qubit via OFD

# Example
```julia
# Create 10-qubit initial state |0⟩^⊗10
state = CAMPSState(10)

# With custom parameters
state = CAMPSState(10; max_bond=512, cutoff=1e-12)
```
"""
mutable struct CAMPSState
    # Core state representation
    clifford::Any               # Destabilizer from QuantumClifford
    mps::Any                    # MPS from ITensorMPS
    sites::Vector{Any}          # Vector{Index} from ITensors

    # Dimensions
    n_qubits::Int

    # Qubit tracking for OFD
    free_qubits::BitVector      # Qubits still in |0⟩
    magic_qubits::BitVector     # Qubits that have absorbed T-gates via OFD

    # History tracking for GF(2) analysis
    twisted_paulis::Vector{Any} # Vector{PauliOperator} from QuantumClifford

    # Truncation parameters
    max_bond::Int
    cutoff::Float64

    """
        CAMPSState(n::Int; max_bond::Int=1024, cutoff::Float64=1e-15)

    Create an initial CAMPS state |0⟩^⊗n.

    The Clifford is initialized to identity, the MPS to |0⟩^⊗n,
    all qubits are marked as free, and none are magic.

    Note: This constructor creates placeholder values. The actual initialization
    with QuantumClifford and ITensorMPS objects happens in Phase 2's
    clifford_interface.jl and mps_interface.jl.
    """
    function CAMPSState(n::Int; max_bond::Int=1024, cutoff::Float64=1e-15)
        n > 0 || throw(ArgumentError("number of qubits must be positive"))
        max_bond > 0 || throw(ArgumentError("max_bond must be positive"))
        cutoff > 0 || throw(ArgumentError("cutoff must be positive"))

        # Placeholder values - will be properly initialized in Phase 2
        clifford = nothing  # Will be one(Destabilizer, n)
        mps = nothing       # Will be MPS(sites, "0")
        sites = Any[]       # Will be siteinds("Qubit", n)

        # Initialize tracking
        free_qubits = trues(n)
        magic_qubits = falses(n)
        twisted_paulis = Any[]

        new(clifford, mps, sites, n, free_qubits, magic_qubits,
            twisted_paulis, max_bond, cutoff)
    end
end

#==============================================================================#
# ACCESSOR FUNCTIONS FOR CAMPSState
#==============================================================================#

"""
    n_qubits(state::CAMPSState) -> Int

Return the number of qubits in the CAMPS state.
"""
n_qubits(state::CAMPSState) = state.n_qubits

"""
    num_free_qubits(state::CAMPSState) -> Int

Return the number of free qubits (qubits still in |0⟩ state).
"""
num_free_qubits(state::CAMPSState) = count(state.free_qubits)

"""
    num_magic_qubits(state::CAMPSState) -> Int

Return the number of magic qubits (qubits that have absorbed T-gates via OFD).
"""
num_magic_qubits(state::CAMPSState) = count(state.magic_qubits)

"""
    num_twisted_paulis(state::CAMPSState) -> Int

Return the number of twisted Paulis recorded (= number of non-Clifford gates applied).
"""
num_twisted_paulis(state::CAMPSState) = length(state.twisted_paulis)

"""
    is_free(state::CAMPSState, qubit::Int) -> Bool

Check if the specified qubit is free (in |0⟩ state).
"""
function is_free(state::CAMPSState, qubit::Int)
    1 <= qubit <= state.n_qubits || throw(BoundsError(state.free_qubits, qubit))
    return state.free_qubits[qubit]
end

"""
    is_magic(state::CAMPSState, qubit::Int) -> Bool

Check if the specified qubit is a magic qubit (has absorbed a T-gate via OFD).
"""
function is_magic(state::CAMPSState, qubit::Int)
    1 <= qubit <= state.n_qubits || throw(BoundsError(state.magic_qubits, qubit))
    return state.magic_qubits[qubit]
end

"""
    mark_as_magic!(state::CAMPSState, qubit::Int)

Mark a qubit as magic (after it has been used as OFD control).
This sets free_qubits[qubit] = false and magic_qubits[qubit] = true.
"""
function mark_as_magic!(state::CAMPSState, qubit::Int)
    1 <= qubit <= state.n_qubits || throw(BoundsError(state.magic_qubits, qubit))
    state.free_qubits[qubit] = false
    state.magic_qubits[qubit] = true
    return state
end

"""
    get_free_qubit_indices(state::CAMPSState) -> Vector{Int}

Return indices of all free qubits.
"""
get_free_qubit_indices(state::CAMPSState) = findall(state.free_qubits)

"""
    get_magic_qubit_indices(state::CAMPSState) -> Vector{Int}

Return indices of all magic qubits.
"""
get_magic_qubit_indices(state::CAMPSState) = findall(state.magic_qubits)

#==============================================================================#
# TYPE DISPLAY
#==============================================================================#

function Base.show(io::IO, ::OFDStrategy)
    print(io, "OFDStrategy()")
end

function Base.show(io::IO, s::OBDStrategy)
    print(io, "OBDStrategy(max_sweeps=$(s.max_sweeps), threshold=$(s.improvement_threshold))")
end

function Base.show(io::IO, s::HybridStrategy)
    print(io, "HybridStrategy(obd_sweeps=$(s.obd_sweeps_on_failure), threshold=$(s.obd_improvement_threshold))")
end

function Base.show(io::IO, ::NoDisentangling)
    print(io, "NoDisentangling()")
end

function Base.show(io::IO, g::CliffordGate)
    print(io, "CliffordGate(qubits=$(g.qubits))")
end

function Base.show(io::IO, g::RotationGate)
    # Pretty print common angles
    angle_str = if g.angle ≈ π/4
        "π/4"
    elseif g.angle ≈ -π/4
        "-π/4"
    elseif g.angle ≈ π/2
        "π/2"
    elseif g.angle ≈ -π/2
        "-π/2"
    elseif g.angle ≈ π
        "π"
    else
        string(round(g.angle, digits=4))
    end
    print(io, "R$(g.axis)($(angle_str)) on qubit $(g.qubit)")
end

function Base.show(io::IO, state::CAMPSState)
    print(io, "CAMPSState(n=$(state.n_qubits), ")
    print(io, "free=$(num_free_qubits(state)), ")
    print(io, "magic=$(num_magic_qubits(state)), ")
    print(io, "t_gates=$(num_twisted_paulis(state)), ")
    print(io, "max_bond=$(state.max_bond))")
end

function Base.show(io::IO, ::MIME"text/plain", state::CAMPSState)
    println(io, "CAMPSState")
    println(io, "  Qubits: $(state.n_qubits)")
    println(io, "  Free qubits: $(num_free_qubits(state)) $(get_free_qubit_indices(state))")
    println(io, "  Magic qubits: $(num_magic_qubits(state)) $(get_magic_qubit_indices(state))")
    println(io, "  Non-Clifford gates applied: $(num_twisted_paulis(state))")
    println(io, "  Max bond dimension: $(state.max_bond)")
    println(io, "  Truncation cutoff: $(state.cutoff)")
    
    if state.mps !== nothing
        println(io, "  MPS initialized: yes")
    else
        println(io, "  MPS initialized: no (call initialize! after loading ITensorMPS)")
    end
    
    if state.clifford !== nothing
        println(io, "  Clifford initialized: yes")
    else
        print(io, "  Clifford initialized: no (call initialize! after loading QuantumClifford)")
    end
end
