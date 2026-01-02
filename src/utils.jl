# CAMPS.jl/src/utils.jl
# Utility functions for CAMPS.jl

#==============================================================================#
# PAULI SYMBOL CONVERSION
#==============================================================================#

"""
    xz_to_symbol(x::Bool, z::Bool) -> Symbol

Convert binary (x, z) representation to Pauli symbol.

QuantumClifford uses the binary encoding:
- I = (x=0, z=0)
- X = (x=1, z=0)
- Y = (x=1, z=1)
- Z = (x=0, z=1)

# Arguments
- `x::Bool`: x-bit (true if Pauli has X component)
- `z::Bool`: z-bit (true if Pauli has Z component)

# Returns
- `Symbol`: One of :I, :X, :Y, :Z

# Examples
```julia
xz_to_symbol(false, false)  # :I
xz_to_symbol(true, false)   # :X
xz_to_symbol(true, true)    # :Y
xz_to_symbol(false, true)   # :Z
```
"""
function xz_to_symbol(x::Bool, z::Bool)::Symbol
    if !x && !z
        return :I
    elseif x && !z
        return :X
    elseif x && z
        return :Y
    else  # !x && z
        return :Z
    end
end

"""
    symbol_to_xz(σ::Symbol) -> Tuple{Bool, Bool}

Convert Pauli symbol to binary (x, z) representation.

Inverse of `xz_to_symbol`.

# Arguments
- `σ::Symbol`: Pauli symbol (:I, :X, :Y, or :Z)

# Returns
- `Tuple{Bool, Bool}`: (x-bit, z-bit)

# Examples
```julia
symbol_to_xz(:I)  # (false, false)
symbol_to_xz(:X)  # (true, false)
symbol_to_xz(:Y)  # (true, true)
symbol_to_xz(:Z)  # (false, true)
```

# Throws
- `ArgumentError` if σ is not a valid Pauli symbol
"""
function symbol_to_xz(σ::Symbol)::Tuple{Bool, Bool}
    if σ == :I
        return (false, false)
    elseif σ == :X
        return (true, false)
    elseif σ == :Y
        return (true, true)
    elseif σ == :Z
        return (false, true)
    else
        throw(ArgumentError("Unknown Pauli symbol: $σ. Expected :I, :X, :Y, or :Z"))
    end
end

#==============================================================================#
# PHASE CONVERSION
#==============================================================================#

"""
    phase_to_complex(phase_byte::UInt8) -> ComplexF64

Convert QuantumClifford phase byte to complex number.

QuantumClifford encodes phases as:
- 0x00 → +1
- 0x01 → +i
- 0x02 → -1
- 0x03 → -i

# Arguments
- `phase_byte::UInt8`: Phase encoding from QuantumClifford (P.phase[])

# Returns
- `ComplexF64`: The complex phase value

# Examples
```julia
phase_to_complex(0x00)  # 1.0 + 0.0im
phase_to_complex(0x01)  # 0.0 + 1.0im
phase_to_complex(0x02)  # -1.0 + 0.0im
phase_to_complex(0x03)  # 0.0 - 1.0im
```

# Note
The phase byte should be obtained from a PauliOperator P as `P.phase[]`.
Only the lower 2 bits are used (values 0-3).
"""
function phase_to_complex(phase_byte::UInt8)::ComplexF64
    # Map: 0 → +1, 1 → +i, 2 → -1, 3 → -i
    # This is exp(i * π/2 * phase_byte)
    phases = (
        1.0 + 0.0im,   # 0x00: +1
        0.0 + 1.0im,   # 0x01: +i
        -1.0 + 0.0im,  # 0x02: -1
        0.0 - 1.0im    # 0x03: -i
    )
    return phases[(phase_byte & 0x03) + 1]
end

"""
    complex_to_phase(c::Complex) -> UInt8

Convert complex phase to QuantumClifford phase byte.

Inverse of `phase_to_complex`. The input must be ±1 or ±i.

# Arguments
- `c::Complex`: Complex phase (+1, +i, -1, or -i)

# Returns
- `UInt8`: Phase encoding for QuantumClifford

# Examples
```julia
complex_to_phase(1.0 + 0.0im)   # 0x00
complex_to_phase(0.0 + 1.0im)   # 0x01
complex_to_phase(-1.0 + 0.0im)  # 0x02
complex_to_phase(0.0 - 1.0im)   # 0x03
```

# Throws
- `ArgumentError` if the input is not ±1 or ±i
"""
function complex_to_phase(c::Complex)::UInt8
    tol = 1e-10

    if abs(c - 1.0) < tol
        return 0x00
    elseif abs(c - im) < tol
        return 0x01
    elseif abs(c + 1.0) < tol
        return 0x02
    elseif abs(c + im) < tol
        return 0x03
    else
        throw(ArgumentError("Invalid phase: $c. Expected ±1 or ±i"))
    end
end

# Method for real numbers - converts to Complex and delegates
function complex_to_phase(c::Real)::UInt8
    return complex_to_phase(Complex(c))
end

#==============================================================================#
# PAULI MATRIX UTILITIES
#==============================================================================#

"""
    pauli_matrix(σ::Symbol) -> Matrix{ComplexF64}

Return the 2×2 Pauli matrix for the given symbol.

# Arguments
- `σ::Symbol`: Pauli symbol (:I, :X, :Y, or :Z)

# Returns
- `Matrix{ComplexF64}`: 2×2 Pauli matrix

# Examples
```julia
pauli_matrix(:I)  # [1 0; 0 1]
pauli_matrix(:X)  # [0 1; 1 0]
pauli_matrix(:Y)  # [0 -im; im 0]
pauli_matrix(:Z)  # [1 0; 0 -1]
```
"""
function pauli_matrix(σ::Symbol)::Matrix{ComplexF64}
    if σ == :I
        return ComplexF64[1 0; 0 1]
    elseif σ == :X
        return ComplexF64[0 1; 1 0]
    elseif σ == :Y
        return ComplexF64[0 -im; im 0]
    elseif σ == :Z
        return ComplexF64[1 0; 0 -1]
    else
        throw(ArgumentError("Unknown Pauli symbol: $σ"))
    end
end

"""
    rotation_matrix(axis::Symbol, θ::Real) -> Matrix{ComplexF64}

Return the 2×2 rotation matrix R_axis(θ) = exp(-i θ σ_axis / 2).

# Arguments
- `axis::Symbol`: Rotation axis (:X, :Y, or :Z)
- `θ::Real`: Rotation angle in radians

# Returns
- `Matrix{ComplexF64}`: 2×2 rotation matrix

# Formulas
- Rx(θ) = cos(θ/2)I - i sin(θ/2)X = [cos(θ/2), -i sin(θ/2); -i sin(θ/2), cos(θ/2)]
- Ry(θ) = cos(θ/2)I - i sin(θ/2)Y = [cos(θ/2), -sin(θ/2); sin(θ/2), cos(θ/2)]
- Rz(θ) = cos(θ/2)I - i sin(θ/2)Z = [exp(-iθ/2), 0; 0, exp(iθ/2)]

# Examples
```julia
rotation_matrix(:Z, π/4)   # T gate matrix
rotation_matrix(:X, π)     # X gate (Pauli X)
```
"""
function rotation_matrix(axis::Symbol, θ::Real)::Matrix{ComplexF64}
    c = cos(θ/2)
    s = sin(θ/2)
    
    if axis == :X
        return ComplexF64[c -im*s; -im*s c]
    elseif axis == :Y
        return ComplexF64[c -s; s c]
    elseif axis == :Z
        return ComplexF64[exp(-im*θ/2) 0; 0 exp(im*θ/2)]
    else
        throw(ArgumentError("Unknown axis: $axis. Expected :X, :Y, or :Z"))
    end
end

#==============================================================================#
# ROTATION PARAMETER EXTRACTION
#==============================================================================#

"""
    rotation_coefficients(θ::Real) -> Tuple{ComplexF64, ComplexF64}

Compute (α, β) coefficients for rotation gate application.

A rotation R_axis(θ) = exp(-i θ σ_axis / 2) can be written as:
    R_axis(θ) = αI + βσ_axis

where:
    α = cos(θ/2)
    β = -i sin(θ/2)

This decomposition is used when applying twisted rotations to MPS:
    R|ψ⟩ = (αI + βP)|ψ⟩ = α|ψ⟩ + β(P|ψ⟩)

where P is the twisted Pauli.

# Arguments
- `θ::Real`: Rotation angle in radians

# Returns
- `Tuple{ComplexF64, ComplexF64}`: (α, β) coefficients

# Examples
```julia
α, β = rotation_coefficients(π/4)  # T gate coefficients
# α ≈ 0.9239, β ≈ -0.3827im
```
"""
function rotation_coefficients(θ::Real)::Tuple{ComplexF64, ComplexF64}
    α = ComplexF64(cos(θ/2))
    β = ComplexF64(-im * sin(θ/2))
    return (α, β)
end

#==============================================================================#
# ANGLE UTILITIES
#==============================================================================#

"""
    is_clifford_angle(θ::Real; tol::Float64=1e-10) -> Bool

Check if a rotation angle corresponds to a Clifford gate.

Clifford rotations have θ = k * π/2 for integer k.

# Arguments
- `θ::Real`: Rotation angle in radians
- `tol::Float64`: Tolerance for floating point comparison

# Returns
- `Bool`: true if the angle is a Clifford angle

# Examples
```julia
is_clifford_angle(π/2)   # true (S gate)
is_clifford_angle(π)     # true (Pauli)
is_clifford_angle(π/4)   # false (T gate)
is_clifford_angle(π/3)   # false
```
"""
function is_clifford_angle(θ::Real; tol::Float64=1e-10)::Bool
    # θ is Clifford if θ/(π/2) is an integer
    ratio = θ / (π/2)
    return abs(ratio - round(ratio)) < tol
end

"""
    normalize_angle(θ::Real) -> Float64

Normalize angle to the range [0, 2π).

# Arguments
- `θ::Real`: Rotation angle in radians

# Returns
- `Float64`: Angle in [0, 2π)

# Examples
```julia
normalize_angle(3π)    # π
normalize_angle(-π/4)  # 7π/4
```
"""
function normalize_angle(θ::Real)::Float64
    θ_norm = mod(Float64(θ), 2π)
    return θ_norm
end

#==============================================================================#
# BIT STRING UTILITIES
#==============================================================================#

"""
    int_to_bits(x::Integer, n::Int) -> BitVector

Convert integer to n-bit binary representation (little-endian).

# Arguments
- `x::Integer`: Non-negative integer to convert
- `n::Int`: Number of bits

# Returns
- `BitVector`: Little-endian bit representation

# Examples
```julia
int_to_bits(5, 4)  # BitVector [1, 0, 1, 0] (binary: 0101)
int_to_bits(0, 3)  # BitVector [0, 0, 0]
```
"""
function int_to_bits(x::Integer, n::Int)::BitVector
    x >= 0 || throw(ArgumentError("x must be non-negative"))
    n > 0 || throw(ArgumentError("n must be positive"))
    
    bits = BitVector(undef, n)
    for i in 1:n
        bits[i] = (x >> (i-1)) & 1 == 1
    end
    return bits
end

"""
    bits_to_int(bits::BitVector) -> Int

Convert bit vector to integer (little-endian).

# Arguments
- `bits::BitVector`: Bit representation

# Returns
- `Int`: Integer value

# Examples
```julia
bits_to_int(BitVector([1, 0, 1, 0]))  # 5
bits_to_int(BitVector([0, 0, 0]))     # 0
```
"""
function bits_to_int(bits::BitVector)::Int
    result = 0
    for i in eachindex(bits)
        if bits[i]
            result += (1 << (i-1))
        end
    end
    return result
end

"""
    bitstring_to_vector(s::String) -> Vector{Int}

Convert a string of 0s and 1s to a vector of integers.

# Arguments
- `s::String`: String containing only '0' and '1' characters

# Returns
- `Vector{Int}`: Vector of 0s and 1s

# Examples
```julia
bitstring_to_vector("0101")  # [0, 1, 0, 1]
bitstring_to_vector("110")   # [1, 1, 0]
```
"""
function bitstring_to_vector(s::String)::Vector{Int}
    result = Vector{Int}(undef, length(s))
    for (i, c) in enumerate(s)
        if c == '0'
            result[i] = 0
        elseif c == '1'
            result[i] = 1
        else
            throw(ArgumentError("Invalid character '$c' in bitstring. Expected '0' or '1'."))
        end
    end
    return result
end

"""
    vector_to_bitstring(v::Vector{<:Integer}) -> String

Convert a vector of 0s and 1s to a string.

# Arguments
- `v::Vector{<:Integer}`: Vector containing only 0s and 1s

# Returns
- `String`: Bitstring representation

# Examples
```julia
vector_to_bitstring([0, 1, 0, 1])  # "0101"
vector_to_bitstring([1, 1, 0])     # "110"
```
"""
function vector_to_bitstring(v::Vector{<:Integer})::String
    return join(v)
end

#==============================================================================#
# CIRCUIT UTILITIES
#==============================================================================#

"""
    count_gates_by_type(circuit::Vector{<:Gate}) -> Dict{Symbol, Int}

Count gates in a circuit by type.

# Arguments
- `circuit::Vector{<:Gate}`: Vector of gates

# Returns
- `Dict{Symbol, Int}`: Dictionary with gate type counts
    - :clifford → number of CliffordGate
    - :rotation → number of RotationGate
    - :t_gate → number of T gates (Rz(±π/4))
    - :total → total number of gates

# Examples
```julia
circuit = [HGate(1), TGate(1), CNOTGate(1,2), TGate(2)]
counts = count_gates_by_type(circuit)
# Dict(:clifford => 2, :rotation => 2, :t_gate => 2, :total => 4)
```
"""
function count_gates_by_type(circuit::Vector{<:Gate})::Dict{Symbol, Int}
    counts = Dict{Symbol, Int}(
        :clifford => 0,
        :rotation => 0,
        :t_gate => 0,
        :total => length(circuit)
    )
    
    for gate in circuit
        if gate isa CliffordGate
            counts[:clifford] += 1
        elseif gate isa RotationGate
            counts[:rotation] += 1
            # Check for T gate: Rz(±π/4)
            if gate.axis == :Z && (abs(abs(gate.angle) - π/4) < 1e-10)
                counts[:t_gate] += 1
            end
        end
    end
    
    return counts
end

"""
    gate_depth(circuit::Vector{<:Gate}, n_qubits::Int) -> Int

Compute the circuit depth (number of layers).

A layer consists of gates that can be executed in parallel (act on different qubits).

# Arguments
- `circuit::Vector{<:Gate}`: Vector of gates
- `n_qubits::Int`: Number of qubits in the circuit

# Returns
- `Int`: Circuit depth

# Note
This is a simple implementation that doesn't optimize for minimal depth.
Each gate is placed in the earliest layer where its qubits are available.
"""
function gate_depth(circuit::Vector{<:Gate}, n_qubits::Int)::Int
    isempty(circuit) && return 0
    
    # Track when each qubit is next available
    qubit_available = zeros(Int, n_qubits)
    
    for gate in circuit
        qubits = if gate isa CliffordGate
            gate.qubits
        else  # RotationGate
            [gate.qubit]
        end
        
        # Find the earliest layer where all qubits are available
        layer = maximum(qubit_available[q] for q in qubits) + 1
        
        # Mark qubits as busy until this layer completes
        for q in qubits
            qubit_available[q] = layer
        end
    end
    
    return maximum(qubit_available)
end

#==============================================================================#
# VALIDATION UTILITIES
#==============================================================================#

"""
    validate_qubit_index(qubit::Int, n_qubits::Int; name::String="qubit")

Validate that a qubit index is in the valid range [1, n_qubits].

# Arguments
- `qubit::Int`: Qubit index to validate
- `n_qubits::Int`: Total number of qubits
- `name::String`: Name of the parameter for error messages

# Throws
- `ArgumentError` if qubit is out of range

# Examples
```julia
validate_qubit_index(3, 5)  # OK
validate_qubit_index(0, 5)  # Error
validate_qubit_index(6, 5)  # Error
```
"""
function validate_qubit_index(qubit::Int, n_qubits::Int; name::String="qubit")
    if qubit < 1 || qubit > n_qubits
        throw(ArgumentError("$name index $qubit out of range [1, $n_qubits]"))
    end
    return nothing
end

"""
    validate_circuit(circuit::Vector{<:Gate}, n_qubits::Int)

Validate that all gates in a circuit act on valid qubits.

# Arguments
- `circuit::Vector{<:Gate}`: Circuit to validate
- `n_qubits::Int`: Total number of qubits

# Throws
- `ArgumentError` if any gate acts on invalid qubits

# Examples
```julia
circuit = [HGate(1), CNOTGate(1, 2)]
validate_circuit(circuit, 3)  # OK
validate_circuit(circuit, 1)  # Error: CNOT acts on qubit 2
```
"""
function validate_circuit(circuit::Vector{<:Gate}, n_qubits::Int)
    for (i, gate) in enumerate(circuit)
        qubits = if gate isa CliffordGate
            gate.qubits
        else  # RotationGate
            [gate.qubit]
        end
        
        for q in qubits
            if q < 1 || q > n_qubits
                throw(ArgumentError("Gate $i acts on qubit $q, which is out of range [1, $n_qubits]"))
            end
        end
    end
    return nothing
end

#==============================================================================#
# NUMERICAL UTILITIES
#==============================================================================#

"""
    isapprox_zero(x::Number; atol::Real=1e-14) -> Bool

Check if a number is approximately zero.

# Arguments
- `x::Number`: Number to check
- `atol::Real`: Absolute tolerance

# Returns
- `Bool`: true if |x| < atol
"""
isapprox_zero(x::Number; atol::Real=1e-14)::Bool = abs(x) < atol

"""
    isapprox_one(x::Number; atol::Real=1e-14) -> Bool

Check if a number is approximately one.

# Arguments
- `x::Number`: Number to check
- `atol::Real`: Absolute tolerance

# Returns
- `Bool`: true if |x - 1| < atol
"""
isapprox_one(x::Number; atol::Real=1e-14)::Bool = abs(x - 1) < atol

"""
    safe_log(x::Real; min_val::Real=1e-300) -> Float64

Compute log(x) safely, avoiding log(0).

# Arguments
- `x::Real`: Value to take log of
- `min_val::Real`: Minimum value to use (if x < min_val, use min_val)

# Returns
- `Float64`: log(max(x, min_val))
"""
function safe_log(x::Real; min_val::Real=1e-300)::Float64
    return log(max(x, min_val))
end
