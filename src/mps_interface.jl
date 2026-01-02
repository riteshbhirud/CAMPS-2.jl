# CAMPS.jl/src/mps_interface.jl
# Interface with ITensorMPS.jl for Matrix Product State operations

#==============================================================================#
# IMPORTS (these are imported in the main CAMPS.jl module)
# From ITensors: ITensor, Index, dag, array, delta, prime, noprime, 
#                commonind, uniqueinds, hasind
# From ITensorMPS: MPS, siteinds, linkind, maxlinkdim, orthogonalize, 
#                  truncate!, inner, expect
# Also uses: LinearAlgebra.svd, LinearAlgebra.norm, LinearAlgebra.diag
#==============================================================================#

#==============================================================================#
# MPS INITIALIZATION
#==============================================================================#

"""
    initialize_mps(n::Int) -> Tuple{MPS, Vector{Index}}

Create an MPS representing |0⟩^⊗n and return it with its site indices.

# Arguments
- `n::Int`: Number of qubits

# Returns
- `Tuple{MPS, Vector{Index}}`: (MPS for |0⟩^⊗n, site indices)

# Example
```julia
mps, sites = initialize_mps(5)
```
"""
function initialize_mps(n::Int)
    # Create qubit site indices
    sites = siteinds("Qubit", n)
    
    # Create product state |0⟩^⊗n
    # In ITensor convention, "0" corresponds to the |0⟩ state
    mps = MPS(sites, "0")
    
    return mps, sites
end

"""
    get_mps_bond_dimension(mps::MPS) -> Int

Get the maximum bond dimension of an MPS.

# Arguments
- `mps::MPS`: Matrix Product State

# Returns
- `Int`: Maximum bond dimension
"""
function get_mps_bond_dimension(mps::MPS)::Int
    return maxlinkdim(mps)
end

"""
    get_mps_norm(mps::MPS) -> Float64

Compute the norm of an MPS: √⟨ψ|ψ⟩

# Arguments
- `mps::MPS`: Matrix Product State

# Returns
- `Float64`: Norm of the MPS
"""
function get_mps_norm(mps::MPS)::Float64
    return real(sqrt(inner(mps, mps)))
end

"""
    normalize_mps!(mps::MPS) -> MPS

Normalize an MPS in place.

# Arguments
- `mps::MPS`: Matrix Product State (modified in place)

# Returns
- `MPS`: The normalized MPS
"""
function normalize_mps!(mps::MPS)::MPS
    n = get_mps_norm(mps)
    if n > 1e-15
        mps[1] = mps[1] / n
    end
    return mps
end

#==============================================================================#
# GATE CONSTRUCTION
#==============================================================================#

"""
    pauli_to_itensor(σ::Symbol, s::Index) -> ITensor

Build a single-site Pauli gate as an ITensor.

# Arguments
- `σ::Symbol`: Pauli symbol (:I, :X, :Y, or :Z)
- `s::Index`: Site index

# Returns
- `ITensor`: 2×2 Pauli gate tensor

# Convention
The ITensor is constructed with indices (s', dag(s)) where s' is the output
(primed) index and dag(s) is the input index.
"""
function pauli_to_itensor(σ::Symbol, s::Index)::ITensor
    mat = pauli_matrix(σ)
    return ITensor(mat, s', dag(s))
end

"""
    rotation_to_itensor(axis::Symbol, θ::Real, s::Index) -> ITensor

Build a single-site rotation gate R_axis(θ) as an ITensor.

R_axis(θ) = exp(-i θ σ_axis / 2)

# Arguments
- `axis::Symbol`: Rotation axis (:X, :Y, or :Z)
- `θ::Real`: Rotation angle in radians (any real number type, including π)
- `s::Index`: Site index

# Returns
- `ITensor`: 2×2 rotation gate tensor
"""
function rotation_to_itensor(axis::Symbol, θ::Real, s::Index)::ITensor
    mat = rotation_matrix(axis, Float64(θ))
    return ITensor(mat, s', dag(s))
end

"""
    identity_itensor(s::Index) -> ITensor

Build a single-site identity gate as an ITensor.

# Arguments
- `s::Index`: Site index

# Returns
- `ITensor`: 2×2 identity tensor
"""
function identity_itensor(s::Index)::ITensor
    mat = ComplexF64[1 0; 0 1]
    return ITensor(mat, s', dag(s))
end

#==============================================================================#
# SINGLE-SITE GATE APPLICATION
#==============================================================================#

"""
    apply_single_site_gate!(mps::MPS, gate::ITensor, site::Int) -> MPS

Apply a single-site gate to an MPS at the specified site.

# Arguments
- `mps::MPS`: Matrix Product State (modified)
- `gate::ITensor`: Single-site gate tensor
- `site::Int`: Site index (1-indexed)

# Returns
- `MPS`: The modified MPS
"""
function apply_single_site_gate!(mps::MPS, gate::ITensor, site::Int)::MPS
    # Contract gate with MPS tensor at site
    new_tensor = gate * mps[site]
    
    # Remove prime from site index
    new_tensor = noprime(new_tensor)
    
    # Update MPS
    mps[site] = new_tensor
    
    return mps
end

"""
    apply_local_rotation!(mps::MPS, sites::AbstractVector, qubit::Int,
                          axis::Symbol, θ::Real) -> MPS

Apply a single-qubit rotation to an MPS.

# Arguments
- `mps::MPS`: Matrix Product State (modified)
- `sites::AbstractVector`: Site indices
- `qubit::Int`: Target qubit (1-indexed)
- `axis::Symbol`: Rotation axis (:X, :Y, or :Z)
- `θ::Real`: Rotation angle (any real number type, including π)

# Returns
- `MPS`: The modified MPS
"""
function apply_local_rotation!(mps::MPS, sites::AbstractVector, qubit::Int,
                               axis::Symbol, θ::Real)::MPS
    gate = rotation_to_itensor(axis, Float64(θ), sites[qubit])
    return apply_single_site_gate!(mps, gate, qubit)
end

#==============================================================================#
# PAULI STRING APPLICATION
#==============================================================================#

"""
    apply_pauli_to_mps!(mps::MPS, sites::AbstractVector, σ::Symbol, qubit::Int) -> MPS

Apply a single-qubit Pauli to an MPS.

# Arguments
- `mps::MPS`: Matrix Product State (modified)
- `sites::AbstractVector`: Site indices
- `σ::Symbol`: Pauli symbol (:I, :X, :Y, or :Z)
- `qubit::Int`: Target qubit (1-indexed)

# Returns
- `MPS`: The modified MPS
"""
function apply_pauli_to_mps!(mps::MPS, sites::AbstractVector, σ::Symbol, qubit::Int)::MPS
    if σ == :I
        return mps  # Identity does nothing
    end
    
    gate = pauli_to_itensor(σ, sites[qubit])
    return apply_single_site_gate!(mps, gate, qubit)
end

"""
    apply_pauli_string!(mps::MPS, P::PauliOperator, sites::Vector{<:Index}) -> MPS

Apply a Pauli string operator to an MPS.

This applies each single-qubit Pauli in the string, then multiplies by the phase.

# Arguments
- `mps::MPS`: Matrix Product State (modified)
- `P::PauliOperator`: Pauli string from QuantumClifford
- `sites::Vector{<:Index}`: Site indices

# Returns
- `MPS`: The modified MPS (P|ψ⟩)

# Example
```julia
P = P"XYZ"  # X ⊗ Y ⊗ Z
apply_pauli_string!(mps, P, sites)  # Applies X⊗Y⊗Z to mps
```
"""
function apply_pauli_string!(mps::MPS, P::PauliOperator, sites::AbstractVector)::MPS
    n = length(sites)
    
    # Apply each single-qubit Pauli
    for j in 1:n
        σ = get_pauli_at(P, j)
        if σ != :I
            gate = pauli_to_itensor(σ, sites[j])
            apply_single_site_gate!(mps, gate, j)
        end
    end
    
    # Apply overall phase
    phase = get_pauli_phase(P)
    if phase != 1.0 + 0.0im
        mps[1] = phase * mps[1]
    end
    
    return mps
end

"""
    apply_pauli_string_to_copy(mps::MPS, P::PauliOperator, sites::Vector{<:Index}) -> MPS

Apply a Pauli string operator to a copy of an MPS.

Same as `apply_pauli_string!` but doesn't modify the input MPS.

# Arguments
- `mps::MPS`: Matrix Product State (not modified)
- `P::PauliOperator`: Pauli string from QuantumClifford
- `sites::Vector{<:Index}`: Site indices

# Returns
- `MPS`: New MPS representing P|ψ⟩
"""
function apply_pauli_string_to_copy(mps::MPS, P::PauliOperator, sites::AbstractVector)::MPS
    mps_copy = copy(mps)
    return apply_pauli_string!(mps_copy, P, sites)
end

#==============================================================================#
# TWISTED ROTATION APPLICATION
#==============================================================================#

"""
    apply_twisted_rotation!(mps::MPS, sites::Vector{<:Index}, P::PauliOperator,
                            θ::Real; max_bond::Int=1024, cutoff::Float64=1e-15) -> MPS

Apply a twisted rotation (αI + βP) to an MPS.

For rotation R_P(θ) = exp(-i θ P / 2) = cos(θ/2)I - i sin(θ/2)P,
this computes: |ψ'⟩ = α|ψ⟩ + β(P|ψ⟩) where α = cos(θ/2), β = -i sin(θ/2).

This is the core operation for non-Clifford gates in CAMPS when OFD
cannot be applied. It increases bond dimension (doubles in worst case).

# Arguments
- `mps::MPS`: Matrix Product State (modified)
- `sites::Vector{<:Index}`: Site indices
- `P::PauliOperator`: Twisted Pauli operator
- `θ::Real`: Rotation angle (any real number type, including π)
- `max_bond::Int`: Maximum bond dimension for truncation
- `cutoff::Float64`: Singular value cutoff for truncation

# Returns
- `MPS`: The modified MPS

# Note
Bond dimension can increase significantly. Use truncation parameters to control.
"""
function apply_twisted_rotation!(mps::MPS, sites::AbstractVector, P::PauliOperator,
                                  θ::Real; max_bond::Int=1024, cutoff::Float64=1e-15)::MPS
    # Compute rotation coefficients
    α, β = rotation_coefficients(Float64(θ))

    # If β ≈ 0, rotation is approximately identity
    if abs(β) < 1e-14
        mps[1] = α * mps[1]
        return mps
    end

    # Compute P|ψ⟩
    mps_P = apply_pauli_string_to_copy(mps, P, sites)

    # Compute α|ψ⟩ + β(P|ψ⟩) using ITensor's add function
    # Scale the MPS tensors
    mps[1] = α * mps[1]
    mps_P[1] = β * mps_P[1]

    # Add the two MPS
    result = add(mps, mps_P; cutoff=cutoff, maxdim=max_bond)

    # Copy result back to mps
    for i in 1:length(mps)
        mps[i] = result[i]
    end

    return mps
end

"""
    apply_twisted_rotation_to_copy(mps::MPS, sites::Vector{<:Index}, P::PauliOperator,
                                    θ::Real; max_bond::Int=1024, cutoff::Float64=1e-15) -> MPS

Apply a twisted rotation to a copy of an MPS.

Same as `apply_twisted_rotation!` but doesn't modify the input MPS.

# Arguments
- `mps::MPS`: Matrix Product State (not modified)
- `sites::Vector{<:Index}`: Site indices
- `P::PauliOperator`: Twisted Pauli operator
- `θ::Real`: Rotation angle (any real number type, including π)
- `max_bond::Int`: Maximum bond dimension
- `cutoff::Float64`: Singular value cutoff

# Returns
- `MPS`: New MPS representing the rotated state
"""
function apply_twisted_rotation_to_copy(mps::MPS, sites::AbstractVector, P::PauliOperator,
                                         θ::Real; max_bond::Int=1024, cutoff::Float64=1e-15)::MPS
    mps_copy = copy(mps)
    return apply_twisted_rotation!(mps_copy, sites, P, θ; max_bond=max_bond, cutoff=cutoff)
end

#==============================================================================#
# MPS TRUNCATION
#==============================================================================#

"""
    truncate_mps!(mps::MPS; max_bond::Int=1024, cutoff::Float64=1e-15) -> MPS

Truncate an MPS to a specified maximum bond dimension.

# Arguments
- `mps::MPS`: Matrix Product State (modified)
- `max_bond::Int`: Maximum bond dimension
- `cutoff::Float64`: Singular value cutoff

# Returns
- `MPS`: The truncated MPS
"""
function truncate_mps!(mps::MPS; max_bond::Int=1024, cutoff::Float64=1e-15)::MPS
    truncate!(mps; maxdim=max_bond, cutoff=cutoff)
    return mps
end

#==============================================================================#
# ENTANGLEMENT ENTROPY
#==============================================================================#

"""
    entanglement_entropy(mps::MPS, bond::Int) -> Float64

Compute the von Neumann entanglement entropy at the specified bond.

The entropy is computed as S = -Σᵢ λᵢ² log(λᵢ²) where λᵢ are the singular
values at the bond.

# Arguments
- `mps::MPS`: Matrix Product State
- `bond::Int`: Bond index (between sites bond and bond+1)

# Returns
- `Float64`: Entanglement entropy in nats (natural log)

# Example
```julia
S = entanglement_entropy(mps, 3)  # Entropy at bond between sites 3 and 4
```
"""
function entanglement_entropy(mps::MPS, bond::Int)::Float64
    n = length(mps)

    # Validate bond index
    if bond < 1 || bond >= n
        throw(ArgumentError("bond must be in [1, n-1] where n=$n"))
    end

    # Orthogonalize MPS to the bond (brings into mixed canonical form centered at bond)
    psi = orthogonalize(mps, bond)

    # Get link index at the bond
    link = linkind(psi, bond)

    # If no link (product state), entropy is 0
    if isnothing(link)
        return 0.0
    end

    # Contract the two tensors at the bond
    combined = psi[bond] * psi[bond + 1]

    # Get the indices for SVD
    # We want to split at the bond, so left indices are from site bond
    left_inds = uniqueinds(psi[bond], psi[bond + 1])

    # Perform SVD - ITensor returns diagonal S tensor
    U, S, V = svd(combined, left_inds)

    # Extract singular values using ITensor's storage_svs function
    # or by converting S to a dense array and extracting diagonal
    singular_values = extract_singular_values(S)

    # Filter out near-zero singular values
    singular_values = filter(sv -> sv > 1e-15, singular_values)

    if isempty(singular_values)
        return 0.0
    end

    # Normalize singular values (they should already be normalized for proper MPS)
    total = sum(sv^2 for sv in singular_values)
    if total < 1e-15
        return 0.0
    end

    # Compute entropy: S = -Σ p_i log(p_i) where p_i = λ_i²
    entropy = 0.0
    for sv in singular_values
        p = (sv^2) / total
        if p > 1e-15
            entropy -= p * log(p)
        end
    end

    return entropy
end

"""
    extract_singular_values(S::ITensor) -> Vector{Float64}

Extract singular values from a diagonal ITensor returned by SVD.

# Arguments
- `S::ITensor`: Diagonal tensor from SVD

# Returns
- `Vector{Float64}`: Vector of singular values
"""
function extract_singular_values(S::ITensor)::Vector{Float64}
    s_inds = inds(S)

    # S should have exactly 2 indices (left and right link)
    if length(s_inds) != 2
        # Single element or unusual case
        return [abs(S[])]
    end

    # Get the dimension (both indices should have same dim for diagonal tensor)
    s_dim = minimum(dim.(s_inds))

    # Extract diagonal elements
    singular_values = Float64[]
    for i in 1:s_dim
        # Access diagonal element S[i,i]
        val = abs(S[s_inds[1] => i, s_inds[2] => i])
        push!(singular_values, val)
    end

    return singular_values
end

"""
    bond_dimension_at(mps::MPS, bond::Int) -> Int

Get the bond dimension at a specific bond.

# Arguments
- `mps::MPS`: Matrix Product State
- `bond::Int`: Bond index (between sites bond and bond+1)

# Returns
- `Int`: Bond dimension at the specified bond
"""
function bond_dimension_at(mps::MPS, bond::Int)::Int
    n = length(mps)
    if bond < 1 || bond >= n
        throw(ArgumentError("bond must be in [1, n-1] where n=$n"))
    end

    link = linkind(mps, bond)
    if isnothing(link)
        return 1  # Product state has virtual bond dim 1
    end
    return dim(link)
end

"""
    all_bond_dimensions(mps::MPS) -> Vector{Int}

Get all bond dimensions of an MPS.

# Arguments
- `mps::MPS`: Matrix Product State

# Returns
- `Vector{Int}`: Bond dimensions [χ₁, χ₂, ..., χ_{n-1}]
"""
function all_bond_dimensions(mps::MPS)::Vector{Int}
    n = length(mps)
    if n <= 1
        return Int[]
    end
    return [bond_dimension_at(mps, bond) for bond in 1:(n-1)]
end

"""
    entanglement_entropy_all_bonds(mps::MPS) -> Vector{Float64}

Compute entanglement entropy at all bonds.

# Arguments
- `mps::MPS`: Matrix Product State

# Returns
- `Vector{Float64}`: Entropy at each bond [S₁, S₂, ..., S_{n-1}]
"""
function entanglement_entropy_all_bonds(mps::MPS)::Vector{Float64}
    n = length(mps)
    return [entanglement_entropy(mps, bond) for bond in 1:(n-1)]
end

"""
    max_entanglement_entropy(mps::MPS) -> Float64

Get the maximum entanglement entropy across all bonds.

# Arguments
- `mps::MPS`: Matrix Product State

# Returns
- `Float64`: Maximum entropy
"""
function max_entanglement_entropy(mps::MPS)::Float64
    entropies = entanglement_entropy_all_bonds(mps)
    return isempty(entropies) ? 0.0 : maximum(entropies)
end

#==============================================================================#
# MPS SAMPLING
#==============================================================================#

"""
    sample_mps(mps::MPS) -> Vector{Int}

Sample a computational basis state from the MPS probability distribution.

# Arguments
- `mps::MPS`: Matrix Product State

# Returns
- `Vector{Int}`: Sampled bitstring (0s and 1s)

# Note
ITensor's sample returns 1-indexed values (1 or 2), which we convert to (0 or 1).
"""
function sample_mps(mps::MPS)::Vector{Int}
    # ITensor's sample requires a well-defined orthogonality center
    # Orthogonalize to site 1 before sampling
    mps_ortho = orthogonalize(mps, 1)

    # ITensor's sample returns values in {1, 2} for Qubit sites
    raw_sample = sample(mps_ortho)

    # Convert to {0, 1}
    return [s - 1 for s in raw_sample]
end

"""
    sample_mps_multiple(mps::MPS, num_samples::Int) -> Vector{Vector{Int}}

Sample multiple computational basis states from the MPS.

# Arguments
- `mps::MPS`: Matrix Product State
- `num_samples::Int`: Number of samples to draw

# Returns
- `Vector{Vector{Int}}`: Vector of sampled bitstrings
"""
function sample_mps_multiple(mps::MPS, num_samples::Int)::Vector{Vector{Int}}
    return [sample_mps(mps) for _ in 1:num_samples]
end

#==============================================================================#
# MPS INNER PRODUCTS
#==============================================================================#

"""
    mps_overlap(mps1::MPS, mps2::MPS) -> ComplexF64

Compute the overlap ⟨ψ₁|ψ₂⟩ between two MPS.

# Arguments
- `mps1::MPS`: First MPS (bra)
- `mps2::MPS`: Second MPS (ket)

# Returns
- `ComplexF64`: Overlap ⟨ψ₁|ψ₂⟩
"""
function mps_overlap(mps1::MPS, mps2::MPS)::ComplexF64
    return inner(mps1, mps2)
end

"""
    mps_probability(mps::MPS, bitstring::Vector{Int}, sites::Vector{<:Index}) -> Float64

Compute the probability of measuring a specific bitstring.

# Arguments
- `mps::MPS`: Matrix Product State
- `bitstring::Vector{Int}`: Bitstring (0s and 1s)
- `sites::Vector{<:Index}`: Site indices

# Returns
- `Float64`: Probability |⟨bitstring|ψ⟩|²
"""
function mps_probability(mps::MPS, bitstring::Vector{Int}, sites::AbstractVector)::Float64
    n = length(mps)
    length(bitstring) == n || throw(ArgumentError("bitstring length must match MPS length"))
    
    # Create product state MPS for the bitstring
    # ITensor uses "0" and "1" state labels
    state_labels = [b == 0 ? "0" : "1" for b in bitstring]
    product_mps = MPS(sites, state_labels)
    
    # Compute overlap
    amp = inner(product_mps, mps)
    
    return abs2(amp)
end

"""
    mps_amplitude(mps::MPS, bitstring::Vector{Int}, sites::Vector{<:Index}) -> ComplexF64

Compute the amplitude ⟨bitstring|ψ⟩ for a specific bitstring.

# Arguments
- `mps::MPS`: Matrix Product State
- `bitstring::Vector{Int}`: Bitstring (0s and 1s)
- `sites::Vector{<:Index}`: Site indices

# Returns
- `ComplexF64`: Amplitude ⟨bitstring|ψ⟩
"""
function mps_amplitude(mps::MPS, bitstring::Vector{Int}, sites::AbstractVector)::ComplexF64
    n = length(mps)
    length(bitstring) == n || throw(ArgumentError("bitstring length must match MPS length"))

    # Convert sites to proper Index vector if needed
    sites_vec = Vector{Index}([s for s in sites])

    # Create product state MPS for the bitstring
    state_labels = [b == 0 ? "0" : "1" for b in bitstring]
    product_mps = MPS(sites_vec, state_labels)

    # Compute overlap
    return inner(product_mps, mps)
end

#==============================================================================#
# TWO-QUBIT GATES (for OBD)
#==============================================================================#

"""
    apply_two_qubit_gate!(mps::MPS, gate::ITensor, site1::Int, site2::Int;
                          max_bond::Int=1024, cutoff::Float64=1e-15) -> MPS

Apply a two-qubit gate to an MPS.

# Arguments
- `mps::MPS`: Matrix Product State (modified)
- `gate::ITensor`: Two-qubit gate tensor
- `site1::Int`: First site index
- `site2::Int`: Second site index
- `max_bond::Int`: Maximum bond dimension after SVD
- `cutoff::Float64`: Singular value cutoff

# Returns
- `MPS`: The modified MPS

# Note
Sites must be adjacent (|site1 - site2| = 1).
"""
function apply_two_qubit_gate!(mps::MPS, gate::ITensor, site1::Int, site2::Int;
                                max_bond::Int=1024, cutoff::Float64=1e-15)::MPS
    # Ensure site1 < site2
    if site1 > site2
        site1, site2 = site2, site1
    end
    
    # Sites must be adjacent for simple application
    if site2 != site1 + 1
        throw(ArgumentError("Two-qubit gate must be applied to adjacent sites"))
    end
    
    # Contract gate with MPS tensors
    combined = mps[site1] * mps[site2]
    combined = gate * combined
    combined = noprime(combined)
    
    # SVD to split back into two tensors
    left_inds = uniqueinds(mps[site1], mps[site2])
    U, S, V = svd(combined, left_inds; maxdim=max_bond, cutoff=cutoff)
    
    # Update MPS tensors
    mps[site1] = U
    mps[site2] = S * V
    
    return mps
end

"""
    matrix_to_two_qubit_itensor(U::Matrix{ComplexF64}, s1::Index, s2::Index) -> ITensor

Convert a 4×4 unitary matrix to a two-qubit ITensor.

# Arguments
- `U::Matrix{ComplexF64}`: 4×4 unitary matrix
- `s1::Index`: First site index
- `s2::Index`: Second site index

# Returns
- `ITensor`: Two-qubit gate tensor

# Convention
Matrix ordering: |00⟩, |01⟩, |10⟩, |11⟩ (binary, first qubit is MSB)
"""
function matrix_to_two_qubit_itensor(U::Matrix{ComplexF64}, s1::Index, s2::Index)::ITensor
    @assert size(U) == (4, 4) "Matrix must be 4×4"
    
    # Create tensor with proper index structure
    # Indices: (s1', s2', dag(s1), dag(s2))
    # where primed are output, unprimed are input
    return ITensor(U, s2', s1', dag(s2), dag(s1))
end
