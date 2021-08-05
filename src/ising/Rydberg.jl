using Base.Iterators


abstract type AbstractRydberg{O <: AbstractOperatorSampler} <: AbstractLTFIM{O} end

struct Rydberg{O,M <: UpperTriangular{Float64},UΩ <: AbstractVector{Float64}, Uδ <: AbstractVector{Float64}, L <: Lattice} <: AbstractRydberg{O}
    op_sampler::O
    V::M
    Ω::UΩ
    δ::Uδ
    lattice::L
    energy_shift::Float64
end
nspins(H::Rydberg) = nspins(H.lattice)

@inline diagonaloperator(::Type{<:AbstractRydberg}) = Diagonal([0, 1])
@inline diagonaloperator(H::AbstractRydberg) = interactionoperator(typeof(H))


function make_prob_vector(H::Type{<:AbstractRydberg}, V::UpperTriangular{T}, Ω::AbstractVector{T}, δ::AbstractVector{T}; epsilon=0.0) where T
    @assert length(Ω) == length(δ) == size(V, 1) == size(V, 2)

    ops = Vector{NTuple{4, Int}}()
    p = Vector{T}()
    energy_shift = zero(T)

    for i in eachindex(Ω)
        if !iszero(Ω[i])
            push!(ops, makediagonalsiteop(AbstractLTFIM, i))
            push!(p, Ω[i] / 2)
            energy_shift += Ω[i] / 2
        end
    end

    Ns = length(Ω)
    bond_spins = Set{NTuple{2,Int}}()
    coordination_numbers = zeros(Int, Ns)
    for j in axes(V, 2), i in axes(V, 1)
        if i < j && !iszero(V[i, j])
            push!(bond_spins, (i, j))
            coordination_numbers[i] += 1
            coordination_numbers[j] += 1
        end
    end

    n = diagonaloperator(H)
    I = Diagonal(LinearAlgebra.I, 2)

    # TODO: add fictitious bonds if there's a z-field on an "unbonded" site
    for (site1, site2) in bond_spins
        # by this point we can assume site1 < site2
        δb1 = δ[site1] / coordination_numbers[site1]
        δb2 = δ[site2] / coordination_numbers[site2]
        local_H = V[site1, site2]*kron(n, n) - δb1*kron(n, I) - δb2*kron(I, n)

        p_spins = -diag(local_H)
        C = abs(min(0, minimum(p_spins))) + epsilon
        p_spins .+= C
        energy_shift += C

        for (t, p_t) in enumerate(p_spins)
            push!(p, p_t)
            push!(ops, (t, length(p), site1, site2))
        end
    end

    return ops, p, energy_shift
end


###############################################################################

# function BlockadeRydberg(dims::NTuple{N, Int}, J::Float64, hx::Float64, hz::Float64, pbc=true) where N
# end

function Rydberg(dims::NTuple{D, Int}, R_b, Ω, δ; pbc=true) where D
    if D == 1
        lat = Chain(dims[1], 1.0, pbc)
    elseif D == 2
        lat = Rectangle(dims[1], dims[2], 1.0, 1.0, pbc)
    else
        error("Unsupported number of dimensions. 1- and 2-dimensional lattices are supported.")
    end
    return Rydberg(lat, R_b, Ω, δ)
end


function Rydberg(lat::Lattice, R_b::Float64, Ω::Float64, δ::Float64)
    Ns = nspins(lat)
    V = zeros(Float64, Ns, Ns)

    @inbounds for i in 1:(Ns-1)
        for j in (i+1):Ns
            # a zero entry in distance_matrix means there should be no bond
            V[i, j] = lat.distance_matrix[i, j] != 0.0 ? Ω * (R_b / lat.distance_matrix[i, j])^6 : 0.0
        end
    end
    V = UpperTriangular(triu!(V, 1))

    return Rydberg(V, Ω*ones(Ns), δ*ones(Ns), lat)
end

function Rydberg(V::AbstractMatrix{T}, Ω::AbstractVector{T}, δ::AbstractVector{T}, lattice::Lattice; epsilon=zero(T)) where T
    ops, p, energy_shift = make_prob_vector(AbstractRydberg, V, Ω, δ, epsilon=epsilon)
    op_sampler = ImprovedOperatorSampler(AbstractLTFIM, ops, p)
    return Rydberg{typeof(op_sampler), typeof(V), typeof(Ω), typeof(δ), typeof(lattice)}(op_sampler, V, Ω, δ, lattice, energy_shift)
end

total_hx(H::Rydberg)::Float64 = sum(H.Ω) / 2
haslongitudinalfield(H::AbstractRydberg) = !iszero(H.δ)
