using Base.Iterators


abstract type AbstractRydberg{O <: AbstractOperatorSampler} <: AbstractLTFIM{O} end


struct Rydberg{O,M <: UpperTriangular{Float64},UΩ <: AbstractVector{Float64}, Uδ <: AbstractVector{Float64}} <: AbstractRydberg{O}
    op_sampler::O
    V::M
    Ω::UΩ
    δ::Uδ
    Ns::Int
    energy_shift::Float64
end


###############################################################################

abstract type Lattice end

struct Rectangle <: Lattice
    nX::Int
    nY::Int
    aX::Float64
    aY::Float64
    distance_matrix::Array{Float64, 2}
end

function Rectangle(nX::Int, nY::Int, aX::Float64, aY::Float64, PBC::Bool)
    N = nX*nY
    distance_matrix = zeros(Float64, N, N)

    for i in 1:(N-1)
        x1 = rem(i, nX) > 0 ? rem(i, nX) : nX
        y1 = rem(i, nX) > 0 ? div(i, nX) + 1 : div(i, nX)

        for j in (i+1):N
            x2 = rem(j, nX) > 0 ? rem(j, nX) : nX
            y2 = rem(j, nX) > 0 ? div(j, nX) + 1 : div(j, nX)

            if PBC
                dy = abs(y1 - y2) > 0.5*nY ? 0.5*nY - rem(abs(y1-y2), 0.5*nY) : abs(y1-y2)
                dx = abs(x1 - x2) > 0.5*nX ? 0.5*nX - rem(abs(x1-x2), 0.5*nX) : abs(x1-x2)
            else
                dy = abs(y1 - y2)
                dx = abs(x1 - x2)
            end

            dx *= aX
            dy *= aY

            distance_matrix[i, j] = sqrt(dy^2 + dx^2)

        end
    end
    return Rectangle(nX, nY, aX, aY, distance_matrix)
end


struct Chain <: Lattice
    nX::Int
    aX::Float64
    distance_matrix::Array{Float64, 2}
end

function Chain(nX::Int, aX::Float64, PBC::Bool)
    distance_matrix = zeros(Float64, nX, nX)

    for i in 1:(nX-1)
        x1 = rem(i, nX) > 0 ? rem(i, nX) : nX

        for j in (i+1):nX
            x2 = rem(j, nX) > 0 ? rem(j, nX) : nX

            if PBC
                dx = abs(x1-x2) > 0.5*nX ? 0.5*nX - rem(abs(x1-x2), 0.5*nX) : abs(x1-x2)
            else
                dx = abs(x1-x2)
            end

            dx *= aX
            distance_matrix[i, j] = dx
        end
    end
    return Chain(nX, aX, distance_matrix)
end

nspins(lattice::Rectangle) = lattice.nX * lattice.nY
nspins(lattice::Chain) = lattice.nX



###############################################################################


function make_prob_vector(::Type{<:AbstractRydberg}, V::UpperTriangular{T}, Ω::AbstractVector{T}, δ::AbstractVector{T}; epsilon=0.0) where T
    @assert length(Ω) == length(δ) == size(V, 1) == size(V, 2)

    ops = Vector{NTuple{3, Int}}()
    p = Vector{T}()
    energy_shift = zero(T)

    for i in eachindex(Ω)
        if !iszero(Ω[i])
            push!(ops, makediagonalsiteop(AbstractLTFIM, i))
            push!(p, Ω[i])
            energy_shift += Ω[i]
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

    n = [0 0; 0 1]
    I = [1 0; 0 1]

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
            if !iszero(p_t)
                push!(ops, (t, site1, site2))
                push!(p, p_t)
            end
        end
    end

    return ops, p, energy_shift
end


###############################################################################

# function BlockadeRydberg(dims::NTuple{N, Int}, J::Float64, hx::Float64, hz::Float64, pbc=true) where N
# end

function Rydberg(dims::NTuple{N, Int}, C::Float64, Ω::Float64, δ::Float64, pbc=true) where N
    if N == 1
        lat = Chain(dims[1], 1.0, pbc)
    elseif N == 2
        lat = Rectangle(dims[1], dims[2], 1.0, 1.0, pbc)
    else
        error("Unsupported number of dimensions")
    end

    Ns = nspins(lat)
    V = zeros(Float64, Ns, Ns)

    @inbounds for i in 1:(Ns-1)
        for j in (i+1):Ns
            V[i, j] = C / lat.distance_matrix[i, j]^6
        end
    end
    V = UpperTriangular(triu!(V, 1))

    return Rydberg(V, Ω*ones(Ns), δ*ones(Ns))
end

function Rydberg(V::AbstractMatrix{T}, Ω::AbstractVector{T}, δ::AbstractVector{T}; epsilon=zero(T)) where T
    ops, p, energy_shift = make_prob_vector(AbstractRydberg, V, Ω, δ, epsilon=epsilon)
    op_sampler = ImprovedOperatorSampler(AbstractLTFIM, ops, p)
    return Rydberg{typeof(op_sampler), typeof(V), typeof(Ω), typeof(δ)}(op_sampler, V, Ω, δ, length(Ω), energy_shift)
end

total_hx(H::Rydberg)::Float64 = sum(H.Ω)
haslongitudinalfield(H::AbstractRydberg) = !iszero(H.δ)
