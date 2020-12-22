using Base.Iterators


abstract type AbstractRydberg{O <: AbstractOperatorSampler} <: AbstractLTFIM{O} end


struct Rydberg{O,M <: UpperTriangular{Float64},U <: AbstractVector{Float64}} <: AbstractRydberg{O}
    op_sampler::O
    V::M
    Ω::U
    δ::Float64
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

# function make_prob_vector(V::UpperTriangular{T}, Ω::AbstractVector{T}, δ::T; epsilon=0.0) where T
#     @assert size(V, 1) == size(V, 2)

#     ops = Vector{NTuple{3, Int}}()
#     p = Vector{T}()
#     energy_shift = zero(T)
#     Ns = size(V, 1)

#     for i in 1:Ns
#         if !iszero(Ω[i])
#             push!(ops, makediagonalsiteop(AbstractLTFIM, i))
#             push!(p, Ω[i])
#             energy_shift += Ω[i]
#         end
#     end

#     bond_spins = Set{NTuple{2,Int}}()
#     for j in axes(V, 2), i in axes(V, 1)
#         if !iszero(V[i, j]) && i < j
#             push!(bond_spins, (i, j))
#         end
#     end

#     δ_b = δ / (Ns - 1)
#     for (site1, site2) in bond_spins
#         # by this point we can assume site1 <= site2
#         #   order:    DD,   DU,   UD, UU
#         p_spins   = [0.0, -δ_b, -δ_b, V[site1, site2] - 2*δ_b]
#         C = abs(min(0, minimum(p_spins))) + epsilon
#         p_spins .+= C

#         for (t, p_t) in enumerate(p_spins)
#             if !iszero(p_t)
#                 push!(ops, (t, site1, site2))
#                 push!(p, p_t)
#             end
#         end

#         energy_shift += C
#     end

#     return ops, p, energy_shift
# end

function make_prob_vector(::Type{<:AbstractRydberg}, V::UpperTriangular{T}, Ω::AbstractVector{T}, δ::T; epsilon=0.0) where T
    @assert length(Ω) == size(V, 1) == size(V, 2)

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
    nbonds_per_site = OrderedDict{Int,Int}(i => 0 for i in 1:Ns)
    for j in axes(V, 2), i in axes(V, 1)
        if i < j && !iszero(V[i, j])
            push!(bond_spins, (i, j))
            nbonds_per_site[i] += 1
            nbonds_per_site[j] += 1
        end
    end

    fictitious_bonds = Set{NTuple{2,Int}}()
    if !iszero(δ)
        max_nbonds = maximum(values(nbonds_per_site))
        underfull = sort(
            filter(pair -> pair[2] < max_nbonds, nbonds_per_site),
            byvalue = true,
            order = Base.Order.Reverse
        )
        while !isempty(underfull)
            k = collect(keys(underfull))
            i = k[1]

            l = 2
            j = k[l]
            while true
                j = k[l]
                i, j = (i < j) ? (i, j) : (j, i)
                if !((i, j) in fictitious_bonds)
                    break
                end
                l += 1
            end

            push!(fictitious_bonds, (i, j))
            nbonds_per_site[i] += 1
            nbonds_per_site[j] += 1

            underfull = sort(
                filter(pair -> pair[2] < max_nbonds, nbonds_per_site),
                byvalue = true,
                order = Base.Order.Reverse
            )
        end
    end

    @assert all(x -> x == max_nbonds, values(nbonds_per_site))

    δ_b = (δ * Ns) / (2 * (length(bond_spins) + length(fictitious_bonds)))
    for (site1, site2) in bond_spins
        # by this point we can assume site1 <= site2
        #   order:   DD,     DU,   UD, UU
        p_spins   = -[0.0, -δ_b, -δ_b, V[site1, site2] - 2*δ_b]
        @show (site1, site2)
        @show p_spins
        C = abs(min(0, minimum(p_spins))) + epsilon
        @show C
        p_spins .+= C
        @show p_spins

        for (t, p_t) in enumerate(p_spins)
            if !iszero(p_t)
                push!(ops, (t, site1, site2))
                push!(p, p_t)
            end
        end

        energy_shift += C
    end

    if !iszero(δ)
        for (site1, site2) in fictitious_bonds
            #   order:   DD,      DU,  UD, UU
            p_spins_e = -[0.0, -δ_b, -δ_b, -2*δ_b]
            C_e = abs(min(0, minimum(p_spins_e))) + epsilon
            p_spins_e .+= C_e

            for (t, p_t) in enumerate(p_spins_e)
                if !iszero(p_t)
                    push!(ops, (t, site1, site2))
                    push!(p, p_t)
                end
            end

            energy_shift += C_e
        end
    end

    return ops, p, energy_shift
end


###############################################################################

# function BlockadeRydberg(dims::NTuple{N, Int}, J::Float64, hx::Float64, hz::Float64, pbc=true) where N
# end

# function Rydberg(dims::NTuple{N, Int}, C::Float64, Ω::Float64, δ::Float64, pbc=true) where N
#     if N == 1
#         lat = Chain(dims[1], 1.0, pbc)
#     else
#         error("Unsupported number of dimensions")
#     end

#     @show dims, C, Ω, δ, pbc
#     Ns = nspins(lat)
#     V = zeros(Float64, Ns, Ns)

#     for i in 1:(Ns-1)
#         for j in (i+1):Ns
#             V[i, j] = C / lat.distance_matrix[i, j]^6
#         end
#     end
#     @show V
#     V = UpperTriangular(V)

#     ops, p, energy_shift, Nb = make_prob_vector(V, Ω, δ)
#     op_sampler = ImprovedOperatorSampler(AbstractLTFIM, ops, p)
#     return Rydberg{typeof(op_sampler)}(op_sampler, C, Ω, δ, Ns, Nb, energy_shift)
# end

function Rydberg(dims::NTuple{N, Int}, C::Float64, Ω::Float64, δ::Float64, pbc=true) where N
    @assert pbc == true
    bond_spins, Ns, Nb = lattice_bond_spins(dims, pbc)
    V_, Ω_ = make_uniform_tfim(bond_spins, Ns, -C, Ω)
    ops, p, energy_shift = make_prob_vector(AbstractRydberg, V_, Ω_, δ, epsilon=0.0)
    println(energy_shift)
    op_sampler = ImprovedOperatorSampler(AbstractLTFIM, ops, p)
    return Rydberg{typeof(op_sampler), typeof(V_), typeof(Ω_)}(op_sampler, V_, Ω_, δ, Ns, energy_shift)
end

total_hx(H::Rydberg)::Float64 = sum(H.Ω)
haslongitudinalfield(H::AbstractRydberg) = !iszero(H.δ)
