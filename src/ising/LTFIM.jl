using Base.Iterators


abstract type AbstractLTFIM{O <: AbstractOperatorSampler} <: AbstractIsing{O} end

struct GeneralLTFIM{O,M <: UpperTriangular{Float64},V <: AbstractVector{Float64}} <: AbstractLTFIM{O}
    op_sampler::O
    J::M
    hx::V
    hz::Float64
    Ns::Int
    energy_shift::Float64
end


struct LTFIM{O} <: AbstractLTFIM{O}
    op_sampler::O
    J::Float64
    hx::Float64
    hz::Float64
    Ns::Int
    energy_shift::Float64
end
###############################################################################

# LTFIM ops:
#  (-2,i,i) is an off-diagonal site operator h(sigma^+_i + sigma^-_i)
#  (-1,i,i) is a diagonal site operator h
#  (0,0,0) is the identity operator I - NOT USED IN THE PROJECTOR CASE
#  (t,i,j) is a diagonal bond operator J(sigma^z_i sigma^z_j) + hzb(sigma^z_i + sigma^z_j)
#    t denotes the spin config at sites i,j, subtract 1 then convert to binary
#    t = 1 -> 00 -> down-down
#    t = 2 -> 01 -> down-up
#    t = 3 -> 10 -> up-down
#    t = 4 -> 11 -> up-up
#    spin_config(t) = divrem(t - 1, 2)
@inline getbondtype(::AbstractLTFIM, s1::Bool, s2::Bool) = (s1<<1 | s2) + 1
@inline spin_config(::AbstractLTFIM, t::Int)::NTuple{2,Int} = divrem(t - 1, 2)
@inline spin_config(H::AbstractLTFIM, op::NTuple{3, Int}) = @inbounds spin_config(H, op[1])

###############################################################################

function make_prob_vector(J::UpperTriangular{T}, hx::AbstractVector{T}, hz::T; epsilon=0.0) where T
    @assert length(hx) == size(J, 1) == size(J, 2)

    ops = Vector{NTuple{3, Int}}()
    p = Vector{T}()
    energy_shift = zero(T)

    for i in eachindex(hx)
        if !iszero(hx[i])
            push!(ops, makediagonalsiteop(AbstractLTFIM, i))
            push!(p, hx[i])
            energy_shift += hx[i]
        end
    end

    Ns = length(hx)
    bond_spins = Set{NTuple{2,Int}}()
    nbonds_per_site = OrderedDict{Int,Int}(i => 0 for i in 1:Ns)
    for j in axes(J, 2), i in axes(J, 1)
        if i < j && !iszero(J[i, j])
            site1, site2 = (i, j)
            push!(bond_spins, (site1, site2))
            nbonds_per_site[i] += 1
            nbonds_per_site[j] += 1
        end
    end

    fictitious_bonds = Set{NTuple{2,Int}}()
    if !iszero(hz)
        max_nbonds = maximum(values(nbonds_per_site))
        underfull = sort!(
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

            underfull = sort!(
                filter(pair -> pair[2] < max_nbonds, nbonds_per_site),
                byvalue = true,
                order = Base.Order.Reverse
            )
        end
    end

    hzb = (hz * Ns) / (2 * (length(bond_spins) + length(fictitious_bonds)))
    for (site1, site2) in bond_spins
        # by this point we can assume site1 <= site2
        J_ = -J[site1, site2]
        #   order:   DD,          DU,  UD, UU
        p_spins   = [J_ - 2*hzb, -J_, -J_, J_ + 2*hzb]
        C = abs(min(0, minimum(p_spins))) + epsilon
        p_spins .+= C

        for (t, p_t) in enumerate(p_spins)
            if !iszero(p_t)
                push!(ops, (t, site1, site2))
                push!(p, p_t)
            end
        end

        energy_shift += C
    end

    if !iszero(hz)
        for (site1, site2) in fictitious_bonds
            #   order:   DD,      DU,  UD, UU
            p_spins_e = [-2*hzb, 0.0, 0.0, 2*hzb]
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

function LTFIM(dims::NTuple{N, Int}, J::Float64, hx::Float64, hz::Float64, pbc=true) where N
    bond_spins, Ns, Nb = lattice_bond_spins(dims, pbc)
    J_, hx_ = make_uniform_tfim(bond_spins, Ns, J, hx)
    ops, p, energy_shift = make_prob_vector(J_, hx_, hz)
    op_sampler = ImprovedOperatorSampler(AbstractLTFIM, ops, p)
    return LTFIM{typeof(op_sampler)}(op_sampler, J, hx, hz, Ns, Nb, energy_shift)
end

function GeneralLTFIM(dims::NTuple{N, Int}, J::Float64, hx::Float64, hz::Float64, pbc=true) where N
    bond_spins, Ns, Nb = lattice_bond_spins(dims, pbc)
    J_, hx_ = make_uniform_tfim(bond_spins, Ns, J, hx)
    ops, p, energy_shift = make_prob_vector(J_, hx_, hz)
    op_sampler = ImprovedOperatorSampler(AbstractLTFIM, ops, p)
    return GeneralLTFIM{typeof(op_sampler),typeof(J_),typeof(hx_)}(op_sampler, J_, hx_, hz, Ns, Nb, energy_shift)
end

total_hx(H::GeneralLTFIM)::Float64 = sum(H.hx)
total_hx(H::LTFIM)::Float64 = H.hx * nspins(H)
haslongitudinalfield(H::AbstractLTFIM) = !iszero(H.hz)