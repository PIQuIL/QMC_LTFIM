using Base.Iterators


abstract type AbstractLTFIM{N,O} <: AbstractIsing{N,O} end

struct LTFIM{N,O} <: AbstractLTFIM{N,O}
    op_sampler::O
    J::Float64
    hx::Float64
    hz::Float64
    P_normalization::Float64
    Ns::Int
    Nb::Int
    energy_shift::Float64
end

###############################################################################

# LTFIM ops:
#  (-2,i,0) is an off-diagonal site operator h(sigma^+_i + sigma^-_i)
#  (-1,i,0) is a diagonal site operator h
#  (0,0,0) is the identity operator I - NOT USED IN THE PROJECTOR CASE
#  (t,i,j) is a diagonal bond operator J(sigma^z_i sigma^z_j) + hzb(sigma^z_i + sigma^z_j)
#    t denotes the spin config at sites i,j, subtract 1 then convert to binary
#    t = 1 -> 00 -> down-down
#    t = 2 -> 01 -> down-up
#    t = 3 -> 10 -> up-down
#    t = 4 -> 11 -> up-up
#    spin_config(t) = divrem(t - 1, 2)
#
@inline isdiagonal(::LTFIM, op::NTuple{3,Int}) = @inbounds (op[1] != -2)
@inline isidentity(::LTFIM, op::NTuple{3,Int}) = @inbounds (op[1] == 0)
@inline issiteoperator(::LTFIM, op::NTuple{3,Int}) = @inbounds (op[1] < 0)
@inline isbondoperator(::LTFIM, op::NTuple{3,Int}) = @inbounds (op[1] > 0)

@inline getbondsites(::LTFIM, op::NTuple{3, Int}) = @inbounds (op[2], op[3])
@inline getbondtype(::LTFIM, s1::Bool, s2::Bool) = (s1<<1 | s2) + 1

@inline makeidentity(::Type{<:LTFIM}) = (0, 0, 0)
@inline makediagonalsiteop(::Type{<:LTFIM}, i::Int) = (-1, i, i)
@inline makeoffdiagonalsiteop(::Type{<:LTFIM}, i::Int) = (-2, i, i)
@inline makeidentity(H::LTFIM) = makeidentity(typeof(H))
@inline makediagonalsiteop(H::LTFIM, i::Int) = makediagonalsiteop(typeof(H), i)
@inline makeoffdiagonalsiteop(H::LTFIM, i::Int) = makeoffdiagonalsiteop(typeof(H), i)

@inline spin_config(::LTFIM, t::Int)::NTuple{2,Int} = divrem(t - 1, 2)
@inline spin_config(H::LTFIM, op::NTuple{3, Int}) = @inbounds spin_config(H, op[1])

###############################################################################

function make_prob_vector(dims::NTuple{D, Int}, J::T, hx::T, hz::T, pbc=true, epsilon=0.0) where {D, T}
    bond_spins, Ns, Nb = lattice_bond_spins(dims, pbc)
    bond_spins = Set(bond_spins)
    edge_sites = Set{Int}()
    edge_bonds = Set{NTuple{2,Int}}()

    if !pbc
        pbc_s = Set(lattice_bond_spins(dims, true)[1])
        edge_bonds = setdiff(pbc_s, bond_spins)
        edge_sites = Set(flatten(edge_bonds))
    end

    ops = Vector{NTuple{3, Int}}()
    p = Vector{T}()
    energy_shift = Ns*float(hx)

    if !iszero(hx)
        for i in 1:Ns
            push!(ops, makediagonalsiteop(LTFIM, i))
            push!(p, hx)
        end
    end

    if !(iszero(J) && iszero(hz))
        hzb = hz / (2 * D)  # using Nb from the PBC case
        #   order:   DD,        DU,       UD,       UU
        p_spins   = [J - 2*hzb, -J,       -J,       J + 2*hzb]
        p_spins_l = [ -2*hzb, 0, 0, 2*hzb]
        # p_spins_l = [J - 3*hzb, -J - hzb, -J + hzb, J + 3*hzb]
        # p_spins_r = [J - 3*hzb, -J + hzb, -J - hzb, J + 3*hzb]
        C   = abs(min(0, minimum(p_spins))) + epsilon
        C_e = abs(min(0, minimum(p_spins_l))) + epsilon
        p_spins   .+= C
        p_spins_l .+= C_e
        # p_spins_r .+= C_e

        for t in eachindex(p_spins)
            for (site1, site2) in bond_spins
                # if !(site1 in edge_sites || site2 in edge_sites)
                    p_t = p_spins[t]
                    energy_shift += C/4
                # elseif site1 in edge_sites
                #     p_t = p_spins_l[t]
                #     energy_shift += C_e/4
                # elseif site2 in edge_sites
                #     p_t = p_spins_r[t]
                #     energy_shift += C_e/4
                # end
                if !iszero(p_t)
                    push!(ops, (t, site1, site2))
                    push!(p, p_t)
                end
            end
            for (site1, site2) in edge_bonds
                p_t = p_spins_l[t]
                energy_shift += C_e/4
                if !iszero(p_t)
                    push!(ops, (t, site1, site2))
                    push!(p, p_t)
                end
            end
        end

    end

    return ops, p, Ns, Nb, energy_shift
end

###############################################################################

function LTFIM(dims::NTuple{N, Int}, J::Float64, hx::Float64, hz::Float64, pbc=true) where N
    ops, p, Ns, Nb, energy_shift = make_prob_vector(dims, J, hx, hz, pbc)
    op_sampler = OperatorSampler(ops, p)
    return LTFIM{N, typeof(op_sampler)}(op_sampler, J, hx, hz, sum(p), Ns, Nb, energy_shift)
end