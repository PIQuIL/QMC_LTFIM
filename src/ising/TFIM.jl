abstract type AbstractIsing{O} <: Hamiltonian{2,O} end
abstract type AbstractTFIM{O} <: AbstractIsing{O} end

struct TFIM{F,O} <: AbstractTFIM{O}
    op_sampler::O
    J::Float64
    h::Float64
    P_normalization::Float64
    Ns::Int
    Nb::Int
    bonds::Vector{NTuple{2,Int}}
    energy_shift::Float64
end


struct ArbitraryInteractionTFIM{O,M <: AbstractMatrix{Float64},V <: AbstractVector{Float64}} <: AbstractTFIM{O}
    op_sampler::O
    J::M
    h::V
    P_normalization::Float64
    Ns::Int
    Nb::Int
    energy_shift::Float64
end


###############################################################################

# TFIM ops:
#  (-2,i,i) is an off-diagonal site operator h*sigma^x_i
#  (-1,i,i) is a diagonal site operator h
#  (0,0,0) is the identity operator I - NOT USED IN THE PROJECTOR CASE
#  (1,i,j) is a diagonal bond operator J(sigma^z_i sigma^z_j)
@inline isdiagonal(::AbstractTFIM, op::NTuple{3,Int}) = @inbounds (op[1] != -2)
@inline isidentity(::AbstractTFIM, op::NTuple{3,Int}) = @inbounds (op[1] == 0)
@inline issiteoperator(::AbstractTFIM, op::NTuple{3,Int}) = @inbounds (op[1] < 0)
@inline isbondoperator(::AbstractTFIM, op::NTuple{3,Int}) = @inbounds (op[1] > 0)

@inline getbondsites(::AbstractTFIM, op::NTuple{3, Int}) = @inbounds (op[2], op[3])
@inline getbondtype(::AbstractTFIM, s1::Bool, s2::Bool) = 1

@inline makeidentity(::Type{<:AbstractTFIM}) = (0, 0, 0)
@inline makediagonalsiteop(::Type{<:AbstractTFIM}, i::Int) = (-1, i, i)
@inline makeoffdiagonalsiteop(::Type{<:AbstractTFIM}, i::Int) = (-2, i, i)
@inline makeidentity(H::AbstractTFIM) = makeidentity(typeof(H))
@inline makediagonalsiteop(H::AbstractTFIM, i::Int) = makediagonalsiteop(typeof(H), i)
@inline makeoffdiagonalsiteop(H::AbstractTFIM, i::Int) = makeoffdiagonalsiteop(typeof(H), i)

###############################################################################

function make_prob_vector(J::AbstractMatrix{T}, hx::AbstractVector{T}) where T
    ops = Vector{NTuple{3, Int}}(undef, 0)
    p = Vector{T}(undef, 0)

    k = 0
    for i in eachindex(hx)
        if hx[i] != 0
            push!(ops, makediagonalsiteop(TFIM, i))
            push!(p, hx[i])
        end
    end

    # only take J_ij terms from upper diagonal since it's symmetric
    for j in axes(J, 2), i in axes(J, 1)
        if i <= j
            if J[i, j] != 0
                push!(ops, (1, i, j))
                push!(p, 2*abs(J[i, j]))
            end
        end
    end

    return ops, p
end

function make_prob_vector(bond_spins::Vector{NTuple{2,Int}}, Ns::Int, J::T, h::T) where T
    ops = Vector{NTuple{3, Int}}(undef, 0)
    p = Vector{T}(undef, 0)

    if !iszero(h)
        for i in 1:Ns
            push!(ops, makediagonalsiteop(TFIM, i))
            push!(p, h)
        end
    end

    if !iszero(J)
        for op in bond_spins
            site1, site2 = op
            site1, site2 = (site1 <= site2) ? (site1, site2) : (site2, site1)
            push!(ops, (1, site1, site2))
            push!(p, 2*abs(J))
        end
    end

    return ops, p
end

###############################################################################

function TFIM(bond_spin, Ns::Int, Nb::Int, h::Float64, J::Float64)
    ops, p = make_prob_vector(bond_spin, Ns, J, h)
    op_sampler = OperatorSampler(ops, p)
    energy_shift = h*Ns + abs(J)*Nb
    F = !signbit(J)  # true if J > 0 (ferromagnetic)
    return TFIM{F, typeof(op_sampler)}(op_sampler, J, h, sum(p), Ns, Nb, bond_spin, energy_shift)
end

total_hx(H::TFIM) = H.h * nspins(H)
total_hx(H::ArbitraryInteractionTFIM) = sum(H.h)

function energy(::BinaryGroundState, H::AbstractIsing, ns::Vector{<:Real})
    hx = total_hx(H)

    if !iszero(hx)
        E = -hx * jackknife(inv, ns)
    else
        E = zero(H.energy_shift)
    end

    return H.energy_shift + E
end

function ArbitraryInteractionTFIM(J::AbstractMatrix{Float64}, h::AbstractVector{Float64})
    @assert length(h) == size(J, 1) == size(J, 2)

    ops, p = make_prob_vector(J, h)
    Ns = length(h)
    Nb = count(isbondoperator, ops)
    op_sampler = HierarchicalOperatorSampler(ops, p)

    energy_shift = sum(h) + sum(abs, J)

    return ArbitraryInteractionTFIM{typeof(op_sampler), typeof(J), typeof(h)}(
        op_sampler, J, h, sum(p), Ns, Nb, energy_shift
    )
end
