abstract type AbstractIsing{N,O} <: Hamiltonian{2,N,O} end
abstract type AbstractTFIM{N,O} <: AbstractIsing{N,O} end

struct TFIM{N,F,O} <: AbstractTFIM{N,O}
    op_sampler::O
    J::Float64
    h::Float64
    P_normalization::Float64
    Ns::Int
    Nb::Int
    bonds::Vector{NTuple{2,Int}}
    energy_shift::Float64
end


struct ArbitraryInteractionTFIM{N,O} <: AbstractTFIM{N,O}
    op_sampler::O
    J::Matrix{Float64}
    h::Vector{Float64}
    P_normalization::Float64
    Ns::Int
    Nb::Int
end


###############################################################################

# TFIM ops:
#  (-2,i,i) is an off-diagonal site operator h(sigma^+_i + sigma^-_i)
#  (-1,i,i) is a diagonal site operator h
#  (0,0,0) is the identity operator I - NOT USED IN THE PROJECTOR CASE
#  (1,i,j) is a diagonal bond operator J(sigma^z_i sigma^z_j)
@inline isdiagonal(::TFIM, op::NTuple{3,Int}) = @inbounds (op[1] != -2)
@inline isidentity(::TFIM, op::NTuple{3,Int}) = @inbounds (op[1] == 0)
@inline issiteoperator(::TFIM, op::NTuple{3,Int}) = @inbounds (op[1] < 0)
@inline isbondoperator(::TFIM, op::NTuple{3,Int}) = @inbounds (op[1] > 0)
@inline getbondsites(::TFIM, op::NTuple{3, Int}) = @inbounds (op[2], op[3])

@inline makeidentity(::Type{<:TFIM}) = (0, 0, 0)
@inline makediagonalsiteop(::Type{<:TFIM}, i::Int) = (-1, i, i)
@inline makeoffdiagonalsiteop(::Type{<:TFIM}, i::Int) = (-2, i, i)
@inline makeidentity(H::TFIM) = makeidentity(typeof(H))
@inline makediagonalsiteop(H::TFIM, i::Int) = makediagonalsiteop(typeof(H), i)
@inline makeoffdiagonalsiteop(H::TFIM, i::Int) = makeoffdiagonalsiteop(typeof(H), i)

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

function TFIM(bond_spin, Dim::Int, Ns::Int, Nb::Int, h::Float64, J::Float64)
    ops, p = make_prob_vector(bond_spin, Ns, J, h)
    op_sampler = OperatorSampler(ops, p)
    energy_shift = h*Ns + abs(J)*Nb
    F = !signbit(J)  # true if J > 0 (ferromagnetic)
    return TFIM{Dim, F, typeof(op_sampler)}(op_sampler, J, h, sum(p), Ns, Nb, bond_spin, energy_shift)
end

function ArbitraryInteractionTFIM(J::AbstractMatrix{Float64}, h::AbstractVector{Float64}; dim::Int=1)
    @assert length(h) == size(J, 1) == size(J, 2)

    ops, p = make_prob_vector(J, h)
    Ns = length(h)
    Nb = count(isbondoperator, ops)
    op_sampler = HierarchicalOperatorSampler(ops, p)

    return ArbitraryInteractionTFIM{dim, typeof(op_sampler)}(op_sampler, Matrix(J), Vector(h), sum(p), Ns, Nb)
end
