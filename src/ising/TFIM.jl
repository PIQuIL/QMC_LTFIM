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
#  (-2,i) is an off-diagonal site operator h(sigma^+_i + sigma^-_i)
#  (-1,i) is a diagonal site operator h
#  (0,0) is the identity operator I - NOT USED IN THE PROJECTOR CASE
#  (i,j) is a diagonal bond operator J(sigma^z_i sigma^z_j)
@inline isdiagonal(::TFIM, op::NTuple{2,Int}) = @inbounds (op[1] != -2)
@inline isidentity(::TFIM, op::NTuple{2,Int}) = @inbounds (op[1] == 0)
@inline issiteoperator(::TFIM, op::NTuple{2,Int}) = @inbounds (op[1] < 0)
@inline isbondoperator(::TFIM, op::NTuple{2,Int}) = @inbounds (op[1] > 0)
@inline getbondsites(::TFIM, op::NTuple{2, Int}) = op

@inline makeidentity(::TFIM) = (0, 0)
@inline makediagonalsiteop(::TFIM, i::Int) = (-1, i)
@inline makeoffdiagonalsiteop(::TFIM, i::Int) = (-2, i)

###############################################################################

function make_prob_vector(J::AbstractMatrix{T}, hx::AbstractVector{T}) where T
    ops = Vector{NTuple{2, Int}}(undef, 0)
    p = Vector{T}(undef, 0)

    k = 0
    for i in eachindex(hx)
        if hx[i] != 0
            push!(ops, (-1, i))
            push!(p, hx[i])
        end
    end

    # only take J_ij terms from upper diagonal since it's symmetric
    for j in axes(J, 2), i in axes(J, 1)
        if i < j
            if J[i, j] != 0
                push!(ops, op)
                push!(p, 2*abs(J[i, j]))
            end
        end
    end

    return ops, p
end

function make_prob_vector(bond_spins::Vector{NTuple{2,Int}}, Ns::Int, J::T, h::T) where T
    ops = Vector{NTuple{2, Int}}(undef, 0)
    p = Vector{T}(undef, 0)

    if !iszero(h)
        for i in 1:Ns
            push!(ops, (-1, i))
            push!(p, h)
        end
    end

    if !iszero(J)
        for op in bond_spins
            push!(ops, op)
            push!(p, 2*abs(J))
        end
    end

    return ops, p
end

###############################################################################

function TFIM(bond_spin, Dim::Int, Ns::Int, Nb::Int, h::Float64, J::Float64)
    ops, p = make_prob_vector(bond_spin, Ns, J, h)
    op_sampler = OperatorSampler(ops, p)
    F = !signbit(J)  # true if J > 0 (ferromagnetic)
    return TFIM{Dim, F, typeof(op_sampler)}(op_sampler, J, h, sum(p), Ns, Nb, bond_spin)
end

function ArbitraryInteractionTFIM(J::AbstractMatrix{Float64}, h::AbstractVector{Float64}; dim::Int=1)
    @assert length(h) == size(J, 1) == size(J, 2)

    ops, p = make_prob_vector(J, h)
    Ns = length(h)
    Nb = count(op -> op[1] > 0, ops)
    op_sampler = HierarchicalOperatorSampler(ops, p)

    return ArbitraryInteractionTFIM{dim, typeof(op_sampler)}(op_sampler, Matrix(J), Vector(h), sum(p), Ns, Nb)
end
