function init_op_list(length, K=2)
    operator_list::Vector{NTuple{K,Int}} = [ntuple(_->0, K) for _ in 1:length]
    return operator_list
end

function resize_op_list!(operator_list::Vector{NTuple{K, Int}}, new_size::Int) where K
    len = length(operator_list)

    if len < new_size
        tail = init_op_list(new_size - len, K)
        append!(operator_list, tail)
    end
end

abstract type AbstractQMCState{D,N,H<:Hamiltonian{D,N}} end

struct BinaryQMCState{N,H,K} <: AbstractQMCState{2,N,H}
    left_config::BitArray{N}
    right_config::BitArray{N}
    operator_list::Vector{NTuple{K,Int}}
end

function BinaryQMCState(H::Hamiltonian{2,N,O}, M::Int) where {N, K, O <: AbstractOperatorSampler{K}}
    BinaryQMCState{N,typeof(H),K}(zero(H), zero(H), init_op_list(2*M, K))
end


struct ClusterData
    linked_list::Vector{Int}
    leg_types::BitVector
    associates::Vector{NTuple{3,Int}}
    flipping_weights::Vector{Float64}
    first::Vector{Int}
    last::Union{Vector{Int}, Nothing}
end