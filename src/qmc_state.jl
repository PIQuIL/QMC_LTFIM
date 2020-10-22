function init_op_list(length)
    operator_list::Vector{NTuple{2,Int}} = [(0, 0) for _ in 1:length]
    return operator_list
end

function resize_op_list!(operator_list::Vector{NTuple{2, Int}}, new_size::Int)
    len = length(operator_list)

    if len < new_size
        tail = init_op_list(new_size - len)
        append!(operator_list, tail)
    end
end

abstract type AbstractQMCState{D,N,H<:Hamiltonian{D,N}} end

struct BinaryQMCState{N,H} <: AbstractQMCState{2,N,H}
    left_config::BitArray{N}
    right_config::BitArray{N}
    operator_list::Vector{NTuple{2,Int}}
end

function BinaryQMCState(H::Hamiltonian{2,N}, M::Int) where {N}
    BinaryQMCState{N,typeof(H)}(zero(H), zero(H), init_op_list(2*M))
end


struct ClusterData
    linked_list::Vector{Int}
    leg_types::BitVector
    associates::Vector{NTuple{3,Int}}
    flipping_weights::Vector{Float64}
    first::Vector{Int}
    last::Union{Vector{Int}, Nothing}
end