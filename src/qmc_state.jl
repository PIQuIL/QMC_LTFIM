function init_op_list(length, K=2)
    operator_list::Vector{NTuple{K,Int}} = [ntuple(_->0, K) for _ in 1:length]
    return operator_list
end

abstract type AbstractQMCState{D,N,K} end

abstract type AbstractGroundState{D,N,K} <: AbstractQMCState{D,N,K} end
abstract type AbstractThermalState{D,N,K} <: AbstractQMCState{D,N,K} end



struct BinaryGroundState{N,K} <: AbstractGroundState{2,N,K}
    left_config::BitArray{N}
    right_config::BitArray{N}
    propagated_config::BitArray{N}

    operator_list::Vector{NTuple{K,Int}}

    linked_list::Vector{Int}
    leg_types::BitVector
    associates::Vector{NTuple{3,Int}}
    flipping_weights::Vector{Float64}

    first::Vector{Int}
end


function BinaryGroundState(H::Hamiltonian{2,N,O}, M::Int) where {N, K, O <: AbstractOperatorSampler{K}}
    operator_list = init_op_list(2*M, K)

    len = 2*nspins(H) + 4*length(operator_list)
    linked_list = zeros(Int, len)
    leg_types = falses(len)
    associates = [(0, 0, 0) for _ in 1:len]
    flipping_weights = ones(len)

    first = zeros(Int, nspins(H))

    BinaryGroundState{N,K}(zero(H), zero(H), zero(H),
                           operator_list,
                           linked_list, leg_types, associates, flipping_weights,
                           first)
end


function BinaryGroundState(left_config::BitArray{N}, right_config::BitArray{N}, operator_list::Vector{NTuple{K,Int}}) where {N, K}
    @assert left_config !== right_config "left_config and right_config can't be the same array!"

    len = 2*length(left_config) + 4*length(operator_list)
    linked_list = zeros(Int, len)
    leg_types = falses(len)
    associates = [(0, 0, 0) for _ in 1:len]

    first = zeros(Int, length(left_config))

    BinaryGroundState{N,K}(left_config, right_config, copy(left_config),
                           operator_list,
                           linked_list, leg_types, associates,
                           first)
end


struct BinaryThermalState{N,K} <: AbstractThermalState{2,N,K}
    left_config::BitArray{N}
    right_config::BitArray{N}
    propagated_config::BitArray{N}

    operator_list::Vector{NTuple{K,Int}}

    linked_list::Vector{Int}
    leg_types::BitVector
    associates::Vector{NTuple{3,Int}}
    flipping_weights::Vector{Float64}

    first::Vector{Int}
    last::Vector{Int}
end


function BinaryThermalState(H::Hamiltonian{2,N,O}, cutoff::Int) where {N, K, O <: AbstractOperatorSampler{K}}
    operator_list = init_op_list(cutoff, K)

    len = 4*length(operator_list)
    linked_list = zeros(Int, len)
    leg_types = falses(len)
    associates = [(0, 0, 0) for _ in 1:len]
    flipping_weights = ones(len)

    first = zeros(Int, nspins(H))
    last = zeros(Int, nspins(H))

    BinaryThermalState{N,K}(zero(H), zero(H), zero(H),
                            operator_list,
                            linked_list, leg_types, associates, flipping_weights,
                            first, last)
end


function BinaryThermalState(left_config::BitArray{N}, right_config::BitArray{N}, operator_list::Vector{NTuple{K,Int}}) where {N, K}
    @assert left_config !== right_config "left_config and right_config can't be the same array!"

    len = 4*length(operator_list)
    linked_list = zeros(Int, len)
    leg_types = falses(len)
    associates = [(0, 0, 0) for _ in 1:len]

    first = zeros(Int, length(left_config))
    last = copy(first)

    BinaryThermalState{N,K}(left_config, right_config, copy(left_config),
                            operator_list,
                            linked_list, leg_types, associates,
                            first, last)
end


const BinaryQMCState{N,K} = Union{BinaryGroundState{N,K}, BinaryThermalState{N,K}}
