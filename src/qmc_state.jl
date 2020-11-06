function init_op_list(length, K=2)
    operator_list::Vector{NTuple{K,Int}} = [ntuple(_->0, K) for _ in 1:length]
    return operator_list
end

abstract type AbstractQMCState{T,K} end

abstract type AbstractGroundState{T,K} <: AbstractQMCState{T,K} end
abstract type AbstractThermalState{T,K} <: AbstractQMCState{T,K} end



struct BinaryGroundState{K,BV <: AbstractVector{Bool}} <: AbstractGroundState{Bool,K}
    left_config::BV
    right_config::BV
    propagated_config::BV

    operator_list::Vector{NTuple{K,Int}}

    linked_list::Vector{Int}
    leg_types::BV
    associates::Vector{NTuple{3,Int}}
    flipping_weights::Vector{Float64}

    in_cluster::BV
    cstack::PushVector{Int, Vector{Int}}
    current_cluster::PushVector{Int, Vector{Int}}

    first::Vector{Int}
end

function BinaryGroundState(left_config::BV, right_config::BV, operator_list::Vector{NTuple{K,Int}}) where {K, BV <: AbstractVector{Bool}}
    @assert left_config !== right_config "left_config and right_config can't be the same array!"
    @assert size(left_config) === size(right_config) "left_config and right_config must be the same size!"

    len = 2*length(left_config) + 4*length(operator_list)
    link_list = zeros(Int, len)
    leg_types = zeros(Bool, len)
    associates = [(0, 0, 0) for _ in 1:len]
    flipping_weights = zeros(len)

    in_cluster = zeros(Bool, len)
    cstack = PushVector{Int}(nextpow(2, length(left_config)))
    current_cluster = PushVector{Int}(nextpow(2, length(left_config)))

    first = zeros(Int, length(left_config))

    BinaryGroundState{K,BV}(
        left_config, right_config, copy(left_config),
        operator_list,
        link_list, leg_types, associates, flipping_weights,
        in_cluster, cstack, current_cluster,
        first
    )
end


struct BinaryThermalState{K,BV <: AbstractVector{Bool}} <: AbstractThermalState{Bool,K}
    left_config::BV
    right_config::BV
    propagated_config::BV

    operator_list::Vector{NTuple{K,Int}}

    linked_list::Vector{Int}
    leg_types::BV
    associates::Vector{NTuple{3,Int}}
    flipping_weights::Vector{Float64}

    in_cluster::BV
    cstack::PushVector{Int, Vector{Int}}
    current_cluster::PushVector{Int, Vector{Int}}

    first::Vector{Int}
    last::Vector{Int}
end

function BinaryThermalState(left_config::BV, right_config::BV, operator_list::Vector{NTuple{K,Int}}) where {K, BV <: AbstractVector{Bool}}
    @assert left_config !== right_config "left_config and right_config can't be the same array!"
    @assert size(left_config) === size(right_config) "left_config and right_config must have the same size!"

    len = 4*length(operator_list)
    link_list = zeros(Int, len)
    leg_types = zeros(Bool, len)
    associates = [(0, 0, 0) for _ in 1:len]
    flipping_weights = zeros(len)

    in_cluster = zeros(Bool, len)
    cstack = PushVector{Int}(nextpow(2, length(left_config)))
    current_cluster = PushVector{Int}(nextpow(2, length(left_config)))

    first = zeros(Int, length(left_config))
    last = copy(first)

    BinaryThermalState{K,BV}(
        left_config, right_config, copy(left_config),
        operator_list,
        link_list, leg_types, associates, flipping_weights,
        in_cluster, cstack, current_cluster,
        first, last
    )
end



const BinaryQMCState{K, BV} = Union{BinaryGroundState{K, BV}, BinaryThermalState{K, BV}}

# TODO: conversion methods
