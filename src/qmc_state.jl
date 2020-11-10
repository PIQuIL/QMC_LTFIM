function init_op_list(length, K=Val{3}())
    operator_list = [ntuple(_->0, K) for _ in 1:length]
    return operator_list
end


abstract type AbstractStateType end
struct Thermal <: AbstractStateType; end
struct Ground <: AbstractStateType; end

abstract type AbstractQMCState{S<:AbstractStateType,T,K} end

const AbstractGroundState = AbstractQMCState{Ground}
const AbstractThermalState = AbstractQMCState{Thermal}

struct QMCState{S,T,K,V <: AbstractVector{T}} <: AbstractQMCState{S,T,K}
    left_config::V
    right_config::V
    propagated_config::V

    operator_list::Vector{NTuple{K,Int}}

    linked_list::Vector{Int}
    leg_types::V
    associates::Vector{NTuple{3,Int}}
    flipping_weights::Vector{Float64}

    in_cluster::V
    cstack::PushVector{Int, Vector{Int}}
    current_cluster::PushVector{Int, Vector{Int}}

    first::Vector{Int}
    last::Union{Vector{Int}, Nothing}

    function QMCState{S, T, K, V}(
            left_config::V, right_config::V, propagated_config::V,
            operator_list,
            link_list, leg_types, associates, flipping_weights,
            in_cluster, cstack, current_cluster,
            first, last
        ) where {S, K, T, V}

        if S isa Type{<:Thermal}
            @assert last isa Vector{Int}
        else
            @assert last === nothing
        end

        new{S, T, K, V}(
            left_config, right_config, propagated_config,
            operator_list,
            link_list, leg_types, associates, flipping_weights,
            in_cluster, cstack, current_cluster,
            first, last
        )
    end

    function QMCState{S, T, K, V}(
        left_config::V, right_config::V, operator_list::Vector{NTuple{K,Int}}
    ) where {S, T, K, V <: AbstractVector{T}}
        @assert left_config !== right_config "left_config and right_config can't be the same array!"
        @assert size(left_config) === size(right_config) "left_config and right_config must have the same size!"

        if S isa Type{<:Ground}
            len = 2*length(left_config) + 4*length(operator_list)
        else
            len = 4*length(operator_list)
        end
        link_list = zeros(Int, len)
        leg_types = similar(left_config, T, len)
        associates = [(0, 0, 0) for _ in 1:len]
        flipping_weights = zeros(len)

        in_cluster = similar(left_config, T, len)
        cstack = PushVector{Int}(nextpow(2, length(left_config)))
        current_cluster = PushVector{Int}(nextpow(2, length(left_config)))

        first = zeros(Int, length(left_config))
        last =  (S isa Type{<:Thermal}) ? copy(first) : nothing
        args = [
            left_config, right_config, copy(left_config),
            operator_list,
            link_list, leg_types, associates, flipping_weights,
            in_cluster, cstack, current_cluster,
            first, last
        ]

        QMCState{S, T, K, V}(args...)
    end
end

QMCState{S, T, K, V}(left_config::V, operator_list) where {S, T, K, V} =
    QMCState{S, T, K, V}(left_config, copy(left_config), operator_list)

QMCState{S, T, K}(left_config::V, right_config::V, operator_list) where {S, T, K, V} =
    QMCState{S, T, K, V}(left_config, right_config, operator_list)
QMCState{S, T, K}(left_config::V, operator_list) where {S, T, K, V} =
    QMCState{S, T, K, V}(left_config, operator_list)

QMCState{S, T}(left_config, right_config, operator_list::Vector{NTuple{K,Int}}) where {S, T, K} =
    QMCState{S, T, K}(left_config, right_config, operator_list)
QMCState{S, T}(left_config, operator_list::Vector{NTuple{K,Int}}) where {S, T, K} =
    QMCState{S, T, K}(left_config, operator_list)

QMCState{S}(left_config, right_config, operator_list) where S =
    QMCState{S, eltype(left_config)}(left_config, right_config, operator_list)
QMCState{S}(left_config, operator_list) where S =
    QMCState{S, eltype(left_config)}(left_config, operator_list)


const GroundState{T,K,V} = QMCState{Ground,T,K,V}
const ThermalState{T,K,V} = QMCState{Thermal,T,K,V}

const BinaryQMCState{K,V <: AbstractVector{Bool}} = QMCState{S, Bool, K, V} where {S <: AbstractStateType}
const BinaryGroundState{K,V <: AbstractVector{Bool}} = GroundState{Bool, K, V}
const BinaryThermalState{K,V <: AbstractVector{Bool}} = ThermalState{Bool, K, V}


function convert(::Type{QMCState{S′}}, state::QMCState{S, T, K, V}) where {S, S′,T, K, V}
    if S′ == S
        return state
    elseif S′ isa Type{<:Thermal}
        len = 4*length(state.operator_list)
        last = copy(state.first)
    else
        len = 2*length(state.left_config) + 4*length(state.operator_list)
        last = nothing
    end

    resize!(state.link_list, len)
    resize!(state.leg_types, len)
    resize!(state.associates, len)
    resize!(state.flipping_weights, len)
    resize!(state.in_cluster, len)

    args = [getfield(state, field) for field in fieldnames(typeof(state))]
    args[end] = last

    return QMCState{S′, T, K, V}(args...)
end
