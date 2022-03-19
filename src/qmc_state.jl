function init_op_list(length, K=Val{4}())
    operator_list = [ntuple(_->0, K) for _ in 1:length]
    return operator_list
end


struct ClusterData{V <: AbstractVector}
    linked_list::Vector{Int}
    leg_types::V
    associates::Vector{Int}
    leg_sites::Vector{Int}
    op_indices::Vector{Int}

    in_cluster::Vector{Int}
    cstack::PushVector{Int, Vector{Int}}
    current_cluster::PushVector{Int, Vector{Int}}

    first::Vector{Int}
    last::Vector{Int}

    function ClusterData(len::Int, left_config::V) where {T, V <: AbstractVector{T}}
        link_list = zeros(Int, len)
        leg_types = similar(left_config, T, len)
        associates = zeros(Int, len)
        leg_sites = zeros(Int, len)
        op_indices = zeros(Int, len)

        in_cluster = zeros(Int, len)
        cstack = PushVector{Int}(nextpow(2, length(left_config)))
        current_cluster = PushVector{Int}(nextpow(2, length(left_config)))

        first = zeros(Int, length(left_config))
        last = copy(first)

        new{V}(
            link_list, leg_types, associates, leg_sites, op_indices,
            in_cluster, cstack, current_cluster, first, last
        )
    end
end

function Base.resize!(cd::ClusterData, len::Int)
    resize!(cd.linked_list, len)
    resize!(cd.leg_types, len)
    resize!(cd.associates, len)
    resize!(cd.leg_sites, len)
    resize!(cd.op_indices, len)
    resize!(cd.in_cluster, len)
end


abstract type AbstractProjector end
struct Exponential <: AbstractProjector; end
struct Power <: AbstractProjector; end

struct QMCState{P <: AbstractProjector, T, K, B <: Union{Nothing, AbstractTrialState{Float64, T}}, V <: AbstractVector{T}}
    left_config::V
    right_config::V
    propagated_config::V

    operator_list::Vector{NTuple{K,Int}}
    cluster_data::ClusterData{V}

    trialstate::B

    function QMCState{P, T, K, B, V}(
        left_config::V, right_config::V, propagated_config::V,
        operator_list::Vector{NTuple{K, Int}}, cluster_data::ClusterData{V},
        trialstate::B
    ) where {P, T, K, V <: AbstractVector{T}, B <: Union{Nothing, AbstractTrialState{Float64, T}}}
        @assert left_config !== right_config "left_config and right_config can't be the same array!"
        @assert size(left_config) === size(right_config) "left_config and right_config must have the same size!"
        new{P, T, K, B, V}(left_config, right_config, propagated_config, operator_list, cluster_data, trialstate)
    end

    function QMCState{P, T, K, B, V}(
        left_config::V, right_config::V, operator_list::Vector{NTuple{K, Int}}, trialstate::B=nothing
    ) where {P, T, K, V <: AbstractVector{T}, B <: Union{Nothing, AbstractTrialState{Float64, T}}}

        len = (trialstate isa AbstractTrialState) ? 2*length(left_config) : 0
        len += 4*length(operator_list)

        QMCState{P, T, K, B, V}(
            left_config, right_config, copy(left_config),
            operator_list,
            ClusterData(len, left_config),
            trialstate
        )
    end
end

QMCState{P, T, K, B, V}(left_config::V, operator_list::Vector{NTuple{K, Int}}, trialstate::B=nothing) where {P, T, K, V, B} =
    QMCState{P, T, K, B, V}(left_config, copy(left_config), operator_list, trialstate)

QMCState{P, T, K, B}(left_config::V, right_config::V, operator_list::Vector{NTuple{K, Int}}, trialstate::B=nothing) where {P, T, K, V, B} =
    QMCState{P, T, K, B, V}(left_config, right_config, operator_list, trialstate)
QMCState{P, T, K, B}(left_config::V, operator_list::Vector{NTuple{K, Int}}, trialstate::B=nothing) where {P, T, K, V, B} =
    QMCState{P, T, K, B, V}(left_config, operator_list, trialstate)

QMCState{P, T, K}(left_config::V, right_config::V, operator_list::Vector{NTuple{K, Int}}, trialstate::B=nothing) where {P, T, K, B, V} =
    QMCState{P, T, K, B, V}(left_config, right_config, operator_list, trialstate)
QMCState{P, T, K}(left_config::V, operator_list::Vector{NTuple{K, Int}}, trialstate::B=nothing) where {P, T, K, B, V} =
    QMCState{P, T, K, B, V}(left_config, operator_list, trialstate)

QMCState{P, T}(left_config::V, right_config::V, operator_list::Vector{NTuple{K, Int}}, trialstate::B=nothing) where {P, T, K, B, V} =
    QMCState{P, T, K, B, V}(left_config, right_config, operator_list, trialstate)
QMCState{P, T}(left_config::V, operator_list::Vector{NTuple{K, Int}}, trialstate::B=nothing) where {P, T, K, B, V} =
    QMCState{P, T, K, B, V}(left_config, operator_list, trialstate)

QMCState{P}(left_config::V, right_config::V, operator_list::Vector{NTuple{K, Int}}, trialstate::B=nothing) where {P, K, B, V} =
    QMCState{P, eltype(left_config), K, B, V}(left_config, right_config, operator_list, trialstate)
QMCState{P}(left_config::V, operator_list::Vector{NTuple{K, Int}}, trialstate::B=nothing) where {P, K, B, V} =
    QMCState{P, eltype(left_config), K, B, V}(left_config, operator_list, trialstate)


const BinaryQMCState{P,K,B,V <: AbstractVector{Bool}} = QMCState{P, Bool, K, B, V}


# function convert(::Type{QMCState{P′, T, K, B′, V}}, state::QMCState{P, T, K, B, V}) where {P, P′, B′, B, T, K, V}
#     if (P′ == P) && (B′ == B)
#         return state
#     elseif B′ === Nothing
#         operator_list = copy(state.operator_list)
#         len = 4*length(operator_list)
#         trialstate = nothing
#     else
#         # make the operator list length even by adding one identity operator
#         operator_list = copy(state.operator_list)
#         if isodd(length(operator_list))
#             push!(operator_list, ntuple(_ -> 0, K))
#         end
#         len = 2*length(state.left_config) + 4*length(operator_list)
#         trialstate = B′()  # may fail if not enough args, nbd since i dont think we'll be converting often
#     end

#     cluster_data = deepcopy(state.cluster_data)
#     resize!(cluster_data, len)

#     return QMCState{P′, T, K, B′, V}(
#         deepcopy(state.left_config), deepcopy(state.right_config), deepcopy(state.propagated_config),
#         operator_list, cluster_data, trialstate
#     )
# end

struct QMCStateSerialization{P, T, K, B, V}
    left_config::V
    operator_list::Vector{NTuple{K,Int}}
    trialstate::B
end

JLD2.writeas(::Type{QMCState{P, T, K, B, V}}) where {P, T, K, B, V} = QMCStateSerialization{P, T, K, B, V}

function JLD2.wconvert(::Type{QMCStateSerialization{P, T, K, B, V}}, state::QMCState{P, T, K, B, V}) where {P, T, K, B, V}
    QMCStateSerialization{P, T, K, B, V}(state.left_config, state.operator_list, state.trialstate)
end

function JLD2.rconvert(::Type{QMCState{P, T, K, B, V}}, saved_state::QMCStateSerialization{P, T, K, B, V}) where {P, T, K, B, V}
    QMCState{P, T, K, B, V}(saved_state.left_config, saved_state.operator_list, saved_state.trialstate)
end