function init_op_list(length, K=Val{ISING_OP_SIZE}())
    # operator_list = [ntuple(_->0, K) for _ in 1:length]
    return zeros(Int, length)
end


abstract type AbstractStateType end
struct Thermal <: AbstractStateType; end
struct Ground <: AbstractStateType; end

abstract type AbstractQMCState{S<:AbstractStateType,T} end

const AbstractGroundState = AbstractQMCState{Ground}
const AbstractThermalState = AbstractQMCState{Thermal}

struct QMCState{S,T,V <: AbstractVector{T},P <: Union{Nothing, AbstractTrialState{Float64, T}}} <: AbstractQMCState{S,T}
    left_config::V
    right_config::V
    propagated_config::V

    operator_list::Vector{Int}

    linked_list::Vector{Int}
    leg_types::V
    associates::Vector{Int}
    leg_sites::Vector{Int}
    op_indices::Vector{Int}

    in_cluster::Vector{Int}
    cstack::PushVector{Int, Vector{Int}}
    current_cluster::PushVector{Int, Vector{Int}}

    first::Vector{Int}
    last::Union{Vector{Int}, Nothing}

    trialstate::P

    function QMCState{S, T, V, P}(
            left_config::V, right_config::V, propagated_config::V,
            operator_list,
            link_list, leg_types::V, associates, leg_sites, op_indices,
            in_cluster, cstack, current_cluster,
            first, last, trialstate::P
        ) where {S, T, V <: AbstractVector{T}, P <: Union{Nothing, AbstractTrialState{Float64, T}}}

        if S isa Type{<:Thermal}
            @assert last isa Vector{Int}
            @assert P == Nothing
        else
            @assert last === nothing
            @assert P <: AbstractTrialState
        end

        new{S, T, V, P}(
            left_config, right_config, propagated_config,
            operator_list,
            link_list, leg_types, associates, leg_sites, op_indices,
            in_cluster, cstack, current_cluster,
            first, last, trialstate
        )
    end

    function QMCState{S, T, V}(
        left_config::V, right_config::V, operator_list::Vector{Int},
        trialstate::Union{Nothing, AbstractTrialState{Float64, T}}=nothing
    ) where {S, T, V <: AbstractVector{T}}
        @assert left_config !== right_config "left_config and right_config can't be the same array!"
        @assert size(left_config) === size(right_config) "left_config and right_config must have the same size!"

        if S isa Type{<:Ground}
            len = 2*length(left_config) + 4*length(operator_list)
            if trialstate === nothing
                trialstate = PlusState{Float64, T}()
            end
        else
            len = 4*length(operator_list)
            trialstate = nothing
        end
        link_list = zeros(Int, len)
        leg_types = similar(left_config, T, len)
        associates = zeros(Int, len)
        leg_sites = zeros(Int, len)
        op_indices = zeros(Int, len)

        in_cluster = zeros(Int, len)
        cstack = PushVector{Int}(nextpow(2, length(left_config)))
        current_cluster = PushVector{Int}(nextpow(2, length(left_config)))

        first = zeros(Int, length(left_config))
        last = (S isa Type{<:Thermal}) ? copy(first) : nothing
        args = [
            left_config, right_config, copy(left_config),
            operator_list,
            link_list, leg_types, associates, leg_sites, op_indices,
            in_cluster, cstack, current_cluster,
            first, last, trialstate
        ]

        new{S, T, V, typeof(trialstate)}(args...)
    end
end

QMCState{S, T, V}(left_config::V, operator_list, trialstate::Union{Nothing, AbstractTrialState{Float64, T}}=nothing) where {S, T, V} =
    QMCState{S, T, V}(left_config, copy(left_config), operator_list, trialstate)

QMCState{S, T}(left_config::V, right_config::V, operator_list, trialstate::Union{Nothing, AbstractTrialState{Float64, T}}=nothing) where {S, T, V} =
    QMCState{S, T, V}(left_config, right_config, operator_list, trialstate)
QMCState{S, T}(left_config::V, operator_list, trialstate::Union{Nothing, AbstractTrialState{Float64, T}}=nothing) where {S, T, V} =
    QMCState{S, T, V}(left_config, operator_list, trialstate)

# QMCState{S, T}(left_config::V, right_config::V, operator_list::Vector{NTuple{K,Int}}, trialstate::Union{Nothing, AbstractTrialState{Float64, T}}=nothing) where {S, T, K, V} =
#     QMCState{S, T, K}(left_config, right_config, operator_list, trialstate)
# QMCState{S, T}(left_config, operator_list::Vector{NTuple{K,Int}}, trialstate::Union{Nothing, AbstractTrialState{Float64, T}}=nothing) where {S, T, K} =
#     QMCState{S, T, K}(left_config, operator_list, trialstate)

QMCState{S}(left_config::V, right_config::V, operator_list, trialstate::Union{Nothing, AbstractTrialState}=nothing) where {S, V} =
    QMCState{S, eltype(left_config)}(left_config, right_config, operator_list, trialstate)
QMCState{S}(left_config, operator_list, trialstate::Union{Nothing, AbstractTrialState}=nothing) where S =
    QMCState{S, eltype(left_config)}(left_config, operator_list, trialstate)


const GroundState{T,V} = QMCState{Ground,T,V}
const ThermalState{T,V} = QMCState{Thermal,T,V}

const BinaryQMCState{V <: AbstractVector{Bool}} = QMCState{S, Bool, V} where {S <: AbstractStateType}
const BinaryGroundState{V <: AbstractVector{Bool}} = GroundState{Bool, V}
const BinaryThermalState{V <: AbstractVector{Bool}} = ThermalState{Bool, V}


function convert(::Type{QMCState{S′, T, V}}, state::QMCState{S, T, V}) where {S, S′,T, V}
    if S′ == S
        return state
    elseif S′ isa Type{<:Thermal}
        len = 4*length(state.operator_list)
        last = copy(state.first)
        trialstate = nothing
    else
        # make the operator list length even by adding one identity operator
        if isodd(length(state.operator_list))
            push!(state.operator_list, 0)
        end
        len = 2*length(state.left_config) + 4*length(state.operator_list)
        last = nothing
        trialstate = PlusState()
    end

    resize!(state.linked_list, len)
    resize!(state.leg_types, len)
    resize!(state.associates, len)
    resize!(state.leg_sites, len)
    resize!(state.op_indices, len)
    resize!(state.in_cluster, len)

    args = [getfield(state, field) for field in fieldnames(typeof(state))]

    return QMCState{S′, T, V}(args[1:end-2]..., last, trialstate)
end
