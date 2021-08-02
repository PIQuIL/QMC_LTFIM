abstract type AbstractOperatorSampler{K, T, P <: AbstractProbabilityVector{T}} end
firstindex(::AbstractOperatorSampler) = 1
lastindex(os::AbstractOperatorSampler) = length(os)
@inline normalization(os::AbstractOperatorSampler) = normalization(os.pvec)

# function getlogweight(os::AbstractOperatorSampler{K, T}, op::NTuple{K, Int}) where {K, T}
#     return @inbounds os.op_log_weights[conv_op_to_idx(op, os.strides, os.shifts)]
# end

###############################################################################
include("operatordict.jl")
###############################################################################

struct OperatorSampler{K, T, P} <: AbstractOperatorSampler{K, T, P}
    operators::Vector{NTuple{K, Int}}
    pvec::P
    op_log_weights::Vector{T} #OperatorDict{K, T}
end


function OperatorSampler(operators::Vector{NTuple{K, Int}}, p::Vector{T}) where {K, T <: AbstractFloat}
    @assert length(operators) == length(p) "Given vectors must have the same length!"
    pvec = probability_vector(p)

    op_log_weights = log.(p)
    return OperatorSampler{K, T, typeof(pvec)}(operators, pvec, op_log_weights)
end

@inline rand(rng::AbstractRNG, os::OperatorSampler{K}) where K = @inbounds os.operators[rand(rng, os.pvec)]


function rand_with_logweight(rng::AbstractRNG, os::OperatorSampler{K}) where K
    i = rand(rng, os.pvec)
    op = @inbounds os.operators[i]
    return @inbounds (op, os.op_log_weights[op[2]])
end


@inline getlogweight(os::OperatorSampler{K, T}, op::NTuple{K, Int}) where {K, T} =
    @inbounds os.op_log_weights[op[2]]


@inline length(os::OperatorSampler) = length(os.operators)

##############################################################################

include("hierarchical_op_sampler.jl")
