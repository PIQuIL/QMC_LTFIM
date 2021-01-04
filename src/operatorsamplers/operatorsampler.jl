abstract type AbstractOperatorSampler{K, T, P <: AbstractProbabilityVector{T}} end
firstindex(::AbstractOperatorSampler) = 1
lastindex(os::AbstractOperatorSampler) = length(os)
@inline normalization(os::AbstractOperatorSampler) = normalization(os.pvec)

function getlogweight(os::AbstractOperatorSampler{K, T}, op::NTuple{K, Int}) where {K, T}
    return @inbounds os.op_log_weights[conv_op_to_idx(op, os.strides, os.shifts)]
end

###############################################################################
include("operatordict.jl")
###############################################################################

struct OperatorSampler{K, T, P} <: AbstractOperatorSampler{K, T, P}
    operators::Vector{NTuple{K, Int}}
    pvec::P
    op_log_weights::OperatorDict{K, T}
end


function OperatorSampler(operators::Vector{NTuple{K, Int}}, p::Vector{T}) where {K, T <: AbstractFloat}
    @assert length(operators) == length(p) "Given vectors must have the same length!"
    pvec = probability_vector(p)

    op_log_weights = OperatorDict(operators, log.(p), default=T(-Inf))
    return OperatorSampler{K, T, typeof(pvec)}(operators, pvec, op_log_weights)
end

@inline rand(rng::AbstractRNG, os::OperatorSampler{K}) where K = @inbounds os.operators[rand(rng, os.pvec)]


function rand_with_logweight(rng::AbstractRNG, os::OperatorSampler{K}) where K
    i = rand(rng, os.pvec)
    # can retrieve logweight straight from pvec since the indices line up
    # in this case; skips the index computation for op_sampler's getlogweight
    return @inbounds (os.operators[i], getlogweight(os.pvec, i))
end


Base.@propagate_inbounds getlogweight(os::OperatorSampler{K, T}, op::NTuple{K, Int}) where {K, T} =
    get(os.op_log_weights, op)


@inline length(os::OperatorSampler) = length(os.operators)

##############################################################################

include("hierarchical_op_sampler.jl")
