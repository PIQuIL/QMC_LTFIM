
abstract type AbstractOperatorSampler{K, T, P <: AbstractProbabilityVector{T}} end
firstindex(::AbstractOperatorSampler) = 1
lastindex(os::AbstractOperatorSampler) = length(os)
@inline normalization(os::AbstractOperatorSampler) = normalization(os.pvec)

function getlogweight(os::AbstractOperatorSampler{K, T}, op::NTuple{K, Int}) where {K, T}
    return @inbounds os.op_log_weights[conv_op_to_idx(op, os.strides, os.shifts)]
end


struct OperatorSampler{K, T, P} <: AbstractOperatorSampler{K, T, P}
    operators::Vector{NTuple{K, Int}}
    pvec::P
    op_log_weights::Vector{T}
    strides::NTuple{K, Int}
    shifts::NTuple{K, Int}
end


function conv_op_to_idx(op::NTuple{K, Int}, strides::NTuple{K, Int}, shifts::NTuple{K, Int}) where K
    idx = 0
    @inbounds if K > 1
        # compute packed storage format for last two indices (i, j),
        #  assuming i <= j which we've enforced in all of our constructors
        i, j = op[end-1] - shifts[end-1], op[end] - shifts[end]
        p_ij = i + ((j * (j-1)) ÷ 2)
        idx += p_ij
        idx *= strides[end-1]

        for i in reverse(eachindex(op[1:end-2]))
            idx += (op[i] - shifts[i])
            idx *= strides[i]
        end
    elseif K == 1
        idx += (op[1] - shifts[1])
        idx *= strides[1]
    end

    return idx + 1
end

function OperatorSampler(operators::Vector{NTuple{K, Int}}, p::Vector{T}) where {K, T <: AbstractFloat}
    @assert length(operators) == length(p) "Given vectors must have the same length!"
    pvec = probability_vector(p)

    axs = ntuple(i -> extrema(x->x[i], operators), K)
    strides = tuple(1, [max - min + 1 for (min, max) in axs[1:end-1]]...)
    shifts = tuple([min for (min, _) in axs]...)

    l = conv_op_to_idx(tuple([max for (_, max) in axs]...), strides, shifts)

    op_log_weights = log.(zeros(T, l))
    for (i, op) in enumerate(operators)
        idx = conv_op_to_idx(op, strides, shifts)
        op_log_weights[idx] = getlogweight(pvec, i)
    end
    return OperatorSampler{K, T, typeof(pvec)}(operators, pvec, op_log_weights, strides, shifts)
end

@inline rand(rng::AbstractRNG, os::OperatorSampler{K}) where K = @inbounds os.operators[rand(rng, os.pvec)]

function rand_with_logweight(rng::AbstractRNG, os::OperatorSampler{K}) where K
    i = rand(rng, os.pvec)
    return @inbounds (os.operators[i], os.op_log_weights[i])
end

@inline length(os::OperatorSampler) = length(os.operators)

##############################################################################

include("hierarchical_op_sampler.jl")
include("improved_op_sampler.jl")
