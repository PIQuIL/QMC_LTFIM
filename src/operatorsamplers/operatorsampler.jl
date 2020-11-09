abstract type AbstractOperatorSampler{K, T, P <: AbstractProbabilityVector{T}} end
firstindex(::AbstractOperatorSampler) = 1
lastindex(os::AbstractOperatorSampler) = length(os)
@inline normalization(os::AbstractOperatorSampler) = normalization(os.pvec)

function getlogweight(os::AbstractOperatorSampler{K, T}, op::NTuple{K, Int}) where {K, T}
    return @inbounds os.op_log_weights[conv_op_to_idx(op, os.strides, os.shifts)]
end

# Using HashTuple as dict keys seems to work fine, it's still slower than
# the array scheme though
struct HashTuple{K}
    tup::NTuple{K, Int}
end
Base.hash(op::HashTuple{K}) where K = foldl((r,i) -> 31r + i, op.tup, init=1)
Base.:(==)(a::HashTuple{K}, b::HashTuple{K}) where K = (a.tup === b.tup)
Base.:(==)(a::HashTuple, b::HashTuple) = false

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


struct OperatorDict{K, NT <: NTuple{K, Int}, V} #where {K, O <: NTuple{K, Int}, V}
    values::Vector{V}
    strides::NTuple{K, Int}
    shifts::NTuple{K, Int}
    axes::NTuple{K, UnitRange{Int}}

    function OperatorDict(operators::Vector{NTuple{K, Int}}, values::Vector{T}; default=zero(T)) where {K, T}
        axs = ntuple(i -> extrema(x->x[i], operators), K)
        axes_ = ntuple(i -> UnitRange(axs[i]...), K)
        strides = tuple(1, [max - min + 1 for (min, max) in axs[1:end-1]]...)
        shifts = tuple([min for (min, _) in axs]...)

        l = conv_op_to_idx(tuple([max for (_, max) in axs]...), strides, shifts)

        vals = fill!(zeros(T, l), default)
        for (i, op) in enumerate(operators)
            idx = conv_op_to_idx(op, strides, shifts)
            vals[idx] = values[i]
        end

        new{K, NTuple{K, Int}, T}(vals, strides, shifts, axes_)
    end
end

function Base.haskey(op_dict::OperatorDict{K, NTuple{K, Int}}, op::NTuple{K, Int}) where K
    all(i -> op[i] in op_dict.axes[i], 1:K)
end

@inline function Base.get(op_dict::OperatorDict{K, NTuple{K, Int}}, op::NTuple{K, Int}) where K
    @boundscheck haskey(op_dict, op)
    strides, shifts = op_dict.strides, op_dict.shifts
    idx = conv_op_to_idx(op, strides, shifts)
    return @inbounds op_dict.values[idx]
end


struct OperatorSampler{K, T, P} <: AbstractOperatorSampler{K, T, P}
    operators::Vector{NTuple{K, Int}}
    pvec::P
    op_log_weights::OperatorDict{K, NTuple{K, Int}, T}
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
include("improved_op_sampler.jl")
