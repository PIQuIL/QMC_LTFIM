
abstract type AbstractOperatorSampler{K, T, P <: AbstractProbabilityVector{T}} end
firstindex(::AbstractOperatorSampler) = 1
lastindex(os::AbstractOperatorSampler) = length(os)

function getweight(os::AbstractOperatorSampler{K, T}, op::NTuple{K, Int}) where {K, T}
    # idx = get(os.op_indices, op, 0)
    idx = os.op_indices[conv_op_to_idx(op, os.strides, os.shifts)]
    # idx = get(os.op_indices, conv_op_to_idx(op, os.strides, os.shifts), 0)
    if iszero(idx)
        return zero(T)
    else
        return getweight(os.pvec, idx)
    end
end


function getlogweight(os::AbstractOperatorSampler{K, T}, op::NTuple{K, Int}) where {K, T}
    # idx = get(os.op_indices, op, 0)
    idx = os.op_indices[conv_op_to_idx(op, os.strides, os.shifts)]
    # idx = get(os.op_indices, conv_op_to_idx(op, os.strides, os.shifts), 0)
    if iszero(idx)
        return zero(T)
    else
        return getlogweight(os.pvec, idx)
    end
end


struct OperatorSampler{K, T, P} <: AbstractOperatorSampler{K, T, P}
    operators::Vector{NTuple{K, Int}}
    pvec::P
    op_indices::Vector{Int}
    strides::NTuple{K, Int}
    shifts::NTuple{K, Int}
end


function conv_op_to_idx(op::NTuple{K, Int}, strides::NTuple{K, Int}, shifts::NTuple{K, Int}) where K
    idx = 0
    @inbounds for i in reverse(eachindex(op))
        idx += (op[i] - shifts[i])
        idx *= strides[i]
    end
    return idx + 1
end

function OperatorSampler(operators::Vector{NTuple{K, Int}}, p::Vector{T}) where {K, T <: Real}
    @assert length(operators) == length(p) "Given vectors must have the same length!"
    pvec = probability_vector(p)

    axs = ntuple(i -> extrema(x->x[i], operators), K)
    strides = tuple(1, [max - min + 1 for (min, max) in axs[1:end-1]]...)
    shifts = tuple([min for (min, _) in axs]...)

    l = conv_op_to_idx(tuple([max for (_, max) in axs]...), strides, shifts)

    op_indices = zeros(Int, l)
    for (i, op) in enumerate(operators)
        idx = conv_op_to_idx(op, strides, shifts)
        op_indices[idx] = i
    end
    return OperatorSampler{K, T, typeof(pvec)}(operators, pvec, op_indices, strides, shifts)
end
rand(rng::AbstractRNG, os::OperatorSampler{K}) where K = @inbounds os.operators[rand(rng, os.pvec)]
@inline length(os::OperatorSampler) = length(os.operators)

###############################################################################

# Leave HierarchicalOperatorSampler using dictionaries for now just to have an
#  easily accessible "ground-truth" comparison.


function cluster_probs_vec(operators::Vector{NTuple{K, Int}}, p::AbstractVector{T}) where {K, T <: Real}
    perm = sortperm(p)
    p = p[perm]
    operators = operators[perm]

    uniq_p = T[]
    uniq_ops = Vector{Vector{NTuple{K, Int}}}(undef, 0)
    op_bins = Dict{NTuple{K, Int}, Int}()

    for i in axes(p, 1)
        if length(uniq_p) == 0
            push!(uniq_p, p[i])
            push!(uniq_ops, [operators[i]])
        elseif !(last(uniq_p) ≈ p[i])
            push!(uniq_p, p[i])
            push!(uniq_ops, [operators[i]])
        else
            push!(uniq_ops[end], operators[i])
        end
        push!(op_bins, operators[i] => length(uniq_p))
    end

    # rescale uniq_p
    for i in eachindex(uniq_p, uniq_ops)
        uniq_p[i] *= length(uniq_ops[i])
    end
    return uniq_ops, uniq_p, op_bins
end



struct HierarchicalOperatorSampler{K, T, P} <: AbstractOperatorSampler{K, T, P}
    operator_bins::Vector{Vector{NTuple{K, Int}}}
    pvec::P
    op_indices::Dict{NTuple{K, Int}, Int}
end

function HierarchicalOperatorSampler(operators::Vector{NTuple{K, Int}}, p::AbstractVector{T}) where {K, T <: Real}
    @assert length(operators) == length(p) "Given vectors must have the same length!"
    operator_bins, p, op_indices = cluster_probs_vec(operators, p)
    pvec = probability_vector(p)
    return HierarchicalOperatorSampler{K, T, typeof(pvec)}(operator_bins, pvec, op_indices)
end
length(os::HierarchicalOperatorSampler) = sum(length, os.operator_bins)

function rand(rng::AbstractRNG, os::HierarchicalOperatorSampler{K})::NTuple{K, Int} where K
    @inbounds ops_list = os.operator_bins[rand(rng, os.pvec)]
    l = rand(rng, 1:length(ops_list))

    return @inbounds ops_list[l]
end

function getweight(os::HierarchicalOperatorSampler{K, T}, op::NTuple{K, Int}) where {K, T}
    idx = get(os.op_indices, op, 0)
    if iszero(idx)
        return zero(T)
    else
        return getweight(os.pvec, idx) / length(os.operator_bins[idx])
    end
end
