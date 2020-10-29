
abstract type AbstractOperatorSampler{K, T, P <: AbstractProbabilityVector{T}} end
firstindex(::AbstractOperatorSampler) = 1
lastindex(os::AbstractOperatorSampler) = length(os)

getweight(os::AbstractOperatorSampler{K}, op::NTuple{K, Int}) where K = haskey(os.op_indices, op) ? getweight(os.pvec, os.op_indices[op]) : 0


struct OperatorSampler{K, T, P} <: AbstractOperatorSampler{K, T, P}
    operators::Vector{NTuple{K, Int}}
    pvec::P
    op_indices::Dict{NTuple{K, Int}, Int}
end


function OperatorSampler(operators::Vector{NTuple{K, Int}}, p::Vector{T}) where {K, T <: Real}
    @assert length(operators) == length(p) "Given vectors must have the same length!"
    pvec = probability_vector(p)
    op_indices = Dict{NTuple{K, Int}, Int}(op => i for (i, op) in enumerate(operators))
    return OperatorSampler{K, T, typeof(pvec)}(operators, pvec, op_indices)
end
rand(rng::AbstractRNG, os::OperatorSampler{K}) where K = @inbounds os.operators[rand(rng, os.pvec)]
@inline length(os::OperatorSampler) = length(os.operators)




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

function getweight(os::HierarchicalOperatorSampler{K}, op::NTuple{K, Int}) where K
    if !haskey(os.op_indices, op)
        return 0
    end
    idx = os.op_indices[op]
    return getweight(os.pvec, idx) / length(os.operator_bins[idx])
end
