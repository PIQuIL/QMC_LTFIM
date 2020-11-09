# TODO: better name
struct ImprovedOperatorSampler{K, T, P} <: AbstractOperatorSampler{K, T, P}
    operators::Vector{NTuple{K, Int}}
    pvec::P
    op_weights::Dict{HashTuple{K, Int}, T}
    op_log_weights::Dict{HashTuple{K, Int}, T}
    # op_log_weights::Vector{T}
    # strides::NTuple{K, Int}
    # shifts::NTuple{K, Int}
end

# only supports the LTFIM case for now
function ImprovedOperatorSampler(operators::Vector{NTuple{3, Int}}, p::Vector{T}) where {T <: AbstractFloat}
    @assert length(operators) == length(p) "Given vectors must have the same length!"

    # axs = ntuple(i -> extrema(x->x[i], operators), 3)
    # strides = tuple(1, [max - min + 1 for (min, max) in axs[1:end-1]]...)
    # shifts = tuple([min for (min, _) in axs]...)

    # l = conv_op_to_idx(tuple([max for (_, max) in axs]...), strides, shifts)

    # op_log_weights = log.(zeros(T, l))
    # for (i, op) in enumerate(operators)
    #     idx = conv_op_to_idx(op, strides, shifts)
    #     op_log_weights[idx] = log(p[i])
    # end
    op_log_weights = Dict(HashTuple(op) => log(p[i]) for (i, op) in enumerate(operators))
    op_weights = Dict(HashTuple(op) => p[i] for (i, op) in enumerate(operators))

    max_mel_ops = Vector{NTuple{3, Int}}()
    p_modified = Vector{T}()

    # fill with all the site operators first
    for (i, op) in enumerate(operators)
        if op[1] < 0
            push!(max_mel_ops, op)
            push!(p_modified, @inbounds p[i])
        end
    end
    idx = findall(op -> op[1] > 0, operators)
    ops = operators[idx]
    p_mod = p[idx]

    perm = sortperm(ops, by=op -> (op[2], op[3]))
    ops = ops[perm]
    p_mod = p_mod[perm]

    op_groups = Dict{NTuple{2, Int}, Vector{NTuple{3, Int}}}()
    p_groups = Dict{NTuple{2, Int}, Vector{T}}()

    while !isempty(ops)
        _, site1, site2 = op = pop!(ops)
        p_ = pop!(p_mod)

        if haskey(op_groups, (site1, site2))
            push!(op_groups[(site1, site2)], op)
            push!(p_groups[(site1, site2)], p_)
        else
            op_groups[(site1, site2)] = [op]
            p_groups[(site1, site2)] = [p_]
        end
    end

    for (k, p_gr) in p_groups
        i = argmax(p_gr)
        push!(max_mel_ops, op_groups[k][i])
        push!(p_modified, p_gr[i])
    end

    pvec = probability_vector(p_modified)
    return ImprovedOperatorSampler{3, T, typeof(pvec)}(max_mel_ops, pvec, op_weights, op_log_weights)#, strides, shifts)
end


getlogweight(os::ImprovedOperatorSampler{K, T}, op::NTuple{K, Int}) where {K, T} =
    get(os.op_log_weights, HashTuple(op), T(-Inf))
getweight(os::ImprovedOperatorSampler{K, T}, op::NTuple{K, Int}) where {K, T} =
    get(os.op_weights, HashTuple(op), zero(T))


@inline rand(rng::AbstractRNG, os::ImprovedOperatorSampler{K}) where K =
    @inbounds os.operators[rand(rng, os.pvec)]

function rand_with_logweight(rng::AbstractRNG, os::ImprovedOperatorSampler{K}) where K
    i = rand(rng, os.pvec)
    # can retrieve logweight straight from pvec since the indices line up
    # in this case which would skip the index computation for op_sampler's getlogweight
    op = os.operators[i]
    return (op, getlogweight(os.pvec, i))
end


function rand_with_weight(rng::AbstractRNG, os::ImprovedOperatorSampler{K}) where K
    i = rand(rng, os.pvec)
    # can retrieve logweight straight from pvec since the indices line up
    # in this case which would skip the index computation for op_sampler's getlogweight
    op = os.operators[i]
    return (op, getweight(os.pvec, i))
end

@inline length(os::ImprovedOperatorSampler) = length(os.operators)

##############################################################################
