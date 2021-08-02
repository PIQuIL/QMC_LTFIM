abstract type AbstractImprovedOperatorSampler{K, T, P} <: AbstractOperatorSampler{K, T, P} end

struct ImprovedOperatorSampler{K, T, P} <: AbstractImprovedOperatorSampler{K, T, P}
    operators::Vector{NTuple{K, Int}}
    pvec::P
    op_log_weights::Vector{T} #OperatorDict{K, T}
end

# only supports the LTFIM/Rydberg cases for now
function ImprovedOperatorSampler(H::Type{<:Hamiltonian{2, <:AbstractOperatorSampler}}, operators::Vector{NTuple{4, Int}}, p::Vector{T}) where {T <: AbstractFloat}
    @assert length(operators) == length(p) "Given vectors must have the same length!"

    op_log_weights = log.(p)

    max_mel_ops = Vector{NTuple{4, Int}}()
    p_modified = Vector{T}()

    # fill with all the site operators first
    for (i, op) in enumerate(operators)
        if issiteoperator(H, op) && isdiagonal(H, op)
            push!(max_mel_ops, op)
            push!(p_modified, @inbounds p[i])
        end
    end
    idx = findall(isbondoperator(H), operators)
    ops = operators[idx]
    p_mod = p[idx]

    perm = sortperm(ops, by=getbondsites(H))
    ops = ops[perm]
    p_mod = p_mod[perm]

    op_groups = Dict{NTuple{2, Int}, Vector{NTuple{4, Int}}}()
    p_groups = Dict{NTuple{2, Int}, Vector{T}}()

    while !isempty(ops)
        op = pop!(ops)
        site1, site2 = getbondsites(H, op)
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
    return ImprovedOperatorSampler{4, T, typeof(pvec)}(max_mel_ops, pvec, op_log_weights)
end

@inline getlogweight(os::ImprovedOperatorSampler{K, T}, op::NTuple{K, Int}) where {K, T} =
    @inbounds os.op_log_weights[op[2]]

@inline rand(rng::AbstractRNG, os::ImprovedOperatorSampler{K}) where K =
    @inbounds os.operators[rand(rng, os.pvec)]

function rand_with_logweight(rng::AbstractRNG, os::ImprovedOperatorSampler{K}) where K
    i = rand(rng, os.pvec)
    op = @inbounds os.operators[i]
    return @inbounds (op, os.op_log_weights[op[2]])
end


function rand_with_weight(rng::AbstractRNG, os::ImprovedOperatorSampler{K}) where K
    i = rand(rng, os.pvec)
    # can retrieve logweight straight from pvec since the indices line up
    # in this case; skips the index computation for op_sampler's getlogweight
    return @inbounds (os.operators[i], getweight(os.pvec, i))
end

@inline length(os::ImprovedOperatorSampler) = length(os.operators)

##############################################################################
