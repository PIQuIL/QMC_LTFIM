abstract type AbstractImprovedOperatorSampler{K, T, P} <: AbstractOperatorSampler{K, T, P} end

struct ImprovedOperatorSampler{K, T, P} <: AbstractImprovedOperatorSampler{K, T, P}
    operators::Vector{Int}
    pvec::P
    op_log_weights::Vector{T}
    op_rel_log_weights::Vector{T}
    operator_tuples::Vector{NTuple{K, Int}}

    function ImprovedOperatorSampler{K, T}(
            ops::Vector{Int}, pvec::Vector{T}, 
            op_log_weights::Vector{T}, 
            op_rel_log_weights::Vector{T}, 
            operator_tuples::Vector{NTuple{K, Int}}) where {K, T}
        pvec = probability_vector(pvec)
        return new{K, T, typeof(pvec)}(ops, pvec, op_log_weights, op_rel_log_weights, operator_tuples)
    end
end

# only supports the LTFIM/Rydberg cases for now
function ImprovedOperatorSampler(H::Type{<:Hamiltonian{2, <:AbstractOperatorSampler}}, operator_tuples::Vector{NTuple{K, Int}}, p::Vector{T}) where {T <: AbstractFloat, K}
    @assert length(operator_tuples) == length(p) "Given vectors must have the same length!"

    op_rel_log_weights = log.(p)
    op_log_weights = log.(p)

    max_mel_ops = Vector{NTuple{K, Int}}()
    p_modified = Vector{T}()
    
    op_groups = Dict{NTuple{2, Int}, Vector{NTuple{K, Int}}}()
    p_groups = Dict{NTuple{2, Int}, Vector{T}}()

    for (i, op) in enumerate(operator_tuples)
        if issiteoperator(H, op) && isdiagonal(H, op)
            push!(max_mel_ops, op)
            push!(p_modified, @inbounds p[i])
            op_rel_log_weights[i] = zero(p[i])  # subtracting off max mat-elem leaves zero
        elseif isbondoperator(H, op)
            site1, site2 = getbondsites(H, op)
            p_ = p[i]
            if haskey(op_groups, (site1, site2))
                push!(op_groups[(site1, site2)], op)
                push!(p_groups[(site1, site2)], p_)
            else
                op_groups[(site1, site2)] = [op]
                p_groups[(site1, site2)] = [p_]
            end
        end
    end

    for (k, p_gr) in p_groups
        i = argmax(p_gr)
        push!(max_mel_ops, op_groups[k][i])
        push!(p_modified, p_gr[i])

        max_mel_logw = op_rel_log_weights[getweightindex(H, op_groups[k][i])]
        for op in op_groups[k]  # subtract off the maximum matrix element weight
            op_rel_log_weights[getweightindex(H, op)] -= max_mel_logw
        end
    end

    return ImprovedOperatorSampler{K, T}(
        [getweightindex(H, w) for w in max_mel_ops],
        # max_mel_ops,
        p_modified, op_log_weights, op_rel_log_weights, operator_tuples
    )
end

@inline rand(rng::AbstractRNG, os::ImprovedOperatorSampler) = @inbounds os.operator_tuples[os.operators[rand(rng, os.pvec)]]

Base.@propagate_inbounds getrelativelogweight(os::ImprovedOperatorSampler, w::Int) = os.op_rel_log_weights[w]
Base.@propagate_inbounds getlogweight(os::ImprovedOperatorSampler, w::Int) = os.op_log_weights[w]
Base.@propagate_inbounds getoperatortuple(os::ImprovedOperatorSampler, w::Int) = os.operator_tuples[w]

@inline length(os::ImprovedOperatorSampler) = length(os.operators)

##############################################################################
