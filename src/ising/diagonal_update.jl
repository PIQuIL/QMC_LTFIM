Base.@propagate_inbounds alignment_check(H::AbstractTFIM, op::NTuple{K, Int}, s1::Bool, s2::Bool) where K =
    xor(isferromagnetic(H, getbondsites(H, op)), s1, s2)

Base.@propagate_inbounds alignment_check(H::AbstractLTFIM, op::NTuple{K, Int}, s1::Bool, s2::Bool) where K =
    (getoperatortype(H, op) == getbondtype(H, s1, s2))


function accept_diagonal_operator(rng::AbstractRNG, H::AbstractIsing{<:AbstractOperatorSampler{K, T}}, spin_prop, op::NTuple{K, Int}) where {K, T}
    site1, site2 = getbondsites(H, op)
    @inbounds if issiteoperator(H, op) || alignment_check(H, op, spin_prop[site1], spin_prop[site2])
        return op
    else
        return nothing
    end
end



function accept_diagonal_operator_replacement(
        rng::AbstractRNG, H::AbstractLTFIM{<:AbstractImprovedOperatorSampler{K, T}}, 
        spin_prop, op::NTuple{K, Int}, old_op::Union{NTuple{K, Int}, Nothing}, 
        stat::AbstractUpdateStat)::Union{NTuple{K, Int}, Nothing} where {K, T}
    @inbounds if issiteoperator(H, op)
        add_unit_prob!(stat)
        return op
    elseif op === old_op
        add_unit_prob!(stat)
        return op
    else
        t = getoperatortype(H, op)
        site1, site2 = getbondsites(H, op)
        real_t = getbondtype(H, spin_prop[site1], spin_prop[site2])
        if t == real_t  # largest matrix element turned out to be the correct one
            add_unit_prob!(stat)
            return op
        end

        op = convertbondtype(H, op, real_t)
        lw = getrelativelogweight(H, op, old_op)

        return (check_logprob(rng, lw, stat) ? op : nothing)
    end
end

@inline function accept_diagonal_operator_insertion(
        rng::AbstractRNG, H::AbstractLTFIM{<:AbstractImprovedOperatorSampler{K, T}}, 
        spin_prop, op::NTuple{K, Int},
        stat::AbstractUpdateStat)::Union{NTuple{K, Int}, Nothing} where {K, T}
    return accept_diagonal_operator_replacement(rng, H, spin_prop, op, nothing, stat)
end

function accept_diagonal_operator_replacement(
        rng::AbstractRNG, H::AbstractLTFIM{<:AbstractImprovedOperatorSampler{K, T}}, 
        spin_prop, op::NTuple{K, Int}, old_op::Union{NTuple{K, Int}, Nothing}, raw_removal_prob::T,
        stat::AbstractUpdateStat)::Union{NTuple{K, Int}, Nothing} where {K, T}
    @inbounds if issiteoperator(H, op)
        add_unit_prob!(stat)
        return op
    elseif op === old_op
        add_unit_prob!(stat)
        return op
    else
        t = getoperatortype(H, op)
        site1, site2 = getbondsites(H, op)
        real_t = getbondtype(H, spin_prop[site1], spin_prop[site2])
        op = convertbondtype(H, op, real_t)
        
        lw_old = getrelativelogweight(H, old_op)
        lw_new = getrelativelogweight(H, op)

        if lw_old == lw_new
            # short-circuit (SC) if relative weights are the same
            #   this covers the case of both old_op and op being maximal matrix elements
            #   but is also a bit more general
            # only using regular equality instead of approx equality bc 
            #   dont wanna spend too much compute on a short-circuit check,
            #   if the two values are identically 0 (i.e. maximal mat-elems) then we'll SC
            #   otherwise just do the math
            add_unit_prob!(stat)
            return op
        end

        prob = exp(lw_new - lw_old) * (1 - min(1, raw_removal_prob/exp(lw_new)))/(1 - min(1, raw_removal_prob/exp(lw_old)))

        return (check_prob(rng, prob, stat) ? op : nothing)
    end
end


#################################################

function diagonal_operator_replacement!(rng::AbstractRNG, qmc_state::BinaryQMCState, H::AbstractIsing, spin_prop, n::Int, d::AbstractDiagonalUpdateStats; max_iter::Int=5)
    current_op = @inbounds getoperatortuple(H.op_sampler, qmc_state.operator_list[n])
    for i in 1:max_iter
        op = accept_diagonal_operator_replacement(
            rng, H, spin_prop, rand(rng, H.op_sampler), current_op, d.replace
        )
        if op !== nothing
            @inbounds qmc_state.operator_list[n] = getweightindex(H, op)
            return i
        end
    end
    return 0
end

function diagonal_operator_replacement!(rng::AbstractRNG, qmc_state::BinaryThermalState, H::AbstractIsing, spin_prop, n::Int, raw_removal_prob, d::AbstractDiagonalUpdateStats; max_iter::Int=5)
    current_op = @inbounds getoperatortuple(H.op_sampler, qmc_state.operator_list[n])
    for i in 1:max_iter
        op = accept_diagonal_operator_replacement(
            rng, H, spin_prop, rand(rng, H.op_sampler), current_op, raw_removal_prob, d.replace
        )
        if op !== nothing
            @inbounds qmc_state.operator_list[n] = getweightindex(H, op)
            return i
        end
    end
    return 0
end

#################################################


function full_diagonal_update_threestep!(rng::AbstractRNG, qmc_state::BinaryThermalState, H::AbstractIsing, beta::Real, d::AbstractDiagonalUpdateStats; eq::Bool = false, max_iter::Int = 5, kw...)
    P_norm = beta * diag_update_normalization(H)

    num_ids = count(iszero, qmc_state.operator_list)

    spin_prop = copyto!(qmc_state.propagated_config, qmc_state.left_config)  # the propagated spin state

    @inbounds for (n, op_index) in enumerate(qmc_state.operator_list)
        op = getoperatortuple(H, op_index)
        if !isdiagonal(H, op)
            spin_prop[getsite(H, op)] ⊻= 1  # spinflip
        elseif !isidentity(H, op)
            if check_prob(rng, (num_ids + 1)/(P_norm), d.removal)  # remove operator, insert identity
                qmc_state.operator_list[n] = 0 #makeidentity(H)
                num_ids += 1
            else
                iters = diagonal_operator_replacement!(rng, qmc_state, H, spin_prop, n, d, max_iter=max_iter)
                if iters > 0
                    update_replacement_attempts!(d, iters)
                end
            end
        else
            if check_prob(rng, P_norm/num_ids, d.operator_insertion)  # insert the diagonal op
                op = accept_diagonal_operator_insertion(rng, H, spin_prop, rand(rng, H.op_sampler), d.matelem_insertion)
                if op !== nothing
                    qmc_state.operator_list[n] = getweightindex(H, op)
                    num_ids -= 1
                end
            end
        end
    end

    end_step!(d)

    total_list_size = length(qmc_state.operator_list)
    num_ops = total_list_size - num_ids

    if eq && (1.2*num_ops > total_list_size)
        resize_op_list!(qmc_state, H, round(Int, 1.5*num_ops, RoundUp))
    end

    return num_ops
end



function full_diagonal_update_twostep!(rng::AbstractRNG, qmc_state::BinaryThermalState, H::AbstractIsing, beta::Real, d::AbstractDiagonalUpdateStats; eq::Bool = false, max_iter::Int=5, kw...)
    P_norm = beta * diag_update_normalization(H)

    num_ids = count(isidentity(H), qmc_state.operator_list)

    spin_prop = copyto!(qmc_state.propagated_config, qmc_state.left_config)  # the propagated spin state

    @inbounds for (n, op_index) in enumerate(qmc_state.operator_list)
        op = getoperatortuple(H, op_index)
        if !isdiagonal(H, op)
            spin_prop[getsite(H, op)] ⊻= 1  # spinflip
        elseif !isidentity(H, op)
            rw = exp(getrelativelogweight(H, op))
            raw_removal_prob = (num_ids + 1)/P_norm
            removal_prob = raw_removal_prob/rw
            if check_prob(rng, removal_prob, d.removal)  # remove operator, insert identity
                qmc_state.operator_list[n] = 0 #getweightindex(makeidentity(H))
                num_ids += 1
            else
                iters = diagonal_operator_replacement!(rng, qmc_state, H, spin_prop, n, raw_removal_prob, d, max_iter=max_iter)
                if iters > 0
                    update_replacement_attempts!(d, iters)
                end
            end
        else
            op = rand(rng, H.op_sampler)

            if isbondoperator(H, op)
                site1, site2 = getbondsites(H, op)
                real_t = getbondtype(H, spin_prop[site1], spin_prop[site2])
                op = convertbondtype(H, op, real_t)
            end
            rw = exp(getrelativelogweight(H, op))
            add_prob!(d.matelem_insertion, rw)

            insert_prob = P_norm*rw/num_ids
            if check_prob(rng, insert_prob, d.operator_insertion)  # insert the diagonal op
                qmc_state.operator_list[n] = getweightindex(H, op)
                num_ids -= 1
            end
        end
    end

    end_step!(d)

    total_list_size = length(qmc_state.operator_list)
    num_ops = total_list_size - num_ids

    if eq && (1.2*num_ops > total_list_size)
        resize_op_list!(qmc_state, H, round(Int, 1.5*num_ops, RoundUp))
    end

    return num_ops
end

#################################################

function full_diagonal_operator_replacement!(rng::AbstractRNG, qmc_state::BinaryQMCState, H::AbstractIsing, d::AbstractDiagonalUpdateStats; max_iter::Int=5, kw...)
    spin_prop = copyto!(qmc_state.propagated_config, qmc_state.left_config)  # the propagated spin state
    @inbounds for (n, op_index) in enumerate(qmc_state.operator_list)
        op = getoperatortuple(H, op_index)
        if !isdiagonal(H, op)
            spin_prop[getsite(H, op)] ⊻= 1  # spinflip
        elseif qmc_state isa BinaryGroundState || !isidentity(H, op)
            if max_iter < 0 && !isidentity(H, op)
                iters = diagonal_operator_replacement!(rng, qmc_state, H, spin_prop, n, d, max_iter=max_iter)
            end
            if iters > 0
                update_replacement_attempts!(d, iters)
            end
        end
    end
    end_step!(d)
    return nothing
end


function full_diagonal_update!(rng::AbstractRNG, qmc_state::BinaryGroundState, H::AbstractIsing, d::AbstractDiagonalUpdateStats; kw...)
    full_diagonal_operator_replacement!(rng, qmc_state, H, d; kw...)
end

