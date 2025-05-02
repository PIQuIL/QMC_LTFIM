########################## finite-beta #######################################

function mc_step_beta!(f::Function, rng::AbstractRNG, qmc_state::BinaryThermalState, H::AbstractIsing, beta::Real, d::Diagnostics; eq::Bool = false, threestep::Bool=false, kw...)
    if threestep
        num_ops = full_diagonal_update_threestep!(rng, qmc_state, H, beta, d.runstats.diagonal_update; eq=eq, kw...)
    else
        num_ops = full_diagonal_update_twostep!(rng, qmc_state, H, beta, d.runstats.diagonal_update; eq=eq, kw...)
    end
    # full_diagonal_operator_replacement!(rng, qmc_state, H, beta, d; kw...)
    lsize = cluster_update!(rng, qmc_state, H, d; kw...)
    f(lsize, qmc_state, H)
    return num_ops
end

mc_step_beta!(f::Function, qmc_state, H, beta, d::Diagnostics; eq = false, kw...) = mc_step_beta!(f, Random.GLOBAL_RNG, qmc_state, H, beta, d; eq = eq, kw...)
mc_step_beta!(rng::AbstractRNG, qmc_state, H, beta, d::Diagnostics; eq = false, kw...) = mc_step_beta!((args...) -> nothing, rng, qmc_state, H, beta, d; eq = eq, kw...)
mc_step_beta!(qmc_state, H, beta, d::Diagnostics; eq = false, kw...) = mc_step_beta!(Random.GLOBAL_RNG, qmc_state, H, beta, d; eq = eq, kw...)

function resize_op_list!(qmc_state::BinaryThermalState, H::AbstractIsing, new_size::Int)
    operator_list = filter!(!iszero, qmc_state.operator_list)
    len = length(operator_list)

    if len < new_size
        tail = init_op_list(new_size - len)
        append!(operator_list, tail)
    end
    # we do not reduce the length of the operator list further than len
    #   as this risks creating an inconsistent configuration by throwing out 
    #   off-diagonal operators
    # anyway, when calling this function `new_size` will be `c*len`, with c > 1
    #   so the resulting operator list will always be "big enough"

    len = 4*length(operator_list)
    # these are going to be overwritten by the cluster update which will be
    # called right after the diagonal update that called this function
    resize!(qmc_state.linked_list, len)
    resize!(qmc_state.leg_types, len)
    resize!(qmc_state.associates, len)
    resize!(qmc_state.leg_sites, len)
    resize!(qmc_state.op_indices, len)
    resize!(qmc_state.in_cluster, len)

    return length(operator_list)
end