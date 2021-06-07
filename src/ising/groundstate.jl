cluster_update!(rng, qmc_state, H::Hamiltonian, runstats; kw...) = multibranch_update!(rng, qmc_state, H, runstats)
function cluster_update!(rng, qmc_state, H::AbstractRydberg, runstats; p::Float64=0.0, kw...)
    if rand(rng) < p
        # occasionally do a multibranch update to maintain ergodicity
        multibranch_update!(rng, qmc_state, H, runstats)
    else
        line_update!(rng, qmc_state, H, runstats)
    end
end

function mc_step!(f::Function, rng::AbstractRNG, qmc_state::BinaryGroundState, H::Hamiltonian, runstats::AbstractRunStats=NoStats(); kw...)
    full_diagonal_update!(rng, qmc_state, H, runstats)
    lsize = cluster_update!(rng, qmc_state, H, runstats; kw...)
    f(lsize, qmc_state, H)
end
mc_step!(f::Function, qmc_state::BinaryGroundState, H::Hamiltonian, runstats::AbstractRunStats=NoStats(); kw...) = mc_step!(f, Random.GLOBAL_RNG, qmc_state, H, runstats; kw...)
mc_step!(rng::AbstractRNG, qmc_state::BinaryGroundState, H::Hamiltonian, runstats::AbstractRunStats=NoStats(); kw...) = mc_step!((args...) -> nothing, rng, qmc_state, H, runstats; kw...)
mc_step!(qmc_state::BinaryGroundState, H::Hamiltonian, runstats::AbstractRunStats=NoStats(); kw...) = mc_step!(Random.GLOBAL_RNG, qmc_state, H, runstats; kw...)


Base.@propagate_inbounds alignment_check(H::AbstractTFIM, op::NTuple{3, Int}, s1::Bool, s2::Bool) =
    xor(isferromagnetic(H, getbondsites(H, op)), s1, s2)

Base.@propagate_inbounds alignment_check(H::AbstractLTFIM, op::NTuple{3, Int}, s1::Bool, s2::Bool) =
    (op[1] == getbondtype(H, s1, s2))


# insert_diagonal_operator! returns true if operator insertion succeeded
# returns true if operator insertion succeeded
function insert_diagonal_operator!(rng::AbstractRNG, qmc_state::BinaryQMCState{K, V}, H::AbstractIsing{O}, spin_prop::V, n::Int) where {K, V, T, O <: AbstractOperatorSampler{K, T}}
    op = rand(rng, H.op_sampler)
    site1, site2 = getbondsites(H, op)
    @inbounds if issiteoperator(H, op) || alignment_check(H, op, spin_prop[site1], spin_prop[site2])
        qmc_state.operator_list[n] = op
        return op, zero(T)
    else
        return nothing, zero(T)
    end
end

function insert_diagonal_operator!(rng::AbstractRNG, qmc_state::BinaryQMCState{K, V}, H::AbstractLTFIM{<:AbstractImprovedOperatorSampler}, spin_prop::V, n::Int) where {K, V}
    op, lw1 = rand_with_logweight(rng, H.op_sampler)

    @inbounds if issiteoperator(H, op)
        qmc_state.operator_list[n] = op
        return op, zero(lw1)
    else
        t, site1, site2 = op
        real_t = getbondtype(H, spin_prop[site1], spin_prop[site2])
        if t == real_t
            qmc_state.operator_list[n] = op
            return op, lw1
        end

        op2 = (real_t, site1, site2)
        lw2 = getlogweight(H.op_sampler, op2)

        if rand(rng) < exp(lw2 - lw1)
            qmc_state.operator_list[n] = op2
            return op2, lw2
        else
            return nothing, zero(lw1)
        end
    end
end

insert_diagonal_operator!(qmc_state, H, spin_prop, n) = insert_diagonal_operator!(Random.GLOBAL_RNG, qmc_state, H, spin_prop, n)

#############################################################################

function full_diagonal_update!(rng::AbstractRNG, qmc_state::BinaryGroundState, H::AbstractIsing, runstats::AbstractRunStats=NoStats())
    spin_prop = copyto!(qmc_state.propagated_config, qmc_state.left_config)  # the propagated spin state

    if !(runstats isa NoStats)
        failures = 0
        count = 0
    end

    for (n, op) in enumerate(qmc_state.operator_list)
        if !isdiagonal(H, op)
            @inbounds spin_prop[op[2]] ⊻= 1  # spinflip
        else
            op = nothing
            if !(runstats isa NoStats); i = -1; end
            while op === nothing
                op, _ = insert_diagonal_operator!(rng, qmc_state, H, spin_prop, n)
                if !(runstats isa NoStats); i += 1; end
            end
            if !(runstats isa NoStats); failures += i; count += 1; end
        end
    end

    # @debug statements are not run unless debug logging is enabled
    @debug("Diagonal Update basis state propagation status: $(spin_prop == qmc_state.right_config)",
           spin_prop,
           qmc_state.right_config)

    if !(runstats isa NoStats)
        fit!(runstats, :diag_update_fails, failures/count)
    end
end
full_diagonal_update!(qmc_state, H, runstats::AbstractRunStats=NoStats()) = full_diagonal_update!(Random.GLOBAL_RNG, qmc_state, H, runstats)
