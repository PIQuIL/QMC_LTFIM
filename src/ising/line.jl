line_link_list_update!(rng::AbstractRNG, qmc_state::BinaryQMCState, H::AbstractIsing, runstats::AbstractRunStats=NoStats()) =
    multibranch_link_list_update!(rng, qmc_state, H, runstats)
line_link_list_update!(qmc_state, H, runstats::AbstractRunStats=NoStats()) =
    line_link_list_update!(Random.GLOBAL_RNG, qmc_state, H, runstats)


#############################################################################


function line_cluster_update!(rng::AbstractRNG, lsize::Int, qmc_state::BinaryQMCState, H::AbstractIsing, runstats::AbstractRunStats=NoStats())
    Ns = nspins(H)
    operator_list = qmc_state.operator_list

    LinkList = qmc_state.linked_list
    LegType = qmc_state.leg_types
    Associates = qmc_state.associates
    flipping_weights = qmc_state.flipping_weights

    in_cluster = fill!(qmc_state.in_cluster, false)
    cstack = qmc_state.cstack # This is the stack of vertices in a cluster
    current_cluster = qmc_state.current_cluster

    if !(runstats isa NoStats)
        ccount = 0  # cluster number counter
    end

    @inbounds for i in 1:lsize
        # Add a new leg onto the cluster
        if !in_cluster[i] && Associates[i] == 0
            if !(runstats isa NoStats); ccount += 1; end
            push!(cstack, i)
            in_cluster[i] = true

            empty!(current_cluster)
            push!(current_cluster, i)
            lnA = 0.0

            while !isempty(cstack)
                leg = LinkList[pop!(cstack)]

                if !in_cluster[leg]
                    in_cluster[leg] = true  # add the new leg and flip it
                    push!(current_cluster, leg)
                    a = Associates[leg]

                    a == 0 && continue
                    # from this point on, we know we're on a bond op

                    # TODO: check if this is inputting the spins in the correct order
                    if isodd(leg - Ns)
                        preflip_bond_type = getbondtype(H, LegType[leg], LegType[a])
                        postflip_bond_type = getbondtype(H, !LegType[leg], LegType[a])
                    else
                        preflip_bond_type = getbondtype(H, LegType[a], LegType[leg])
                        postflip_bond_type = getbondtype(H, LegType[a], !LegType[leg])
                    end

                    # now add the straight-through leg to the cluster
                    straight_thru = Associates[a]
                    in_cluster[straight_thru] = true
                    push!(cstack, straight_thru)
                    push!(current_cluster, straight_thru)

                    w = flipping_weights[leg]
                    lnA += (
                        H.op_sampler.op_log_weights[w + postflip_bond_type]
                        - H.op_sampler.op_log_weights[w + preflip_bond_type]
                    )
                end
            end

            if qmc_state isa BinaryGroundState && !(qmc_state.trialstate isa AbstractProductState)
                left_flips = empty!(qmc_state.trialstate.left_flips)
                right_flips = empty!(qmc_state.trialstate.right_flips)
                for i in current_cluster
                    if i <= Ns
                        push!(left_flips, i)
                    elseif i > (lsize - Ns)
                        push!(right_flips, i - lsize + Ns)
                    end
                end

                lnA += logweightchange(qmc_state.trialstate, qmc_state.left_config, left_flips)
                lnA += logweightchange(qmc_state.trialstate, qmc_state.right_config, right_flips)
            end

            A = exp(min(lnA, zero(lnA)))
            flip = rand(rng) < A

            fit!(runstats, :cluster_update_accept, A)

            if flip
                @inbounds for i in current_cluster
                    LegType[i] ⊻= 1  # spinflip
                end
            end

            fit!(runstats, :cluster_sizes, float(length(current_cluster)))
        end
    end

    if !(runstats isa NoStats)
        fit!(runstats, :cluster_count, float(ccount))
    end

    # map back basis states and operator list
    ocount = _map_back_basis_states!(rng, lsize, qmc_state, H)
    _map_back_operator_list!(ocount, qmc_state, H)

    return lsize
end

line_cluster_update!(lsize, qmc_state, H, runstats::AbstractRunStats=NoStats()) = line_cluster_update!(Random.GLOBAL_RNG, lsize, qmc_state, H, runstats)


#############################################################################


function line_update!(rng::AbstractRNG, qmc_state::BinaryQMCState, H::AbstractIsing, runstats::AbstractRunStats=NoStats())
    lsize = line_link_list_update!(rng, qmc_state, H, runstats)
    return line_cluster_update!(rng, lsize, qmc_state, H, runstats)
end
line_update!(qmc_state::BinaryQMCState, H::AbstractIsing, runstats::AbstractRunStats=NoStats()) = line_update!(Random.GLOBAL_RNG, qmc_state, H, runstats)
