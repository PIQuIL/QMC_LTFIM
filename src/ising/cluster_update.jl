function cluster_update!(rng::AbstractRNG, update_kernel!::Function, acceptance::Function, lsize::Int, qmc_state::BinaryQMCState, H::AbstractIsing, d::Diagnostics)
    Ns = nspins(H)
    operator_list = qmc_state.operator_list

    LinkList = qmc_state.linked_list
    LegType = qmc_state.leg_types
    Associates = qmc_state.associates
    op_indices = qmc_state.op_indices

    in_cluster = fill!(qmc_state.in_cluster, 0)
    cstack = qmc_state.cstack # This is the stack of vertices in a cluster
    current_cluster = qmc_state.current_cluster

    ccount = 0  # cluster number counter

    C = d.runstats.cluster_update

    @inbounds for i in 1:lsize
        # Add a new leg onto the cluster
        if iszero(in_cluster[i]) && iszero(Associates[i])
            ccount += 1
            push!(cstack, i)
            in_cluster[i] = ccount

            empty!(current_cluster)
            push!(current_cluster, i)
            lnA = 0.0

            while !isempty(cstack)
                leg = LinkList[pop!(cstack)]

                if iszero(in_cluster[leg])
                    in_cluster[leg] = ccount  # add the new leg and flip it
                    push!(current_cluster, leg)
                    a = Associates[leg]

                    a == 0 && continue
                    # from this point on, we know we're on a bond op
                    op_index = operator_list[op_indices[leg]]
                    op = getoperatortuple(H, op_index)
                    # op = operator_list[div(leg-1, 4) + 1]
                    w = op_index - getoperatortype(H, op)
                    preflip_bond_type, postflip_bond_type = update_kernel!(qmc_state, H, ccount, leg, a)
                    lnA += (
                        getlogweight(H.op_sampler, w + postflip_bond_type)
                        - getlogweight(H.op_sampler, w + preflip_bond_type)
                    )
                end
            end

            A = acceptance(H, lnA)

            if check_prob(rng, A, C.cluster_update_accept)
                @inbounds for j in current_cluster
                    LegType[j] ⊻= 1  # spinflip
                end
                add_accepted_cluster_size!(C, length(current_cluster))
            else
                add_rejected_cluster_size!(C, length(current_cluster))
            end
        end
    end

    end_step!(C)

    # map back basis states and operator list
    ocount = _map_back_basis_states!(rng, lsize, qmc_state, H)
    _map_back_operator_list!(ocount, qmc_state, H, d)

    return lsize
end
