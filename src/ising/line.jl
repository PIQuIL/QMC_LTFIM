function line_link_list_update!(::AbstractRNG, qmc_state::BinaryQMCState, H::AbstractIsing, ::AbstractRunStats=NoStats())
    Ns = nspins(H)
    spin_left = qmc_state.left_config

    # retrieve linked list data structures
    LinkList = qmc_state.linked_list  # needed for cluster update
    LegType = qmc_state.leg_types

    # A diagonal bond operator has non trivial associates for cluster building
    Associates = qmc_state.associates

    flipping_weights = qmc_state.flipping_weights

    if qmc_state isa BinaryGroundState
        First = qmc_state.first

        # The first N elements of the linked list are the spins of the LHS basis state
        @inbounds for i in 1:Ns
            LegType[i] = spin_left[i]
            Associates[i] = 0
            First[i] = i
        end
        idx = Ns
    else
        First = fill!(qmc_state.first, 0)  #initialize the First list
        Last = fill!(qmc_state.last, 0)   #initialize the Last list
        idx = 0
    end

    spin_prop = copyto!(qmc_state.propagated_config, spin_left)  # the propagated spin state

    # Now, add the 2M operators to the linked list. Each has either 2 or 4 legs
    @inbounds for op in qmc_state.operator_list
        if issiteoperator(H, op)
            site = op[3]
            # lower or left leg
            idx += 1
            F = First[site]
            LinkList[idx] = F
            if qmc_state isa BinaryGroundState || !iszero(F)
                LinkList[F] = idx  # completes backwards link
            else
                Last[site] = idx
            end

            LegType[idx] = spin_prop[site]
            Associates[idx] = 0
            if H isa AbstractLTFIM
                flipping_weights[idx] = 0.0
            end

            if !isdiagonal(H, op)  # off-diagonal site operator
                spin_prop[site] ⊻= 1  # spinflip
            end

            # upper or right leg
            idx += 1
            First[site] = idx
            LegType[idx] = spin_prop[site]
            Associates[idx] = 0
            if H isa AbstractLTFIM
                flipping_weights[idx] = 0.0
            end
        elseif qmc_state isa BinaryGroundState || isbondoperator(H, op)  # diagonal bond operator
            site1, site2 = bond = getbondsites(H, op)
            spins = spin_prop[site1], spin_prop[site2]
            num_sites = 2  # length(bond)
            num_legs = 4   # 2*num_sites

            # vertex leg indices
            # thinking of imaginary time as increasing as we move upward,
            # these indices refer to the
            # lower left, lower right, upper left, upper right legs respectively
            # v1, v2, v3, v4 = idx + 1, idx + 2, idx + 3, idx + 4

            @simd for i in 1:num_sites
                v = idx + i
                st = bond[i]
                F = First[st]
                LinkList[v] = F
                if qmc_state isa BinaryGroundState || !iszero(F)
                    LinkList[F] = v  # completes backwards link
                else
                    Last[st] = v
                end
                First[st] = v + num_sites
            end

            @simd for i in 1:num_legs
                v = idx + i
                LegType[v] = spins[mod(i, 1:num_sites)]
                Associates[v] = v + 1
                if H isa AbstractLTFIM
                    flipping_weights[v] = 0.0
                end
            end
            Associates[idx + num_legs] = idx + 1

            if H isa AbstractLTFIM
                @simd for i in 1:num_legs
                    lw = getlogweight(H.op_sampler, (i, op[2] - op[1] + i, site1, site2))
                    flipping_weights[idx + i] = lw
                end
            end

            idx += num_legs
        end
    end

    if qmc_state isa BinaryGroundState
        # The last N elements of the linked list are the final spin state
        @inbounds for i in 1:Ns
            idx += 1
            F = First[i]
            LinkList[idx] = F
            LinkList[F] = idx
            LegType[idx] = spin_prop[i]
            Associates[idx] = 0
            if H isa AbstractLTFIM
                flipping_weights[idx] = 0.0
            end
        end
    else
        #Periodic boundary conditions for finite-beta
        @inbounds for i in 1:Ns
            F = First[i]
            if !iszero(F)  #This might be encountered at high temperatures
                L = Last[i]
                LinkList[F] = L
                LinkList[L] = F
            end
        end
    end
    # @debug statements are not run unless debug logging is enabled
    @debug("Link List basis state propagation status: $(spin_prop == qmc_state.right_config)",
           spin_prop,
           qmc_state.right_config)

    return idx
end
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

                    min_assoc = min(leg, a, straight_thru, Associates[straight_thru])

                    lnA += (
                        flipping_weights[min_assoc - 1 + postflip_bond_type]
                        - flipping_weights[min_assoc - 1 + preflip_bond_type]
                    )
                end
            end
            # heat bath: inv(1 + inv(A))) = W2/(W1 + W2) not good
            # metropolis: A (equiv to min(A, 1)) pretty good
            # scaled metropolis: min(A, 1)/2 also good
            # |M| and M^2 seem to converge better to 99% CIs
            #  when using metropolis (not the scaled variant)
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
