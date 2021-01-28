function line_link_list_update!(rng::AbstractRNG, qmc_state::BinaryQMCState, H::AbstractIsing, runstats=Val{false}())
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
    @inbounds for (n, op) in enumerate(qmc_state.operator_list)
        if issiteoperator(H, op)
            site = op[2]
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
            spin1, spin2 = spins = spin_prop[site1], spin_prop[site2]
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
                lw1 = getlogweight(H.op_sampler, op)

                flip_site1 = getbondtype(H, !spin1, spin2)
                lw2 = getlogweight(H.op_sampler, (flip_site1, site1, site2))
                flipping_weights[idx + 1] = lw2 - lw1

                flip_site2 = getbondtype(H, spin1, !spin2)
                lw2 = getlogweight(H.op_sampler, (flip_site2, site1, site2))
                flipping_weights[idx + 2] = lw2 - lw1
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
line_link_list_update!(qmc_state, H, runstats=Val{false}()) = line_link_list_update!(Random.GLOBAL_RNG, qmc_state, H, runstats)


#############################################################################


function line_cluster_update!(rng::AbstractRNG, lsize::Int, qmc_state::BinaryQMCState, H::AbstractIsing, runstats=Val{false}())
    Ns = nspins(H)
    operator_list = qmc_state.operator_list

    LinkList = qmc_state.linked_list
    LegType = qmc_state.leg_types
    Associates = qmc_state.associates
    flipping_weights = qmc_state.flipping_weights

    in_cluster = fill!(qmc_state.in_cluster, false)
    cstack = qmc_state.cstack # This is the stack of vertices in a cluster
    current_cluster = qmc_state.current_cluster

    if runstats isa Val{true}
        ccount = 0  # cluster number counter
        cluster_sizes = PushVector{Int}()
        acceptance = PushVector{Float64}()
    end

    @inbounds for i in rand(rng, 1:lsize, lsize ÷ 4)  # tune denominator until abort ratio < 10%
        abort = false
        # Add a new leg onto the cluster
        if !in_cluster[i] && Associates[i] == 0
            if runstats isa Val{true}; ccount += 1; end
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
                    lnA += flipping_weights[leg]

                    # now check all associates and add them to the cluster
                    a = Associates[leg]
                    j = 1
                    while a != 0 && !in_cluster[a]
                        in_cluster[a] = true

                        if j == 2
                            push!(cstack, a)
                            push!(current_cluster, a)
                            lnA += flipping_weights[a]
                        end

                        a = Associates[a]
                        j += 1
                    end
                elseif Associates[leg] != 0
                    abort = true
                end
            end
            # heat bath: inv(1 + inv(A))) = W2/(W1 + W2) not good
            # metropolis: A (equiv to min(A, 1)) pretty good
            # scaled metropolis: min(A, 1)/2 also good
            # |M| and M^2 seem to converge better to 99% CIs
            #  when using metropolis (not the scaled variant)
            if abort
                # TODO: count number of aborts
                continue
            end

            # in the TFIM case, acceptance rate is exactly 1
            #   so we set it to 1/2 to ensure ergodicity
            A = exp(min(lnA, zero(lnA)))
            flip = rand(rng) < A

            if runstats isa Val{true}; push!(acceptance, A); end

            if flip
                @inbounds for i in current_cluster
                    LegType[i] ⊻= 1  # spinflip
                end
            end

            if runstats isa Val{true}; push!(cluster_sizes, length(current_cluster)); end
        end
    end

    # map back basis states and operator list
    ocount = _map_back_basis_states!(rng, lsize, qmc_state, H)
    _map_back_operator_list!(ocount, qmc_state, H)

    if runstats isa Val{true}
        return lsize, mean(acceptance), ccount, mean(cluster_sizes)
    else
        return lsize
    end
end

line_cluster_update!(lsize, qmc_state, H, runstats=Val{false}()) = line_cluster_update!(Random.GLOBAL_RNG, lsize, qmc_state, H, runstats)


#############################################################################


function line_update!(rng::AbstractRNG, qmc_state::BinaryQMCState, H::AbstractIsing, runstats=Val{false}())
    lsize = line_link_list_update!(rng, qmc_state, H, runstats)
    return line_cluster_update!(rng, lsize, qmc_state, H, runstats)
end
line_update!(qmc_state::BinaryQMCState, H::AbstractIsing, runstats=Val{false}()) = line_update!(Random.GLOBAL_RNG, qmc_state, H, runstats)
