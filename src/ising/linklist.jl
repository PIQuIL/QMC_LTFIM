function link_list_update!(::AbstractRNG, qmc_state::BinaryQMCState, H::AbstractIsing, ::Diagnostics)
    Ns = nspins(H)
    spin_left = qmc_state.left_config

    # retrieve linked list data structures
    LinkList = qmc_state.linked_list  # needed for cluster update
    LegType = qmc_state.leg_types

    # A diagonal bond operator has non trivial associates for cluster building
    Associates = qmc_state.associates

    op_indices = qmc_state.op_indices
    leg_sites = qmc_state.leg_sites

    if qmc_state isa BinaryGroundState
        First = qmc_state.first

        # The first N elements of the linked list are the spins of the LHS basis state
        @inbounds for i in 1:Ns
            LegType[i] = spin_left[i]
            Associates[i] = 0
            First[i] = i
            if H isa AbstractLTFIM
                op_indices[i] = 0
                leg_sites[i] = 0
            end
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
            site = getsite(H, op)
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
                op_indices[idx] = n
                leg_sites[idx] = 1
            end

            if !isdiagonal(H, op)  # off-diagonal site operator
                spin_prop[site] ⊻= 1  # spinflip
            end

            # upper or right leg
            idk += 1
            First[site] = idx
            LegType[idx] = spin_prop[site]
            Associates[idx] = 0
            if H isa AbstractLTFIM
                op_indices[idx] = n
                leg_sites[idx] = 1
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
                m = mod(i, 1:num_sites)
                LegType[v] = spins[m]
                Associates[v] = v + 1
                if H isa AbstractLTFIM
                    op_indices[v] = n
                    leg_sites[v] = m
                end
            end
            Associates[idx + num_legs] = idx + 1
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
                op_indices[idx] = 0
                leg_sites[idx] = 0
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
link_list_update!(qmc_state, H, d::Diagnostics) =
    link_list_update!(Random.GLOBAL_RNG, qmc_state, H, d)


#############################################################################


@inline function _map_back_operator_list!(ocount::Int, qmc_state::BinaryQMCState, H::AbstractIsing, d::Diagnostics)
    operator_list = qmc_state.operator_list
    LegType = qmc_state.leg_types

    # if we build an array that maps n to ocount, this loop will become
    #   very easy to parallelize
    @inbounds for (n, op) in enumerate(operator_list)
        if isbondoperator(H, op)
            if H isa AbstractLTFIM
                s1, s2 = LegType[ocount], LegType[ocount+1]
                t = getbondtype(H, s1, s2)
                operator_list[n] = convertoperatortype(H, op, t)
                fit!(d.tmatrix, op, operator_list[n])
            end
            ocount += 4
        elseif issiteoperator(H, op)
            if LegType[ocount] == LegType[ocount+1]  # diagonal
                operator_list[n] = makediagonalsiteop(H, getsite(H, op))
            else  # off-diagonal
                operator_list[n] = makeoffdiagonalsiteop(H, getsite(H, op))
            end
            fit!(d.tmatrix, op, operator_list[n])
            ocount += 2
        end
    end
end


@inline function _map_back_basis_states!(rng::AbstractRNG, lsize::Int, qmc_state::BinaryGroundState, H::AbstractIsing)
    Ns = nspins(H)
    spin_left, spin_right = qmc_state.left_config, qmc_state.right_config
    LegType = qmc_state.leg_types

    @inbounds for i in 1:Ns
        spin_left[i] = LegType[i]  # left basis state
        spin_right[i] = LegType[lsize-Ns+i]  # right basis state
    end
    return Ns + 1  # next one is leg Ns + 1
end

@inline function _map_back_basis_states!(rng::AbstractRNG, lsize::Int, qmc_state::BinaryThermalState, H::AbstractIsing)
    Ns = nspins(H)
    spin_left, spin_right = qmc_state.left_config, qmc_state.right_config
    LegType = qmc_state.leg_types

    First = qmc_state.first
    Last = qmc_state.last
    @inbounds for i in 1:Ns
        F = First[i]
        if !iszero(F)
            spin_left[i] = LegType[Last[i]]  # left basis state
            spin_right[i] = LegType[F]  # right basis state
        else
            #randomly flip spins not connected to operators
            spin_left[i] = spin_right[i] = rand(rng, Bool)
        end
    end
    return 1  # first leg
end

