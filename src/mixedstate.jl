
########################## finite-beta #######################################

function mc_step_beta!(f::Function, rng::AbstractRNG, qmc_state::BinaryThermalState, H::Hamiltonian, beta::Real; eq::Bool = false)
    num_ops = diagonal_update_beta!(rng, qmc_state, H, beta; eq = eq)

    lsize = link_list_update_beta!(qmc_state, H)

    f(lsize, qmc_state, H)

    cluster_update_beta!(rng, lsize, qmc_state, H)

    return num_ops
end
mc_step_beta!(f::Function, qmc_state, H, beta; eq = false) = mc_step_beta!(f, Random.GLOBAL_RNG, qmc_state, H, beta; eq = eq)
mc_step_beta!(rng::AbstractRNG, qmc_state, H, beta; eq = false) = mc_step_beta!((args...) -> nothing, rng, qmc_state, H, beta; eq = eq)
mc_step_beta!(qmc_state, H, beta; eq = false) = mc_step_beta!(Random.GLOBAL_RNG, qmc_state, H, beta; eq = eq)

function resize_op_list!(qmc_state::BinaryThermalState{N, K}, H::AbstractIsing, new_size::Int) where {N, K}
    operator_list = filter!(op -> !isidentity(H, op), qmc_state.operator_list)
    len = length(operator_list)

    if len < new_size
        tail = init_op_list(new_size - len, K)
        append!(operator_list, tail)
    end

    len = 4*length(operator_list)
    # these are going to be overwritten by link_list_update_beta!
    # which will be called right after the diagonal update that called this function
    resize!(qmc_state.linked_list, len)
    resize!(qmc_state.leg_types, len)
    resize!(qmc_state.associates, len)
end


function diagonal_update_beta!(rng::AbstractRNG, qmc_state::BinaryThermalState, H::TFIM, beta::Real; eq::Bool = false)
    P_norm = beta * H.P_normalization

    num_ids = count(op -> isidentity(H, op), qmc_state.operator_list)
    P_remove = (num_ids + 1) / P_norm
    P_accept = P_norm / num_ids

    spin_prop = copyto!(qmc_state.propagated_config, qmc_state.left_config)  # the propagated spin state

    @inbounds for (n, op) in enumerate(qmc_state.operator_list)
        if !isdiagonal(H, op)
            spin_prop[op[2]] ⊻= 1  # spinflip
        elseif !isidentity(H, op)
            if rand(rng) < P_remove
                qmc_state.operator_list[n] = (0, 0)
                num_ids += 1
                P_remove = (num_ids + 1) / P_norm
                P_accept = P_norm / num_ids
            end
        else
            if rand(rng) < P_accept
                success = insert_diagonal_operator!(rng, qmc_state, H, spin_prop, n)

                if success
                    # save one operation lol
                    P_remove = num_ids / P_norm
                    num_ids -= 1
                    P_accept = P_norm / num_ids
                end
            end
        end
    end

    # DEBUG
    # if spin_prop != qmc_state.right_config  # check the spin propagation for error
    #     error("Basis state propagation error in diagonal update!")
    # end

    total_list_size = length(qmc_state.operator_list)
    num_ops = total_list_size - num_ids

    if eq && 1.2*num_ops > total_list_size
        resize_op_list!(qmc_state, H, round(Int, 1.5*num_ops, RoundUp))
    end

    return num_ops
end
diagonal_update_beta!(qmc_state, H, beta; eq = false) = diagonal_update_beta!(Random.GLOBAL_RNG, qmc_state, H, beta; eq = eq)

#############################################################################

function link_list_update_beta!(qmc_state::BinaryThermalState, H::TFIM)
    Ns = nspins(H)
    spin_left, spin_right = qmc_state.left_config, qmc_state.right_config

    len = 0
    @simd for op in qmc_state.operator_list
        if issiteoperator(H, op)
            len += 2
        elseif isbondoperator(H, op)
            len += 4
        end
    end

    # initialize linked list data structures
    LinkList = qmc_state.linked_list  # needed for cluster update
    LegType = qmc_state.leg_types

    # A diagonal bond operator has non trivial associates for cluster building
    Associates = qmc_state.associates

    First = fill!(qmc_state.first, 0)  #initialize the First list
    Last = fill!(qmc_state.last, 0)   #initialize the Last list
    idx = 0

    spin_prop = copyto!(qmc_state.propagated_config, spin_left)  # the propagated spin state

    # Now, add the 2M operators to the linked list. Each has either 2 or 4 legs
    @inbounds for op in qmc_state.operator_list
        if issiteoperator(H, op)
            site = op[2]
            # lower or left leg
            idx += 1
            LinkList[idx] = First[site]
            LegType[idx] = spin_prop[site]
            current_link = idx

            if !isdiagonal(H, op)  # off-diagonal site operator
                spin_prop[site] ⊻= 1  # spinflip
            end

            if First[site] != 0
                LinkList[First[site]] = current_link  # completes backwards link
            else
                Last[site] = current_link
            end
            First[site] = current_link + 1
            Associates[idx] = (0, 0, 0)

            # upper or right leg
            idx += 1
            LegType[idx] = spin_prop[site]
            Associates[idx] = (0, 0, 0)
        elseif isbondoperator(H, op)  # diagonal bond operator
            site1, site2 = getbondsites(H, op)

            # lower left
            idx += 1
            LinkList[idx] = First[site1]
            LegType[idx] = spin_prop[site1]
            current_link = idx

            if First[site1] != 0
                LinkList[First[site1]] = current_link  # completes backwards link
            else
                Last[site1] = current_link
            end

            First[site1] = current_link + 2
            vertex1 = current_link
            Associates[idx] = (vertex1 + 1, vertex1 + 2, vertex1 + 3)

            # lower right
            idx += 1
            LinkList[idx] = First[site2]
            LegType[idx] = spin_prop[site2]
            current_link = idx

            if First[site2] != 0
                LinkList[First[site2]] = current_link  # completes backwards link
            else
                Last[site2] = current_link
            end

            First[site2] = current_link + 2
            Associates[idx] = (vertex1, vertex1 + 2, vertex1 + 3)

            # upper left
            idx += 1
            LegType[idx] = spin_prop[site1]
            Associates[idx] = (vertex1, vertex1 + 1, vertex1 + 3)

            # upper right
            idx += 1
            LegType[idx] = spin_prop[site2]
            Associates[idx] = (vertex1, vertex1 + 1, vertex1 + 2)
        end
    end

    #Periodic boundary conditions for finite-beta
    @inbounds for i in 1:Ns
        if First[i] != 0  #This might be encountered at high temperatures
            LinkList[First[i]] = Last[i]
            LinkList[Last[i]] = First[i]
        end
    end

    # DEBUG
    # if spin_prop != spin_right
    #     @debug "Basis state propagation error: LINKED LIST"
    # end

    return len
end

#############################################################################

function cluster_update_beta!(rng::AbstractRNG, lsize::Int, qmc_state::BinaryThermalState, H::TFIM)
    Ns = nspins(H)
    spin_left, spin_right = qmc_state.left_config, qmc_state.right_config
    operator_list = qmc_state.operator_list

    LinkList = qmc_state.linked_list
    LegType = qmc_state.leg_types
    Associates = qmc_state.associates

    in_cluster = falses(lsize)
    cstack = Stack{Int}()  # This is the stack of vertices in a cluster
    ccount = 0  # cluster number counter

    @inbounds for i in 1:lsize
        # Add a new leg onto the cluster
        if (!in_cluster[i] && Associates[i] === (0, 0, 0))
            # ccount += 1
            push!(cstack, i)
            in_cluster[i] = true

            flip = rand(rng, Bool)  # flip a coin for the SW cluster flip
            if flip
                LegType[i] ⊻= 1  # spinflip
            end

            while !isempty(cstack)
                leg = LinkList[pop!(cstack)]

                if !in_cluster[leg]
                    in_cluster[leg] = true  # add the new leg and flip it
                    if flip
                        LegType[leg] ⊻= 1
                    end

                    # now check all associates and add to cluster
                    assoc = Associates[leg]  # a 3-tuple
                    if assoc !== (0, 0, 0)
                        for a in assoc
                            push!(cstack, a)
                            in_cluster[a] = true
                            if flip
                                LegType[a] ⊻= 1
                            end
                        end
                    end
                end

            end
        end
    end

    # map back basis states and operator list
    First = qmc_state.first
    Last = qmc_state.last
    @inbounds for i in 1:Ns
        if First[i] != 0
            spin_left[i] = LegType[Last[i]]  # left basis state
            spin_right[i] = LegType[First[i]]  # right basis state
        else
            #randomly flip spins not connected to operators
            spin_left[i] = spin_right[i] = rand(rng, Bool)
        end

    end

    ocount = 1  # first leg
    @inbounds for (n, op) in enumerate(operator_list)
        if isbondoperator(H, op)
            ocount += 4
        elseif !isidentity(H, op)
            if LegType[ocount] == LegType[ocount+1]  # diagonal
                operator_list[n] = (-1, op[2])
            else  # off-diagonal
                operator_list[n] = (-2, op[2])
            end
            ocount += 2
        end
    end
end
cluster_update_beta!(lsize, qmc_state, H) = cluster_update_beta!(Random.GLOBAL_RNG, lsize, qmc_state, H)
