
########################## finite-beta #######################################

function mc_step_beta!(f::Function, rng::AbstractRNG, qmc_state::BinaryThermalState{N}, H::AbstractIsing{N}, beta::Real; eq::Bool = false) where N
    num_ops = diagonal_update_beta!(rng, qmc_state, H, beta; eq = eq)

    lsize = link_list_update_beta!(qmc_state, H)

    f(lsize, qmc_state, H)

    cluster_update!(rng, lsize, qmc_state, H)

    return num_ops
end
mc_step_beta!(f::Function, qmc_state, H, beta; eq = false) = mc_step_beta!(f, Random.GLOBAL_RNG, qmc_state, H, beta; eq = eq)
mc_step_beta!(rng::AbstractRNG, qmc_state, H, beta; eq = false) = mc_step_beta!((args...) -> nothing, rng, qmc_state, H, beta; eq = eq)
mc_step_beta!(qmc_state, H, beta; eq = false) = mc_step_beta!(Random.GLOBAL_RNG, qmc_state, H, beta; eq = eq)

function resize_op_list!(qmc_state::BinaryThermalState{N, K}, H::AbstractIsing{N}, new_size::Int) where {N, K}
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


function diagonal_update_beta!(rng::AbstractRNG, qmc_state::BinaryThermalState{N}, H::AbstractIsing{N}, beta::Real; eq::Bool = false) where N
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
                qmc_state.operator_list[n] = makeidentity(H)
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

function link_list_update_beta!(qmc_state::BinaryThermalState{N}, H::AbstractIsing{N}) where N
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

    flipping_weights = qmc_state.flipping_weights

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
            if H isa LTFIM
                flipping_weights[idx] = 0.0
            end

            # upper or right leg
            idx += 1
            LegType[idx] = spin_prop[site]
            Associates[idx] = (0, 0, 0)
            if H isa LTFIM
                flipping_weights[idx] = 0.0
            end
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

            if H isa LTFIM
                s1, s2 = spin_prop[site1], spin_prop[site2]
                if xor(s1, s2)
                    # no weight change if spins are anti-parallel
                    # NOTE: this simplification does not apply in the
                    #       case of a non-uniform z-field
                    @simd for l in 0:3
                        flipping_weights[idx + l] = 0.0
                    end
                else
                    lw1 = getlogweight(H.op_sampler, op)
                    new_t = getbondtype(H, !s1, !s2)
                    lw2 = getlogweight(H.op_sampler, (new_t, site1, site2))
                    flipping_weights[idx] = lw2 - lw1
                    @simd for l in 1:3
                        flipping_weights[idx + l] = 0.0
                    end
                end
            end

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
