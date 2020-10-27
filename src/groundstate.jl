# updates.jl
#
# Defines the functions that perform the diagonal update, and also
# that build the linked list and operator cluster update

# TFIM ops:
#  (-2,i) is an off-diagonal site operator h(sigma^+_i + sigma^-_i)
#  (-1,i) is a diagonal site operator h
#  (0,0) is the identity operator I - NOT USED IN THE PROJECTOR CASE
#  (i,j) is a diagonal bond operator J(sigma^z_i sigma^z_j)
@inline isdiagonal(H::TFIM, op::NTuple{2,Int}) = @inbounds (op[1] != -2)
@inline isidentity(H::TFIM, op::NTuple{2,Int}) = @inbounds (op[1] == 0)
@inline issiteoperator(H::TFIM, op::NTuple{2,Int}) = @inbounds (op[1] < 0)
@inline isbondoperator(H::TFIM, op::NTuple{2,Int}) = @inbounds (op[1] > 0)
@inline getbondsites(H::TFIM, op::NTuple{2, Int}) = op

# LTFIM ops:
#  (-2,i,0) is an off-diagonal site operator h(sigma^+_i + sigma^-_i)
#  (-1,i,0) is a diagonal site operator h
#  (0,0,0) is the identity operator I - NOT USED IN THE PROJECTOR CASE
#  (t,i,j) is a diagonal bond operator J(sigma^z_i sigma^z_j) + hzb(sigma^z_i + sigma^z_j)
#    t denotes the spin config at sites i,j, subtract 1 then convert to binary
#    t = 1 -> 00 -> down-down
#    t = 2 -> 01 -> down-up
#    t = 3 -> 10 -> up-down
#    t = 4 -> 11 -> up-up
#    spin_config(t) = divrem(t - 1, 2)
#
@inline isdiagonal(H::LTFIM, op::NTuple{3,Int}) = @inbounds (op[1] != -2)
@inline isidentity(H::LTFIM, op::NTuple{3,Int}) = @inbounds (op[1] == 0)
@inline issiteoperator(H::LTFIM, op::NTuple{3,Int}) = @inbounds (op[1] < 0)
@inline isbondoperator(H::LTFIM, op::NTuple{3,Int}) = @inbounds (op[1] > 0)
@inline getbondsites(H::LTFIM, op::NTuple{3, Int}) = @inbounds (op[2], op[3])
@inline getbondtype(H::LTFIM, s1::Bool, s2::Bool) = 2*s1 + s2 + 1

# could try converting to bool as well so we can use ===
@inline spin_config(H::LTFIM, t::Int)::NTuple{2,Int} = divrem(t - 1, 2)
@inline spin_config(H::LTFIM, op::NTuple{3, Int}) = @inbounds spin_config(H, op[1])


function mc_step!(f::Function, qmc_state::BinaryGroundState, H::TFIM)
    diagonal_update!(qmc_state, H)

    cluster_data = linked_list_update(qmc_state, H)

    f(cluster_data, qmc_state, H)

    cluster_update!(cluster_data, qmc_state, H)
end

mc_step!(qmc_state, H) = mc_step!((args...) -> nothing, qmc_state, H)

########################## finite-beta #######################################

function mc_step_beta!(f::Function, qmc_state::BinaryThermalState, H::TFIM, beta::Real; eq::Bool = false)
    num_ops = diagonal_update_beta!(qmc_state, H, beta; eq = eq)

    cluster_data = linked_list_update_beta(qmc_state, H)

    f(cluster_data, qmc_state, H)

    cluster_update_beta!(cluster_data, qmc_state, H)

    return num_ops
end

mc_step_beta!(qmc_state, H, beta; eq = false) = mc_step_beta!((args...) -> nothing, qmc_state, H, beta; eq = eq)


@inline alignment_check(::TFIM{N,true}, op::NTuple{2, Int}, s1::Bool, s2::Bool) where N = !xor(s1, s2)
@inline alignment_check(::TFIM{N,false}, op::NTuple{2, Int}, s1::Bool, s2::Bool) where N = xor(s1, s2)


@inline function alignment_check(H::LTFIM, op::NTuple{3, Int}, s1::Bool, s2::Bool)
    # l = ((true, true), (true, false), (false, true), (false, false))[rand(H.p_spins)]
    # return l === (s1, s2) || l === (!s1, !s2)
    # return !iszero(bond_weight(s1, s2, H))
    return spin_config(H, op) == (s1, s2)
end


# insert_diagonal_operator! returns true if operator insertion succeeded
# returns true if operator insertion succeeded
function insert_diagonal_operator!(qmc_state::BinaryQMCState{N}, H::TFIM{N}, spin_prop::BitArray{N}, n::Int) where N
    op = rand(H.op_sampler)
    site1, site2 = getbondsites(H, op)
    @inbounds if issiteoperator(H, op) || alignment_check(H, op, spin_prop[site1], spin_prop[site2])
        qmc_state.operator_list[n] = op
        return true
    else
        return false
    end
end

function insert_diagonal_operator!(qmc_state::BinaryQMCState{N}, H::ArbitraryInteractionTFIM{N}, spin_prop::BitArray{N}, n::Int) where N
    op = rand(H.op_sampler)
    site1, site2 = getbondsites(H, op)
    @inbounds F = !signbit(H.J[site1, site2])
    @inbounds if issiteoperator(H, op) || xor(F, spin_prop[site1], spin_prop[site2])
        qmc_state.operator_list[n] = op
        return true
    else
        return false
    end
end

#############################################################################

function diagonal_update!(qmc_state::BinaryGroundState{N}, H::AbstractIsing{N}) where N
    spin_prop = copyto!(qmc_state.propagated_config, qmc_state.left_config)  # the propagated spin state

    for (n, op) in enumerate(qmc_state.operator_list)
        if !isdiagonal(H, op)
            @inbounds spin_prop[op[2]] ⊻= 1  # spinflip
        else
            success = false
            while !success
                success = insert_diagonal_operator!(qmc_state, H, spin_prop, n)
            end
        end
    end

    # DEBUG
    # if spin_prop != qmc_state.right_config  # check the spin propagation for error
    #     error("Basis state propagation error in diagonal update!")
    # end
end


#############################################################################

function linked_list_update(qmc_state::BinaryGroundState{N}, H::AbstractIsing{N}) where N
    Ns = nspins(H)
    spin_left = qmc_state.left_config

    len = 2 * Ns
    @simd for op in qmc_state.operator_list
        len += ifelse(issiteoperator(H, op), 2, 4)
    end

    # retrieve linked list data structures
    LinkList = qmc_state.linked_list  # needed for cluster update
    LegType = qmc_state.leg_types

    # A diagonal bond operator has non trivial associates for cluster building
    Associates = qmc_state.associates

    flipping_weights = qmc_state.flipping_weights

    First = qmc_state.first

    # The first N elements of the linked list are the spins of the LHS basis state
    @inbounds for i in 1:Ns
        LegType[i] = spin_left[i]
        First[i] = i
        Associates[i] = (0, 0, 0)
        flipping_weights[i] = 1.0
    end

    spin_prop = copyto!(qmc_state.propagated_config, spin_left)  # the propagated spin state
    idx = Ns

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

            LinkList[First[site]] = current_link  # completes backwards link
            First[site] = current_link + 1
            Associates[idx] = (0, 0, 0)
            flipping_weights[idx] = 1.0

            # upper or right leg
            idx += 1
            LegType[idx] = spin_prop[site]
            Associates[idx] = (0, 0, 0)
            flipping_weights[idx] = 1.0
        else  # diagonal bond operator
            site1, site2 = getbondsites(H, op)

            # lower left
            idx += 1
            LinkList[idx] = First[site1]
            LegType[idx] = spin_prop[site1]
            current_link = idx

            LinkList[First[site1]] = current_link  # completes backwards link
            First[site1] = current_link + 2
            vertex1 = current_link
            Associates[idx] = (vertex1 + 1, vertex1 + 2, vertex1 + 3)
            flipping_weights[idx] = 1.0

            if H isa LTFIM
                s1, s2 = spin_prop[site1], spin_prop[site2]
                w1 = getweight(H.op_sampler, op)
                new_t = getbondtype(H, !s1, !s2)
                w2 = getweight(H.op_sampler, (new_t, site1, site2))
                flipping_weights[idx] = w1/w2
            end

            # lower right
            idx += 1
            LinkList[idx] = First[site2]
            LegType[idx] = spin_prop[site2]
            current_link = idx

            LinkList[First[site2]] = current_link  # completes backwards link
            First[site2] = current_link + 2
            Associates[idx] = (vertex1, vertex1 + 2, vertex1 + 3)
            flipping_weights[idx] = 1.0

            # upper left
            idx += 1
            LegType[idx] = spin_prop[site1]
            Associates[idx] = (vertex1, vertex1 + 1, vertex1 + 3)
            flipping_weights[idx] = 1.0

            # upper right
            idx += 1
            LegType[idx] = spin_prop[site2]
            Associates[idx] = (vertex1, vertex1 + 1, vertex1 + 2)
            flipping_weights[idx] = 1.0
        end
    end

    # The last N elements of the linked list are the final spin state
    @inbounds for i in 1:Ns
        idx += 1
        LinkList[idx] = First[i]
        LegType[idx] = spin_prop[i]
        LinkList[First[i]] = idx
        Associates[idx] = (0, 0, 0)
        flipping_weights[idx] = 1.0
    end

    # DEBUG
    # if spin_prop != spin_right
    #     @debug "Basis state propagation error: LINKED LIST"
    # end

    return len
end

#############################################################################

function cluster_update!(lsize::Int, qmc_state::BinaryGroundState{N}, H::LTFIM{N}) where N
    Ns = nspins(H)
    spin_left, spin_right = qmc_state.left_config, qmc_state.right_config
    operator_list = qmc_state.operator_list

    LinkList = qmc_state.linked_list
    LegType = qmc_state.leg_types
    Associates = qmc_state.associates
    flipping_weights = qmc_state.flipping_weights

    in_cluster = zeros(Int, lsize)
    cstack = Stack{Int}()  # This is the stack of vertices in a cluster
    current_cluster = Vector{Int}()
    ccount = 0  # cluster number counter

    @inbounds for i in 1:lsize
        # Add a new leg onto the cluster
        if (in_cluster[i] == 0 && Associates[i] === nullt)
            ccount += 1
            push!(cstack, i)
            in_cluster[i] = ccount

            push!(current_cluster, i)

            while !isempty(cstack)
                leg = LinkList[pop!(cstack)]

                if in_cluster[leg] == 0
                    in_cluster[leg] = ccount  # add the new leg and flip it
                    push!(current_cluster, leg)

                    # now check all associates and add to cluster
                    assoc = Associates[leg]  # a 3-tuple
                    if assoc !== nullt
                        for a in assoc
                            push!(cstack, a)
                            in_cluster[a] = ccount
                            push!(current_cluster, a)
                        end
                    end
                end
            end

            invA = prod(flipping_weights[current_cluster])
            flip = rand() < inv(1 + invA)
            if flip
                @inbounds for i in current_cluster
                    LegType[i] ⊻= 1  # spinflip
                end
            end
            empty!(current_cluster)
        end
    end

    # map back basis states and operator list
    @inbounds for i in 1:Ns
        spin_left[i] = LegType[i]  # left basis state
        spin_right[i] = LegType[lsize-Ns+i]  # right basis state
    end

    ocount = Ns + 1  # next on is leg Ns + 1
    @inbounds for (n, op) in enumerate(operator_list)
        if isbondoperator(H, op)
            s1, s2 = LegType[ocount], LegType[ocount+1]
            t = getbondtype(H, s1, s2)
            site1, site2 = getbondsites(H, op)
            operator_list[n] = (t, site1, site2)
            ocount += 4
        else
            if LegType[ocount] == LegType[ocount+1]  # diagonal
                operator_list[n] = (-1, op[2], 0)
            else  # off-diagonal
                operator_list[n] = (-2, op[2], 0)
            end
            ocount += 2
        end
    end

end


function cluster_update!(lsize::Int, qmc_state::BinaryGroundState{N}, H::TFIM{N}) where N
    Ns = nspins(H)
    spin_left, spin_right = qmc_state.left_config, qmc_state.right_config
    operator_list = qmc_state.operator_list

    LinkList = qmc_state.linked_list
    LegType = qmc_state.leg_types
    Associates = qmc_state.associates

    in_cluster = zeros(Int, lsize)
    cstack = Stack{Int}()  # This is the stack of vertices in a cluster
    ccount = 0  # cluster number counter

    @inbounds for i in 1:lsize
        # Add a new leg onto the cluster
        if (in_cluster[i] == 0 && Associates[i] === (0, 0, 0))
            ccount += 1
            push!(cstack, i)
            in_cluster[i] = ccount

            flip = rand(Bool)  # flip a coin for the SW cluster flip
            if flip
                LegType[i] ⊻= 1  # spinflip
            end

            while !isempty(cstack)
                leg = LinkList[pop!(cstack)]

                if in_cluster[leg] == 0
                    in_cluster[leg] = ccount  # add the new leg and flip it
                    if flip
                        LegType[leg] ⊻= 1
                    end

                    # now check all associates and add to cluster
                    assoc = Associates[leg]  # a 3-tuple
                    if assoc !== (0, 0, 0)
                        for a in assoc
                            push!(cstack, a)
                            in_cluster[a] = ccount
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
    @inbounds for i in 1:Ns
        spin_left[i] = LegType[i]  # left basis state
        spin_right[i] = LegType[lsize-Ns+i]  # right basis state
    end

    ocount = Ns + 1  # next on is leg Ns + 1
    @inbounds for (n, op) in enumerate(operator_list)
        if isbondoperator(H, op)
            ocount += 4
        else
            if LegType[ocount] == LegType[ocount+1]  # diagonal
                operator_list[n] = (-1, op[2])
            else  # off-diagonal
                operator_list[n] = (-2, op[2])
            end
            ocount += 2
        end
    end

end
