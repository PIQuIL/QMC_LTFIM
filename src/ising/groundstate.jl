function mc_step!(f::Function, rng::AbstractRNG, qmc_state::BinaryGroundState, H::Hamiltonian)
    # diag_update_fails = diagonal_update!(rng, qmc_state, H)
    diagonal_update!(rng, qmc_state, H)

    lsize = link_list_update!(qmc_state, H)

    f(lsize, qmc_state, H)

    # cluster_update_accept, num_clusters, cluster_sizes = cluster_update!(rng, lsize, qmc_state, H)
    cluster_update!(rng, lsize, qmc_state, H)

    # return diag_update_fails, cluster_update_accept, num_clusters, cluster_sizes
end
mc_step!(f::Function, qmc_state, H) = mc_step!(f, Random.GLOBAL_RNG, qmc_state, H)
mc_step!(rng::AbstractRNG, qmc_state, H) = mc_step!((args...) -> nothing, rng, qmc_state, H)
mc_step!(qmc_state, H) = mc_step!(Random.GLOBAL_RNG, qmc_state, H)


@inline alignment_check(::TFIM{N,true}, op::NTuple{2, Int}, s1::Bool, s2::Bool) where N = !xor(s1, s2)
@inline alignment_check(::TFIM{N,false}, op::NTuple{2, Int}, s1::Bool, s2::Bool) where N = xor(s1, s2)


@inline alignment_check(H::LTFIM, op::NTuple{3, Int}, s1::Bool, s2::Bool) =
    (op[1] == getbondtype(H, s1, s2))


# insert_diagonal_operator! returns true if operator insertion succeeded
# returns true if operator insertion succeeded
function insert_diagonal_operator!(rng::AbstractRNG, qmc_state::BinaryQMCState{N}, H::AbstractIsing{N}, spin_prop::BitArray{N}, n::Int) where N
    op = rand(rng, H.op_sampler)
    site1, site2 = getbondsites(H, op)
    @inbounds if issiteoperator(H, op) || alignment_check(H, op, spin_prop[site1], spin_prop[site2])
        qmc_state.operator_list[n] = op
        return true
    else
        return false
    end
end

# function insert_diagonal_operator!(rng::AbstractRNG, qmc_state::BinaryQMCState{N}, H::TFIM{N}, spin_prop::BitArray{N}, n::Int) where N
#     P_h = H.h*nspins(H) / (H.h*nspins(H) + 2*H.J*nbonds(H))

#     if rand(rng) < P_h
#         qmc_state.operator_list[n] = (-1, rand(rng, 1:nspins(H)))
#         return true
#     else
#         site1, site2 = H.bonds[rand(rng, 1:nbonds(H))]
#         if spin_prop[site1] == spin_prop[site2]
#             qmc_state.operator_list[n] = (site1, site2)
#             return true
#         end
#     end
#     return false
# end


function insert_diagonal_operator!(rng::AbstractRNG, qmc_state::BinaryQMCState{N}, H::ArbitraryInteractionTFIM{N}, spin_prop::BitArray{N}, n::Int) where N
    op = rand(rng, H.op_sampler)
    site1, site2 = getbondsites(H, op)
    @inbounds if issiteoperator(H, op) || xor(!signbit(H.J[site1, site2]), spin_prop[site1], spin_prop[site2])
        qmc_state.operator_list[n] = op
        return true
    else
        return false
    end
end
insert_diagonal_operator!(qmc_state, H, spin_prop, n) = insert_diagonal_operator!(Random.GLOBAL_RNG, qmc_state, H, spin_prop, n)

#############################################################################

function diagonal_update!(rng::AbstractRNG, qmc_state::BinaryGroundState{N}, H::AbstractIsing{N}) where N
    spin_prop = copyto!(qmc_state.propagated_config, qmc_state.left_config)  # the propagated spin state

    # failures = Vector{Int}()
    for (n, op) in enumerate(qmc_state.operator_list)
        if !isdiagonal(H, op)
            @inbounds spin_prop[op[2]] ⊻= 1  # spinflip
        else
            success = false
            # i = -1
            while !success
                success = insert_diagonal_operator!(rng, qmc_state, H, spin_prop, n)
                # i += 1
            end
            # push!(failures, i)
        end
    end

    # @debug statements are not run unless debug logging is enabled
    @debug("Diagonal Update basis state propagation status: $(spin_prop == qmc_state.right_config)",
           spin_prop,
           qmc_state.right_config)
    # return mean(failures)
end
diagonal_update!(qmc_state, H) = diagonal_update!(Random.GLOBAL_RNG, qmc_state, H)


#############################################################################

function link_list_update!(qmc_state::BinaryGroundState{N}, H::AbstractIsing{N}) where N
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
        flipping_weights[i] = 0.0
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

            if H isa LTFIM
                s1, s2 = spin_prop[site1], spin_prop[site2]
                lw1 = getlogweight(H.op_sampler, op)
                new_t = getbondtype(H, !s1, !s2)
                lw2 = getlogweight(H.op_sampler, (new_t, site1, site2))
                flipping_weights[idx] = lw2 - lw1
                @simd for l in 1:3
                    flipping_weights[idx + l] = 0.0
                end
            end

            # lower right
            idx += 1
            LinkList[idx] = First[site2]
            LegType[idx] = spin_prop[site2]
            current_link = idx

            LinkList[First[site2]] = current_link  # completes backwards link
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

    # The last N elements of the linked list are the final spin state
    @inbounds for i in 1:Ns
        idx += 1
        LinkList[idx] = First[i]
        LegType[idx] = spin_prop[i]
        LinkList[First[i]] = idx
        Associates[idx] = (0, 0, 0)
        flipping_weights[idx] = 0.0
    end

    # DEBUG
    # @debug statements are not run unless debug logging is enabled
    @debug("Link List basis state propagation status: $(spin_prop == qmc_state.right_config)",
           spin_prop,
           qmc_state.right_config)

    return len
end

#############################################################################

function op_list_weight(qmc_state::BinaryGroundState{N}, H::AbstractIsing{N}) where N
    return prod(op -> getweight(H.op_sampler, op), qmc_state.operator_list)
end

function cluster_update!(rng::AbstractRNG, lsize::Int, qmc_state::BinaryQMCState{N}, H::AbstractLTFIM{N}) where N
    Ns = nspins(H)
    spin_left, spin_right = qmc_state.left_config, qmc_state.right_config
    operator_list = qmc_state.operator_list

    LinkList = qmc_state.linked_list
    LegType = qmc_state.leg_types
    Associates = qmc_state.associates
    flipping_weights = qmc_state.flipping_weights

    in_cluster = falses(lsize)
    cstack = Stack{Int}()  # This is the stack of vertices in a cluster
    current_cluster = Vector{Int}()
    # ccount = 0  # cluster number counter
    # cluster_sizes = Vector{Int}()
    # acceptance = Vector{Float64}()

    @inbounds for i in 1:lsize
        # Add a new leg onto the cluster
        if (!in_cluster[i] && Associates[i] === (0, 0, 0))
            # ccount += 1
            push!(cstack, i)
            in_cluster[i] = true

            empty!(current_cluster)
            push!(current_cluster, i)
            lnA = flipping_weights[i]

            while !isempty(cstack)
                leg = LinkList[pop!(cstack)]

                if !in_cluster[leg]
                    in_cluster[leg] = true  # add the new leg and flip it

                    push!(current_cluster, leg)
                    lnA += flipping_weights[leg]

                    # now check all associates and add to cluster
                    assoc = Associates[leg]  # a 3-tuple
                    if assoc !== (0, 0, 0)
                        for a in assoc
                            push!(cstack, a)
                            in_cluster[a] = true
                            push!(current_cluster, a)
                            lnA += flipping_weights[a]
                        end
                    end
                end
            end
            A = exp(min(lnA, zero(lnA)))
            # push!(acceptance, A)
            # heat bath: inv(1 + inv(A))) = W2/(W1 + W2) not good
            # metropolis: A (equiv to min(A, 1)) pretty good
            # scaled metropolis: min(A, 1)/2 also good
            # |M| and M^2 only seem to converge to 99% CIs
            #  when using metropolis (not the scaled variant)
            flip = rand(rng) < A
            if flip
                @inbounds for i in current_cluster
                    LegType[i] ⊻= 1  # spinflip
                end
            end

            # push!(cluster_sizes, length(current_cluster))
        end
    end

    # map back basis states and operator list
    ocount = _map_back_basis_states!(rng, lsize, qmc_state, H)

    @inbounds for (n, op) in enumerate(operator_list)
        if isbondoperator(H, op)
            s1, s2 = LegType[ocount], LegType[ocount+1]
            t = getbondtype(H, s1, s2)
            site1, site2 = getbondsites(H, op)
            operator_list[n] = (t, site1, site2)
            ocount += 4
        elseif issiteoperator(H, op)
            if LegType[ocount] == LegType[ocount+1]  # diagonal
                operator_list[n] = makediagonalsiteop(H, op[2])
            else  # off-diagonal
                operator_list[n] = makeoffdiagonalsiteop(H, op[2])
            end
            ocount += 2
        end
    end

    # return mean(acceptance), ccount, mean(cluster_sizes)
end


@inline function _map_back_basis_states!(rng::AbstractRNG, lsize::Int, qmc_state::BinaryGroundState, H::AbstractIsing)
    Ns = nspins(H)
    spin_left, spin_right = qmc_state.left_config, qmc_state.right_config
    LegType = qmc_state.leg_types

    @inbounds for i in 1:Ns
        spin_left[i] = LegType[i]  # left basis state
        spin_right[i] = LegType[lsize-Ns+i]  # right basis state
    end
    return Ns + 1  # next on is leg Ns + 1
end

@inline function _map_back_basis_states!(rng::AbstractRNG, lsize::Int, qmc_state::BinaryThermalState, H::AbstractIsing)
    Ns = nspins(H)
    spin_left, spin_right = qmc_state.left_config, qmc_state.right_config
    LegType = qmc_state.leg_types

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
    return 1  # first leg
end


function cluster_update!(rng::AbstractRNG, lsize::Int, qmc_state::BinaryQMCState{N}, H::AbstractTFIM{N}) where N
    Ns = nspins(H)
    spin_left, spin_right = qmc_state.left_config, qmc_state.right_config
    operator_list = qmc_state.operator_list

    LinkList = qmc_state.linked_list
    LegType = qmc_state.leg_types
    Associates = qmc_state.associates

    in_cluster = falses(lsize)
    cstack = Stack{Int}()  # This is the stack of vertices in a cluster
    # ccount = 0  # cluster number counter
    # cluster_sizes = Vector{Int}()

    @inbounds for i in 1:lsize
        # Add a new leg onto the cluster
        # cluster_size = 0
        if (!in_cluster[i] && Associates[i] === (0, 0, 0))
            # ccount += 1
            push!(cstack, i)
            in_cluster[i] = true
            # cluster_size += 1

            flip = rand(rng, Bool)  # flip a coin for the SW cluster flip
            if flip
                LegType[i] ⊻= 1  # spinflip
            end

            while !isempty(cstack)
                leg = LinkList[pop!(cstack)]

                if !in_cluster[leg]
                    in_cluster[leg] = true  # add the new leg and flip it
                    # cluster_size += 1
                    if flip
                        LegType[leg] ⊻= 1
                    end

                    # now check all associates and add to cluster
                    assoc = Associates[leg]  # a 3-tuple
                    if assoc !== (0, 0, 0)
                        for a in assoc
                            push!(cstack, a)
                            in_cluster[a] = true
                            # cluster_size += 1
                            if flip
                                LegType[a] ⊻= 1
                            end
                        end
                    end
                end
            end
            # push!(cluster_sizes, cluster_size)
        end
    end

    # map back basis states and operator list
    ocount = _map_back_basis_states!(rng, lsize, qmc_state, H)

    @inbounds for (n, op) in enumerate(operator_list)
        if isbondoperator(H, op)
            ocount += 4
        elseif issiteoperator(H, op)
            if LegType[ocount] == LegType[ocount+1]  # diagonal
                operator_list[n] = makediagonalsiteop(H, op[2])
            else  # off-diagonal
                operator_list[n] = makeoffdiagonalsiteop(H, op[2])
            end
            ocount += 2
        end
    end

    # return 1/2, ccount, mean(cluster_sizes)
end
cluster_update!(lsize, qmc_state, H) = cluster_update!(Random.GLOBAL_RNG, lsize, qmc_state, H)
