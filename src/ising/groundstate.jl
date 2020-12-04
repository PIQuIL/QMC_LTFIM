function mc_step!(f::Function, rng::AbstractRNG, qmc_state::BinaryGroundState, H::Hamiltonian, runstats=Val{false}())
    if runstats isa Val{true}
        diag_update_fails = full_diagonal_update!(rng, qmc_state, H, runstats)
        lsize = link_list_update!(rng, qmc_state, H, runstats)
        cluster_update_accept, num_clusters, cluster_sizes = cluster_update!(rng, lsize, qmc_state, H, runstats)
        f(lsize, qmc_state, H)
        return diag_update_fails, cluster_update_accept, num_clusters, cluster_sizes
    else
        full_diagonal_update!(rng, qmc_state, H)
        lsize = link_list_update!(rng, qmc_state, H)
        cluster_update!(rng, lsize, qmc_state, H)
        f(lsize, qmc_state, H)
    end
end
mc_step!(f::Function, qmc_state::BinaryGroundState, H::Hamiltonian, runstats=Val{false}()) = mc_step!(f, Random.GLOBAL_RNG, qmc_state, H, runstats)
mc_step!(rng::AbstractRNG, qmc_state::BinaryGroundState, H::Hamiltonian, runstats=Val{false}()) = mc_step!((args...) -> nothing, rng, qmc_state, H, runstats)
mc_step!(qmc_state::BinaryGroundState, H::Hamiltonian, runstats=Val{false}()) = mc_step!(Random.GLOBAL_RNG, qmc_state, H, runstats)


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

function full_diagonal_update!(rng::AbstractRNG, qmc_state::BinaryGroundState, H::AbstractIsing, runstats=Val{false}())
    spin_prop = copyto!(qmc_state.propagated_config, qmc_state.left_config)  # the propagated spin state

    if runstats isa Val{true}
        failures = PushVector{Int}()
    end

    for (n, op) in enumerate(qmc_state.operator_list)
        if !isdiagonal(H, op)
            @inbounds spin_prop[op[2]] ⊻= 1  # spinflip
        else
            op = nothing
            if runstats isa Val{true}; i = -1; end
            while op === nothing
                op, _ = insert_diagonal_operator!(rng, qmc_state, H, spin_prop, n)
                if runstats isa Val{true}; i += 1; end
            end
            if runstats isa Val{true}; push!(failures, i); end
        end
    end

    # @debug statements are not run unless debug logging is enabled
    @debug("Diagonal Update basis state propagation status: $(spin_prop == qmc_state.right_config)",
           spin_prop,
           qmc_state.right_config)
    if runstats isa Val{true}
        return mean(failures)
    end
end
full_diagonal_update!(qmc_state, H, runstats=Val{false}()) = full_diagonal_update!(Random.GLOBAL_RNG, qmc_state, H, runstats)


#############################################################################

function link_list_update!(rng::AbstractRNG, qmc_state::BinaryQMCState, H::AbstractIsing, runstats=Val{false}())
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
                Associates[v] = idx + mod(i+1, 1:num_legs)
                if H isa AbstractLTFIM
                    flipping_weights[v] = 0.0
                end
            end

            if H isa AbstractLTFIM
                lw1 = getlogweight(H.op_sampler, op)
                flip_t = getbondtype(H, !spin1, !spin2)
                lw2 = getlogweight(H.op_sampler, (flip_t, site1, site2))
                flipping_weights[idx + 1] = lw2 - lw1
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
link_list_update!(qmc_state, H, runstats=Val{false}()) = link_list_update!(Random.GLOBAL_RNG, qmc_state, H, runstats)

#############################################################################

function cluster_update!(rng::AbstractRNG, lsize::Int, qmc_state::BinaryQMCState, H::AbstractIsing, runstats=Val{false}())
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

    @inbounds for i in 1:lsize
        # Add a new leg onto the cluster
        if (!in_cluster[i] && Associates[i] == 0)
            if runstats isa Val{true}; ccount += 1; end
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

                    # now check all associates and add them to the cluster
                    a = Associates[leg]
                    while a != 0 && !in_cluster[a]
                        push!(cstack, a)
                        in_cluster[a] = true
                        push!(current_cluster, a)
                        lnA += flipping_weights[a]
                        a = Associates[a]
                    end
                end
            end
            # heat bath: inv(1 + inv(A))) = W2/(W1 + W2) not good
            # metropolis: A (equiv to min(A, 1)) pretty good
            # scaled metropolis: min(A, 1)/2 also good
            # |M| and M^2 only seem to converge to 99% CIs
            #  when using metropolis (not the scaled variant)

            # in the TFIM case, acceptance rate is exactly 1
            #   so we set it to 1/2 to ensure ergodicity
            if H isa AbstractLTFIM
                A = iszero(H.hz) ? 0.5 : exp(min(lnA, zero(lnA)))
                flip = rand(rng) < A
            else
                A = 0.5
                flip = rand(rng, Bool)
            end

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
        return mean(acceptance), ccount, mean(cluster_sizes)
    end
end

@inline function _map_back_operator_list!(ocount::Int, qmc_state::BinaryQMCState, H::AbstractIsing)
    operator_list = qmc_state.operator_list
    LegType = qmc_state.leg_types

    # if we build an array that maps n to ocount, this loop will become
    #   very easy to parallelize
    @inbounds for (n, op) in enumerate(operator_list)
        if isbondoperator(H, op)
            if H isa AbstractLTFIM
                s1, s2 = LegType[ocount], LegType[ocount+1]
                t = getbondtype(H, s1, s2)
                site1, site2 = getbondsites(H, op)
                operator_list[n] = (t, site1, site2)
            end
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


function cluster_update!(rng::AbstractRNG, lsize::Int, qmc_state::BinaryQMCState, H::AbstractTFIM, runstats=Val{false}())
    Ns = nspins(H)
    operator_list = qmc_state.operator_list

    LinkList = qmc_state.linked_list
    LegType = qmc_state.leg_types
    Associates = qmc_state.associates

    in_cluster = fill!(qmc_state.in_cluster, false)
    cstack = qmc_state.cstack  # This is the stack of vertices in a cluster

    if runstats isa Val{true}
        ccount = 0  # cluster number counter
        cluster_sizes = PushVector{Int}()
    end

    @inbounds for i in 1:lsize
        # Add a new leg onto the cluster
        if (!in_cluster[i] && Associates[i] == 0)
            if runstats isa Val{true}
                cluster_size = 1
                ccount += 1
            end
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
                    if runstats isa Val{true}; cluster_size += 1; end
                    if flip
                        LegType[leg] ⊻= 1
                    end

                    # now check all associates and add them to the cluster
                    a = Associates[leg]
                    while a != 0 && !in_cluster[a]
                        push!(cstack, a)
                        in_cluster[a] = true
                        if runstats isa Val{true}; cluster_size += 1; end
                        if flip
                            LegType[a] ⊻= 1
                        end
                        a = Associates[a]
                    end
                end
            end
            if runstats isa Val{true}; push!(cluster_sizes, cluster_size); end
        end
    end

    # map back basis states and operator list
    ocount = _map_back_basis_states!(rng, lsize, qmc_state, H)
    _map_back_operator_list!(ocount, qmc_state, H)

    if runstats isa Val{true}
        return 1/2, ccount, mean(cluster_sizes)
    end
end
cluster_update!(lsize, qmc_state, H, runstats=Val{false}()) = cluster_update!(Random.GLOBAL_RNG, lsize, qmc_state, H, runstats)
