# TODO: there might be a slight performance degradation with this
#       functional approach, might wanna try to inject this function directly
#       into the body of the cluster update
@inline function rydberg_kernel!(qmc_state::BinaryQMCState, H::AbstractIsing, ccount::Int, leg::Int, a::Int)
    Ns = nspins(H)
    LegType, leg_sites = qmc_state.leg_types, qmc_state.leg_sites

    @inbounds ll, la = LegType[leg], LegType[a]
    @inbounds sl, sa = leg_sites[leg], leg_sites[a]
    preflip_bond_type = (sl > sa) ? getbondtype(H, ll, la) : getbondtype(H, la, ll)


    if ll == la
        line_kernel!(qmc_state, H, ccount, leg, a)
        postflip_bond_type = (sl > sa) ? getbondtype(H, !ll, la) : getbondtype(H, la, !ll)
    elseif ll != la
        if ll  # incoming leg is a 1-state -> line update
            line_kernel!(qmc_state, H, ccount, leg, a)
            postflip_bond_type = getbondtype(H, false, false)
        else
            multibranch_kernel!(qmc_state, H, ccount, leg, a)
            postflip_bond_type = (sl > sa) ? getbondtype(H, !ll, !la) : getbondtype(H, !la, !ll)
        end
    else
        return line_kernel!(qmc_state, H, ccount, leg, a)
    end

    return preflip_bond_type, postflip_bond_type
end

rydberg_acceptance(H::AbstractIsing, lnA::T) where {T <: Real} = exp(min(lnA, zero(T)))/2

# function rydberg_cluster_update!(rng::AbstractRNG, lsize::Int, qmc_state::BinaryQMCState, H::AbstractIsing, d::Diagnostics)
#     return cluster_update!(rng, rydberg_kernel!, rydberg_acceptance, lsize, qmc_state, H, d)
# end

function rydberg_cluster_update!(rng::AbstractRNG, lsize::Int, qmc_state::BinaryQMCState, H::AbstractIsing, d::Diagnostics)
    Ns = nspins(H)
    operator_list = qmc_state.operator_list

    LinkList = qmc_state.linked_list
    LegType = qmc_state.leg_types
    leg_sites = qmc_state.leg_sites
    Associates = qmc_state.associates
    op_indices = qmc_state.op_indices

    in_cluster = fill!(qmc_state.in_cluster, 0)
    cstack = qmc_state.cstack # This is the stack of vertices in a cluster
    current_cluster = qmc_state.current_cluster
    runstats = d.runstats

    ccount = 0  # cluster number counter

    num_accept = 0
    num_reject = 0

    @inbounds for i in 1:lsize
        # Add a new leg onto the cluster
        if iszero(in_cluster[i]) && iszero(Associates[i])
            ccount += 1
            push!(cstack, i)
            in_cluster[i] = ccount
            cluster_size = 1

            flip = rand(rng, Bool)
            if flip
                LegType[i] ⊻= 1  # spinflip
            end

            while !isempty(cstack)
                leg = LinkList[pop!(cstack)]

                if in_cluster[leg] != ccount
                    in_cluster[leg] = ccount  # add the new leg and flip it
                    cluster_size += 1
                    a = Associates[leg]
                    if flip
                        LegType[leg] ⊻= 1  # spinflip
                    end
                    if a == 0
                        continue
                    end
                    # from this point on, we know we're on a bond op

                    @inbounds ll, la = LegType[leg], LegType[a]

                    if ll == la || ll
                        @inbounds a = Associates[a]
                        @inbounds in_cluster[a] = ccount
                        push!(cstack, a)
                        cluster_size += 1
                        if flip
                            LegType[a] ⊻= 1
                        end
                    elseif ll != la
                        @inbounds while a != leg #&& iszero(in_cluster[a])
                            push!(cstack, a)
                            in_cluster[a] = ccount
                            cluster_size += 1
                            if flip
                                LegType[a] ⊻= 1
                            end
                            a = Associates[a]
                        end
                    end
                end
            end

            # A = rydberg_acceptance(H, lnA)
            fit!(runstats, :cluster_update_accept, 0.5)

            if flip
                fit!(runstats, :accepted_cluster_sizes, cluster_size)
                num_accept += 1
            else
                fit!(runstats, :rejected_cluster_sizes, cluster_size)
                num_reject += 1
            end
            fit!(runstats, :cluster_sizes, cluster_size)
        end
    end

    if !(runstats isa NoStats)
        fit!(runstats, :accepted_cluster_count, num_accept+1)
        fit!(runstats, :rejected_cluster_count, num_reject+1)
        fit!(runstats, :cluster_count, ccount+1)
    end

    # map back basis states and operator list
    ocount = _map_back_basis_states!(rng, lsize, qmc_state, H)
    _map_back_operator_list!(ocount, qmc_state, H, d)

    return lsize
end





#############################################################################


function rydberg_update!(rng::AbstractRNG, qmc_state::BinaryQMCState, H::AbstractIsing, d::Diagnostics)
    lsize = link_list_update!(rng, qmc_state, H, d)
    return rydberg_cluster_update!(rng, lsize, qmc_state, H, d)
end
rydberg_update!(qmc_state::BinaryQMCState, H::AbstractIsing, d::Diagnostics) = rydberg_update!(Random.GLOBAL_RNG, qmc_state, H, d)
