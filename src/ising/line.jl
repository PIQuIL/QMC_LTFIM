# TODO: there might be a slight performance degradation with this
#       functional approach, might wanna try to inject this function directly
#       into the body of the cluster update
@inline function line_kernel!(qmc_state::BinaryQMCState, H::AbstractIsing, ccount::Int, leg::Int, a::Int)
    Ns = nspins(H)
    cluster_data = qmc_state.cluster_data
    LegType, Associates, leg_sites = cluster_data.leg_types, cluster_data.associates, cluster_data.leg_sites
    in_cluster, cstack, current_cluster = cluster_data.in_cluster, cluster_data.cstack, cluster_data.current_cluster

    @inbounds ll, la = LegType[leg], LegType[a]
    @inbounds sl, sa = leg_sites[leg], leg_sites[a]
    # TODO: check if this is inputting the spins in the correct order
    if sl > sa
        preflip_bond_type = getbondtype(H, ll, la)
        postflip_bond_type = getbondtype(H, !ll, la)
    else
        preflip_bond_type = getbondtype(H, la, ll)
        postflip_bond_type = getbondtype(H, la, !ll)
    end
    # ^using circshift for this is too slow, gonna have to manually expand out
    #  a bunch of ifs for k-local (might need to do some metaprogramming!)
    # using SVectors and sortperm should work (it's super fast!) and is easier
    # also, should short-circuit if order doesn't matter, i.e.
    #  the longitudinal field is uniform + coordination numbers are all equal

    # now add the straight-through leg to the cluster
    @inbounds straight_thru = Associates[a]
    @inbounds in_cluster[straight_thru] = ccount
    push!(cstack, straight_thru)
    push!(current_cluster, straight_thru)

    return preflip_bond_type, postflip_bond_type
end

line_acceptance(H::AbstractIsing, lnA::T) where {T <: Real} = exp(min(lnA, zero(T)))

function line_cluster_update!(rng::AbstractRNG, lsize::Int, qmc_state::BinaryQMCState, H::AbstractIsing, runstats::AbstractRunStats=NoStats())
    return cluster_update!(rng, line_kernel!, line_acceptance, lsize, qmc_state, H, runstats)
end

line_cluster_update!(lsize, qmc_state, H, runstats::AbstractRunStats=NoStats()) = line_cluster_update!(Random.GLOBAL_RNG, lsize, qmc_state, H, runstats)


#############################################################################


function line_update!(rng::AbstractRNG, qmc_state::BinaryQMCState, H::AbstractIsing, runstats::AbstractRunStats=NoStats())
    lsize = link_list_update!(rng, qmc_state, H, runstats)
    return line_cluster_update!(rng, lsize, qmc_state, H, runstats)
end
line_update!(qmc_state::BinaryQMCState, H::AbstractIsing, runstats::AbstractRunStats=NoStats()) = line_update!(Random.GLOBAL_RNG, qmc_state, H, runstats)
