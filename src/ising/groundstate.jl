# cluster_update!(rng, qmc_state, H::Hamiltonian, runstats; kw...) = multibranch_update!(rng, qmc_state, H, runstats)
function cluster_update!(rng, qmc_state, H::Hamiltonian, d::Diagnostics; p::Float64=0.0, kw...)
    if rand(rng) < p
        multibranch_update!(rng, qmc_state, H, d)
    else
        line_update!(rng, qmc_state, H, d)
    end
end

function cluster_update!(rng, qmc_state, H::AbstractRydberg, d::Diagnostics; p::Float64=0.0, kw...)
    if p < 0
        rydberg_update!(rng, qmc_state, H, d)
    elseif rand(rng) < p
        multibranch_update!(rng, qmc_state, H, d)
    else
        line_update!(rng, qmc_state, H, d)
    end
end


function mc_step!(f::Function, rng::AbstractRNG, qmc_state::BinaryGroundState, H::Hamiltonian, d::Diagnostics; kw...)
    full_diagonal_update!(rng, qmc_state, H, d.runstats.diagonal_update)
    lsize = cluster_update!(rng, qmc_state, H, d; kw...)
    f(lsize, qmc_state, H)
end
mc_step!(f::Function, qmc_state::BinaryGroundState, H::Hamiltonian, d::Diagnostics; kw...) = mc_step!(f, Random.GLOBAL_RNG, qmc_state, H, d; kw...)
mc_step!(rng::AbstractRNG, qmc_state::BinaryGroundState, H::Hamiltonian, d::Diagnostics; kw...) = mc_step!((args...) -> nothing, rng, qmc_state, H, d; kw...)
mc_step!(qmc_state::BinaryGroundState, H::Hamiltonian, d::Diagnostics; kw...) = mc_step!(Random.GLOBAL_RNG, qmc_state, H, d; kw...)


