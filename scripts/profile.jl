using QMC
using Random
using RandomNumbers

Rb = 1.0
omega = 1.0
delta = 1.1
L = 10
beta = 10

H = Rydberg((L, L), Rb, omega, delta; pbc = (true, true), epsilon=0.0)

qmc_state = BinaryThermalState(H, round(Int, 2*beta*QMC.diag_update_normalization(H), RoundUp))


rng = Xorshifts.Xoroshiro128Plus(1234)
rand!(rng, qmc_state.left_config)
copyto!(qmc_state.right_config, qmc_state.left_config)

@profview mc_step_beta!(rng, qmc_state, H, beta, Diagnostics(), eq=true)

d = Diagnostics(RunStats())
@profview_allocs [mc_step_beta!(rng, qmc_state, H, beta, d, eq=true) for _ in 1:100] sample_rate=0.1