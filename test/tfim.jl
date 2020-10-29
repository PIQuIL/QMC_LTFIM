using Test, QMC
using Statistics
using Random
using RandomNumbers
using Measurements


expected_values = Dict{Tuple{Bool, Int, Float64}, Dict{String, Float64}}(
    (false, 10, 1.0) => Dict("M" => 6.451643863139102e-16,
                             "|M|" => 0.48188450771036156,
                             "M^2" => 0.32936397228034336,
                             "H" => -1.0774361711997833,
                             "C" => 3.085429815831006),

    (true, 10, 1.0) => Dict("M" => 1.4357874060005093e-16,
                            "|M|" => 0.5526982106525444,
                            "M^2" => 0.41362453160389673,
                            "H" => -1.1247786719410742,
                            "C" => 3.236282462798428),
)

# @testset "Ground State" begin
#     @testset "$N-sites, PBC=$PBC" for PBC in [true, false], N in [5, 10]
#         rng = Xorshifts.Xoroshiro128Plus(1234)

#         H = LTFIM((N,), 1.0, 1.0, 1.0, PBC)

#         gs = BinaryGroundState(H, 1000)

#         MCS = 1_000_000
#         EQ_MCS = 10_000

#         mags = zeros(MCS)
#         ns = zeros(MCS)

#         for i in 1:EQ_MCS
#             mc_step!(rng, gs, H)
#         end

#         for i in 1:MCS # Monte Carlo Production Steps
#             mc_step!(rng, gs, H) do lsize, gs, H
#                 spin_prop = sample(H, gs)
#                 ns[i] = num_single_site_diag(H, gs.operator_list)
#                 mags[i] = magnetization(spin_prop)
#             end
#         end

#         abs_mag = mean_and_stderr(abs, mags)
#         mag_sqr = mean_and_stderr(abs2, mags)

#         energy = jackknife(ns) do n
#             if H.hx != 0
#                 (-H.hx * (1.0 / n)) + H.energy_shift / nspins(H)
#             else
#                 H.energy_shift / nspins(H)
#             end
#         end

#         expected_vals = expected_values[(PBC, N, Inf64)]
#         @test stdscore(abs_mag, expected_vals["|M|"]) < 1.0
#         @test stdscore(mag_sqr, expected_vals["M^2"]) < 1.0
#         @test stdscore(energy, expected_vals["H"]) < 1.0
#     end
# end

@testset "Thermal State" begin
    @testset "$N-sites, PBC=$PBC, β=1.0" for PBC in [true, false], N in [10]
        # rng = Xorshifts.Xoroshiro128Plus(1234)
        rng = MersenneTwister(4321)

        bonds, Ns, Nb = lattice_bond_spins((10,), PBC)
        H = TFIM(bonds, 1, Ns, Nb, 1.0, 1.0)
        th = BinaryThermalState(H, 1000)
        beta = 1.0

        MCS = 1_000_000
        EQ_MCS = 100_000

        mags = zeros(MCS)
        ns = zeros(MCS)

        [mc_step_beta!(rng, th, H, beta) for i in 1:EQ_MCS]

        for i in 1:MCS # Monte Carlo Steps
            ns[i] = mc_step_beta!(rng, th, H, beta) do lsize, th, H
                mags[i] = magnetization(sample(H, th))
            end
        end
        abs_mag = mean_and_stderr(abs, mags)
        mag_sqr = mean_and_stderr(abs2, mags)

        energy = mean_and_stderr(x -> -x/beta, ns) + abs(H.J)*nbonds(H) + H.h*nspins(H)
        energy /= nspins(H)

        expected_vals = expected_values[(PBC, N, 1.0)]
        @test abs(stdscore(abs_mag, expected_vals["|M|"])) < 1.0
        @test abs(stdscore(mag_sqr, expected_vals["M^2"])) < 1.0
        @test abs(stdscore(energy, expected_vals["H"])) < 1.0
    end
end