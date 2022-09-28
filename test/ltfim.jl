using Test, QMC
using Statistics
using Random
using RandomNumbers
using Measurements
using BinningAnalysis
import Distributions: TDist


expected_values = Dict{Tuple{Bool, Int, Float64}, Dict{String, Float64}}(
    (false, 10, Inf64) => Dict("M" => 0.9300787137837009,
                               "|M|" => 0.9300800988262218,
                               "M^2" => 0.879412342055611,
                               "H" => -2.0818751039065506),

    (false, 10, 1.0) => Dict("M" => 0.9152481757564336,
                             "|M|" => 0.9152717961127222,
                             "M^2" => 0.8564384998357667,
                             "H" => -2.051491413329706,
                             "C" => 1.5664917850898519),

    (true, 10, Inf64) => Dict("M" => 0.9424709532842519,
                              "|M|" => 0.9424716425300058,
                              "M^2" => 0.9001728915806667,
                              "H" => -2.1664826655166842),

    (true, 10, 1.0) => Dict("M" => 0.9339108974848992,
                            "|M|" => 0.9339228094722204,
                            "M^2" => 0.8868178763718234,
                            "H" => -2.1469676306665013,
                            "C" => 1.1680236801192336),

    (false, 5, Inf64) => Dict("M" => 0.9176830687768431,
                              "|M|" => 0.9183612642212718,
                              "M^2" => 0.8757671253668294,
                              "H" => -1.9972680527419864),

    (false, 5, 1.0) => Dict("M" => 0.8964725143596023,
                            "|M|" => 0.8991340140307242,
                            "M^2" => 0.8496226242573195,
                            "H" => -1.9559435941739651,
                            "C" => 0.984259674060965),

    (true, 5, Inf64) => Dict("M" => 0.9424692513081341,
                             "|M|" => 0.9427934100320888,
                             "M^2" => 0.9120933231955718,
                             "H" => -2.166482937646996),

    (true, 5, 1.0) => Dict("M" => 0.9337493597827439,
                           "|M|" => 0.9349728326000805,
                           "M^2" => 0.9014248358212859,
                           "H" => -2.1468390919494333,
                           "C" => 0.5884527794434007)
)


@testset "1D LTFIM Ground State $N-sites, PBC=$PBC" for PBC in [true], N in [5, 10]
    rng = Xorshifts.Xoroshiro128Plus(1234)

    H = LTFIM((N,), 1.0, 1.0, 1.0, PBC)

    gs = BinaryGroundState(H, 1000)

    MCS = 1_000_000
    EQ_MCS = 10_000

    d = Diagnostics()

    [mc_step!(rng, gs, H, d) for _ in 1:EQ_MCS]

    mags = zeros(MCS)
    for i in 1:MCS # Monte Carlo Production Steps
        mc_step!(rng, gs, H, d) do lsize, gs, H
            spin_prop = sample(H, gs)
            mags[i] = magnetization(spin_prop)
        end
    end
    A = LogBinner(abs.(mags))
    S = LogBinner(abs2.(mags))
    abs_mag = measurement(mean(A), std_error(A))
    mag_sqr = measurement(mean(S), std_error(S))

    N_eff_A = floor(Int64, MCS / (2tau(A) + 1))
    N_eff_S = floor(Int64, MCS / (2tau(S) + 1))

    expected_vals = expected_values[(PBC, N, Inf64)]
    @test abs(abs_mag.val - expected_vals["|M|"]) < THRESHOLD * abs_mag.err
    @test abs(mag_sqr.val - expected_vals["M^2"]) < THRESHOLD * mag_sqr.err
end


@testset "1D LTFIM Thermal State $N-sites, PBC=$PBC, β=1.0" for PBC in [true], N in [5, 10]
    rng = Xorshifts.Xoroshiro128Plus(1234)

    H = LTFIM((N,), 1.0, 1.0, 1.0, PBC)
    th = BinaryThermalState(H, 1000)
    beta = 1.0

    MCS = 1_000_000
    EQ_MCS = 100_000

    d = Diagnostics()

    [mc_step_beta!(rng, th, H, beta, d; eq=true) for i in 1:EQ_MCS]

    mags = zeros(MCS)
    ns = zeros(Int, MCS)
    for i in 1:MCS # Monte Carlo Steps
        ns[i] = mc_step_beta!(rng, th, H, beta, d) do lsize, th, H
            mags[i] = magnetization(sample(H, th))
        end
    end
    A = LogBinner(abs.(mags))
    S = LogBinner(abs2.(mags))
    E = LogBinner(-ns/beta)
    abs_mag = measurement(mean(A), std_error(A))
    mag_sqr = measurement(mean(S), std_error(S))
    energy = measurement(mean(E), std_error(E)) + H.energy_shift
    energy /= nspins(H)

    N_eff_A = floor(Int64, MCS / (2tau(A) + 1))
    N_eff_S = floor(Int64, MCS / (2tau(S) + 1))
    N_eff_E = floor(Int64, MCS / (2tau(E) + 1))

    expected_vals = expected_values[(PBC, N, 1.0)]
    @test abs(abs_mag.val - expected_vals["|M|"]) < THRESHOLD * abs_mag.err
    @test abs(mag_sqr.val - expected_vals["M^2"]) < THRESHOLD * mag_sqr.err
    @test abs(energy.val - expected_vals["H"]) < THRESHOLD * energy.err
end
