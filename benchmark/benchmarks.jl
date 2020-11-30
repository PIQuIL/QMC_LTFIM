using BenchmarkTools
using QMC

using Random

SUITE = BenchmarkGroup()

SUITE["probability_vector"] = BenchmarkGroup()

for n in 10:10:100
    SUITE["probability_vector"][n] = BenchmarkGroup()
    SUITE["probability_vector"][n]["vector"] =
        @benchmarkable rand(rng, p_vec) setup=(rng = MersenneTwister(1234); p_vec = ProbabilityVector(rand(rng, 1:10, $n)))
    SUITE["probability_vector"][n]["heap"] =
        @benchmarkable rand(rng, p_heap) setup=(rng = MersenneTwister(1234); p_heap = ProbabilityHeap(rand(rng, 1:10, $n)))
    SUITE["probability_vector"][n]["alias"] =
        @benchmarkable rand(rng, p_alias) setup=(rng = MersenneTwister(1234); p_alias = ProbabilityAlias(rand(rng, 1:10, $n)))
end


SUITE["TFIM_groundstate"] = BenchmarkGroup()
SUITE["TFIM_thermalstate"] = BenchmarkGroup()
SUITE["LTFIM_groundstate"] = BenchmarkGroup()
SUITE["LTFIM_thermalstate"] = BenchmarkGroup()

for M = 200:200:1000
    bonds, Ns, Nb = lattice_bond_spins(10)
    HT = TFIM(bonds, Ns, Nb, 1.0, 1.0)
    HL = LTFIM((10,), 1.0, 1.0, 1.0)

    ###########################################################################

    for (H, gs) in [(HT, "TFIM_groundstate"), (HL, "LTFIM_groundstate")]
        SUITE[gs][M] = BenchmarkGroup()

        SUITE[gs][M]["full_diagonal_update"] =
            @benchmarkable(QMC.full_diagonal_update!(rng, groundstate, $H),
                           setup=(rng = MersenneTwister(1234);
                                  groundstate = BinaryGroundState($H, $M)))

        SUITE[gs][M]["linked_list_update"] =
            @benchmarkable(QMC.link_list_update!(rng, groundstate, $H),
                           setup=(rng = MersenneTwister(1234);
                                  groundstate = BinaryGroundState($H, $M);
                                  QMC.full_diagonal_update!(rng, groundstate, $H)))

        SUITE[gs][M]["cluster_update"] =
            @benchmarkable(QMC.cluster_update!(rng, lsize, groundstate, $H),
                           setup=(rng = MersenneTwister(1234);
                                  groundstate = BinaryGroundState($H, $M);
                                  QMC.full_diagonal_update!(rng, groundstate, $H);
                                  lsize = QMC.link_list_update!(rng, groundstate, $H)))

        SUITE[gs][M]["mc_step"] =
            @benchmarkable(QMC.mc_step!(rng, groundstate, $H),
                           setup=(rng = MersenneTwister(1234);
                                  groundstate = BinaryGroundState($H, $M)))
    end

    ###########################################################################


    for (H, ts) in [(HT, "TFIM_thermalstate"), (HL, "LTFIM_thermalstate")]
        SUITE[ts][M] = BenchmarkGroup()
        beta = 10.0

        SUITE[ts][M]["full_diagonal_update"] =
            @benchmarkable(QMC.full_diagonal_update_beta!(rng, thermalstate, $H, $beta),
                           setup=(rng = MersenneTwister(1234);
                                  thermalstate = BinaryThermalState($H, $M)))

        SUITE[ts][M]["linked_list_update"] =
            @benchmarkable(QMC.link_list_update!(rng, thermalstate, $H),
                           setup=(rng = MersenneTwister(1234);
                                  thermalstate = BinaryThermalState($H, $M);
                                  QMC.full_diagonal_update_beta!(rng, thermalstate, $H, $beta)))

        SUITE[ts][M]["cluster_update"] =
            @benchmarkable(QMC.cluster_update!(rng, lsize, thermalstate, $H),
                           setup=(rng = MersenneTwister(1234);
                                  thermalstate = BinaryThermalState($H, $M);
                                  QMC.full_diagonal_update_beta!(rng, thermalstate, $H, $beta);
                                  lsize = QMC.link_list_update!(rng, thermalstate, $H)))

        SUITE[ts][M]["mc_step"] =
            @benchmarkable(QMC.mc_step_beta!(rng, thermalstate, $H, $beta),
                           setup=(rng = MersenneTwister(1234);
                                  thermalstate = BinaryThermalState($H, $M)))
    end
end
