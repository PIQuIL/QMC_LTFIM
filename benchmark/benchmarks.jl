using BenchmarkTools
using QMC

using Random
Random.seed!(1234)

SUITE = BenchmarkGroup()

SUITE["probability_vector"] = BenchmarkGroup()

for n in 10:10:100
    p = rand(1:10, n)

    p_vec = ProbabilityVector(p)
    p_heap = ProbabilityHeap(p)

    SUITE["probability_vector"][n] = BenchmarkGroup()
    SUITE["probability_vector"][n]["vector"] = @benchmarkable rand($p_vec)
    SUITE["probability_vector"][n]["heap"] = @benchmarkable rand($p_heap)
end


SUITE["TFIM_groundstate"] = BenchmarkGroup()
SUITE["TFIM_thermalstate"] = BenchmarkGroup()
SUITE["LTFIM_groundstate"] = BenchmarkGroup()
SUITE["LTFIM_thermalstate"] = BenchmarkGroup()

for M = 200:200:1000
    bonds, Ns, Nb = lattice_bond_spins(10)
    HT = TFIM(bonds, 1, Ns, Nb, 1.0, 1.0)
    HL = LTFIM((10,), 1.0, 1.0, 1.0)

    ###########################################################################

    for (H, gs) in [(HT, "TFIM_groundstate"), (HL, "LTFIM_groundstate")]
        SUITE[gs][M] = BenchmarkGroup()
        groundstate = BinaryGroundState(H, M)

        SUITE[gs][M]["diagonal_update"] =
            @benchmarkable QMC.diagonal_update!($groundstate, $H)
        SUITE[gs][M]["linked_list_update"] =
            @benchmarkable QMC.link_list_update!($groundstate, $H)
        SUITE[gs][M]["cluster_update"] =
            @benchmarkable(QMC.cluster_update!(lsize, $groundstate, $H),
                        setup=(lsize = QMC.link_list_update!($groundstate, $H)))

        SUITE[gs][M]["mc_step"] = @benchmarkable QMC.mc_step!($groundstate, $H)
    end

    ###########################################################################


    for (H, ts) in [(HT, "TFIM_thermalstate")] #, (HL, "LTFIM_thermalstate")]
        SUITE[ts][M] = BenchmarkGroup()
        beta = 10.0
        thermalstate = BinaryThermalState(H, M)

        SUITE[ts][M]["diagonal_update"] =
            @benchmarkable QMC.diagonal_update_beta!($thermalstate, $H, $beta)
        SUITE[ts][M]["linked_list_update"] =
            @benchmarkable QMC.link_list_update_beta!($thermalstate, $H)
        SUITE[ts][M]["cluster_update"] =
            @benchmarkable(QMC.cluster_update_beta!(lsize, $thermalstate, $H),
                        setup=(lsize = QMC.link_list_update_beta!($thermalstate, $H)))

        SUITE[ts][M]["mc_step"] = @benchmarkable QMC.mc_step_beta!($thermalstate, $H, $beta)
    end
end
