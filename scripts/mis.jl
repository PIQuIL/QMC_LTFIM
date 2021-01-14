using DrWatson
@quickactivate "QMC"

using QMC

using RydbergEmulator
using LinearAlgebra
using LightGraphs
using Random
using RandomNumbers
using ProgressMeter

using Measurements
using Statistics
using FFTW

using DelimitedFiles
using JLD2
using Printf

using DataStructures
using ArgParse


function load_instance(n, ff)
    L = ceil(Int, sqrt(n/ff))
    file = joinpath(@__DIR__, "graphs", "$L-$n-$ff.atoms")
    if isfile(file)
        return include(file)
    else
        @warn "cannot find data file: $file, generate new file"
        save(n, ff)
    end
end

function Rydberg(graph, C::Float64, Ω::Float64, δ::Float64)
    V = zeros(20, 20)
    for e in edges(graph)
        V[e.dst, e.src] = C/RydbergEmulator.distance(atoms[e.dst], atoms[e.src])^6
    end
    V = UpperTriangular(V')
    return Rydberg(V, Ω*ones(nv(graph)), δ*ones(nv(graph)))
end

function make_info_file(info_file, samples_file, mc_opts, op_list_length, observables, corr_time)
    M, MCS, EQ_MCS, skip = mc_opts
    if length(observables) == 5
        mag, abs_mag, mag_sqr, energy, heat_capacity = observables
    else
        mag, abs_mag, mag_sqr, energy = observables
        heat_capacity = nothing
    end

    open(info_file, "w") do file_io
        streams = [Base.stdout, file_io]

        for io in streams
            @printf(io, "⟨M⟩   = % .16f +/- %.16f\n", mag.val, mag.err)
            @printf(io, "⟨|M|⟩ = % .16f +/- %.16f\n", abs_mag.val, abs_mag.err)
            @printf(io, "⟨M^2⟩ = % .16f +/- %.16f\n", mag_sqr.val, mag_sqr.err)
            @printf(io, "⟨H⟩   = % .16f +/- %.16f\n", energy.val, energy.err)
            if heat_capacity !== nothing
                @printf(io, "C     = % .16f +/- %.16f\n", heat_capacity.val, heat_capacity.err)
            end
            println(io)

            println(io, "Correlation time: $(corr_time)\n")

            println(io, "Initial Operator list length: $(2*M)")
            println(io, "Final Operator list length: $(op_list_length)")
            println(io, "Number of MC measurements: $(MCS)")
            println(io, "Number of equilibration steps: $(EQ_MCS)")
            println(io, "Number of skips between measurements: $(skip)\n")

            println(io, "Samples outputted to file: $(samples_file)")
        end
    end
end

function save_data(path, mc_opts, qmc_state, measurements, observables, corr_time)
    info_file = path * "_info.txt"
    samples_file = path * "_samples.txt"
    qmc_state_file = path * "_state.jld2"

    open(samples_file, "w") do io
        writedlm(samples_file, measurements, " ")
    end

    @time @save qmc_state_file qmc_state

    M = length(qmc_state.operator_list)

    make_info_file(info_file, samples_file, mc_opts, M, observables, corr_time)
end

function mis(C, Ω, δ)
    runstats = false
    Dim = 2

    # MC parameters
    M = 1000 # length of the operator_list is 2M
    MCS = 100_000 # the number of samples to record
    EQ_MCS = div(MCS, 10)
    skip = 0  # number of MC steps to perform between each msmt

    println("Running Rydberg(C=$C, Ω=$Ω, δ=$δ)")
    atoms = load_instance(20, 0.8)
    graph = unit_disk_graph(atoms, 1.5)
    H = Rydberg(graph, C, Ω, δ)
    d = @ntuple Dim C Ω δ skip M

    mc_opts = @ntuple M MCS EQ_MCS skip
    qmc_state = BinaryGroundState(H, M)

    rng = Xorshifts.Xoroshiro128Plus(1234)
    rand!(rng, qmc_state.left_config)
    copyto!(qmc_state.right_config, qmc_state.left_config)

    M, MCS, EQ_MCS, skip = mc_opts

    sname = savename(d; digits = 4)
    mkpath(datadir("sims", "mis"))
    path = datadir("sims", "mis", sname)

    measurements = zeros(Int, MCS, nspins(H))
    mags = zeros(MCS)
    ns = zeros(MCS)
    if runstats isa Val{true}
        diag_update_fails = zeros(MCS)
        cluster_update_accep = zeros(MCS)
        num_clusters = zeros(Int, MCS)
        cluster_sizes = zeros(MCS)
    end

    @showprogress "Warm up..." for i in 1:EQ_MCS
        mc_step!(rng, qmc_state, H)
    end

    @showprogress "MCMC...   " for i in 1:MCS # Monte Carlo Production Steps
        output = mc_step!(rng, qmc_state, H, runstats) do lsize, qmc_state, H
            spin_prop = sample(H, qmc_state)
            measurements[i, :] = spin_prop

            ns[i] = num_single_site_diag(H, qmc_state.operator_list)
            mags[i] = magnetization(spin_prop)
        end

        if runstats isa Val{true}
            diag_update_fails[i], cluster_update_accep[i], num_clusters[i], cluster_sizes[i] = output
        end

        for _ in 1:skip
            mc_step!(rng, qmc_state, H)
        end
    end

    if runstats isa Val{true}
        println("Diag update acceptance rate:    ", 1 - mean_and_stderr(diag_update_fails))
        println("Cluster update acceptance rate: ", mean_and_stderr(cluster_update_accep))
        println("Average number of clusters: ", mean_and_stderr(num_clusters))
        println("Average cluster size: ", mean_and_stderr(cluster_sizes))
    end

    mag = mean_and_stderr(mags)
    abs_mag = mean_and_stderr(abs, mags)
    mag_sqr = mean_and_stderr(abs2, mags)

    @time energy = energy_density(qmc_state, H, ns)

    observables = (mag, abs_mag, mag_sqr, energy)

    # measure correlation time from equilibriation samples
    @time corr_time = correlation_time(mags .^ 2)

    save_data(path, mc_opts, qmc_state, measurements, observables, corr_time)
end

mis(1.0, 1.0, -1.0)

