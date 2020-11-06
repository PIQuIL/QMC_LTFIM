using DrWatson
@quickactivate "QMC"

using QMC

# main.jl
#
# A projector QMC program for the TFIM

using Random
using RandomNumbers

using ProgressMeter

using Measurements
using Statistics
using FFTW

using DelimitedFiles
using JLD2
using Printf
using Lattices

using DataStructures
using ArgParse


###############################################################################

function init_mc_cli(parsed_args)
    PBC = parsed_args["periodic"]
    hx = parsed_args["hx"]
    hz = parsed_args["hz"]
    J = parsed_args["interaction"]
    runstats = parsed_args["runstats"]

    Dim = length(parsed_args["dims"])
    # NOTE: nX saved as list: ([nX]), so output file name doesn't have nX
    # parsed_args["dims"][1] fixes it but needs a better solution for >1D
    nX = parsed_args["dims"][1]

    BC_name = PBC ? "PBC" : "OBC"

    # MC parameters
    M = parsed_args["M"] # length of the operator_list is 2M
    MCS = parsed_args["measurements"] # the number of samples to record
    EQ_MCS = div(MCS, 10)
    skip = parsed_args["skip"]  # number of MC steps to perform between each msmt

    if hz === "nothing"
        println("Running TFIM(J=$J, hx=$hx)")
        if Dim == 1
            nX = nX[1]
            bond_spin, Ns, Nb = lattice_bond_spins(nX, PBC)
        elseif Dim == 2
            bond_spin, Ns, Nb = lattice_bond_spins(nX, PBC)
            nX = nX[1]
        else
            error("Unsupported number of dimensions")
        end
        H = TFIM(bond_spin, Dim, Ns, Nb, hx, J)
        d = @ntuple Dim nX BC_name J hx hz skip M
    else
        hz = parse(Float64, hz)
        println("Running LTFIM(J=$J, hx=$hx, hz=$hz)")
        H = LTFIM(Tuple(nX), J, hx, hz, PBC)
        d = @ntuple Dim nX BC_name J hx hz skip M
    end

    # path = "$(Dim)D/$(nX)/$(BC_name)/J$(J)/h$(h)/skip$(skip)/"

    mc_opts = @ntuple M MCS EQ_MCS skip

    # NOTE: why is this 2M?
    if haskey(parsed_args, "beta")
        qmc_state = BinaryThermalState(H, 2M)
    else
        qmc_state = BinaryGroundState(H, M)
    end

    rng = Xorshifts.Xoroshiro128Plus(parsed_args["seed"])

    return H, qmc_state, savename(d; digits = 4), mc_opts, rng, Val{runstats}()
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


function mixedstate(parsed_args)
    H, qmc_state, sname, mc_opts, rng, _ = init_mc_cli(parsed_args)
    beta = parsed_args["beta"]

    M, MCS, EQ_MCS, skip = mc_opts

    mkpath(datadir("sims", "mixedstate"))
    path = datadir("sims", "mixedstate", savename((@ntuple beta), sname; digits = 4))

    measurements = zeros(Int, MCS, nspins(H))
    mags = zeros(MCS)
    ns = zeros(MCS)

    max_ns = maximum(@showprogress "Warm up..." [mc_step_beta!(rng, qmc_state, H, beta; eq=true) for i in 1:EQ_MCS])

    # TODO: bug with 3//2. Using 1.5 instead
    #resize_op_list!(qmc_state, H, round(Int, (3//2)*max_ns, RoundUp))
    resize_op_list!(qmc_state, H, round(Int, (1.5)*max_ns, RoundUp))

    @showprogress "MCMC...   " for i in 1:MCS # Monte Carlo Steps
        ns[i] = mc_step_beta!(rng, qmc_state, H, beta) do lsize, qmc_state, H
            spin_prop = qmc_state.left_config
            measurements[i, :] = spin_prop
            mags[i] = magnetization(spin_prop)
        end

        for _ in 1:skip
            mc_step_beta!(rng, qmc_state, H, beta)
        end
    end

    mag = mean_and_stderr(mags)
    abs_mag = mean_and_stderr(abs, mags)
    mag_sqr = mean_and_stderr(abs2, mags)

    energy = mean_and_stderr(x -> -x/beta, ns) + H.energy_shift
    energy /= nspins(H)

    observables = (mag, abs_mag, mag_sqr, energy)

    # measure correlation time from equilibriation samples
    @time corr_time = correlation_time(mags .^ 2)

    save_data(path, mc_opts, qmc_state, measurements, observables, corr_time)
end



function groundstate(parsed_args)
    H, qmc_state, sname, mc_opts, rng, runstats = init_mc_cli(parsed_args)

    M, MCS, EQ_MCS, skip = mc_opts

    mkpath(datadir("sims", "groundstate"))
    path = datadir("sims", "groundstate", sname)

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

    h = (H isa TFIM) ? H.h : H.hx
    @time energy = jackknife(ns) do n
        if h != 0
            (-h / n) + H.energy_shift / nspins(H)
        else
            H.energy_shift / nspins(H)
        end
    end

    observables = (mag, abs_mag, mag_sqr, energy)

    # measure correlation time from equilibriation samples
    @time corr_time = correlation_time(mags .^ 2)

    save_data(path, mc_opts, qmc_state, measurements, observables, corr_time)
end


###############################################################################


s = ArgParseSettings()


@add_arg_table! s begin
    "groundstate"
        help = "Use Projector SSE to simulate the ground state"
        action = :command
    "mixedstate"
        help = "Use vanilla SSE to simulate the system at non-zero temperature"
        action = :command
end


@add_arg_table! s["groundstate"] begin
    "dims"
        help = "The dimensions of the square lattice"
        required = true
        arg_type = Int
        nargs = '+'
    "--periodic", "-p"
        help = "Periodic BCs"
        action = :store_true
    "--hx"
        help = "Strength of the transverse field"
        arg_type = Float64
        default = 1.0
    "--hz"
        help = "Strength of the longitudinal field"
        arg_type = Union{Float64, String}
        default = "nothing"
    "--interaction", "-J"
        help = "Strength of the interaction"
        arg_type = Float64
        default = 1.0

    "-M"
        help = "Half-size of the operator list"
        arg_type = Int
        default = 1000

    "--measurements", "-n"
        help = "Number of samples to record"
        arg_type = Int
        default = 100_000
    "--skip", "-s"
        help = "Number of MC steps to perform between each measurement"
        arg_type = Int
        default = 0

    "--seed"
        help = "Random seed"
        arg_type = Int
        default = 1234

    "--runstats"
        help = "Print run statistics (acceptance rates, cluster sizes, etc."
        action = :store_true
end

import_settings!(s["mixedstate"], s["groundstate"])

@add_arg_table! s["mixedstate"] begin
    "--beta"
        help = "The inverse-temperature parameter for the simulation"
        arg_type = Float64
        default = 10.0
end


parsed_args = parse_args(ARGS, s)

if parsed_args["%COMMAND%"] == "groundstate"
    groundstate(parsed_args["groundstate"])
else
    mixedstate(parsed_args["mixedstate"])
end
