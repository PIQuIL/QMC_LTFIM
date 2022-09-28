using DrWatson
@quickactivate "QMC"

using QMC

# main.jl
#
# A projector QMC program for the TFIM
using LinearAlgebra

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


###############################################################################

function init_mc_cli(parsed_args)
    PBC = parsed_args["periodic"]
    Ω = parsed_args["omega"]
    δ = parsed_args["delta"]
    R_b = parsed_args["radius"]
    runstats = parsed_args["runstats"]

    Dim = length(parsed_args["dims"])
    # NOTE: nX saved as list: ([nX]), so output file name doesn't have nX
    # parsed_args["dims"][1] fixes it but needs a better solution for >1D
    nX = tuple(parsed_args["dims"]...)

    BC_name = PBC ? "PBC" : "OBC"

    # MC parameters
    M = parsed_args["M"] # length of the operator_list is 2M
    MCS = parsed_args["measurements"] # the number of samples to record
    EQ_MCS = div(MCS, 10)
    skip = parsed_args["skip"]  # number of MC steps to perform between each msmt

    println("Running Rydberg(R_b=$R_b, Ω=$Ω, δ=$δ)")
    H = Rydberg(nX, R_b, Ω, δ; pbc = (isone(Dim) ? PBC : (false, false)), epsilon=0.2)
    d = @ntuple Dim nX BC_name R_b Ω δ skip M

    mc_opts = @ntuple M MCS EQ_MCS skip

    if haskey(parsed_args, "beta")
        qmc_state = BinaryThermalState(H, 2M)
    else
        qmc_state = BinaryGroundState(H, M)
    end

    rng = Xorshifts.Xoroshiro128Plus(parsed_args["seed"])
    rand!(rng, qmc_state.left_config)
    copyto!(qmc_state.right_config, qmc_state.left_config)

    return H, qmc_state, savename(d; digits = 4), mc_opts, rng, Val{runstats}()
end


function make_info_file(info_file, samples_file, mc_opts, op_list_length, observables, corr_time)
    M, MCS, EQ_MCS, skip = mc_opts
    if length(observables) == 6
        mag, abs_mag, mag_sqr, binder, energy, heat_capacity = observables
    else
        mag, abs_mag, mag_sqr, binder, energy = observables
        heat_capacity = nothing
    end

    open(info_file, "w") do file_io
        streams = [Base.stdout, file_io]

        for io in streams
            @printf(io, "⟨M⟩   = % .16f +/- %.16f\n", mag.val, mag.err)
            @printf(io, "⟨|M|⟩ = % .16f +/- %.16f\n", abs_mag.val, abs_mag.err)
            @printf(io, "⟨M^2⟩ = % .16f +/- %.16f\n", mag_sqr.val, mag_sqr.err)
            @printf(io, "U_4   = % .16f +/- %.16f\n", binder.val, binder.err)
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


using BinningAnalysis

function mixedstate(parsed_args)
    H, qmc_state, sname, mc_opts, rng, runstats = init_mc_cli(parsed_args)
    beta = parsed_args["beta"]

    if runstats isa Val{true}
        d = Diagnostics(RunStatsHistogram(500), CombinedTransitionMatrix(nspins(H), 1:length(H.op_sampler.op_log_weights)))
    else
        d = Diagnostics()
    end
    M, MCS, EQ_MCS, skip = mc_opts

    mkpath(datadir("sims", "mixedstate"))
    path = datadir("sims", "mixedstate", savename((@ntuple beta), sname; digits = 4))

    measurements = zeros(Int, MCS, nspins(H))
    mags = zeros(MCS)
    smags = zeros(MCS)
    ns = zeros(MCS)

    max_ns = maximum(@showprogress "Warm up..." [mc_step_beta!(rng, qmc_state, H, beta, Diagnostics(); eq=false, p=0.0) for i in 1:EQ_MCS])

    # TODO: bug with 3//2. Using 1.5 instead
    #resize_op_list!(qmc_state, H, round(Int, (3//2)*max_ns, RoundUp))
    # resize_op_list!(qmc_state, H, round(Int, (1.5)*max_ns, RoundUp))

    @showprogress "MCMC...   " for i in 1:MCS  # Monte Carlo Steps
        ns[i] = mc_step_beta!(rng, qmc_state, H, beta, d; p=0.0) do lsize, qmc_state, H
            spin_prop = qmc_state.left_config
            measurements[i, :] = spin_prop
            mags[i] = magnetization(spin_prop)
            smags[i] = staggered_magnetization(H, spin_prop)
        end

        for _ in 1:skip
            mc_step_beta!(rng, qmc_state, H, beta, Diagnostics(); p=0.0)
        end
    end


    println(qmc_state.operator_list)
    qmc_state_line = deepcopy(qmc_state)
    println("Line clusters:")
    lsize = QMC.line_update!(rng, qmc_state_line, H, Diagnostics())
    println(qmc_state_line.operator_list)
    println(collect(zip(1:lsize, qmc_state_line.linked_list[1:lsize])))
    println(collect(zip(1:lsize, qmc_state_line.in_cluster[1:lsize])))

    qmc_state_mb = deepcopy(qmc_state)
    println("MB clusters:")
    lsize = QMC.multibranch_update!(rng, qmc_state_mb, H, Diagnostics())
    println(qmc_state_mb.operator_list)
    println(collect(zip(1:lsize, qmc_state_mb.linked_list[1:lsize])))
    println(collect(zip(1:lsize, qmc_state_mb.in_cluster[1:lsize])))

    mag = mean_and_stderr(smags)
    abs_mag = mean_and_stderr(abs, smags)
    mag_sqr = mean_and_stderr(abs2, smags)

    # energy = energy_density(qmc_state, H, beta, ns)

    binder_cumulant = QMC.jackknife(smags .^ 4, smags .^ 2) do M4, M2
        3 - (M4 / (M2 ^ 2))
    end
    binder_cumulant /= 2

    # pd = normalize(d.tmatrix.pdmatrix)
    # # t = normalize(d.tmatrix.tmatrix)

    # # @show eigvals(pd)
    # @show eigvals(pd)

    # # @show pd
    # @show d.tmatrix.pdmatrix
    @show d.tmatrix.tmatrix

    # @show correlation_time(energy_density.(qmc_state, H, beta, ns))

    lb = LogBinner([(H.energy_shift - (n / beta))/nspins(H) for n in ns])
    @show tau(lb) #, std_error(lb)
    # @show mean(lb)

    observables = (mag, abs_mag, mag_sqr, binder_cumulant, mean(lb) ± std_error(lb))

    # measure correlation time from equilibriation samples
    corr_time = correlation_time(smags .^ 2)

    save_data(path, mc_opts, qmc_state, measurements, observables, corr_time)
end


using Plots

function groundstate(parsed_args)
    H, qmc_state, sname, mc_opts, rng, runstats = init_mc_cli(parsed_args)

    M, MCS, EQ_MCS, skip = mc_opts

    mkpath(datadir("sims", "groundstate"))
    path = datadir("sims", "groundstate", sname)

    measurements = zeros(Int, MCS, nspins(H))
    mags = zeros(MCS)
    smags = zeros(MCS)
    nssd = zeros(MCS)
    nsso = zeros(MCS)
    nbond = zeros(MCS)
    d = Diagnostics()
    if runstats isa Val{true}
        diag_update_fails = zeros(MCS)
        cluster_update_accep = zeros(MCS)
        num_clusters = zeros(Int, MCS)
        cluster_sizes = zeros(MCS)
        abort_rates = zeros(MCS)
        d = Diagnostics(RunStatsHistogram(500), NoTransitionMatrix())
    end

    mbprob = 0.0
    @showprogress "Warm up..." for _ in 1:EQ_MCS
        mc_step!(rng, qmc_state, H, Diagnostics(), p=mbprob)
    end

    @showprogress "MCMC...   " for i in 1:MCS # Monte Carlo Production Steps
        mc_step!(rng, qmc_state, H, d, p=mbprob) do _, qmc_state, H
            spin_prop = sample(H, qmc_state)
            measurements[i, :] = spin_prop

            nssd[i] = num_single_site_diag(H, qmc_state.operator_list)
            nsso[i] = num_single_site_offdiag(H, qmc_state.operator_list)
            nbond[i] = num_two_site_diag(H, qmc_state.operator_list)

            mags[i] = magnetization(spin_prop)
            smags[i] = staggered_magnetization(H, spin_prop)
        end

        for _ in 1:skip
            mc_step!(rng, qmc_state, H, Diagnostics(), p=mbprob)
        end
    end

    # println(qmc_state.operator_list)
    # qmc_state_line = deepcopy(qmc_state)
    # println("Line clusters:")
    # lsize = QMC.line_update!(rng, qmc_state_line, H, Diagnostics())
    # println(qmc_state_line.operator_list)
    # println(collect(zip(1:lsize, qmc_state_line.linked_list[1:lsize])))
    # println(collect(zip(1:lsize, qmc_state_line.in_cluster[1:lsize])))

    # qmc_state_mb = deepcopy(qmc_state)
    # println("MB clusters:")
    # lsize = QMC.multibranch_update!(rng, qmc_state_mb, H, Diagnostics())
    # println(qmc_state_mb.operator_list)
    # println(collect(zip(1:lsize, qmc_state_mb.linked_list[1:lsize])))
    # println(collect(zip(1:lsize, qmc_state_mb.in_cluster[1:lsize])))

    if runstats isa Val{true}
        # println("Diag update acceptance rate:    ", 1 - mean_and_stderr(diag_update_fails))
        # println("Cluster update acceptance rate: ", mean_and_stderr(cluster_update_accep))
        # println("Average number of clusters: ", mean_and_stderr(num_clusters))
        # println("Average cluster size: ", mean_and_stderr(cluster_sizes))
        # println("Average abort rate: ", mean_and_stderr(abort_rates))

        # println(d.runstats.cluster_sizes)
        # gui(plot(d.runstats.cluster_sizes[4:2:end], xaxis=:log))

        p1 = plot(2:2:length(d.runstats.cluster_sizes), d.runstats.cluster_sizes[2:2:end],
                  xaxis=:log, legend=false, title="Cluster Size Histogram")
        p2 = plot(d.runstats.cluster_count,
                  xlims=(findfirst(!iszero, d.runstats.cluster_count) - 10,
                         findlast(!iszero, d.runstats.cluster_count) + 10),
                  legend=false, title="Cluster Count Histogram")

        p3 = plot(2:2:length(d.runstats.accepted_cluster_sizes), d.runstats.accepted_cluster_sizes[2:2:end],
                  xaxis=:log, legend=false, title="Accepted Cluster Size Histogram")
        p4 = plot(d.runstats.accepted_cluster_count,
                  xlims=(findfirst(!iszero, d.runstats.accepted_cluster_count) - 10,
                         findlast(!iszero, d.runstats.accepted_cluster_count) + 10),
                  legend=false, title="Accepted Cluster Count Histogram")

        p5 = plot(2:2:length(d.runstats.rejected_cluster_sizes), d.runstats.rejected_cluster_sizes[2:2:end],
                  xaxis=:log, legend=false, title="Rejected Cluster Size Histogram")
        p6 = plot(d.runstats.rejected_cluster_count,
                  xlims=(findfirst(!iszero, d.runstats.rejected_cluster_count) - 10,
                         findlast(!iszero, d.runstats.rejected_cluster_count) + 10),
                  legend=false, title="Rejected Cluster Count Histogram")
        gui(plot(p1, p2, p3, p4, p5, p6, layout = @layout([a b; c d; e f])))
        sleep(60)
    end

    mag = mean_and_stderr(mags)
    mag = mag.val ± (mag.err * 2 * correlation_time(mags))

    abs_mag = mean_and_stderr(abs, mags)
    abs_mag = abs_mag.val ± (abs_mag.err * 2 * correlation_time(abs.(mags)))

    mag_sqr = mean_and_stderr(abs2, mags)
    mag_sqr = mag_sqr.val ± (mag_sqr.err * 2 * correlation_time(abs2.(mags)))

    nssd_mean = mean_and_stderr(nssd)
    nssd_mean = nssd_mean.val ± (nssd_mean.err * 2 * correlation_time(nssd))

    nsso_mean = mean_and_stderr(nsso)
    nsso_mean = nsso_mean.val ± (nsso_mean.err * 2 * correlation_time(nsso))

    nbond_mean = mean_and_stderr(nbond)
    nbond_mean = nbond_mean.val ± (nbond_mean.err * 2 * correlation_time(nbond))

    @show nssd_mean, nsso_mean, nbond_mean

    binder_cumulant = QMC.jackknife(smags .^ 4, smags .^ 2) do M4, M2
        3 - (M4 / (M2 ^ 2))
    end
    binder_cumulant /= 2

    energy = energy_density(qmc_state, H, nssd)

    observables = (mag, abs_mag, mag_sqr, binder_cumulant, energy)

    # measure correlation time from equilibriation samples
    corr_time = correlation_time(smags .^ 2)

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
    "--omega"
        help = "Strength of the transverse field"
        arg_type = Float64
        default = 1.0
    "--delta"
        help = "Strength of the longitudinal field"
        arg_type = Float64
        default = 1.0
    "--radius", "-R"
        help = "Rydberg blockade radius (in units of the lattice spacing). Control the strength of the interaction"
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
        help = "Print run statistics (acceptance rates, cluster sizes, etc.)"
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
    @time groundstate(parsed_args["groundstate"])
else
    mixedstate(parsed_args["mixedstate"])
end
