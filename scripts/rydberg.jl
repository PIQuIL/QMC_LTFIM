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

using Plots

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
    H = Rydberg(nX, R_b, Ω, δ; pbc = (isone(Dim) ? PBC : (true, true)), epsilon=0.0)
    d = @ntuple Dim nX BC_name R_b Ω δ skip M

    mc_opts = @ntuple M MCS EQ_MCS skip

    if haskey(parsed_args, "beta")
        beta = (!isinf(parsed_args["beta"])) ? parsed_args["beta"] : 10.0
        @show round(Int, 2beta * QMC.diag_update_normalization(H), RoundUp)
        qmc_state = BinaryThermalState(H, round(Int, 2beta * QMC.diag_update_normalization(H), RoundUp))
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

    # @time @save qmc_state_file qmc_state

    M = length(qmc_state.operator_list)

    make_info_file(info_file, samples_file, mc_opts, M, observables, corr_time)
end


using BinningAnalysis
using Distributions
using StatsPlots

function mixedstate(parsed_args)
    H, qmc_state, sname, mc_opts, rng, runstats = init_mc_cli(parsed_args)
    beta = parsed_args["beta"]

    gs = isinf(beta)

    if gs
        beta = 10.0  # set starting beta
    end


    if runstats isa Val{true}
        d = Diagnostics(RunStats(), CombinedTransitionMatrix(nspins(H), 1:length(H.op_sampler.op_log_weights)))
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

    println("Initial operator list length: $(length(qmc_state.operator_list))")
    eq_ns = @showprogress "Warm up..." [mc_step_beta!(rng, qmc_state, H, beta, Diagnostics(); eq=true, p=0.0) for i in 1:EQ_MCS]
    max_ns = maximum(eq_ns)
    mean_ns = mean(eq_ns[end - div(EQ_MCS, 10) : end])
    @show mean(eq_ns), mean_ns
    
    norm_const = QMC.diag_update_normalization(H)

    if gs
        C_converged = false
        i = 0
        while i < 5
            C_err = QMC.jackknife(eq_ns .^ 2, eq_ns) do n2, n
                return n2 - n^2 - n
            end
            if (C_err.err / abs(C_err.val)) > 0.95
                C_converged = true
                delta_beta = abs(C_err.val)/(C_err.err * beta^2)
                @show beta, C_err.val, C_err.err, C_converged

                @show delta_beta
                beta += delta_beta
            else
                C_converged = false
                delta_beta = C_err.val/(C_err.err * beta^2)
                @show beta, C_err.val, C_err.err, C_converged
                
                @show delta_beta
                beta += delta_beta
            end
            
            resize_op_list!(qmc_state, H, round(Int, beta*norm_const + mean_ns, RoundUp))
            eq_ns = @showprogress "Running at beta=$(beta)..." [mc_step_beta!(rng, qmc_state, H, beta, Diagnostics(); eq=true, p=0.0) for i in 1:EQ_MCS]
            eq_ns = eq_ns[div(length(eq_ns), 2):end]  # assume first half of samples are equilibrating

            i += C_converged
        end
    end


    # TODO: bug with 3//2. Using 1.5 instead
    #resize_op_list!(qmc_state, H, round(Int, (3//2)*max_ns, RoundUp))
    # resize_op_list!(qmc_state, H, round(Int, max((1.5)*max_ns, beta*norm_const + mean_ns), RoundUp))
    resize_op_list!(qmc_state, H, round(Int, beta*norm_const + mean_ns, RoundUp))

    @showprogress "MCMC...   " for i in 1:MCS  # Monte Carlo Steps
        ns[i] = mc_step_beta!(rng, qmc_state, H, beta, d; p=0.0) do lsize, qmc_state, H
            spin_prop = qmc_state.left_config
            measurements[i, :] = spin_prop
            mags[i] = magnetization(spin_prop)
            smags[i] = staggered_magnetization(H, spin_prop)
        end
        # if i == 1
        #     @show collect(zip(qmc_state.op_indices, qmc_state.associates))
        # end
        for _ in 1:skip
            mc_step_beta!(rng, qmc_state, H, beta, Diagnostics(); p=0.0)
        end
    end

    gr()
    range = minimum(ns):maximum(ns)
    if beta > 3
        histogram(ns, normalize=:pdf, bins=range[1:2:end], label = "operator count histogram")
    else
        histogram(ns, normalize=:pdf, bins=minimum((length(range), 100)), label = "operator count histogram")
    end
    pois = Poisson(mean(ns))
    plot!(range, pdf(pois, range), color=:red, label="Poisson(<n>)")
    vspan!([mean(ns), max_ns]; alpha = 0.5, label = "ok zone (<n>, n_max)", color=:green)
    vspan!([max_ns, 1.5max_ns]; alpha = 0.5, label = "danger zone (n_max, 1.5*n_max)", color=:red)
    vline!([var(ns)], label="var(n)", color=:red, linestyle=:dash)
    vline!([beta*norm_const], label = "beta * norm", color=:black)
    vline!([2*beta*norm_const], label = "2 * beta * norm", color=:black, linestyle=:dot)
    vline!([beta*norm_const + mean(ns)], label = "beta * norm + <n>", color=:black, linestyle=:dash)
    png("rydberg_test_omega$(parsed_args["omega"])_beta$(parsed_args["beta"])_histogram.png")

    plot(qqplot(ns, pois))
    png("rydberg_test_omega$(parsed_args["omega"])_beta$(parsed_args["beta"])_qq.png")

    @show max_ns, 1.5*max_ns
    @show mean(ns), beta*norm_const, beta*norm_const + mean(ns) 

    @show maximum(ns)
    mag = mean_and_stderr(smags)
    abs_mag = mean_and_stderr(abs, smags)
    mag_sqr = mean_and_stderr(abs2, smags)

    mns = mean(ns)

    function moments_of_H(n, k_max, beta)
        cumprod(m/mns for m in n:-1:(n-k_max+1))
    end

    energy_moments = permutedims(hcat(moments_of_H.(ns, 20, beta)...))[:, end-5:end]
    @show size(energy_moments)
    energy_moments_val = dropdims(mean(energy_moments, dims=1), dims=1)
    energy_moments_std = dropdims(std(energy_moments, dims=1)/sqrt(size(energy_moments, 1)), dims=1)
    energy_moments_err = [std_error(LogBinner(energy_moments[:, k])) for k in 1:size(energy_moments, 2)]
    energy_moments_tau = [2tau(LogBinner(energy_moments[:, k]))+1 for k in 1:size(energy_moments, 2)]
    @show energy_moments_val
    @show energy_moments_std
    @show energy_moments_err
    @show energy_moments_tau

    @show (@. energy_moments_err / abs(energy_moments_val))
    @show (@. energy_moments_std / abs(energy_moments_val))
    # energy = energy_density(qmc_state, H, beta, ns)

    binder_cumulant = QMC.bootstrap_alt(smags) do M
        3 - (mean(x -> x^4, M) / (mean(abs2, M) ^ 2))
    end
    binder_cumulant /= 2

    # pd = normalize(d.tmatrix.pdmatrix)
    # # t = normalize(d.tmatrix.tmatrix)

    # # @show eigvals(pd)
    # @show eigvals(pd)

    # # @show pd
    # @show d.tmatrix.pdmatrix
    # @show d.tmatrix.tmatrix

    # @show correlation_time(energy_density.(qmc_state, H, beta, ns))

    lb = LogBinner([(H.energy_shift - (n / beta))/nspins(H) for n in ns])
    @show tau(lb) #, std_error(lb)
    # @show mean(lb)

    c = QMC.jackknife((@. ns*(ns-1)), ns) do n2, n  # these are already averaged
        # C = <n^2> - <n>^2 - <n> = <n*(n-1)> - <n>^2 = n2 - n^2

        # normalizing C by <n>^2
        # C/(n^2) = n2/n^2 - 1

        # in terms of physical qtys, C = beta^2 (<H^2> - <H>^2)
        # dividing this by <H>^2 gives
        # C/(<H>^2) = beta^2 ()

        return n2/(n^2) - 1
    end 
    # c /= nspins(H)  # normalizing c already takes care of this

    observables = (mag, abs_mag, mag_sqr, binder_cumulant, mean(lb) ± std_error(lb), c)

    # measure correlation time from equilibriation samples
    corr_time = correlation_time(smags .^ 2)

    save_data(path, mc_opts, qmc_state, measurements, observables, corr_time)
end



function groundstate(parsed_args)
    H, qmc_state, sname, mc_opts, rng, runstats = init_mc_cli(parsed_args)

    M, MCS, EQ_MCS, skip = mc_opts

    mkpath(datadir("sims", "groundstate"))
    path = datadir("sims", "groundstate", sname)

    measurements = zeros(Int, MCS, nspins(H))
    mags = zeros(MCS)
    smags = zeros(MCS)
    ns = zeros(MCS)
    if runstats isa Val{true}
        diag_update_fails = zeros(MCS)
        cluster_update_accep = zeros(MCS)
        num_clusters = zeros(Int, MCS)
        cluster_sizes = zeros(MCS)
        abort_rates = zeros(MCS)
        d = Diagnostics(RunStats(), NoTransitionMatrix())
    else
        d = Diagnostics()
    end

    @showprogress "Warm up..." for i in 1:EQ_MCS
        mc_step!(rng, qmc_state, H, Diagnostics())
    end

    @showprogress "MCMC...   " for i in 1:MCS # Monte Carlo Production Steps
        output = mc_step!(rng, qmc_state, H, d) do lsize, qmc_state, H
            spin_prop = sample(H, qmc_state)
            measurements[i, :] = spin_prop

            ns[i] = num_single_site_diag(H, qmc_state.operator_list)
            mags[i] = magnetization(spin_prop)
            smags[i] = staggered_magnetization(H, spin_prop)
        end

        if runstats isa Val{true}
            diag_update_fails[i], cluster_update_accep[i], num_clusters[i], cluster_sizes[i], abort_rates[i] = output
        end

        for _ in 1:skip
            mc_step!(rng, qmc_state, H, Diagnostics())
        end
    end

    if runstats isa Val{true}
        println("Diag update acceptance rate:    ", 1 - mean_and_stderr(diag_update_fails))
        println("Cluster update acceptance rate: ", mean_and_stderr(cluster_update_accep))
        println("Average number of clusters: ", mean_and_stderr(num_clusters))
        println("Average cluster size: ", mean_and_stderr(cluster_sizes))
        println("Average abort rate: ", mean_and_stderr(abort_rates))
    end

    mag = mean_and_stderr(mags)
    mag = mag.val ± (mag.err * 2 * correlation_time(mags))

    abs_mag = mean_and_stderr(abs, mags)
    abs_mag = abs_mag.val ± (abs_mag.err * 2 * correlation_time(abs.(mags)))

    mag_sqr = mean_and_stderr(abs2, mags)
    mag_sqr = mag_sqr.val ± (mag_sqr.err * 2 * correlation_time(abs2.(mags)))

    binder_cumulant = QMC.jackknife(smags .^ 4, smags .^ 2) do M4, M2
        3 - (M4 / (M2 ^ 2))
    end
    binder_cumulant /= 2

    energy = energy_density(qmc_state, H, ns)

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
