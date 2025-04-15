using QMC

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

init_op_list_length(beta, H, c::Int=2) = round(Int, (c * beta) * QMC.diag_update_normalization(H), RoundUp)

function init_mc_cli(parsed_args)
    Ω = parsed_args["omega"]
    δ = parsed_args["delta"]
    R_b = parsed_args["radius"]

    nX = parsed_args["dim"]

    # MC parameters
    MCS = parsed_args["measurements"] # the number of samples to record
    EQ_MCS = div(MCS, 10)

    println("Running Rydberg(R_b=$R_b, Ω=$Ω, δ=$δ)")
    H = Rydberg((nX, nX), R_b, Ω, δ; pbc = (true, true), epsilon=0.0)

    mc_opts = (MCS, EQ_MCS, init_op_list_length(parsed_args["beta"], H, parsed_args["c"]))
    qmc_state = BinaryThermalState(H, mc_opts[end])
    
    rng = Xorshifts.Xoroshiro128Plus(parsed_args["seed"])
    rand!(rng, qmc_state.left_config)
    copyto!(qmc_state.right_config, qmc_state.left_config)

    return H, qmc_state, mc_opts, rng
end


function make_info_file(parsed_args, mc_opts, op_list_length, observables, corr_time)
    MCS, EQ_MCS, M = mc_opts
    if length(observables) == 6
        mag, abs_mag, mag_sqr, binder, energy, heat_capacity = observables
        M_init = mc_opts[end]
    else
        mag, abs_mag, mag_sqr, binder, energy = observables
        heat_capacity = nothing
        M_init = 2*mc_opts[end]
    end

    streams = [Base.stdout]
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

        println(io, "Correlation time of <M^2>: $(corr_time)\n")

        println(io, "Initial Operator list length: $(M_init)")
        println(io, "Final Operator list length: $(op_list_length)")
        println(io, "Number of MC measurements: $(MCS)")
        println(io, "Number of equilibration steps: $(EQ_MCS)")
    end
end


function save_data(parsed_args, mc_opts, qmc_state, measurements, observables, corr_time)
    # info_file = path * "_info.txt"
    # samples_file = path * "_samples.txt"
    # qmc_state_file = path * "_state.jld2"

    # open(samples_file, "w") do io
    #     writedlm(samples_file, measurements, " ")
    # end

    M = length(qmc_state.operator_list)

    make_info_file(parsed_args, mc_opts, M, observables, corr_time)
end


using BinningAnalysis
using Distributions
using StatsPlots
using PrettyPrinting


function thermalstate(parsed_args)
    H, qmc_state, mc_opts, rng = init_mc_cli(parsed_args)
    beta = parsed_args["beta"]
    threestep = parsed_args["threestep"]

    MCS, EQ_MCS, M_init = mc_opts

    measurements = zeros(Int, MCS, nspins(H))
    mags = zeros(MCS)
    smags = zeros(MCS)
    ns = zeros(MCS)

    MAX_ITER = parsed_args["max-replace"]
    P = 0.0

    println("Initial operator list length: $(length(qmc_state.operator_list))")
    eq_ns = @showprogress "Warm up..." [mc_step_beta!(rng, qmc_state, H, beta, Diagnostics(); eq=true, threestep=threestep, p=P, max_iter=MAX_ITER) for i in 1:EQ_MCS]
    max_ns = maximum(eq_ns[div(EQ_MCS, 10) : end])  # drop first tenth of samples just in case of unfavorable initialization
    mean_ns = mean(eq_ns[end - div(EQ_MCS, 10) : end])
    @show mean(eq_ns), mean_ns
    
    norm_const = QMC.diag_update_normalization(H)
    if iszero(parsed_args["M-setting"])
        resize_op_list!(qmc_state, H, round(Int, beta*norm_const + mean_ns, RoundUp))
    else
        resize_op_list!(qmc_state, H, round(Int, parsed_args["M-setting"]*max_ns, RoundUp))
    end

    # d = Diagnostics(parsed_args["runstats-hist"] ? RunStatsHistogram(1000) : RunStats())
    d = Diagnostics(RunStats())
    
    @showprogress "MCMC...   " for i in 1:MCS  # Monte Carlo Steps
        ns[i] = mc_step_beta!(rng, qmc_state, H, beta, d; threestep=threestep, p=P, max_iter=MAX_ITER) do lsize, qmc_state, H
            spin_prop = qmc_state.left_config
            measurements[i, :] = spin_prop
            mags[i] = magnetization(spin_prop)
            smags[i] = staggered_magnetization(H, spin_prop)
        end
    end

    gr()
    range = minimum(ns):maximum(ns)
    if beta > 3
        histogram(ns, normalize=:pdf, bins=range[1:2:end], label = "operator count histogram", linewidth=0)
    else
        histogram(ns, normalize=:pdf, bins=minimum((length(range), 100)), label = "operator count histogram", linewidth=0)
    end

    
    mean_n = mean(ns)
    pois = Poisson(mean_n)
    plot!(range, pdf(pois, range), color=:red, label="Poisson(<n>)")
    vspan!([mean_n, max_ns]; alpha = 0.5, label = "ok zone (<n>, n_max)", color=:green)
    vspan!([max_ns, 1.5max_ns]; alpha = 0.5, label = "danger zone (n_max, 1.5*n_max)", color=:red)
    vline!([var(ns)], label="var(n)", color=:red, linestyle=:dash)
    vline!([beta*norm_const], label = "beta * norm", color=:black)
    vline!([cquantile(pois, 1e-10)], label = "k_star", color=:black, linestyle=:dashdot)
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

    rs = QMC.summarize(d.runstats)
    # @show rs
    pprint(rs)
    # @show rs.diagonal_update.operator_insertion
    # @show rs.diagonal_update.matelem_insertion
    # @show rs.diagonal_update.removal
    # @show rs.diagonal_update.replace
    # @show rs.diagonal_update.replace_steps


    # @show rs.cluster_update_accept
    # @show rs.cluster_sizes
    # @show rs.accepted_cluster_sizes
    # @show rs.rejected_cluster_sizes

    # plot_cluster_sizes(d.runstats)

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
    @show corr_time
    @show 2tau(LogBinner(smags .^ 2)) + 1

    save_data(parsed_args, mc_opts, qmc_state, measurements, observables, corr_time)
end

###############################################################################


s = ArgParseSettings()


@add_arg_table! s begin
    "thermalstate"
        help = "Use vanilla SSE to simulate the system at non-zero temperature"
        action = :command
end


@add_arg_table! s["thermalstate"] begin
    "dim"
        help = "The linear dimension of the square lattice"
        required = true
        arg_type = Int
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

    "--beta"
        help = "The inverse-temperature parameter for the simulation"
        arg_type = Float64
        default = 10.0

    "-c"
        help = "The scaling constant used to set the initial operator list length: c * beta * mathcal{N}."
        arg_type = Int64
        default = 2

    "--M-setting"
        help = "How to set the truncation order M. If zero, uses M_optimal from thesis, otherwise treated as a scaling factor for n_max."
        arg_type = Float64
        default = 1.5
        
    "--max-replace"
        help = "Maximum number of iterations for diagonal replacement move. If negative uses Metropolis Acceptance, otherwise ignores the current config."
        arg_type = Int
        default = 0

    "--threestep"
        help = "Whether to use the Three-Step Diagonal Update Scheme."
        action = :store_true

    "--measurements", "-n"
        help = "Number of samples to record"
        arg_type = Int
        default = 100_000

    "--seed"
        help = "Random seed"
        arg_type = Int
        default = 1234

    "--runstats-hist"
        help = "Build Histograms for Runstats."
        action = :store_true
        
end


parsed_args = parse_args(ARGS, s)

@time thermalstate(parsed_args["thermalstate"])
