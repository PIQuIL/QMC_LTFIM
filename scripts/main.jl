using DrWatson
# @quickactivate "QMC"

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
using OnlineStats
using BinningAnalysis

using DelimitedFiles
using JSON
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

    # MC parameters
    M = parsed_args["M"] # length of the operator_list is 2M
    MCS = parsed_args["measurements"] # the number of samples to record
    EQ_MCS = div(MCS, 10)
    skip = parsed_args["skip"]  # number of MC steps to perform between each msmt

    if J isa Number || tryparse(Float64, J) !== nothing
        J = J isa Number ? J : parse(Float64, J)
        Dim = length(parsed_args["dims"])
        Dim > 2 && error("Unsupported number of dimensions")

        nX = tuple(parsed_args["dims"]...)
        BC_name = PBC ? "PBC" : "OBC"

        if hz == "nothing"
            println("Running TFIM(J=$J, hx=$hx)")
            H = TFIM(lattice_bond_spins(nX, PBC)..., hx, J)
        else
            hz = parse(Float64, hz)
            println("Running LTFIM(J=$J, hx=$hx, hz=$hz)")
            H = LTFIM(nX, J, hx, hz, PBC)
        end
        nX, nY = (Dim == 2) ? nX : (nX[1], nothing)
        d = (Dim = Dim, nX = nX, nY = nY, BC = BC_name, J = J, hx = hx, hz = hz, skip = skip, M = M)
    else
        J = readdlm(J)
        @assert size(J, 1) == size(J, 2) "interaction matrix must be square!"
        J = UpperTriangular(triu(J, 1))
        nX = size(J, 1)

        if hz == "nothing"
            println("Running TFIM(J=custom, hx=$hx)")
            H = TFIM(J, hx*ones(nX))
        else
            hz = parse(Float64, hz)
            println("Running LTFIM(J=custom, hx=$hx, hz=$hz)")
            H = LTFIM(J, hx*ones(nX), hz*ones(nX))
        end
        d = (J = "custom", hx = hx, hz = hz, skip = skip, M = M)
    end

    mc_opts = (M, MCS, EQ_MCS, skip)

    # NOTE: why is this 2M?
    if haskey(parsed_args, "beta")
        qmc_state = BinaryThermalState(H, 2M)
    else
        qmc_state = BinaryGroundState(H, M)
    end

    rng = Xorshifts.Xoroshiro128Plus(parsed_args["seed"])

    if parsed_args["runstats"] > 2
        runstats = RunStatsHistogram(parsed_args["runstats"])
    elseif 0 <= parsed_args["runstats"] <= 2
        runstats = RunStats()
    else
        runstats = NoStats()
    end

    return H, qmc_state, savename(d; digits = 4), mc_opts, rng, runstats
end


function make_info_file(info_file, mc_opts, op_list_length, observables, runtime_stats)
    M, MCS, EQ_MCS, skip = mc_opts

    info = Dict(
        "observables" => observables,
        "qmc_params" => Dict(
            "num_samples" => MCS,
            "equilibration_steps" => EQ_MCS,
            "skips" => skip,
            "initial_operator_list_length" => 2*M,
            "final_operator_list_length" => op_list_length
        ),
        "runtime_stats" => runtime_stats
    )

    open(info_file, "w") do io
        JSON.print(io, info, 2)
    end

    mag = observables["magnetization"]
    abs_mag = observables["absolute magnetization"]
    mag_sqr = observables["squared magnetization"]
    energy = observables["energy"]

    @printf(Base.stdout, "⟨M⟩   = % .16f +/- %.16f\n", mag["value"], mag["error"])
    @printf(Base.stdout, "⟨|M|⟩ = % .16f +/- %.16f\n", abs_mag["value"], abs_mag["error"])
    @printf(Base.stdout, "⟨M^2⟩ = % .16f +/- %.16f\n", mag_sqr["value"], mag_sqr["error"])
    @printf(Base.stdout, "⟨H⟩   = % .16f +/- %.16f\n", energy["value"], energy["error"])

    println("\nInitial Operator list length: $(2*M)")
    println("Final Operator list length: $(op_list_length)")
    println("Number of MC measurements: $(MCS)")
    println("Number of equilibration steps: $(EQ_MCS)")
    println("Number of skips between measurements: $(skip)")
end


function save_data(path, mc_opts, qmc_state, observables, runtime_stats)
    info_file = path * "_info.json"
    qmc_state_file = path * "_state.jld2"

    @time @save qmc_state_file qmc_state

    M = length(qmc_state.operator_list)
    make_info_file(info_file, mc_opts, M, observables, runtime_stats)
end


measurementtodict(M::Measurement) = Dict("value" => M.val, "error" => M.err)
measurementtodict(B::LogBinner, convergence_threshold::Float64 = 0.05) = Dict(
    "value" => mean(B),
    "error" => std_error(B),
    "tau" => tau(B),
    "has_converged" => has_converged(B,
                                     BinningAnalysis._reliable_level(B),
                                     convergence_threshold),
    "convergence" => convergence(B),
    "convergence_threshold" => convergence_threshold,
)
measurementtodict(H::OnlineStats.KHist) = Dict("value" => mean(H), "error" => std(H) / sqrt(nobs(H)))
measurementtodict(V::BinningAnalysis.Variance) = Dict("value" => mean(V), "error" => std_error(V))
measurementtodict(V::OnlineStats.Variance) = Dict("value" => mean(V), "error" => std(V) / sqrt(nobs(V)))


function mixedstate(parsed_args)
    H, qmc_state, sname, mc_opts, rng, runstats = init_mc_cli(parsed_args)
    beta = parsed_args["beta"]

    M, MCS, EQ_MCS, skip = mc_opts

    mkpath(datadir("sims", "mixedstate"))
    path = datadir("sims", "mixedstate", savename((@ntuple beta), sname; digits = 4))

    binner_capacity = nextpow(2, MCS) - 1
    mags = LogBinner(capacity=binner_capacity)
    abs_mags = LogBinner(capacity=binner_capacity)
    sqr_mags = LogBinner(capacity=binner_capacity)
    energy = LogBinner(capacity=binner_capacity)

    max_ns = maximum(@showprogress "Warm up..." [mc_step_beta!(rng, qmc_state, H, beta; eq=true) for i in 1:EQ_MCS])

    resize_op_list!(qmc_state, H, round(Int, (1.5)*max_ns, RoundUp))

    @showprogress "MCMC...   " for i in 1:MCS # Monte Carlo Steps
        n = mc_step_beta!(rng, qmc_state, H, beta, runstats) do lsize, qmc_state, H
            m = magnetization(sample(H, qmc_state))
            push!(mags, m)
            push!(abs_mags, abs(m))
            push!(sqr_mags, m ^ 2)
        end

        push!(energy, energy_density(qmc_state, H, beta, n))

        for _ in 1:skip
            mc_step_beta!(rng, qmc_state, H, beta)
        end
    end

    runtime_stats = Dict{Symbol, Any}([k => measurementtodict(getproperty(runstats, k)) for k in fieldnames(typeof(runstats))])

    observables = Dict{String, Dict{String, Number}}(
        "magnetization" => measurementtodict(mags),
        "absolute magnetization" => measurementtodict(abs_mags),
        "squared magnetization" => measurementtodict(sqr_mags),
        "energy" => measurementtodict(energy)
    )

    save_data(path, mc_opts, qmc_state, observables, runtime_stats)
end


function groundstate(parsed_args)
    H, qmc_state, sname, mc_opts, rng, runstats = init_mc_cli(parsed_args)

    M, MCS, EQ_MCS, skip = mc_opts

    mkpath(datadir("sims", "groundstate"))
    path = datadir("sims", "groundstate", sname)

    binner_capacity = nextpow(2, MCS) - 1
    mags = LogBinner(capacity=binner_capacity)
    abs_mags = LogBinner(capacity=binner_capacity)
    sqr_mags = LogBinner(capacity=binner_capacity)
    ns = zeros(MCS)

    @showprogress "Warm up..." for i in 1:EQ_MCS
        mc_step!(rng, qmc_state, H)
    end

    @showprogress "MCMC...   " for i in 1:MCS # Monte Carlo Production Steps
        mc_step!(rng, qmc_state, H, runstats) do lsize, qmc_state, H
            m = magnetization(sample(H, qmc_state))
            push!(mags, m)
            push!(abs_mags, abs(m))
            push!(sqr_mags, m ^ 2)

            ns[i] = num_single_site_diag(H, qmc_state.operator_list)
        end

        for _ in 1:skip
            mc_step!(rng, qmc_state, H)
        end
    end

    runtime_stats = Dict{Symbol, Any}([k => measurementtodict(getproperty(runstats, k)) for k in fieldnames(typeof(runstats))])

    observables = Dict{String, Dict{String, Number}}(
        "magnetization" => measurementtodict(mags),
        "absolute magnetization" => measurementtodict(abs_mags),
        "squared magnetization" => measurementtodict(sqr_mags),
        "energy" => measurementtodict(energy_density(qmc_state, H, ns))
    )

    save_data(path, mc_opts, qmc_state, observables, runtime_stats)
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
        help = "The dimensions of the square lattice; overridden for custom interaction matrices"
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
        help = "Strength of the interaction or the path of a text file containing an interaction matrix"
        arg_type = Union{Float64, String}
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
        help = """Number of histogram bins for runstats.
                If <=2, only compute the mean and std error of each stat.
                If < 0, don't track runstats at all.
                When continuing a simulation, a value of 0 will re-use the
                same runstats calculation as before.
               """
        arg_type = Int
        default = 0
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
