using DrWatson: savename

using QMC

# main.jl
#
# A projector QMC program for the TFIM

using Random
using RandomNumbers
using LinearAlgebra
using Measurements
using BinningAnalysis
using Statistics
using OnlineStats

using Plots
using StatsPlots
using DelimitedFiles
using JLD2
using JSON
using FileIO
using CSV
using DataFrames
using Printf

using ArgParse


SCRATCH_PATH = "/media/ejaaz/Seagate Expansion Drive/qmc_data/"
# SCRATCH_PATH = "/scratch/ejaazm/"

###############################################################################

function setup_trialstate(type::String, delta::Float64, omega::Float64, V::UpperTriangular{Float64})
    if type == "fields" || type == "fields2"
        H = Hermitian(-delta*[0 0; 0 1] - omega*[0 1; 1 0]/2)

        E, V = eigen(H)
        psi = V[:, 1]
        @assert all(x -> signbit(x) == signbit(psi[1]), psi)

        P = (type == "fields2") ? abs2.(psi) : abs.(psi)
        return ProductState{Float64, Bool}(Dict(false => P[1], true => P[2]))
    elseif type == "mft"
        error("Not yet supported!")
    else
        return nothing
    end
end


function init_mc_cli(parsed_args)
    Ω = parsed_args["omega"]
    δ = parsed_args["delta"]
    R_b = parsed_args["radius"]
    epsilon = parsed_args["epsilon"]
    seed = parsed_args["seed"]

    ts_type = parsed_args["trialstate"]

    truncation = parsed_args["trunc"]

    nY = parsed_args["nY"]
    # nX = nY

    # MC parameters
    M = parsed_args["M"]
    MCS = parsed_args["measurements"] # the number of samples to record per batch
    batches = parsed_args["batches"]
    mb_prob = parsed_args["mb-prob"]

    println("Running Rydberg(($nY,), R_b=$R_b, Ω=$Ω, δ=$δ; trunc=$truncation, epsilon=$epsilon)")

    d = (nY=nY, R_b=R_b, Ω=Ω, δ=δ, seed=seed)

    mc_opts = (M, MCS, batches, mb_prob)

    sname = savename(d; digits = 4)
    path = joinpath(
        SCRATCH_PATH, "qmc_sims",
        "transition_matrices", "no_trivial_clusters",
        "nY=$nY", "delta=$(@sprintf("%.2f", δ))",
        "p=$mb_prob", "epsilon=$epsilon")
    mkpath(path)

    res = parsed_args["restart"] ? nothing : continue_simulation(path, sname, parsed_args)
    if res === nothing
        H = Rydberg((nY,nY), R_b, Ω, δ; pbc=false, trunc=truncation, epsilon=epsilon)
        if M == 0
            qmc_state = BinaryThermalState(H, 2000)
        else
            qmc_state = BinaryGroundState(H, M, setup_trialstate(ts_type, δ, Ω, H.V))
        end

        rng = Xorshifts.Xoroshiro128Plus(seed)
        rand!(rng, qmc_state.left_config)
        copyto!(qmc_state.right_config, qmc_state.left_config)

        starting_batch = 0

        observables = DataFrame(
            batch = zeros(Int, MCS),
            n_ssd = zeros(MCS),
            checkerboard = zeros(MCS),
            mags = zeros(MCS),
        )

        if parsed_args["runstats"] > 2
            runstats = RunStatsHistogram(parsed_args["runstats"])
        elseif 0 <= parsed_args["runstats"] <= 2
            runstats = RunStats()
        else
            runstats = NoStats()
        end
        diagnostics = Diagnostics(runstats, CombinedTransitionMatrix(nspins(H), 1:length(H.op_sampler.op_log_weights)))
    else
        H, qmc_state, rng, observables, diagnostics, starting_batch = res
    end

    path = joinpath(path, sname)

    return H, qmc_state, path, mc_opts, rng, observables, diagnostics, starting_batch
end

function continue_simulation(path, sname, parsed_args)
    checkpoints = filter(readdir(path)) do s
        endswith(s, ".jld2") && contains(s, "batch") && contains(s, sname)
    end
    isempty(checkpoints) && return nothing

    batches = map(checkpoints) do s
        split(split(s, "batch_")[2], '_')[1]
    end
    batches = sort(batches, by=x->parse(Int, x), rev=true)

    for s in batches
        try
            qmc_state_file = joinpath(path, sname) * "_batch_$(s)_state.jld2"
            state = load(qmc_state_file)
            starting_batch = parse(Int, s) + 1

            rng::Xorshifts.Xoroshiro128Plus = state["rng"]
            qmc_state::BinaryGroundState = state["qmc_state"]
            H::Rydberg = state["hamiltonian"]
            observables = state["observables"]
            diagnostics = state["diagnostics"]

            if parsed_args["runstats"] > 2
                runstats2 = RunStatsHistogram(parsed_args["runstats"])
            elseif 0 < parsed_args["runstats"] <= 2
                runstats2 = RunStats()
            elseif parsed_args["runstats"] == 0
                runstats2 = diagnostics.runstats
            else
                runstats2 = NoStats()
            end

            # allow overriding of runstats method when continuing simulation
            if typeof(diagnostics.runstats) != typeof(runstats2)
                diagnostics = Diagnostics(runstats2, diagnostics.tmatrix)
            end

            return H, qmc_state, rng, observables, diagnostics, starting_batch
        catch _
            nothing
        end
    end

    return nothing
end

measurementtodict(V::BinningAnalysis.Variance) = Dict("value" => mean(V), "error" => std_error(V))
measurementtodict(V::OnlineStats.Variance) = Dict("value" => mean(V), "error" => std(V) / sqrt(nobs(V)))
measurementtodict(M::Measurement) = Dict("value" => M.val, "error" => M.err)
measurementtodict(H::OnlineStats.KHist) = Dict("value" => mean(H), "error" => std(H) / sqrt(nobs(H)))
measurementtodict(H::OnlineStats.Hist) = Dict("value" => mean(H), "error" => std(H) / sqrt(nobs(H)))

function groundstate(parsed_args)
    H, qmc_state, path, mc_opts, rng, observables, diagnostics, starting_batch =
        init_mc_cli(parsed_args)

    M, MCS, batches, mb_prob = mc_opts

    l = floor(Int, log10(batches) + 1)

    for b in starting_batch:batches
        # don't include equilibration samples in diagnostics
        d = (b == 0) ? Diagnostics() : diagnostics

        for _ in 1:MCS  # Monte Carlo Production Steps
            mc_step!(rng, qmc_state, H, d; p=mb_prob) #do _, qmc_state, H
                spin_prop = sample(H, qmc_state)

                observables[i, :n_ssd] = num_single_site_diag(H, qmc_state.operator_list)

                observables[i, :mags] = magnetization(spin_prop)
                observables[i, :checkerboard] = staggered_magnetization(H, spin_prop)

                observables[i, :batch] = b
            # end
        end

        println("Batch $b completed")
        # save batch
        qmc_state_file = path * "_batch_$(lpad(b, l, "0"))_state.jld2"
        @save(qmc_state_file,
              rng=rng,
              qmc_state=qmc_state,
              hamiltonian=H,
              observables=observables,
              diagnostics=diagnostics)

        data_file = path * "_batch_$(lpad(b, l, "0"))_raw_observables.csv"
        CSV.write(data_file, observables)

        # delete the previous saved state, if it exists
        old_qmc_state = path * "_batch_$(lpad(b-1, l, "0"))_state.jld2"
        if isfile(old_qmc_state)
            rm(old_qmc_state)
        end
    end

    # runstats_file = path * "_runstats.json"

    # runstats_dict = Dict{Symbol, Any}([k => measurementtodict(getproperty(diagnostics.runstats, k))
    #                                    for k in fieldnames(typeof(diagnostics.runstats))])
    # runstats_dict[:operator_list_length] = length(qmc_state.operator_list)

    # open(runstats_file, "w") do io
    #     JSON.print(io, runstats_dict, 2)
    # end
end


###############################################################################


s = ArgParseSettings()


@add_arg_table! s begin
    "groundstate"
        help = "Use Projector SSE to simulate the ground state"
        action = :command
end


@add_arg_table! s["groundstate"] begin
    "nY"
        help = "The length of the square lattice along the Y axis (PBC dimension)"
        required = true
        arg_type = Int
    "--omega"
        help = "Strength of the transverse field"
        arg_type = Float64
        default = 1.0
    "--delta"
        help = "Strength of the detuning"
        arg_type = Float64
        default = 1.0
    "--radius", "-R"
        help = "Rydberg blockade radius (in units of the lattice spacing). Controls the strength of the interaction."
        arg_type = Float64
        default = 1.2

    "--epsilon"
        help = "Multiplicative Epsilon value for the constant energy shift to the Hamiltonian"
        arg_type = Float64
        default = 0.0

    "--trunc"
        help = """Interaction truncation.
                Passing K > 0 will keep interactions upto and including the K'th nearest neighbour interactions.
                Passing K <= 0 will keep all interactions.
               """
        arg_type = Int
        default = 0

    "-M"
        help = "Projector length"
        arg_type = Int64
        default = 10000

    "--measurements", "-n"
        help = "Number of samples to record per batch"
        arg_type = Int
        default = 10_000

    "--batches", "-b"
        help = "Number of batches to run"
        arg_type = Int
        default = 100

    "--mb-prob"
        help = "Probability of performing a multibranch cluster update"
        arg_type = Float64
        default = 0.0

    "--trialstate"
        help = "Trial state type"
        arg_type = String
        default = "plus"

    "--runstats"
        help = """Number of histogram bins for runstats.
                If <=2, only compute the mean and std error of each stat.
                If < 0, don't track runstats at all.
                When continuing a simulation, a value of 0 will re-use the
                same runstats calculation as before.
               """
        arg_type = Int
        default = 0

    "--seed"
        help = "Random seed"
        arg_type = Int
        default = 1234

    "--restart"
        help = "Ignore saved simulation results and start from scratch"
        action = :store_true

end


parsed_args = parse_args(ARGS, s)

if parsed_args["%COMMAND%"] == "groundstate"
    @time groundstate(parsed_args["groundstate"])
else
    println("thermal state currently not supported")
end
