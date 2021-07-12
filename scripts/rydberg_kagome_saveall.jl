using DrWatson: savename

using QMC

# main.jl
#
# A projector QMC program for the TFIM

using Random
using RandomNumbers

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


SCRATCH_PATH = "../../qmc_data/tests/"

###############################################################################

function init_mc_cli(parsed_args)
    Ω = parsed_args["omega"]
    δ = parsed_args["delta"]
    R_b = parsed_args["radius"]
    trunc = parsed_args["trunc"]
    seed = parsed_args["seed"]

    t = parsed_args["t"]
    n1 = parsed_args["n1"]
    n2 = parsed_args["n2"]

    # MC parameters
    M = parsed_args["M"]
    MCS = parsed_args["measurements"] # the number of samples to record per batch
    batches = parsed_args["batches"]
    mb_prob = parsed_args["mb-prob"]

    lat = Kagome(t, n1, n2, false; trunc=trunc)
    lattice_type = typeof(lat)

    d = (lat=typeof(lat), n1=n1, n2=n2, R_b=R_b, Ω=Ω, δ=δ, seed=seed)
    sname = savename(d; digits = 4)
    mc_opts = (M, MCS, batches, mb_prob)

    println("Running Rydberg(lattice=$lattice_type, n1=$n1, n2=$n2, R_b=$R_b, Ω=$Ω, δ=$δ, p=$mb_prob, trunc=$trunc).")
    println()
    println("Boundary conditions are hard-coded to be open!")

    path = joinpath(
        SCRATCH_PATH, 
        "groundstate", 
        "$lattice_type",
        "trunc=$trunc", 
        "p=$mb_prob",
        "n1=$n1", "n2=$n2", "t=$t",
        "Rb=$R_b", "delta=$δ", "M=$M"
    )
    mkpath(path)

    res = parsed_args["restart"] ? nothing : continue_simulation(path, sname, parsed_args)
    if res === nothing
        H = Rydberg(lat, R_b, Ω, δ)
        qmc_state = BinaryGroundState(H, M)

        rng = Xorshifts.Xoroshiro128Plus(seed)
        rand!(rng, qmc_state.left_config)
        copyto!(qmc_state.right_config, qmc_state.left_config)

        starting_batch = 0

        observables = DataFrame(
            batch = zeros(Int, MCS),
            n_ssd = zeros(MCS),
            mags = zeros(MCS)
        )

        if parsed_args["runstats"] > 2
            runstats = RunStatsHistogram(parsed_args["runstats"])
        elseif 0 <= parsed_args["runstats"] <= 2
            runstats = RunStats()
        else
            runstats = NoStats()
        end
    else
        H, qmc_state, rng, observables, runstats, starting_batch = res
    end

    path = joinpath(path, sname)

    return H, qmc_state, path, mc_opts, rng, observables, runstats, starting_batch
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
            runstats = state["runstats"]

            if parsed_args["runstats"] > 2
                runstats2 = RunStatsHistogram(parsed_args["runstats"])
            elseif 0 < parsed_args["runstats"] <= 2
                runstats2 = RunStats(parsed_args["runstats"])
            elseif parsed_args["runstats"] == 0
                runstats2 = runstats
            else
                runstats2 = NoStats()
            end

            # allow overriding of runstats method when continuing simulation
            if typeof(runstats) != typeof(runstats2)
                runstats = runstats2
            end

            return H, qmc_state, rng, observables, runstats, starting_batch
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
    H, qmc_state, path, mc_opts, rng, observables, runstats, starting_batch =
        init_mc_cli(parsed_args)

    M, MCS, batches, mb_prob = mc_opts
    if starting_batch == 0 && M == 0
        beta = 20.0
        max_ns = maximum([mc_step_beta!(rng, qmc_state, H, beta; eq=true, p=mb_prob) for i in 1:MCS])

        # there's still a lot of identity elements left over, no need to make the simulation cell bigger
        resize_op_list!(qmc_state, H, max_ns)
        qmc_state = convert(BinaryGroundState{3, typeof(qmc_state.left_config)}, qmc_state)
        println("final operator list length: ", length(qmc_state.operator_list))
        println("max ops: ", max_ns)
    end

    l = floor(Int, log10(batches) + 1)

    for b in starting_batch:batches
        for i in 1:MCS  # Monte Carlo Production Steps
            # don't include equilibration samples in runstats
            rs = (b == 0) ? NoStats() : runstats

            mc_step!(rng, qmc_state, H, rs; p=mb_prob) do lsize, qmc_state, H
                spin_prop = sample(H, qmc_state)

                observables[i, :n_ssd] = num_single_site_diag(H, qmc_state.operator_list)
                observables[i, :mags] = magnetization(spin_prop)

                observables[i, :batch] = b
            end
        end

        # save batch
        qmc_state_file = path * "_batch_$(lpad(b, l, "0"))_state.jld2"
        @save(qmc_state_file,
              rng=rng,
              qmc_state=qmc_state,
              hamiltonian=H,
              observables=observables,
              runstats=runstats)

        data_file = path * "_raw_observables.csv"
        CSV.write(data_file, observables; append = (b > 0))

        # delete the previous saved state, if it exists
        old_qmc_state = path * "_batch_$(lpad(b-1, l, "0"))_state.jld2"
        if isfile(old_qmc_state)
            rm(old_qmc_state)
        end
    end

    runstats_file = path * "_runstats.json"

    runstats_dict = Dict{Symbol, Any}([k => measurementtodict(getproperty(runstats, k)) for k in fieldnames(typeof(runstats))])
    runstats_dict[:operator_list_length] = length(qmc_state.operator_list)

    open(runstats_file, "w") do io
        JSON.print(io, runstats_dict, 2)
    end

    if runstats isa RunStatsHistogram
        # plot histograms
        for k in fieldnames(typeof(runstats))
            file = path * "_$(k).png"
            plt = plot(
                getproperty(runstats, k),
                legend = nothing,
                size = (500, 500)
            )
            savefig(plt, file)
        end
    end
end


###############################################################################


s = ArgParseSettings()


@add_arg_table! s begin
    "groundstate"
        help = "Use Projector SSE to simulate the ground state"
        action = :command
end


@add_arg_table! s["groundstate"] begin
    "n1"
        help = "Dimensions of Kagome lattice in x direction."
        required = true
        arg_type = Int
    "n2"
        help = "Dimensions of Kagome lattice in the non-x direction."
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
    "-t"
        help = "The parameter defining the Kagome lattice spacing."
        arg_type = Float64
        default = 1.0
    "--trunc"
        help = """Truncate interactions at this distance. E.g., 
               if trunc = 4, sites that are greater that 4 units away from each other
               do not interact. Measured in units of lattice spacing.
               """
        arg_type = Float64
        default = Inf

    "-M"
        help = "Projector length. If zero, will perform a thermal equilibration at beta=20 to select M."
        arg_type = Int64
        default = 0

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

    "--runstats"
        help = """Number of histogram bins for runstats.
                If <=2, only compute the mean and std error of each stat.
                If < 0, don't track runstats at all.
                When continuing a simulation, a value of 0 will re-use the
                same runstats calculation as before.
               """
        arg_type = Int
        default = -1

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
    # mixedstate(parsed_args["mixedstate"])
    println("thermal state currently not supported")
end
