using QMC

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

###############################################################################


function init_mc_cli(parsed_args)
    Ω = parsed_args["omega"]
    δ = parsed_args["delta"]
    R_b = parsed_args["radius"]
    seed = parsed_args["seed"]
    beta = parsed_args["beta"]

    save_path = parsed_args["path"]

    L = parsed_args["L"]

    # MC parameters
    M = parsed_args["M"]  # initial simulation cell length
    MCS = parsed_args["measurements"] # the number of samples to record per batch
    EQS = parsed_args["equilibration"]
    batches = parsed_args["batches"]
    rand_slice = parsed_args["rand-slice"]
    mb_prob = parsed_args["mb-prob"]

    EQS = (EQS === nothing) ? MCS : EQS

    println("Running Rydberg(($L,$L), R_b=$R_b, Ω=$Ω, δ=$δ)")

    mc_opts = (beta, EQS, MCS, batches, rand_slice, mb_prob)

    path = joinpath(
        save_path,
        "L=$L", 
        "Rb=$(@sprintf("%.2f", R_b))", 
        "omega=$(@sprintf("%.2f", Ω))",
        "delta=$(@sprintf("%.2f", δ))",
        "beta=$beta",
        "seed=$seed")
    mkpath(path)

    res = parsed_args["restart"] ? nothing : continue_simulation(path, parsed_args)
    if res === nothing
        H = Rydberg((L, L), R_b, Ω, δ; pbc=false)
        qmc_state = BinaryThermalState(H, M)

        writedlm(joinpath(path, "V_ij.csv"), Matrix(H.V))
        writedlm(joinpath(path, "energy_shift.csv"), H.energy_shift)

        rng = Xorshifts.Xoroshiro128Plus(seed)
        rand!(rng, qmc_state.left_config)
        copyto!(qmc_state.right_config, qmc_state.left_config)

        starting_batch = 0

        if parsed_args["runstats"] > 2
            runstats = RunStatsHistogram(parsed_args["runstats"])
        elseif 0 <= parsed_args["runstats"] <= 2
            runstats = RunStats()
        else
            runstats = NoStats()
        end
        diagnostics = Diagnostics(runstats, NoTransitionMatrix())
    else
        H, qmc_state, rng, diagnostics, starting_batch = res
    end

    return H, qmc_state, path, mc_opts, rng, diagnostics, starting_batch
end

function continue_simulation(path, parsed_args)
    checkpoints = filter(readdir(path)) do s
        endswith(s, "state.jld2") && contains(s, "batch")
    end
    isempty(checkpoints) && return nothing

    batches = map(checkpoints) do s
        split(split(s, "batch_")[2], '_')[1]
    end
    batches = sort(batches, by=x->parse(Int, x), rev=true)

    for s in batches
        try
            qmc_state_file = joinpath(path, "batch_$(s)_state.jld2")
            state = load(qmc_state_file)
            starting_batch = parse(Int, s) + 1

            rng::Xorshifts.Xoroshiro128Plus = state["rng"]
            qmc_state::BinaryThermalState = state["qmc_state"]
            H::Rydberg = state["hamiltonian"]
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

            return H, qmc_state, rng, diagnostics, starting_batch
        catch e
            println(e)
            nothing
        end
    end

    return nothing
end

function thermalstate(parsed_args)
    H, qmc_state, path, mc_opts, rng, diagnostics, starting_batch =
        init_mc_cli(parsed_args)

    beta, EQS, MCS, batches, rand_slice, mb_prob = mc_opts

    l = floor(Int, log10(batches) + 1)
    nsteps = (starting_batch == 0) ? EQS : MCS
    measurements = zeros(Bool, nspins(H), nsteps)
    observables = DataFrame(
        batch = zeros(Int, nsteps),
        n_sso = zeros(Int, nsteps),
        n_ops = zeros(Int, nsteps)
    )

    for b in starting_batch:batches
        # don't include equilibration samples in diagnostics
        eq = (b == 0)
        d = eq ? Diagnostics() : diagnostics
        nsteps = eq ? EQS : MCS

        for i in 1:nsteps  # Monte Carlo Production Steps
            n_ops = mc_step_beta!(rng, qmc_state, H, beta, d; eq = eq, p=mb_prob) do _, qmc_state, H
                msmt_slice = rand_slice ? rand(rng, 1:length(qmc_state.operator_list)) : 1
                spin_prop = sample(H, qmc_state, msmt_slice)
                measurements[:, i] = spin_prop

                observables[i, :n_sso] = num_single_site_offdiag(H, qmc_state.operator_list)
                observables[i, :batch] = b
            end
            observables[i, :n_ops] = n_ops
        end

        data_file = joinpath(path, "batch_$(lpad(b, l, "0"))_raw_observables.csv")
        CSV.write(data_file, observables)

        samples_file = joinpath(path, "batch_$(lpad(b, l, "0"))_samples.bin")
        write(samples_file, BitMatrix(measurements))

        if eq  # resize measurements arrays if no longer equilibrating
            measurements = zeros(Bool, nspins(H), MCS)
            observables = DataFrame(
                batch = zeros(Int, MCS),
                n_sso = zeros(Int, MCS),
                n_ops = zeros(Int, MCS)
            )
        end

        # save batch
        qmc_state_file = joinpath(path, "batch_$(lpad(b, l, "0"))_state.jld2")
        @save(qmc_state_file,
              rng=rng,
              qmc_state=qmc_state,
              hamiltonian=H,
              diagnostics=diagnostics)

        # delete the previous 5 saved states, if they exist
        #  (trying the last 5 bc sometimes the deletion fails if the job times out)
        for i in 1:5
            old_qmc_state = joinpath(path, "batch_$(lpad(b-i, l, "0"))_state.jld2")
            if isfile(old_qmc_state)
                rm(old_qmc_state)
            end
        end
    end
end


###############################################################################


s = ArgParseSettings()


@add_arg_table! s begin
    "thermal"
        help = "Use SSE to simulate the thermal state"
        action = :command
end


@add_arg_table! s["thermal"] begin
    "L"
        help = "The length of the square lattice along the X or Y axis"
        required = true
        arg_type = Int
    "path"
        help = "Root path to save data to"
        required = true
        arg_type = String
    
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
    "--beta"
        help = "Inverse temperature of the thermal state"
        arg_type = Float64
        default = 20.0

    "-M"
        help = "Initial simulation cell length"
        arg_type = Int64
        default = 10000


    "--equilibration", "-e"
        help = """Number of equilibration steps to run (these will be saved as batch #0).
                If `nothing` is passed, will use the same value as `--measurements`.
               """
        arg_type = Union{Int, Nothing}
        default = nothing
    "--measurements", "-n"
        help = "Number of samples to record per batch"
        arg_type = Int
        default = 10_000
    "--batches", "-b"
        help = "Number of batches to run"
        arg_type = Int
        default = 100


    "--rand-slice"
        help = """Take a diagonal measurement at a random imaginary time slice instead of just the first.
                May help reduce autocorrelation.
               """
        action = :store_true

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
                Runstats are not calculated during the equilibration batch (batch #0).
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

@time thermalstate(parsed_args["thermal"])
