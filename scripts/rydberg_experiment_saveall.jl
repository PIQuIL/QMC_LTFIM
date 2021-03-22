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

using DelimitedFiles
using JLD2
using JSON
using FileIO
using CSV
using DataFrames

using ArgParse


#SCRATCH_PATH = "/media/ejaaz/Seagate Expansion Drive/qmc_data/"
SCRATCH_PATH = "/scratch/ejaazm/"

###############################################################################

function init_mc_cli(parsed_args)
    Ω = parsed_args["omega"]
    δ = parsed_args["delta"]
    R_b = parsed_args["radius"]
    seed = parsed_args["seed"]

    nY = parsed_args["nY"]
    nX = 2*nY

    # MC parameters
    M = parsed_args["M"]
    MCS = parsed_args["measurements"] # the number of samples to record per batch
    batches = parsed_args["batches"]

    println("Running Rydberg(($nX, $nY), R_b=$R_b, Ω=$Ω, δ=$δ)")

    d = (nX=nX, nY=nY, R_b=R_b, Ω=Ω, δ=δ, seed=seed)

    mc_opts = (M, MCS, batches)

    sname = savename(d; digits = 4)
    path = joinpath(
        SCRATCH_PATH, "qmc_sims",
        "disord2checkerboard",
        "groundstate",
        "Rydberg_QCP", "nY=$nY", "delta=$δ", "M=$M")
    mkpath(path)

    res = parsed_args["restart"] ? nothing : continue_simulation(path, sname)
    if res === nothing
        H = Rydberg((nX, nY), R_b, Ω, δ, (false, true))
        if M == 0
            qmc_state = BinaryThermalState(H, 2000)
        else
            qmc_state = BinaryGroundState(H, M)
        end

        rng = Xorshifts.Xoroshiro128Plus(seed)
        rand!(rng, qmc_state.left_config)
        copyto!(qmc_state.right_config, qmc_state.left_config)

        starting_batch = 0

        observables = DataFrame(
            batch = zeros(Int, MCS),
            n_ssd = zeros(MCS),
            smags = zeros(MCS),
            mags = zeros(MCS),
        )

        runstats = (
            diag_update_fails=BinningAnalysis.Variance(),
            cluster_update_accep=BinningAnalysis.Variance(),
            num_clusters=BinningAnalysis.Variance(),
            cluster_sizes=BinningAnalysis.Variance(),
        )
    else
        H, qmc_state, rng, observables, runstats, starting_batch = res
    end

    path = joinpath(path, sname)

    return H, qmc_state, path, mc_opts, rng, observables, runstats, starting_batch
end

function continue_simulation(path, sname)
    checkpoints = filter(contains("batch"),
                         filter(endswith(".jld2"), readdir(path)))
    isempty(checkpoints) && return nothing

    starting_batch_s = maximum(checkpoints) do s
        split(split(s, "batch_")[2], '_')[1]
    end
    starting_batch = parse(Int, starting_batch_s)

    qmc_state_file = joinpath(path, sname) * "_batch_$(starting_batch_s)_state.jld2"
    state = load(qmc_state_file)

    rng::Xorshifts.Xoroshiro128Plus = state["rng"]
    qmc_state::BinaryGroundState = state["qmc_state"]
    H::Rydberg = state["hamiltonian"]
    observables = state["observables"]
    runstats = state["runstats"]

    return H, qmc_state, rng, observables, runstats, starting_batch + 1
end

measurementtodict(V::BinningAnalysis.Variance) = Dict("value" => mean(V), "error" => std_error(V))
measurementtodict(M::Measurement) = Dict("value" => M.val, "error" => M.err)

function groundstate(parsed_args)
    H, qmc_state, path, mc_opts, rng, observables, runstats, starting_batch =
        init_mc_cli(parsed_args)

    M, MCS, batches = mc_opts
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
            diag_update_fail, cluster_stats = mc_step!(rng, qmc_state, H, Val{true}()) do lsize, qmc_state, H
                spin_prop = sample(H, qmc_state)

                observables[i, :n_ssd] = num_single_site_diag(H, qmc_state.operator_list)
                observables[i, :mags] = magnetization(spin_prop)
                observables[i, :smags] = staggered_magnetization(H, spin_prop)

                observables[i, :batch] = b
            end

            if b > 0  # don't include equilibration samples
                push!(runstats.diag_update_fails, diag_update_fail)
                push!(runstats.cluster_update_accep, cluster_stats[1])
                push!(runstats.num_clusters, cluster_stats[2])
                push!(runstats.cluster_sizes, cluster_stats[3])
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

    runstats = Dict{Symbol, Any}([k => measurementtodict(runstats[k]) for k in keys(runstats)])
    runstats[:operator_list_length] = length(qmc_state.operator_list)

    open(runstats_file, "w") do io
        JSON.print(io, runstats, 2)
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
