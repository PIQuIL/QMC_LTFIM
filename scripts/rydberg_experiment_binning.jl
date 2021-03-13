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

using ArgParse


# SCRATCH_PATH = "/media/ejaaz/Seagate Expansion Drive/qmc_data/"
SCRATCH_PATH = "/scratch-deleted-2021-mar-20/ejaazm/"

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
    binsize = parsed_args["binsize"]
    mb_prob = parsed_args["mb-prob"]
    @assert 0.0 <= mb_prob <= 1.0

    println("Running Rydberg(($nX, $nY), R_b=$R_b, Ω=$Ω, δ=$δ)")

    d = (nX=nX, nY=nY, R_b=R_b, Ω=Ω, δ=δ, seed=seed)

    mc_opts = (M, MCS, batches, binsize, mb_prob)

    sname = savename(d; digits = 4)
    path = joinpath(
        SCRATCH_PATH, "qmc_sims",
	"improved_lines",
	"delta_scan",
	"groundstate",
        "Rydberg_QCP", "nY=$nY", "delta=$δ",
        "M=$M", "p=$mb_prob")
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

        starting_batch = 1

        observables = (
            mags=zeros(batches, MCS ÷ binsize), smags=zeros(batches, MCS ÷ binsize),
            mags2=zeros(batches, MCS ÷ binsize), smags2=zeros(batches, MCS ÷ binsize),
            mags4=zeros(batches, MCS ÷ binsize), smags4=zeros(batches, MCS ÷ binsize),
            n_ssd=zeros(batches, MCS ÷ binsize)
        )

        runstats = (
            line_update=(
                diag_update_fails=BinningAnalysis.Variance(),
                cluster_update_accep=BinningAnalysis.Variance(),
                num_clusters=BinningAnalysis.Variance(),
                cluster_sizes=BinningAnalysis.Variance(),
                abort_rates=BinningAnalysis.Variance()
            ),
            multibranch_update=(
                diag_update_fails=BinningAnalysis.Variance(),
                cluster_update_accep=BinningAnalysis.Variance(),
                num_clusters=BinningAnalysis.Variance(),
                cluster_sizes=BinningAnalysis.Variance(),
            )
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

    M, MCS, batches, binsize, mb_prob = mc_opts
    if starting_batch == 1 && M == 0
        beta = 20.0
        max_ns = maximum([mc_step_beta!(rng, qmc_state, H, beta; eq=true, p=mb_prob) for i in 1:MCS])

        resize_op_list!(qmc_state, H, round(Int, (1.5)*max_ns, RoundUp))
        qmc_state = convert(BinaryGroundState{3, typeof(qmc_state.left_config)}, qmc_state)
    end

    if starting_batch == 1  # equilibration step
        for i in 1:MCS
            mc_step!(rng, qmc_state, H)
        end
    end

    # atexit() do
    #     println("Killed by SIGTERM!")
    #     qmc_state_file = path * "_killed_checkpoint.jld2"
    #     @time @save qmc_state_file rng=rng qmc_state=qmc_state hamiltonian=H
    #     exit(69)
    # end

    l = floor(Int, log10(batches) + 1)

    for b in starting_batch:batches
        for i in 1:(MCS÷binsize) # Monte Carlo Production Steps
            for j in 1:binsize
                diag_update_fail, cluster_stats = mc_step!(rng, qmc_state, H, Val{true}(); p=mb_prob) do lsize, qmc_state, H
                    spin_prop = sample(H, qmc_state)

                    n = num_single_site_diag(H, qmc_state.operator_list)
                    mag = magnetization(spin_prop)
                    smag = staggered_magnetization(H, spin_prop)

                    observables.mags[b, i] += mag
                    observables.mags2[b, i] += mag ^ 2
                    observables.mags4[b, i] += mag ^ 4

                    observables.smags[b, i] += smag
                    observables.smags2[b, i] += smag ^ 2
                    observables.smags4[b, i] += smag ^ 4

                    observables.n_ssd[b, i] += n
                end

                if length(cluster_stats) == 4
                    push!(runstats.line_update.diag_update_fails, diag_update_fail)
                    push!(runstats.line_update.cluster_update_accep, cluster_stats[1])
                    push!(runstats.line_update.num_clusters, cluster_stats[2])
                    push!(runstats.line_update.cluster_sizes, cluster_stats[3])
                    push!(runstats.line_update.abort_rates, cluster_stats[4])
                else
                    push!(runstats.multibranch_update.diag_update_fails, diag_update_fail)
                    push!(runstats.multibranch_update.cluster_update_accep, cluster_stats[1])
                    push!(runstats.multibranch_update.num_clusters, cluster_stats[2])
                    push!(runstats.multibranch_update.cluster_sizes, cluster_stats[3])
                end
            end

            for obs in observables
                obs[b, i] /= binsize
            end
        end

        begin
            # save batch
            qmc_state_file = path * "_batch_$(lpad(b, l, "0"))_state.jld2"
            @save(qmc_state_file,
                  rng=rng,
                  qmc_state=qmc_state,
                  hamiltonian=H,
                  observables=observables,
                  runstats=runstats)

            # delete the previous saved state, if it exists
            old_qmc_state = path * "_batch_$(lpad(b-1, l, "0"))_state.jld2"
            if isfile(old_qmc_state)
                rm(old_qmc_state)
            end
        end
    end

    observables_file = path * "_observables.json"
    runstats_file = path * "_runstats.json"

    obs_estimates = Dict{Symbol, Measurement}()
    for k in keys(observables)
        obs_estimates[k] = mean_and_stderr(observables[k][:])
    end

    obs_estimates[:energy] = energy_density(qmc_state, H, observables[:n_ssd][:])
    obs_estimates[:binder] = QMC.jackknife(observables[:smags4][:], observables[:smags2][:]) do M4, M2
        (3 - (M4/(M2^2))) / 2
    end

    open(observables_file, "w") do io
        JSON.print(io,
            Dict([k => measurementtodict(obs_estimates[k]) for k in keys(obs_estimates)]),
            2)
    end
    open(runstats_file, "w") do io
        JSON.print(io,
            Dict([
                k => Dict([k1 => measurementtodict(runstats[k][k1]) for k1 in keys(runstats[k])])
                for k in keys(runstats)
            ]),
            2)
    end

    operator_length_file = path * "_operator_list_length=$(length(qmc_state.operator_list)).txt"
    open(io -> nothing, operator_length_file, "w") #just create the file
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
        help = "Strength of the longitudinal field"
        arg_type = Float64
        default = 1.0
    "--radius", "-R"
        help = "Rydberg blockade radius (in units of the lattice spacing). Control the strength of the interaction"
        arg_type = Float64
        default = 1.2

    "-M"
        help = "Projector length. If zero, will perform a thermal equilibration at beta=20 to select a starting M."
        arg_type = Int64
        default = 0

    "--mb-prob"
        help = "Probability of performing a multibranch update"
        arg_type = Float64
        default = 0.0

    "--measurements", "-n"
        help = "Number of samples to record per batch"
        arg_type = Int
        default = 10_000

    "--batches", "-b"
        help = "Number of batches to run"
        arg_type = Int
        default = 100

    "--binsize"
        help = "Size of the bins to average over before saving"
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
