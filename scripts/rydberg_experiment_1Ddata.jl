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


SCRATCH_PATH = "../../qmc_data/"
#SCRATCH_PATH = "/scratch/ijsdevlu/"

###############################################################################

function init_mc_cli(parsed_args)
    Ω = parsed_args["omega"]
    δ = parsed_args["delta"]
    R_b = parsed_args["radius"]
    seed = parsed_args["seed"]
    mb = parsed_args["multibranch"]

    L = parsed_args["L"]

    # MC parameters
    M = parsed_args["M"]
    MCS = parsed_args["measurements"] # the number of samples to record per batch
    batches = parsed_args["batches"]

    println("Running Rydberg(L=$L, R_b=$R_b, Ω=$Ω, δ=$δ, multibranch=$mb)")

    mb_prob = mb ? 1.0 : 0.0

    # PBCs hard-set to false
    H = Rydberg(tuple(L), R_b, Ω, δ, false)
    #H = Rydberg()
    update_name = mb ? "multibranch" : "line"

    if haskey(parsed_args, "beta")
        beta = parsed_args["beta"]
        qmc_state = BinaryThermalState(H, M)
        path = joinpath(
            SCRATCH_PATH, "qmc_sims",
            "1D",
            "thermalstate", "$update_name",
            "L=$L", "delta=$δ", "M=$M", "beta=$beta")
        
        d = (L=L, R_b=R_b, Ω=Ω, δ=δ, beta=beta, seed=seed)
        sname = savename(d; digits = 4)
        mc_opts = (M, MCS, batches, mb_prob, beta)

        runstats = (
            cluster_update_accep=BinningAnalysis.Variance(),
            num_clusters=BinningAnalysis.Variance(),
            cluster_sizes=BinningAnalysis.Variance()
        )

    else
        qmc_state = BinaryGroundState(H, M)
        path = joinpath(
            SCRATCH_PATH, "qmc_sims",
            "1D",
            "groundstate", "$update_name",
            "L=$L", "delta=$δ", "M=$M")

        d = (L=L, R_b=R_b, Ω=Ω, δ=δ, seed=seed)
        sname = savename(d; digits = 4)
        mc_opts = (M, MCS, batches, mb_prob)

        runstats = (
            diag_update_fails=BinningAnalysis.Variance(),
            cluster_update_accep=BinningAnalysis.Variance(),
            num_clusters=BinningAnalysis.Variance(),
            cluster_sizes=BinningAnalysis.Variance()
        )

    end    
    mkpath(path)
    rng = Xorshifts.Xoroshiro128Plus(seed)
    rand!(rng, qmc_state.left_config)
    copyto!(qmc_state.right_config, qmc_state.left_config)

    starting_batch = 0

    observables = DataFrame(
        batch = zeros(Int, MCS),
        n = zeros(MCS),
        smags = zeros(MCS),
        mags = zeros(MCS),
    )

    path = joinpath(path, sname)

    return H, qmc_state, path, mc_opts, rng, observables, runstats, starting_batch
end

measurementtodict(V::BinningAnalysis.Variance) = Dict("value" => mean(V), "error" => std_error(V))
measurementtodict(M::Measurement) = Dict("value" => M.val, "error" => M.err)

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
            diag_update_fail, cluster_stats = mc_step!(rng, qmc_state, H, Val{true}()) do lsize, qmc_state, H
                spin_prop = sample(H, qmc_state)

                observables[i, :n] = num_single_site_diag(H, qmc_state.operator_list)
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

        data_file_obs = path * "_raw_observables.csv"
        CSV.write(data_file_obs, observables; append = (b > 0))

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


function mixedstate(parsed_args)
    H, qmc_state, path, mc_opts, rng, observables, runstats, starting_batch =
        init_mc_cli(parsed_args)

    M, MCS, batches, mb_prob, beta = mc_opts
    if starting_batch == 0
        max_ns = maximum([mc_step_beta!(rng, qmc_state, H, beta; eq=true, p=mb_prob) for i in 1:MCS])
        resize_op_list!(qmc_state, H, round(Int, (1.5)*max_ns, RoundUp))

        println("final operator list length: ", length(qmc_state.operator_list))
        println("max ops: ", max_ns)
    end

    l = floor(Int, log10(batches) + 1)
    
    for b in starting_batch:batches
        for i in 1:MCS  # Monte Carlo Production Steps
            observables[i, :n], cluster_stats = mc_step_beta!(rng, qmc_state, H, beta, Val{true}()) do lsize, qmc_state, H
                #observables[i, :n], cluster_stats = output
                spin_prop = sample(H, qmc_state)

                #observables[i, :n] = num_single_site_diag(H, qmc_state.operator_list)
                observables[i, :mags] = magnetization(spin_prop)
                observables[i, :smags] = staggered_magnetization(H, spin_prop)

                observables[i, :batch] = b
            end

            if b > 0  # don't include equilibration samples
                #push!(runstats.diag_update_fails, diag_update_fail)
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

        data_file_obs = path * "_raw_observables.csv"
        CSV.write(data_file_obs, observables; append = (b > 0))

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
    "mixedstate"
        help = "Use vanilla SSE to simulate the system at non-zero temperature"
        action = :command
end


@add_arg_table! s["groundstate"] begin
    "L"
        help = "The length of the 1D chain"
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

    "--multibranch", "--mb"
        help = "Do multibranch cluster updates instead of default (line updates)"
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
    @time mixedstate(parsed_args["mixedstate"])
end
