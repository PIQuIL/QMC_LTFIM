using DrWatson: savename

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
#using FFTW

using DelimitedFiles
using JLD2
using Printf

using DataStructures
using ArgParse


SCRATCH_PATH = "/home/ejaazm/scratch"

###############################################################################

function init_mc_cli(parsed_args)
    Ω = parsed_args["omega"]
    δ = parsed_args["delta"]
    R_b = parsed_args["radius"]

    nY = parsed_args["nY"]
    nX = 2*nY

    # MC parameters
    MCS = parsed_args["measurements"] # the number of samples to record per batch
    batches = parsed_args["batches"]

    println("Running Rydberg(($nX, $nY), R_b=$R_b, Ω=$Ω, δ=$δ)")
    H = Rydberg((nX, nY), R_b, Ω, δ, (false, true))
    d = (nX=nX, nY=nY, R_b=R_b, Ω=Ω, δ=δ)

    mc_opts = (MCS, batches)

    qmc_state = BinaryThermalState(H, 2000)

    rng = Xorshifts.Xoroshiro128Plus(parsed_args["seed"])
    rand!(rng, qmc_state.left_config)
    copyto!(qmc_state.right_config, qmc_state.left_config)

    sname = savename(d; digits = 4)
    path = joinpath(SCRATCH_PATH, "qmc_sims", "groundstate", "Rydberg_QCP", "nY=$nY", "delta=$δ")
    mkpath(path)
    path = joinpath(path, sname)

    return H, qmc_state, path, mc_opts, rng
end


function groundstate(parsed_args)
    H, qmc_state, path, mc_opts, rng = init_mc_cli(parsed_args)
    runstats = Val{true}()
    MCS, batches = mc_opts

    ns, mags, smags = zeros(MCS), zeros(MCS), zeros(MCS)

    open(path * "_raw_observables_columns.txt", "w") do io
        writedlm(io, ["ns", "mags", "smags"])
    end

    diag_update_fails = zeros(MCS)
    cluster_update_accep = zeros(MCS)
    num_clusters = zeros(Int, MCS)
    cluster_sizes = zeros(MCS)
    abort_rates = zeros(MCS)

    open(path * "_runstats_columns.txt", "w") do io
        writedlm(io, [
            "diag_update_fails", "cluster_update_accep",
            "num_clusters", "cluster_sizes", "abort_rates"
        ])
    end

    beta = 20.0
    max_ns = maximum([mc_step_beta!(rng, qmc_state, H, beta; eq=true) for i in 1:MCS])

    # TODO: bug with 3//2. Using 1.5 instead
    #resize_op_list!(qmc_state, H, round(Int, (3//2)*max_ns, RoundUp))
    resize_op_list!(qmc_state, H, round(Int, (1.5)*max_ns, RoundUp))
    qmc_state = convert(BinaryGroundState{3, typeof(qmc_state.left_config)}, qmc_state)
    println("operator list length: $(length(qmc_state.operator_list))")

    l = floor(Int, log10(batches) + 1)

    for b in 1:batches
        for i in 1:MCS # Monte Carlo Production Steps
            diag_update_fails[i], cluster_stats = mc_step!(rng, qmc_state, H, Val{true}()) do lsize, qmc_state, H
                spin_prop = sample(H, qmc_state)

                ns[i] = num_single_site_diag(H, qmc_state.operator_list)
                mags[i] = magnetization(spin_prop)
                smags[i] = staggered_magnetization(H, spin_prop)
            end

            cluster_update_accep[i], num_clusters[i], cluster_sizes[i], abort_rates[i] = cluster_stats
        end

        begin
            batch_num = lpad(b, l, "0")
            # save batch
            open(path * "_batch_$(batch_num)_raw_observables.txt", "w") do io
                writedlm(io, zip(ns, mags, smags), ", ")
            end
            open(path * "_batch_$(batch_num)_runstats.txt", "w") do io
                writedlm(
                    io,
                    zip(diag_update_fails,
                        cluster_update_accep,
                        num_clusters,
                        cluster_sizes,
                        abort_rates),
                    ", "
                )
            end

            qmc_state_file = path * "_batch_$(batch_num)_state.jld2"
            @save qmc_state_file rng=rng qmc_state=qmc_state hamiltonian=H
        end
    end

end


###############################################################################


s = ArgParseSettings()


@add_arg_table! s begin
    "groundstate"
        help = "Use Projector SSE to simulate the ground state"
        action = :command
    # "mixedstate"
    #     help = "Use vanilla SSE to simulate the system at non-zero temperature"
    #     action = :command
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

end

# import_settings!(s["mixedstate"], s["groundstate"])

# @add_arg_table! s["mixedstate"] begin
#     "--beta"
#         help = "The inverse-temperature parameter for the simulation"
#         arg_type = Float64
#         default = 10.0
# end


parsed_args = parse_args(ARGS, s)

if parsed_args["%COMMAND%"] == "groundstate"
    @time groundstate(parsed_args["groundstate"])
else
    # mixedstate(parsed_args["mixedstate"])
    println("thermal state currently not supported")
end
