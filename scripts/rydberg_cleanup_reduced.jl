using QMC

# main.jl
#
# A projector QMC program for the TFIM

using Measurements
using BinningAnalysis
using Statistics

using DelimitedFiles
using JLD2
using FileIO
using JSON
using CSV

using DataFrames
using Query
using StatsPlots

using ArgParse

SCRATCH_PATH = "../../qmc_data/production/"
#SCRATCH_PATH = "/scratch/ijsdevlu/1D_qmc_data/preprod/"

RELIABLE_SIZE = 256
TOTAL_MEASUREMENTS = 10_000_000

OBSERVABLES = ["n", "mags", "abs_mags", "mags2", "mags4", "smags", "abs_smags", "smags2", "smags4", "energy"]
LINEAR_OBSERVABLES = ["n", "mags", "abs_mags", "mags2", "mags4", "smags", "abs_smags", "smags2", "smags4"]

###############################################################################

function init_cli(parsed_args)
    δ = parsed_args["delta"]
    L = parsed_args["L"]
    M = parsed_args["M"]
    mb = parsed_args["multibranch"]
    rseed = parsed_args["seed"]
    binsize = parsed_args["binsize"]
    rfactor = parsed_args["rfactor"]

    update_name = mb ? "multibranch" : "line"

    if haskey(parsed_args, "beta")
        beta = parsed_args["beta"]
        path = joinpath(
            SCRATCH_PATH,
            "thermalstate", "$update_name",
            "L=$L", "delta=$δ", "M=$M", "beta=$beta")      
    else
        path = joinpath(
            SCRATCH_PATH,
            "groundstate", "$update_name",
            "L=$L", "delta=$δ", "M=$M")
    end    

    if !isdir(path)
        println("$path doesn't point to a valid directory!")
        exit(1)
    end

    if rseed != nothing
        rseed = parse(Int, rseed)
    end

    return path, L, δ, M, rseed, binsize, rfactor
end

###############################################################################

function save_stats(path::String, bin_file::String, stats::Dict{String, Dict{String,Float64}}, suffix::String)
    stats_file = first(filter(endswith("raw_observables.csv"), readdir(path, join=true, sort=true)))
    stats_file = replace(bin_file, "binned_linear_observables.json" => suffix * "_observables.json")

    open(stats_file, "w") do io
        JSON.print(io, stats, 2)
    end
end

function rebin_data(bins::Dict, binsize::Int, num_bins::Int; rfactor=2)
    new_num_bins = num_bins ÷ rfactor
    new_bins = Dict{String, Vector{Float64}}(lin_obs => zeros(Float64, new_num_bins) for lin_obs in LINEAR_OBSERVABLES)

    stride = 0
    for bin_num in 1:new_num_bins
        for lin_obs in LINEAR_OBSERVABLES
            new_bins[lin_obs][bin_num] = mean(bins[lin_obs][bin_num + i + stride] for i in 0:(rfactor-1))
        end
        stride += rfactor-1
    end

    return new_bins
end

function make_bin!(observable_file::String, bin_num::Int, binsize::Int, bins::Dict)
    # skipto = 100002 leaves out the equilibration phase
    # DO NOT TRY READING THE ENTIRE CSV INTO MEMORY!!!
    df = DataFrame(CSV.File(observable_file, skipto=100_002 + binsize*bin_num, limit=binsize))

    bins["mags"][bin_num] = mean(df.mags)
    bins["smags"][bin_num] = mean(df.smags)
    bins["n"][bin_num] = mean(df.n)

    bins["abs_mags"][bin_num] = mean(abs.(df.mags))
    bins["abs_smags"][bin_num] = mean(abs.(df.smags))
    bins["mags2"][bin_num] = mean(df.mags .^ 2)
    bins["smags2"][bin_num] = mean(df.smags .^ 2)
    bins["mags4"][bin_num] = mean(df.mags .^ 4)
    bins["smags4"][bin_num] = mean(df.smags .^ 4)
end

function bin_data(path::String, binsize::Int; rseed=nothing)
    dir_contents = readdir(path, join=true, sort=true)
    observable_files = filter(endswith("raw_observables.csv"), dir_contents)

    if rseed != nothing
        observable_files = filter(x -> occursin("seed=$rseed", x), observable_files)
    end

    num_bins = TOTAL_MEASUREMENTS ÷ binsize # per random seed
    num_seeds = length(observable_files)
    bins = Dict{String, Vector{Float64}}(lin_obs => zeros(Float64, num_bins * num_seeds) for lin_obs in LINEAR_OBSERVABLES)

    bin_num = 1
    for (i, file) in enumerate(observable_files)
        for _ in 1:num_bins
            make_bin!(file, bin_num, binsize, bins)
            bin_num += 1
        end
    end

    # save the bins
    bin_file = first(filter(endswith("raw_observables.csv"), readdir(path, join=true, sort=true)))
    bin_file = replace(bin_file, "raw_observables.csv" => "binned_linear_observables.json")

    if rseed == nothing
        bin_file = replace(bin_file, r"_seed=\d+" => "_ALLSEEDS")
    else
        bin_file = replace(bin_file, r"_seed=\d+" => "_seed=$rseed")
    end

    open(bin_file, "w") do io
        JSON.print(io, bins, 2)
    end
    
    return bin_file, num_bins
end

function cleanup_single_system(parsed_args)
    path, L, δ, M, rseed, binsize, rfactor = init_cli(parsed_args)
    factor = 2
    bin_file, num_bins = bin_data(path, binsize, rseed=rseed)

    bins = Dict()
    open(bin_file, "r") do f
        bins = JSON.parse(f, dicttype=Dict{String, Vector{Float64}})
    end

    new_bins = rebin_data(bins, binsize, num_bins, rfactor=rfactor)

    stats = Dict(obs => Dict{String, Float64}() for obs in OBSERVABLES)
    rebin_stats = Dict(obs => Dict{String, Float64}() for obs in OBSERVABLES)
    for lin_obs in LINEAR_OBSERVABLES
        stats[lin_obs]["value"] = mean(bins[lin_obs])
        stats[lin_obs]["error"] = stdm(bins[lin_obs], stats[lin_obs]["value"]) / sqrt(num_bins)

        rebin_stats[lin_obs]["value"] = mean(new_bins[lin_obs])
        rebin_stats[lin_obs]["error"] = stdm(new_bins[lin_obs], rebin_stats[lin_obs]["value"]) / sqrt(num_bins ÷ rfactor)
    end

    # get H and qmc_state to calculate energy
    state_file = filter(endswith(".jld2"), readdir(path, join=true, sort=true))[1]
    H, qmc_state = load(state_file, "hamiltonian", "qmc_state")

    E_density = energy(qmc_state, H, bins["n"]) / nspins(H)
    stats["energy"]["value"] = E_density.val
    stats["energy"]["error"] = E_density.err        

    rebin_E_density = energy(qmc_state, H, new_bins["n"]) / nspins(H)
    rebin_stats["energy"]["value"] = rebin_E_density.val
    rebin_stats["energy"]["error"] = rebin_E_density.err

    save_stats(path, bin_file, stats, "processed")
    save_stats(path, bin_file, rebin_stats, "rebinned_processed")
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

    "--delta"
        help = "Strength of the detuning"
        arg_type = Float64

    "-M"
        help = "Projector length."
        arg_type = Int64

    "--seed"
        help = "Specify cleaning up simulations for a single seed. If this is nothing, all possible random seeds will be combined."
        default = nothing

    "--multibranch", "--mb"
        help = "Specify that the update used was the multibranch update, not the (default) line update."
        action = :store_true

    "--binsize"
        help = "How many consecutive MC measurements to average together into a binned value. This value should be greater than correlation time."
        arg_type = Int
        default = 20_000
    "--rfactor"
        help = "The rebinning factor. For example, if 1000 bins are made and rfactor = 2, adjacent bins will be averaged into one new bin to see if error bars have converged."
        arg_type = Int
        default = 2
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
    @time cleanup_single_system(parsed_args["groundstate"])
else
    @time cleanup_single_system(parsed_args["mixedstate"])
end
