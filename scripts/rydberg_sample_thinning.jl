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
using Plots
using StatsPlots
using Printf
using ArgParse


# SCRATCH_PATH = "/media/ejaaz/Seagate Expansion Drive/qmc_data/"
SCRATCH_PATH = "/scratch/ejaazm/"

###############################################################################

function init_cli(parsed_args)
    δ = Float64(parsed_args["delta"])
    R_b = Float64(parsed_args["radius"])
    nY = parsed_args["nY"]
    mb_prob = parsed_args["mb-prob"]
    M = parsed_args["M"]

    ts_type = parsed_args["trialstate"]
    truncation = parsed_args["trunc"]

    path = joinpath(
        SCRATCH_PATH, "qmc_sims",
        "snn_data",
        "nY=$nY",
        "Rb=$(@sprintf("%.2f", R_b))",
        "delta=$(@sprintf("%.2f", δ))",
        "trunc=$truncation",
        "ts=$ts_type",
        "M=$M",
        "p=$mb_prob")

    if !isdir(path)
        println("$path doesn't point to a valid directory!")
        exit(1)
    end

    return path, nY, δ, R_b, M, mb_prob, ts_type, truncation
end


function thin(parsed_args)
    path, nY, δ, R_b, M, mb_prob, ts_type, truncation = init_cli(parsed_args)

    println("Beginning sample thinning for system nY=$nY, R_b=$R_b, delta=$δ, M=$M, p=$mb_prob...")

    dir_contents = readdir(path, join=true, sort=true)

    sample_files = filter(endswith("spin_configs.csv"), dir_contents)
    sample_files = filter(!contains("batch_0"), sample_files)  #drop EQ samples

    seeds = [parse(Int, split(match(r"seed=\d+", file).match, "=")[2])
             for file in sample_files]

    observable_file = only(filter(endswith("observables.json"), dir_contents))
    observable_dict = JSON.parsefile(observable_file)
    taus = Dict([v["seed"] => v["checkerboard"]["tau"]
                 for (k, v) in observable_dict
                 if startswith(k, "chain")])

    thinned_samples_file = replace(observable_file, "observables.json" => "samples.csv")

    for (i, seed) in enumerate(seeds)
        skip = ceil(Int, 2*taus[seed] + 1)
        chain_df = DataFrame()
        for file in filter(contains("seed=$seed"), sample_files)
            df = convert.(Int, DataFrame(CSV.File(file, header=0)))
            append!(chain_df, df)
        end
        CSV.write(thinned_samples_file, chain_df[1:skip:end, :],
                  append=(i > 1), writeheader=false)
    end
end


###############################################################################


s = ArgParseSettings()


@add_arg_table! s begin
    "nY"
        help = "The length of the square lattice along the Y axis (PBC dimension)"
        required = true
        arg_type = Int

    "--delta"
        help = "Strength of the detuning"
        arg_type = Float64
        default = 1.0
    "--radius", "-R"
        help = "Rydberg blockade radius (in units of the lattice spacing). Controls the strength of the interaction."
        arg_type = Float64
        default = 1.2

    "--trunc"
        help = """Interaction truncation.
                Passing K > 0 will keep interactions upto and including the K'th nearest neighbour interactions.
                Passing K <= 0 will keep all interactions.
               """
        arg_type = Int
        default = 0

    "-M"
        help = "Projector length."
        arg_type = Int64
        default = 10_000

    "--mb-prob"
        help = "Probability of performing a multibranch cluster update"
        arg_type = Float64
        default = 0.0

    "--trialstate"
        help = "Trial state type"
        arg_type = String
        default = "plus"
end


parsed_args = parse_args(ARGS, s)

@time thin(parsed_args)
