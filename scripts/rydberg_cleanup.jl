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

using DataFrames
using JSON
using CSV

using ArgParse


SCRATCH_PATH = "/home/ejaazm/scratch"

###############################################################################

function init_mc_cli(parsed_args)
    δ = parsed_args["delta"]
    nY = parsed_args["nY"]

    binsize = parsed_args["bin-size"]
    path = joinpath(SCRATCH_PATH, "qmc_sims", "groundstate", "Rydberg_QCP", "nY=$nY", "delta=$δ")

    return path, binsize, parsed_args["delete"]
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

    "all_varNs" => all_varNs(B),
    "all_taus" => all_taus(B),
    "all_std_errors" => all_std_errors(B),
    "all_means" => all_means(B)
)


function reduce_runstats(delete_files::Bool, dir_contents::Vector{String})
    runstats_files = filter(endswith("runstats.txt"), dir_contents)
    runstats_files = runstats_files[2:end]  # drop equilibriation samples

    runstats_names = only(filter(endswith("runstats_columns.txt"), dir_contents))
    runstats_names = reshape(readdlm(runstats_names, String), :)

    runstats_df = DataFrame()

    for (i, file) in enumerate(runstats_files)
        data = readdlm(file, ',', Float64)
        df = DataFrame(data, runstats_names);

        append!(runstats_df, combine(df, names(df) .=> mean .=> names(df)))

        if delete_files
            rm(file)
        end
    end

    runstats_df = combine(runstats_df, names(runstats_df) .=> mean)[1, :]
    runstats = NamedTuple(runstats_df)

    runstats_file = replace(only(filter(endswith("runstats_columns.txt"), dir_contents)),
                            "runstats_columns.txt" => "runstats.json")

    open(runstats_file, "w") do io
        JSON.print(io, runstats, 2)
    end
end



function cleanup(parsed_args)
    path, binsize, delete_files = init_mc_cli(parsed_args)

    dir_contents = readdir(path, join=true, sort=true)
    reduce_runstats(delete_files, dir_contents)

    observable_files = filter(endswith("raw_observables.txt"), dir_contents)
    observable_files = observable_files[2:end]  # drop equilibriation samples

    observable_names = only(filter(endswith("raw_observables_columns.txt"), dir_contents))
    observable_names = reshape(readdlm(observable_names, String), :)

    binners = Dict()
    num_batches = length(observable_files)

    binned_df = DataFrame()

    for (i, file) in enumerate(observable_files)
        data = readdlm(file, ',', Float64)

        df = DataFrame(data, observable_names);
        df.abs_mags = abs.(df.mags)
        df.abs_smags = abs.(df.smags)
        df.mags2 = df.mags .^ 2
        df.smags2 = df.smags .^ 2
        df.mags4 = df.mags .^ 4
        df.smags4 = df.smags .^ 4;

        if isempty(binners)
            for col in names(df)
                binners[col] = LogBinner()
            end
        end

        for col in names(df)
            append!(binners[col], df[!, col])
        end

        df.block = repeat(1:(size(df, 1) ÷ binsize); inner=binsize)

        gd = groupby(df, :block)
        append!(binned_df, combine(gd, valuecols(gd) .=> mean))

        if delete_files
            rm(file)
        end
    end
    binned_df = select(binned_df, Not(:block))

    state_files = filter(endswith(".jld2"), readdir(path, join=true, sort=true))

    if delete_files
        for f in state_files[1:end-1]
            rm(f)
        end
    end

    state_file = state_files[end]

    H, qmc_state = load(state_file, "hamiltonian", "qmc_state")

    E_density = energy_density(qmc_state, H, binned_df.ns_mean)

    binder_cumulant = QMC.jackknife(binned_df.smags4_mean, binned_df.smags2_mean) do M4, M2
        (3 - (M4 / (M2 ^ 2))) / 2
    end

    msmt_dicts = Dict(
        k => measurementtodict(v) for (k, v) in binners
    )
    msmt_dicts["energy_density"] = measurementtodict(E_density)
    msmt_dicts["binder_cumulant"] = measurementtodict(binder_cumulant)

    msmt_file = replace(only(filter(endswith("raw_observables_columns.txt"), dir_contents)),
                        "raw_observables_columns.txt" => "observables.json")

    open(msmt_file, "w") do io
        JSON.print(io, msmt_dicts, 2)
    end

    binned_file = replace(msmt_file, "observables.json" => "binned_observables.csv")
    CSV.write(binned_file, binned_df)
end


###############################################################################


s = ArgParseSettings()


@add_arg_table! s begin
    "nY"
        help = "The length of the square lattice along the Y axis (PBC dimension)"
        required = true
        arg_type = Int
    "--delta"
        help = "Strength of the longitudinal field"
        arg_type = Float64
        default = 1.0
    "--bin-size", "-b"
        help = "Size of bins to reduce data to"
        arg_type = Int
        default = 100
    "--delete"
        help = "Delete files that are no longer needed"
        action = :store_true
end


parsed_args = parse_args(ARGS, s)

cleanup(parsed_args)