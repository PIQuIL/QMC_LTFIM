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


SCRATCH_PATH = "/media/ejaaz/Seagate Expansion Drive/qmc_data/"
# SCRATCH_PATH = "/scratch-deleted-2021-mar-20/ejaazm/"

RELIABLE_SIZE = 256

###############################################################################

function init_cli(parsed_args)
    δ = parsed_args["delta"]
    nY = parsed_args["nY"]
    M = parsed_args["M"]

    path = joinpath(
        SCRATCH_PATH, "qmc_sims",
        "disord2checkerboard",
        "groundstate",
        "Rydberg_QCP", "nY=$nY", "delta=$δ", "M=$M")

    if !isdir(path)
        println("$path doesn't point to a valid directory!")
        exit(1)
    end

    return path, parsed_args["delete"], nY, δ, M
end

measurementtodict(V::BinningAnalysis.Variance) = Dict("value" => mean(V), "error" => std_error(V))
measurementtodict(M::Measurement) = Dict("value" => M.val, "error" => M.err)
measurementtodict(B::LogBinner{T, N}, convergence_threshold::Float64 = 0.05) where {T, N} = Dict(
    "value" => mean(B),
    "error" => std_error(B),
    "tau" => tau(B),
    "has_converged" => has_converged(B,
                                     something(findlast(x -> x.count >= RELIABLE_SIZE, B.accumulators), 1),
                                     convergence_threshold),
    "convergence" => convergence(B),
    "convergence_threshold" => convergence_threshold,

    "all_varNs" => all_varNs(B),
    "all_taus" => all_taus(B),
    "all_std_errors" => all_std_errors(B),
    "all_means" => all_means(B),
    "all_counts" => [count(B, lvl) for lvl in 1:N if count(B, lvl) > 1]
)



function get_df(path, nY, delta, M)
    dir_contents = readdir(path, join=true, sort=true)

    observable_files = filter(endswith("raw_observables.csv"), dir_contents)

    full_df = DataFrame()

    for (i, file) in enumerate(observable_files)
        df = DataFrame(CSV.File(file));

        df.abs_mags = abs.(df.mags)
        df.abs_smags = abs.(df.smags)
        df.mags2 = df.mags .^ 2
        df.smags2 = df.smags .^ 2
        df.mags4 = df.mags .^ 4
        df.smags4 = df.smags .^ 4

        df[!, :chain] .= i

        seed = split(file, "seed=")[2]
        seed = parse(Int, split(seed, "_")[1])
        df[!, :seed] .= seed

        df[!, :nY] .= nY
        df[!, :delta] .= delta
        df[!, :M] .= M

        append!(full_df, df)
    end

    return full_df
end


function plot_equilibration(plots_path, gdf)
    plots_path = joinpath(plots_path, "equilibration")
    mkpath(plots_path)

    for observable in [:n_ssd, :smags, :abs_smags, :smags2, :smags4, :mags, :abs_mags, :mags2, :mags4]
        for (ch, df) in enumerate(gdf)
            seed = df[1, :seed]

            plt = df |>
                @filter(_.batch < 5) |>
                @df density(
                    df[:, ^(observable)],
                    group = :batch,
                    fmt = :png,
                    legend = true,
                    title = "Density histogram of $observable by batch (chain #$(ch))",
                    size = (750, 500)
                )
            savefig(plt, joinpath(plots_path, "density_$(observable)_chain_$(ch)_seed_$(seed).png"))

            plt = df |>
                @filter(_.batch < 10) |>
                @df violin(
                    :batch,
                    df[:, ^(observable)],
                    fmt = :png,
                    title = "Violin/Boxplot of $observable by batch (chain #$(ch))",
                    xlabel = "Batch",
                    ylabel = "$observable",
                    legend = false,
                    size = (750, 500)
                )
            plt = df |>
                @filter(_.batch < 10) |>
                @df boxplot!(
                    :batch,
                    df[:, ^(observable)],
                    fmt = :png,
                    legend = false,
                    fillalpha = 0.5,
                )
            savefig(plt, joinpath(plots_path, "violin_$(observable)_chain_$(ch)_seed_$(seed).png"))
        end
    end
end


function plot_corr_time_convergence(plots_path, gdf)
    plots_path = joinpath(plots_path, "correlation_times")
    mkpath(plots_path)

    for observable in [:n_ssd, :abs_smags, :smags2, :smags4]
        for (ch, df) in enumerate(gdf)
            seed = df[1, :seed]
            binner = LogBinner(df[!, observable])

            τ = all_taus(binner)
            τ = @. 2*τ + 1
            plt = plot(τ,
                       legend = :outerbottom,
                       title = "Correlation time of $(observable) (chain #$(ch))",
                       xlabel = "binning level, L",
                       ylabel = "\\tau_L",
                       label = "\\tau_L",
                       size = (500, 600))

            vline!([something(findlast(x -> x.count >= RELIABLE_SIZE, binner.accumulators), 1)],
                    label = "highest level with >= $RELIABLE_SIZE samples")
            savefig(plt, joinpath(plots_path, "$(observable)_chain_$(ch)_seed_$(seed).png"))
        end
    end
end


function energy_binning(qmc_state, H, n_ssd::AbstractVector)
    n_ssd = Vector(n_ssd)
    E_density = energy_density(qmc_state, H, n_ssd)
    std_err = E_density.err
    val = E_density.val

    all_std_errs = [std_err]
    all_counts = [length(n_ssd)]

    while length(n_ssd) >= RELIABLE_SIZE
        if iseven(length(n_ssd))
            n_ssd = (n_ssd[1:2:end] + n_ssd[2:2:end]) / 2
        else
            n_ssd = (n_ssd[1:2:end-1] + n_ssd[2:2:end]) / 2
        end

        E_density = energy_density(qmc_state, H, n_ssd)
        std_err = E_density.err
        push!(all_std_errs, std_err)
        push!(all_counts, length(n_ssd))
    end

    return val, all_std_errs, all_counts
end



function binder_binning(smags4::AbstractVector, smags2::AbstractVector)
    smags4, smags2 = Vector(smags4), Vector(smags2)
    binder_cumulant = QMC.jackknife(smags4, smags2) do M4, M2
        (3 - (M4 / (M2 ^ 2))) / 2
    end
    std_err = binder_cumulant.err
    val = binder_cumulant.val

    all_std_errs = [std_err]
    all_counts = [length(smags4)]

    while length(smags4) >= RELIABLE_SIZE
        if iseven(length(smags4))
            smags2 = (smags2[1:2:end] + smags2[2:2:end]) / 2
            smags4 = (smags4[1:2:end] + smags4[2:2:end]) / 2
        else
            smags2 = (smags2[1:2:end-1] + smags2[2:2:end]) / 2
            smags4 = (smags4[1:2:end-1] + smags4[2:2:end]) / 2
        end

        binder_cumulant = QMC.jackknife(smags4, smags2) do M4, M2
            (3 - (M4 / (M2 ^ 2))) / 2
        end

        std_err = binder_cumulant.err
        push!(all_std_errs, std_err)
        push!(all_counts, length(smags4))
    end

    return val, all_std_errs, all_counts
end




function estimate_observables_for_one_chain(state_file, observables, df)
    H, qmc_state = load(state_file, "hamiltonian", "qmc_state")

    msmt_dict = Dict(String(obs) => measurementtodict(LogBinner(df[!, obs]))
                     for obs in observables)

    E_density, E_std_errs, E_counts = energy_binning(qmc_state, H, df.n_ssd)
    msmt_dict["energy_density"] = Dict("value" => E_density,
                                       "error" => maximum(E_std_errs),
                                       "all_std_errors" => E_std_errs,
                                       "all_counts" => E_counts)

    binder_cumulant, U_std_errs, U_counts = binder_binning(df.smags4, df.smags2)
    msmt_dict["binder_cumulant"] = Dict("value" => binder_cumulant,
                                        "error" => maximum(U_std_errs),
                                        "all_std_errors" => U_std_errs,
                                        "all_counts" => U_counts)

    return msmt_dict
end


function estimate_observables(path, gdf)
    # any state object works, we just need the Hamiltonian struct and the qmc_state's type
    state_files = filter(endswith(".jld2"), readdir(path, join=true, sort=true))
    state_file = last(state_files)

    observables = [:n_ssd, :smags, :abs_smags, :smags2, :smags4]

    msmt_dicts = Dict(
        "chain_$chain" => estimate_observables_for_one_chain(state_file, observables, df)
        for (chain, df) in enumerate(gdf)
    )

    mean_df = deepcopy(select(gdf[1], observables))
    for i in 2:gdf.ngroups
        mean_df .+= select(gdf[i], observables)
    end
    mean_df ./= gdf.ngroups

    msmt_dicts["combined"] = estimate_observables_for_one_chain(state_file, observables, mean_df)

    msmt_file = first(filter(endswith("raw_observables.csv"), readdir(path, join=true, sort=true)))
    msmt_file = replace(msmt_file, "raw_observables.csv" => "observables.json")
    msmt_file = replace(msmt_file, r"_seed=\d+" => "")
    open(msmt_file, "w") do io
        JSON.print(io, msmt_dicts, 2)
    end
end


###############################################################################

function cleanup_single_system(parsed_args)
    path, delete_files, nY, δ, M = init_cli(parsed_args)

    df = get_df(path, nY, δ, M)

    println("Beginning cleanup for system nY=$nY, delta=$δ, M=$M...")

    plots_path = joinpath(path, "plots")
    mkpath(plots_path)

    gdf = groupby(df, :seed)

    println("Generating equilibration plots...")
    plot_equilibration(plots_path, gdf)

    println("Generating correlation time plots...")
    plot_corr_time_convergence(plots_path, gdf)

    println("Estimating observables...")

    # drop equilibration samples
    df = df |> @filter(_.batch > 0) |> DataFrame
    gdf = groupby(df, :seed)
    estimate_observables(path, gdf)

    if delete_files
        println("Deleting raw observables files...")
        foreach(rm, filter(endswith("raw_observables.csv"), readdir(path, join=true)))
    end
end





###############################################################################


s = ArgParseSettings()


@add_arg_table! s begin
    "nY"
        help = "The length of the square lattice along the Y axis (PBC dimension)"
        required = true
        arg_type = Int

    "delta"
        help = "Strength of the detuning"
        arg_type = Float64

    "M"
        help = "Projector length."
        arg_type = Int64

    "--delete"
        help = "Delete files that are no longer needed"
        action = :store_true
end


parsed_args = parse_args(ARGS, s)

cleanup_single_system(parsed_args)