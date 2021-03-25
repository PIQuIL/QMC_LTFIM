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


SCRATCH_PATH = "../../qmc_data/"

RELIABLE_SIZE = 256

###############################################################################

function init_cli(parsed_args)
    δ = parsed_args["delta"]
    L = parsed_args["L"]
    M = parsed_args["M"]
    mb = parsed_args["multibranch"]

    update_name = mb ? "multibranch" : "line"

    if haskey(parsed_args, "beta")
        beta = parsed_args["beta"]
        path = joinpath(
            SCRATCH_PATH, "qmc_sims",
            "1D",
            "thermalstate", "$update_name",
            "L=$L", "delta=$δ", "M=$M", "beta=$beta")
        
    else
        path = joinpath(
            SCRATCH_PATH, "qmc_sims",
            "1D",
            "groundstate", "$update_name",
            "L=$L", "delta=$δ", "M=$M")

    end    

    if !isdir(path)
        println("$path doesn't point to a valid directory!")
        exit(1)
    end

    return path, L, δ, M
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



function get_df(path, L, delta, M)
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

        df[!, :L] .= L
        df[!, :delta] .= delta
        df[!, :M] .= M

        append!(full_df, df)
    end

    return full_df
end


function energy_binning(qmc_state, H, n::AbstractVector; beta=0.0)
    # n = n_ssd (0-Temp), expansion order (finite temp)
    n = Vector(n)

    if qmc_state isa BinaryThermalState
        E_density = energy_density(qmc_state, H, beta, n)
    else
        E_density = energy_density(qmc_state, H, n)
    end

    std_err = E_density.err
    val = E_density.val

    all_std_errs = [std_err]
    all_counts = [length(n)]

    while length(n) >= RELIABLE_SIZE
        if iseven(length(n))
            n = (n[1:2:end] + n[2:2:end]) / 2
        else
            n = (n[1:2:end-1] + n[2:2:end]) / 2
        end

        E_density = energy_density(qmc_state, H, n)
        std_err = E_density.err
        push!(all_std_errs, std_err)
        push!(all_counts, length(n))
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




function estimate_observables_for_one_chain(state_file, observables, df; beta=0.0)
    H, qmc_state = load(state_file, "hamiltonian", "qmc_state")

    msmt_dict = Dict(String(obs) => measurementtodict(LogBinner(df[!, obs]))
                     for obs in observables)

    E_density, E_std_errs, E_counts = energy_binning(qmc_state, H, df.n; beta=beta)
    msmt_dict["energy_density"] = Dict("value" => E_density,
                                       "error" => maximum(E_std_errs),
                                       "all_std_errors" => E_std_errs,
                                       "all_counts" => E_counts)

    #=
    binder_cumulant, U_std_errs, U_counts = binder_binning(df.smags4, df.smags2)
    msmt_dict["binder_cumulant"] = Dict("value" => binder_cumulant,
                                        "error" => maximum(U_std_errs),
                                        "all_std_errors" => U_std_errs,
                                        "all_counts" => U_counts)
    =#
    return msmt_dict
end


function estimate_observables(path, gdf; beta=0.0)
    # any state object works, we just need the Hamiltonian struct and the qmc_state's type
    state_files = filter(endswith(".jld2"), readdir(path, join=true, sort=true))
    state_file = last(state_files)

    observables = [:n, :mags, :smags]

    msmt_dicts = Dict(
        "chain_$chain" => estimate_observables_for_one_chain(state_file, observables, df; beta=beta)
        for (chain, df) in enumerate(gdf)
    )

    mean_df = deepcopy(select(gdf[1], observables))
    for i in 2:gdf.ngroups
        mean_df .+= select(gdf[i], observables)
    end
    mean_df ./= gdf.ngroups

    msmt_dicts["combined"] = estimate_observables_for_one_chain(state_file, observables, mean_df; beta=beta)

    msmt_file = first(filter(endswith("raw_observables.csv"), readdir(path, join=true, sort=true)))
    msmt_file = replace(msmt_file, "raw_observables.csv" => "observables.json")
    msmt_file = replace(msmt_file, r"_seed=\d+" => "")
    open(msmt_file, "w") do io
        JSON.print(io, msmt_dicts, 2)
    end
end

###############################################################################

function cleanup_single_system(parsed_args)
    path, L, δ, M = init_cli(parsed_args)

    df = get_df(path, L, δ, M)

    if haskey(parsed_args, "beta")
        beta = parsed_args["beta"]
        println("Beginning cleanup for system L=$L, delta=$δ, M=$M, beta=$beta...")
    else
        beta = 0.
        println("Beginning cleanup for system L=$L, delta=$δ, M=$M...")
    end

    gdf = groupby(df, :seed)

    println("Estimating observables...")

    # drop equilibration samples
    df = df |> @filter(_.batch > 0) |> DataFrame
    gdf = groupby(df, :seed)
    
    estimate_observables(path, gdf; beta=beta)

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
    @time cleanup_single_system(parsed_args["groundstate"])
else
    @time cleanup_single_system(parsed_args["mixedstate"])
end