using QMC

using FileIO
using LinearAlgebra
using JLD2
using ArgParse
using DataFrames
using DelimitedFiles
using BinningAnalysis
using Printf


function block_average!(v::AbstractVector{T}, blocksize::Int=2) where T <: AbstractFloat
    L, rem = divrem(length(v), blocksize)

    @inbounds for i in 0:(L-1)
        offset = blocksize*i + 1
        v[i + 1] = v[offset]
        for j in 1:(blocksize-1)
            v[i + 1] += v[offset + j]
        end
        v[i + 1] /= blocksize
    end

    if !iszero(rem)
        offset = blocksize*L + 1
        v[L + 1] = v[offset]
        for j in 1:(rem - 1)
            v[L + 1] += v[offset + j]
        end
        v[L + 1] /= rem
    end

    resize!(v, L + Int(!iszero(rem)))
    return v
end

function energy_binning(qmc_state, H, n_ssd::AbstractVector)
    n_ssd = Vector(n_ssd)
    E_density = QMC.energy_density(qmc_state, H, n_ssd)
    std_err = E_density.err
    val = E_density.val

    all_std_errs = [std_err]
    all_counts = [length(n_ssd)]

    while length(n_ssd) >= 256
        block_average!(n_ssd, 2)

        E_density = QMC.energy_density(qmc_state, H, n_ssd)
        std_err = E_density.err
        push!(all_std_errs, std_err)
        push!(all_counts, length(n_ssd))
    end

    return val, all_std_errs, all_counts
end

function bin_observables(delta::Float64)
    root_path = "/home/ejaazm/scratch/qmc_sims/scipost/groundstate/1D-redo/nY=51/"

    batches = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]

    delta_s = @sprintf("%.2f", delta)
    jldopen(root_path * "summary/Rb=1.20_delta=$(delta_s).jld2", "w") do file
        for p in [0.0, 0.5, 1.0], epsilon in [0.1] #, truncation in 0:20
            chk, mag = LogBinner(), LogBinner()
            n_ssd = Vector{Float64}(undef, 0)

            for batch in batches
                data, cols = readdlm(root_path * "Rb=1.20/delta=$(delta_s)/p=$p/eps=$epsilon/R_b=1.2_nY=51_seed=1234_Ω=1.0_δ=$(delta)_batch_$(batch)_raw_observables.csv", ',', Float64, header=true)
                df = DataFrame(data, cols[:])
                append!(chk, abs.(df.checkerboard))
                append!(mag, abs.(df.mags))
                append!(n_ssd, df.n_ssd)
            end

            qmc_state, H = load(root_path * "Rb=1.20/delta=$(delta_s)/p=$p/eps=$epsilon/R_b=1.2_nY=51_seed=1234_Ω=1.0_δ=$(delta)_batch_$(batches[end])_state.jld2",
                                "qmc_state", "hamiltonian")
            E = energy_binning(qmc_state, H, n_ssd)

            file["p=$(p)/epsilon=$(epsilon)"] = (:chk => chk, :mag => mag,
                                                 :n_ssd => LogBinner(n_ssd),
                                                 :energy => E)
        end
    end
end


function run_binning(parsed_args)
    delta = parsed_args["delta"]
    root_path = "/home/ejaazm/scratch/qmc_sims/scipost/groundstate/1D-redo/nY=51/summary/"
    mkpath(root_path)

    bin_observables(delta)
end



s = ArgParseSettings()


@add_arg_table! s begin
    "delta"
        help = "detuning"
        required = true
        arg_type = Float64
end


parsed_args = parse_args(ARGS, s)
@time run_binning(parsed_args)
