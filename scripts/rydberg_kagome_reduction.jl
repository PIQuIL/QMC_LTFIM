using QMC

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

function bin_observables(nX::Int, Rb::Float64, delta::Float64, trunc::Float64, M::Int, seed::Int)
    batches = ["1", "2", "3", "4", "5"]
    root_path = "/home/ejaazm/scratch/qmc_sims/kagome/groundstate/pa1=true/pa2=true/trunc=$(trunc)/p=0.0/n1=$nX/n2=$nX/t=1.0/Rb=$Rb/delta=$delta/M=$M/"

    mags, nematic = LogBinner(), LogBinner()

    n_ssd = Vector{Float64}(undef, 0)

    delta_s = @sprintf("%.2f", delta)
    for batch in batches
        data, cols = nothing, nothing
        try
            data, cols = readdlm(root_path * "R_b=$(Rb)_lat=Kagome_n1=$(nX)_n2=$(nX)_seed=$(seed)_Ω=1.0_δ=$(delta)_batch_$(batch)_raw_observables.csv", ',', Float64, header=true)
        catch e
            println("Error reading csv file of nX=$nX, Rb=$Rb, delta=$delta, trunc=$trunc, M=$M, seed=$seed, batch=$batch")
            println("Error was", e)
            continue
        end
        df = DataFrame(data, cols[:])
        append!(n_ssd, df.n_ssd)
        append!(mags, abs.(df.mags))
        append!(nematic, @. sqrt(df.nematic_real^2 + df.nematic_imag^2))
    end

    state = jldopen(root_path * "R_b=$(Rb)_lat=Kagome_n1=$(nX)_n2=$(nX)_seed=$(seed)_Ω=1.0_δ=$(delta)_batch_5_state.jld2")

    qmc_state = state["qmc_state"]
    H = state["hamiltonian"]

    E = energy_binning(qmc_state, H, n_ssd)
    return E, mags, nematic
end


function run_binning(parsed_args)
    nX = parsed_args["nX"]
    Rb = parsed_args["Rb"]
    trunc = parsed_args["trunc"]

    delta = 3.3

    root_path = "/home/ejaazm/scratch/qmc_sims/kagome/reduced/"
    mkpath(root_path)

    for M in 10000:5000:40000, seed in 1000:1000:5000
        folder = root_path * "nX_$(nX)/Rb_$(Rb)/trunc_$(trunc)/"
        mkpath(folder)
        file = folder * "M_$(M)_seed_$(seed).jld2"
        if !isfile(file)
            E, mags, nematic = bin_observables(nX, Rb, delta, trunc, M, seed)
            @save(file, energy=E, mags=mags, nematic=nematic)
        end
    end
end



s = ArgParseSettings()


@add_arg_table! s begin
    "nX"
        help = "The length of the kagome lattice along the X axis"
        required = true
        arg_type = Int

    "Rb"
        help = "Blockade radius"
        required = true
        arg_type = Float64

    "trunc"
        help = "Truncation distance"
        required = true
        arg_type = Float64
end


parsed_args = parse_args(ARGS, s)
@time run_binning(parsed_args)
