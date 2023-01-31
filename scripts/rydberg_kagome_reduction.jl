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


function combine_two_pt_fn(two_pt_A::Matrix{T}, one_pt_A::Vector{T}, n_A::Int,
                           two_pt_B::Matrix{T}, one_pt_B::Vector{T}, n_B::Int) where T <: AbstractFloat
    n_AB = n_A + n_B
    w_A, w_B = n_A/n_AB, n_B/n_AB

    one_pt = (@. (w_A * one_pt_A) + (w_B * one_pt_B))

    two_pt = (@. (w_A * two_pt_A) + (w_B * two_pt_B))
    LinearAlgebra.BLAS.syr!('U', w_A*w_B, (@. one_pt_A - one_pt_B), two_pt)

    return two_pt, one_pt, n_AB
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
    batches = map(i -> lpad(i, 3, "0"), 1:100)
    root_path = "/home/ejaazm/scratch/qmc_sims/kagome/groundstate/pa1=true/pa2=true/trunc=$(trunc)/p=0.0/n1=$nX/n2=$nX/t=1.0/Rb=$Rb/delta=$delta/M=$M/"

    mags, nematic = LogBinner(), LogBinner()

    n_ssd = Vector{Float64}(undef, 0)

    two_pt = zeros(3*nX*nX, 3*nX*nX)
    one_pt = zeros(3*nX*nX)
    n = 0

    delta_s = @sprintf("%.2f", delta)
    for batch in batches
        data, cols = nothing, ["batch" "n_ssd" "mags" "nematic_real" "nematic_imag"]
        try
            data, _ = readdlm(root_path * "R_b=$(Rb)_lat=Kagome_n1=$(nX)_n2=$(nX)_seed=$(seed)_Ω=1.0_δ=$(delta)_batch_$(batch)_raw_observables.csv", ',', Float64, header=true)
        catch e
            println("Error reading csv file of nX=$nX, Rb=$Rb, delta=$delta, trunc=$trunc, M=$M, seed=$seed, batch=$batch")
            println("Error was", e)
            continue
        end
        df = DataFrame(data, cols[:])
        append!(n_ssd, df.n_ssd)
        append!(mags, abs.(df.mags))
        append!(nematic, @. sqrt(df.nematic_real^2 + df.nematic_imag^2))

        try
            pt2 = readdlm(root_path * "R_b=$(Rb)_lat=Kagome_n1=$(nX)_n2=$(nX)_seed=$(seed)_Ω=1.0_δ=$(delta)_batch_$(batch)_2_pt_fn.csv", ' ', Float64, header=false)
            pt1 = readdlm(root_path * "R_b=$(Rb)_lat=Kagome_n1=$(nX)_n2=$(nX)_seed=$(seed)_Ω=1.0_δ=$(delta)_batch_$(batch)_1_pt_fn.csv", ' ', Float64, header=false)[:]

            two_pt, one_pt, n = combine_two_pt_fn(two_pt, one_pt, n, pt2, pt1, length(df.n_ssd))
        catch e
            println("Error aggregating one_pt or two_pt fns of nX=$nX, Rb=$Rb, delta=$delta, trunc=$trunc, M=$M, seed=$seed, batch=$batch")
#             rethrow(e)
            println("A file may be corrupted")
            println(e)
            continue
        end
    end

    state = jldopen(root_path * "R_b=$(Rb)_lat=Kagome_n1=$(nX)_n2=$(nX)_seed=$(seed)_Ω=1.0_δ=$(delta)_batch_$(batches[end])_state.jld2")

    qmc_state = state["qmc_state"]
    H = state["hamiltonian"]

    E = energy_binning(qmc_state, H, n_ssd)
    return E, mags, nematic, two_pt, one_pt, n
end


function run_binning(parsed_args)
    nX = parsed_args["nX"]
    Rb = parsed_args["Rb"]
    trunc = parsed_args["trunc"]

    delta = 3.3

    root_path = "/home/ejaazm/scratch/qmc_sims/kagome/reduced_new/"
    mkpath(root_path)

    # 50000 20000 150000

    for M in 50000:20000:150000, seed in 1000:1000:5000
        folder = root_path * "nX_$(nX)/Rb_$(Rb)/trunc_$(trunc)/"
        mkpath(folder)
        file = folder * "M_$(M)_seed_$(seed).jld2"
        E, mags, nematic, two_pt, one_pt, n = bin_observables(nX, Rb, delta, trunc, M, seed)
        @save(file,
              energy=E, mags=mags, nematic=nematic,
              two_pt=two_pt, one_pt=one_pt, n=n)
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
