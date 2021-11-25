using DrWatson
@quickactivate "QMC"

using QMC


using Random
using RandomNumbers

using DelimitedFiles
using JLD2
using Printf

using ArgParse


###############################################################################

function init_mc_cli(parsed_args)
    PBC = parsed_args["periodic"]
    Ω = parsed_args["omega"]
    δ = parsed_args["delta"]
    R_b = parsed_args["radius"]
    mb_prob = parsed_args["mb-prob"]

    Dim = length(parsed_args["dims"])
    # NOTE: nX saved as list: ([nX]), so output file name doesn't have nX
    # parsed_args["dims"][1] fixes it but needs a better solution for >1D
    nX = tuple(parsed_args["dims"]...)

    BC_name = PBC ? "PBC" : "OBC"

    # MC parameters
    M = parsed_args["M"] # length of the operator_list is 2M
    MCS = parsed_args["measurements"] # the number of samples to record per batch
    batches = parsed_args["batches"]
    EQ_MCS = div(batches*MCS, 10)

    if Dim == 1
        size = "$(nX[1])"
    elseif Dim == 2
        size = "$(nX[1]),$(nX[2])"
    end

    println("Running Rydberg(R_b=$R_b, Ω=$Ω, δ=$δ)")
    H = Rydberg(nX, R_b, Ω, δ; pbc = PBC)
    d = @ntuple Dim size BC_name R_b Ω δ M mb_prob

    mc_opts = @ntuple M batches MCS EQ_MCS mb_prob

    qmc_state = BinaryGroundState(H, M)

    rng = Xorshifts.Xoroshiro128Plus(parsed_args["seed"])
    rand!(rng, qmc_state.left_config)
    copyto!(qmc_state.right_config, qmc_state.left_config)

    return H, qmc_state, savename(d; digits = 4), mc_opts, rng
end



function groundstate(parsed_args)
    H, qmc_state, sname, mc_opts, rng = init_mc_cli(parsed_args)

    M, batches, MCS, EQ_MCS, mb_prob = mc_opts

    mkpath(datadir("sims", "groundstate"))
    path = datadir("sims", "groundstate", sname)

    ns = 0.0
    occupations = similar(qmc_state.propagated_config, Float64)

    for _ in 1:EQ_MCS  # equilibration step
        mc_step!(rng, qmc_state, H; p=mb_prob)
    end

    l = floor(Int, log10(batches) + 1)

    for b in 1:batches
        ns = 0.0
        fill!(occupations, 0.0)

        for i in 1:MCS
            mc_step!(rng, qmc_state, H; p=mb_prob) do _, qmc_state, H
                spin_prop = sample(H, qmc_state)
                occupations .+= spin_prop
                ns += num_single_site_diag(H, qmc_state.operator_list)
            end
        end

        occupations ./= MCS
        ns /= MCS

        # save batch
        qmc_state_file = path * "_batch_$(lpad(b, l, "0"))_state.jld2"
        @save(qmc_state_file,
              rng=rng,
              qmc_state=qmc_state,
              hamiltonian=H)

        # delete the previous saved state, if it exists
        old_qmc_state = path * "_batch_$(lpad(b-1, l, "0"))_state.jld2"
        if isfile(old_qmc_state)
            rm(old_qmc_state)
        end

        data_file = path * "_occupations.csv"
        open(data_file, (b > 1) ? "a" : "w") do io
            writedlm(io, occupations', ",")
        end

        data_file = path * "_energy.csv"
        open(data_file, (b > 1) ? "a" : "w") do io
            writedlm(io, ns, " ")
        end
    end
end


###############################################################################


s = ArgParseSettings()


@add_arg_table! s begin
    "groundstate"
        help = "Use Power-Iteration SSE to simulate the ground state"
        action = :command
end


@add_arg_table! s["groundstate"] begin
    "dims"
        help = "The dimensions of the square lattice"
        required = true
        arg_type = Int
        nargs = '+'
    "--periodic", "-p"
        help = "Periodic BCs"
        action = :store_true
    "--omega"
        help = "Strength of the transverse field"
        arg_type = Float64
        default = 1.0
    "--delta"
        help = "Strength of the longitudinal field"
        arg_type = Float64
        default = 1.0
    "--radius", "-R"
        help = "Rydberg blockade radius (in units of the lattice spacing). Controls the strength of the interaction."
        arg_type = Float64
        default = 1.0

    "-M"
        help = "Half-size of the operator list"
        arg_type = Int
        default = 1000

    "--mb-prob"
        help = "Probability of performing a multibranch update"
        arg_type = Float64
        default = 0.0

    "--measurements", "-n"
        help = "Number of MC steps to perform per batch"
        arg_type = Int
        default = 100
    "--batches", "-b"
        help = "Number of batches to generate"
        arg_type = Int
        default = 1000

    "--seed"
        help = "Random seed"
        arg_type = Int
        default = 1234
end


parsed_args = parse_args(ARGS, s)
@time groundstate(parsed_args["groundstate"])
