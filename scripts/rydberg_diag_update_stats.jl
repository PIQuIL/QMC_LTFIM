using ArgParse
using CSV
using DataFrames
using DelimitedFiles
using Distributions
using Format
using JSON
using LambertW
using LinearAlgebra
using Printf
using QMC
using Random
using RandomNumbers
using Serialization
using StatsBase

###############################################################################

@inline coordinates(lat::Rectangle) = divrem.(0:(nspins(lat)-1), lat.n1)
@inline smag_signs(lat::Rectangle, coords::Vector{NTuple{2, Int}}) = [(-1)^(i+j) for (i,j) in coords]
@inline smag_signs(lat::Rectangle) = smag_signs(lat, coordinates(lat))

staggered_magnetization(Ns::Int, signs::Vector{Int}, spin_props::AbstractVecOrMat{Bool}) =
    ((@. 2*spin_props - 1)' * signs) / Ns
staggered_magnetization(lat::Rectangle, signs::Vector{Int}, spin_props) =
    staggered_magnetization(nspins(lat), signs, spin_props)
staggered_magnetization(lat::Rectangle, coords::Vector{NTuple{2, Int}}, spin_props) =
    staggered_magnetization(lat, smag_signs(lat, coords), spin_props)
staggered_magnetization(lat::Rectangle, spin_props) =
    staggered_magnetization(lat, smag_signs(lat), spin_props)



@inline init_op_list_length(beta, H) = round(Int, (2 * beta) * QMC.diag_update_normalization(H), RoundUp)

function init_mc_cli(parsed_args)
    Ω = parsed_args["omega"]
    δ = parsed_args["delta"]
    R_b = parsed_args["radius"]
    seed = parsed_args["seed"]
    beta = parsed_args["beta"]

    L = parsed_args["L"]
    save_path = parsed_args["path"]

    # MC parameters
    MCS = parsed_args["measurements"] # the number of samples to record per batch
    batches = parsed_args["batches"]

    mc_opts = (beta, seed, MCS, batches)

    path = joinpath(
        save_path,
        "L=$L",
        "Rb=$(@sprintf("%.2f", R_b))",
        "omega=$(@sprintf("%.2f", Ω))",
        "delta=$(@sprintf("%.2f", δ))",
        "beta=$beta", 
        "seed=$seed",
        parsed_args["threestep"] ? "3step" : "2step",
        "M_init=$(parsed_args["M-init"])",
        "M_setting=$(parsed_args["M-setting"])",
        "max_replace=$(parsed_args["max-replace"])"
    )
    mkpath(path)
    mkpath(joinpath(path, "observables"))
    mkpath(joinpath(path, "measurements"))

    res = continue_simulation(path, parsed_args)
    if res === nothing
        H = Rydberg((L, L), R_b, Ω, δ; pbc = (true, true))
        lat = H.lattice

        if parsed_args["M-init"] == 0
            M = init_op_list_length(beta, H)
        elseif parsed_args["M-init"] == -1
            M = init_op_list_length(beta/2, H)
        else
            M = parsed_args["M-init"]
        end
        qmc_state = BinaryThermalState(H, M)
        
        rng = Xorshifts.Xoroshiro128Plus(seed)
        rand!(rng, qmc_state.left_config)
        copyto!(qmc_state.right_config, qmc_state.left_config)

        writedlm(joinpath(path, "V_ij.csv"), Matrix(H.V))
        writedlm(joinpath(path, "energy_shift.csv"), H.energy_shift)
        writedlm(joinpath(path, "coordinates.csv"), coordinates(lat))

        starting_batch = 0

        diagnostics = Diagnostics(RunStatsHistogram(1000))
    else
        H, lat, qmc_state, rng, diagnostics, starting_batch = res
    end

    println("Running Rydberg(Square($L, $L, PBC), R_b=$R_b, Ω=$Ω, δ=$δ) at β=$beta, starting at batch $(starting_batch)...")

    return H, lat, qmc_state, path, mc_opts, rng, diagnostics, starting_batch
end

function continue_simulation(path, parsed_args)
    checkpoints = filter(readdir(path)) do s
        endswith(s, "state.bin") && contains(s, "batch")
    end
    isempty(checkpoints) && return nothing

    batches = map(checkpoints) do s
        split(split(s, "batch_")[2], '_')[1]
    end
    batches = sort(batches, by=x->parse(Int, x), rev=true)

    for s in batches
        try
            qmc_state_file = joinpath(path, "batch_$(s)_state.bin")
            state = deserialize(qmc_state_file)
            starting_batch = parse(Int, s) + 1

            rng::Xorshifts.Xoroshiro128Plus = state[:rng]
            qmc_state::BinaryThermalState = state[:qmc_state]
            H::Rydberg = state[:hamiltonian]
            diagnostics = state[:diagnostics]
            lat = state[:lattice]

            return H, lat, qmc_state, rng, diagnostics, starting_batch
        catch e
            println(e)
            println(catch_backtrace())
        end
    end

    return nothing
end

function max_poisson_1(λ, n)
    logn = log(n)
    x0 = logn/lambertw(logn/(ℯ * λ))
    numer = log(λ) - λ - (log(2π)/2) - (3/2)log(x0)
    denom = log(x0) - log(λ)
    return x0 + (numer/denom)
end


function get_M_info(eq_ns::Vector{Int}, eq_diagnostics::Vector{Diagnostics{R, M}}, beta::Float64, norm_const::Float64; frac::Float64=0.1) where {R <: AbstractRunStats, M}
    info_dict = Dict{String, Float64}()
    info_dict["frac"] = frac
    info_dict["beta"] = beta
    info_dict["norm_const"] = norm_const

    # use only the end of the eq data
    end_ns = @views eq_ns[round(Int, (1.0 - frac)*length(eq_ns)):end]
    info_dict["mean_ns"] = mean_ns = mean(end_ns)
    info_dict["mean_ns_full"] = mean_ns_full = mean(eq_ns)
    info_dict["var_ns"] = var_ns = var(end_ns)
    info_dict["var_ns_full"] = var_ns_full = var(eq_ns)

    # drop first few samples, but otherwise keep most of it
    info_dict["max_ns"] = max_ns = maximum(@views eq_ns[round(Int, frac*length(eq_ns)):end])
    info_dict["max_ns_full"] = max_ns_full = maximum(eq_ns)

    start = round(Int, (1.0 - frac)*length(eq_diagnostics))
    avg_mat_elem = deepcopy(eq_diagnostics[start].runstats.diagonal_update.matelem_insertion.prob)
    for i in (start+1):length(eq_diagnostics)
        merge!(avg_mat_elem, eq_diagnostics[i].runstats.diagonal_update.matelem_insertion.prob)
    end
    info_dict["M_opt for 2step"] = beta*norm_const*mean(avg_mat_elem) + mean_ns
    info_dict["M_opt for 3step"] = beta*norm_const + mean_ns

    for p in ("1e-6", "1e-9", "1e-10", "1e-12", "1e-15", "1e-18", "1e-20")
        info_dict["Poisson(λ=mean_ns) cquantile $p"] = cquantile(Poisson(mean_ns), parse(Float64, p))
        info_dict["Poisson(λ=mean_ns_full) cquantile $p"] = cquantile(Poisson(mean_ns_full), parse(Float64, p))
        info_dict["Poisson(λ=var_ns) cquantile $p"] = cquantile(Poisson(var_ns), parse(Float64, p))
        info_dict["Poisson(λ=var_ns_full) cquantile $p"] = cquantile(Poisson(var_ns_full), parse(Float64, p))
        
        info_dict["Normal(μ=mean_ns, σ²=var_ns) cquantile $p"] = 
            cquantile(Normal(mean_ns, sqrt(var_ns)), parse(Float64, p))
        info_dict["Normal(μ=mean_ns, σ²=mean_ns) cquantile $p"] = 
            cquantile(Normal(mean_ns, sqrt(mean_ns)), parse(Float64, p))
        info_dict["Normal(μ=mean_ns_full, σ²=var_ns_full) cquantile $p"] = 
            cquantile(Normal(mean_ns_full, sqrt(var_ns_full)), parse(Float64, p))
        info_dict["Normal(μ=mean_ns_full, σ²=mean_ns_full) cquantile $p"] = 
            cquantile(Normal(mean_ns_full, sqrt(mean_ns_full)), parse(Float64, p))
    end

    MCS = length(eq_ns)
    max_poisson_from_mean = round(Int, max_poisson_1(mean_ns, MCS), RoundUp)
    info_dict["MaxPoisson(λ=mean_ns)"] = max_poisson_from_mean
    info_dict["Poisson(λ=mean_ns) log_10-prob of anything larger than MaxPoisson(λ=mean_ns)"] = 
        (logccdf(Poisson(mean_ns), max_poisson_from_mean)/log(10))
    info_dict["Poisson(λ=mean_ns) log_10-prob of anything larger than max_ns"] = logccdf(Poisson(mean_ns), max_ns)/log(10)
    
    max_poisson_from_mean = round(Int, max_poisson_1(mean_ns_full, MCS), RoundUp)
    info_dict["MaxPoisson(λ=mean_ns_full)"] = max_poisson_from_mean
    info_dict["Poisson(λ=mean_ns_full) log_10-prob of anything larger than MaxPoisson(λ=mean_ns_full)"] = 
        (logccdf(Poisson(mean_ns_full), max_poisson_from_mean)/log(10))
    info_dict["Poisson(λ=mean_ns_full) log_10-prob of anything larger than max_ns_full"] = logccdf(Poisson(mean_ns_full), max_ns_full)/log(10)

    max_poisson_from_var = round(Int, max_poisson_1(var_ns, MCS), RoundUp)
    info_dict["MaxPoisson(λ=var_ns)"] = max_poisson_from_var
    info_dict["Poisson(λ=var_ns) log_10-prob of anything larger than MaxPoisson(λ=var_ns)"] = 
        (logccdf(Poisson(var_ns), max_poisson_from_var)/log(10))
    info_dict["Poisson(λ=var_ns) log_10-prob of anything larger than max_ns"] = logccdf(Poisson(var_ns), max_ns)/log(10)
    
    max_poisson_from_var = round(Int, max_poisson_1(var_ns_full, MCS), RoundUp)
    info_dict["MaxPoisson(λ=var_ns_full)"] = max_poisson_from_var
    info_dict["Poisson(λ=var_ns_full) log_10-prob of anything larger than MaxPoisson(λ=var_ns_full)"] = 
        (logccdf(Poisson(var_ns_full), max_poisson_from_var)/log(10))
    info_dict["Poisson(λ=var_ns_full) log_10-prob of anything larger than max_ns_full"] = logccdf(Poisson(var_ns_full), max_ns_full)/log(10)

    return info_dict
end


function get_M_final(info_dict::Dict{String, Float64}, M_setting::Float64)
    if iszero(M_setting)
        return info_dict["M_opt for 3step"]
    elseif isone(-M_setting)
        return info_dict["M_opt for 2step"]
    elseif M_setting > 0
        return round(Int, M_setting*info_dict["max_ns"], RoundUp)
    end
    @assert M_setting >= 0.0 || M_setting == -1.0
end


function thermalstate(parsed_args)
    H, lat, qmc_state, path, mc_opts, rng, diagnostics, starting_batch =
        init_mc_cli(parsed_args)

    beta, seed, MCS, batches = mc_opts

    l = floor(Int, log10(batches) + 1)
    nsteps = MCS
    measurements_trace = zeros(Bool, nspins(H), nsteps)
    measurements_slice = zeros(Bool, nspins(H), nsteps)
    observables = DataFrame(
        batch = zeros(Int, nsteps),
        beta = zeros(Float64, nsteps),
        n_sso = zeros(Int, nsteps),
        n_ops = zeros(Int, nsteps),
        M = zeros(Int, nsteps),
    )

    MAX_ITER = parsed_args["max-replace"]
    threestep = parsed_args["threestep"]

    norm_const = QMC.diag_update_normalization(H)
    coords = coordinates(lat)
    signs = smag_signs(lat, coords)

    formatter = generate_formatter("%.$(l)i")

    for b in starting_batch:batches
        eq = (b == 0)
        if eq
            eq_diagnostics = [Diagnostics(RunStats()) for _ in 1:nsteps]
            combined_eq_runstats = NoStats()
        end

        for i in 1:nsteps  # Monte Carlo Production Steps
            d = eq ? eq_diagnostics[i] : diagnostics
            
            observables[i, :n_ops] = mc_step_beta!(rng, qmc_state, H, beta, d; threestep=threestep, eq = eq, p=0.0, max_iter=MAX_ITER) do _, qmc_state, H
                measurements_slice[:, i] = QMC.sample(H, qmc_state, rand(rng, 1:length(qmc_state.operator_list)))
                measurements_trace[:, i] = qmc_state.left_config

                observables[i, :n_sso] = num_single_site_offdiag(H, qmc_state.operator_list)
                observables[i, :batch] = b
                observables[i, :beta] = beta
            end

            observables[i, :M] = length(qmc_state.operator_list)
            if eq
                merge!(combined_eq_runstats, eq_diagnostics[i].runstats)
            end
        end

        bs = formatter(b)
        if eq
            info_dict = get_M_info(observables[:, :n_ops], eq_diagnostics, beta, norm_const; frac=0.1)
            M = get_M_final(info_dict, parsed_args["M-setting"])
            info_dict = merge(Dict{String, Union{Bool, Int, Float64, String}}(), info_dict, parsed_args)
            info_dict["M_final_attempted"] = M
            info_dict["M_final"] = resize_op_list!(qmc_state, H, M)

            open(joinpath(path, "run_info.json"), "w") do io
                JSON.print(io, info_dict)
            end
            
            serialize(joinpath(path, "eq_diagnostics.bin"),
                      (all_steps=eq_diagnostics,
                       combined=combined_eq_runstats))
        end

        for (t, measurements) in ((:slice, measurements_slice), (:trace, measurements_trace))
            observables[:, Symbol(:staggered_magnetization_, t)] = staggered_magnetization(nspins(H), signs, measurements)
            observables[:, Symbol(:occupation_, t)] = dropdims(sum(measurements, dims=1), dims=1)/nspins(H)

            write(joinpath(path, "measurements", "batch_$(bs)_$(t)_samples.bin"), 
                  BitMatrix(measurements))
        end
        CSV.write(joinpath(path, "observables", "batch_$(bs)_raw_observables.csv"), 
                  observables)

        # save MC state
        #   This *must* be the last thing we save in the batch in case the job gets killed while
        #   saving. For example, if the job gets killed while saving measurements for batch 10
        #   but the state file for batch 10 is already saved, when we restart the job, the QMC
        #   will start at batch 11, even though we've lost some of our data for batch 10.
        qmc_state_file = joinpath(path, "batch_$(bs)_state.bin")
        serialize(qmc_state_file,
                  (rng=rng,
                   qmc_state=qmc_state,
                   hamiltonian=H,
                   diagnostics=diagnostics))

        # delete the previous 5 saved states, if they exist
        #  (trying the last 5 bc sometimes the deletion fails if the job times out)
        for i in 1:5
            bs = formatter(b-i)
            old_qmc_state = joinpath(path, "batch_$(bs)_state.bin")
            if isfile(old_qmc_state)
                rm(old_qmc_state)
            end
        end
    end
end


###############################################################################


s = ArgParseSettings()


@add_arg_table! s begin
    "thermal"
        help = "Use Thermal SSE"
        action = :command
end


@add_arg_table! s["thermal"] begin
    "L"
        help = "Dimensions of square lattice along the X or Y direction."
        required = true
        arg_type = Int
    "path"
        help = "Root path to save data to"
        required = true
        arg_type = String

    "--omega"
        help = "Strength of the transverse field"
        arg_type = Float64
        default = 1.0
    "--delta"
        help = "Strength of the detuning"
        arg_type = Float64
        default = 1.0
    "--radius", "-R"
        help = "Rydberg blockade radius (in units of the lattice spacing). Controls the strength of the interaction."
        arg_type = Float64
        default = 1.2

    "--beta"
        help = "Inverse temperature of the thermal state."
        arg_type = Float64
        default = 20.0
        
    "--max-replace"
        help = "Maximum number of iterations for diagonal replacement move."
        arg_type = Int
        default = 0

    "--threestep"
        help = "Whether to use the Three-Step Diagonal Update Scheme."
        action = :store_true

    "--measurements", "-n"
        help = "Number of samples to record per batch"
        arg_type = Int
        default = 100_000

    "--batches", "-b"
        help = "Number of batches to run"
        arg_type = Int
        default = 5

    "--M-init"
        help = "Initial operator list length. If 0 uses 2*beta*diag_update_normalization, if -1 uses beta*diag_update_normalization."
        arg_type = Int
        default = 0

    "--M-setting"
        help = """How to set the operator list length after equilibration has finished. 
        If positive, M propto n_max, with this option setting the scaling factor.
        If 0, uses M_opt for the threestep algorithm, -1 uses M_opt for the twostep algorithm.
        """
        arg_type = Float64
        default = 0

    "--seed"
        help = "Random seed"
        arg_type = Int
        default = 1234
end


parsed_args = parse_args(ARGS, s)

@time thermalstate(parsed_args["thermal"])
