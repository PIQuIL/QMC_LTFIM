using LinearAlgebra
using ArgParse
using Printf
using JSON
using DataStructures


s = ArgParseSettings()

@add_arg_table! s begin
    "N"
        help = "The length of the chain"
        required = true
        arg_type = Int

    "--hx"
        help = "Strength of the transverse field"
        arg_type = Float64
        default = 1.0

    "--hz"
        help = "Strength of the longitudinal field"
        arg_type = Float64
        default = 0.0

    "--interaction", "-J"
        help = "Strength of the interaction"
        arg_type = Float64
        default = 1.0

    "--periodic", "-p"
        help = "Periodic BCs"
        action = :store_true

    "--thermal"
        help = "Include Thermal State calculations for various beta values"
        action = :store_true

    "--json"
        help = "Output statistics in JSON format"
        action = :store_true
end


parsed_args = parse_args(ARGS, s)


N = parsed_args["N"]

@assert N < 20 "Can't do more than 20 sites!"

J = parsed_args["interaction"] #exchange interaction
hx = parsed_args["hx"] #transverse field
hz = parsed_args["hz"] #longitudinal field
PBC = parsed_args["periodic"]


Dim = 2^N

Hamiltonian = zeros(Dim,Dim)   #This is your 2D Hamiltonian matrix

for Ket = 0:Dim-1  #Loop over Hilbert Space
    Diagonal = 0.
    Spin2 = 0
    for SpinIndex = 0:N-2  #Loop over spin index (base zero, stop one spin before the end of the chain)
        Spin1 = 2*((Ket>>SpinIndex)&1) - 1
        NextIndex = SpinIndex + 1
        Spin2 = 2*((Ket>>NextIndex)&1) - 1
        Diagonal = Diagonal - J*Spin1*Spin2 - hz*Spin1 #spins are +1 and -1
    end
    if PBC
        Spin1 = 2*((Ket>>0)&1) - 1
        Diagonal = Diagonal - J*Spin1*Spin2 - hz*Spin2
    else
        Diagonal = Diagonal - hz*Spin2 #this is the spin at the end of the chain
    end

    Hamiltonian[Ket+1,Ket+1] = Diagonal

    for SpinIndex = 0:N-1
        bit = 2^SpinIndex   #The "label" of the bit to be flipped
        Bra = Ket ⊻ bit    #Binary XOR flips the bit
        Hamiltonian[Bra+1,Ket+1] = -hx
    end
end
Hamiltonian = Hermitian(Hamiltonian);

Diag = eigen(Hamiltonian);

GroundState = Diag.vectors[:, 1];  #this gives the groundstate eigenvector

##### Calculate the groundstate magnetization in the Z direction
magnetization = zeros(Dim)
abs_mag = zeros(Dim)
mag_squared = zeros(Dim)

SumSz = dropdims(sum(@. (2 * (((0:Dim-1) >> (0:N-1)') & 1) - 1); dims=2); dims=2)
AbsSumSz = abs.(SumSz)
SumSzSq = abs2.(SumSz)

magnetization = SumSz' * abs2.(Diag.vectors)
abs_mag = AbsSumSz' * abs2.(Diag.vectors)
mag_squared = SumSzSq' * abs2.(Diag.vectors)


################################## Renyi ######################################
using QuantumInformation


function renyi(ρ::AbstractMatrix, region::AbstractVector{Int}; α=2)
    N = Int(log2(size(ρ, 1)))
    ρ_B = ptrace(ρ, repeat([2], N), collect(region))
    if α > 1
        trace = sum(λ -> λ^α, eigvals(ρ_B))
        return log(trace) / (1 - α)
    elseif α == 1
        trace = sum(eigvals(ρ_B)) do λ
            (iszero(λ) || isone(λ)) ? zero(λ) : -λ * log(λ)
        end
        return trace
    else
        throw(ArgumentError("α must be at least 1!"))
    end
end

###############################################################################

ρ = GroundState * GroundState'
S2 = OrderedDict(["S2_$i" => renyi(ρ, 1:i; α=2) for i in 1:(N-1)])

gs_statistics = OrderedDict{String, Float64}("M" => magnetization[1] / N,
                                             "|M|" => abs_mag[1] / N,
                                             "M^2" => mag_squared[1] / (N*N),
                                             "H" => Diag.values[1] / N,
                                             S2...)

if !parsed_args["json"]
    println("Ground State Properties: \n")
    @printf("⟨M⟩    = % .16f\n", gs_statistics["M"])
    @printf("⟨|M|⟩  = % .16f\n", gs_statistics["|M|"])
    @printf("⟨M^2⟩  = % .16f\n", gs_statistics["M^2"])
    @printf("⟨H⟩    = % .16f\n", gs_statistics["H"])
    println()
    println("-"^70)
end




if parsed_args["thermal"]
    beta_vals = [20,15,10,5,3,2,1,0.8,0.5,0.2]
    th_statistics = OrderedDict{Float64, OrderedDict{String, Float64}}()
    th_statistics[Inf64] = gs_statistics

    if !parsed_args["json"]
        println("Thermal State Properties: \n")
    end
    for β in beta_vals
        weights = exp.(-β * Diag.values)
        Z = sum(weights)
        E = dot(Diag.values, weights) / (N*Z)
        C = (β^2 * ((dot(Diag.values .^2, weights) / Z) - (N*E)^2))

        # magnetization of thermal state
        M = dot(weights, magnetization) / (N*Z)
        M_abs = dot(weights, abs_mag) / (N*Z)
        M2 = dot(weights, mag_squared) / (N*N*Z)

        th_statistics[β] = OrderedDict{String, Float64}("M" => M, "|M|" => M_abs,
                                                        "M^2" => M2,
                                                        "H" => E, "C" => C)

        if !parsed_args["json"]
            @printf("β      = % .16f\n", β)

            @printf("⟨M⟩    = % .16f\n", M)
            @printf("⟨|M|⟩  = % .16f\n", M_abs)
            @printf("⟨M^2⟩  = % .16f\n", M2)
            @printf("⟨H⟩    = % .16f\n", E)
            @printf("C      = % .16f\n", C)
            println()
        end
    end
    if !parsed_args["json"]
        println("-"^70)
    else
        JSON.print(Base.stdout, th_statistics, 4)
    end
else
    if parsed_args["json"]
        JSON.print(Base.stdout, gs_statistics, 4)
    end
end