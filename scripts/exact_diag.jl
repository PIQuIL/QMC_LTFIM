using LinearAlgebra
using ArgParse
using Printf


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

println("Ground State Properties: \n")
@printf("⟨M⟩    = % .16f\n", (magnetization[1] / N))
@printf("⟨|M|⟩  = % .16f\n", (abs_mag[1] / N))
@printf("⟨M^2⟩  = % .16f\n", (mag_squared[1] / (N*N)))
@printf("⟨H⟩    = % .16f\n", Diag.values[1] / N)
println()
println("-"^70)

if parsed_args["thermal"]
    beta_vals = [20,15,10,5,3,2,1,0.8,0.5,0.2]

    println("Thermal State Properties: \n")
    for β in beta_vals
        weights = exp.(-β * Diag.values)
        Z = sum(weights)
        E = dot(Diag.values, weights) / (N*Z)
        C = (β^2 * ((dot(Diag.values .^2, weights) / Z) - (N*E)^2))

        # magnetization of thermal state
        M = dot(weights, magnetization) / (N*Z)
        M_abs = dot(weights, abs_mag) / (N*Z)
        M2 = dot(weights, mag_squared) / (N*N*Z)

        @printf("β      = % .16f\n", β)

        @printf("⟨M⟩    = % .16f\n", M)
        @printf("⟨|M|⟩  = % .16f\n", M_abs)
        @printf("⟨M^2⟩  = % .16f\n", M2)
        @printf("⟨H⟩    = % .16f\n", E)
        @printf("C      = % .16f\n", C)
        println()
    end
    println("-"^70)
end