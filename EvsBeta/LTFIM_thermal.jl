using LinearAlgebra

struct LTFIM_1D
    N::Int
    hx::Float64
    hz::Float64
    β::Float64
    Hamiltonian::Array{Float64,2}
    Energy::Float64
    Magnetization::Float64
end

function H(N::Int, hx::Float64, hz::Float64; PBC=true)
    
    Dim = 2^N
    H_ = zeros(Dim,Dim)   #This is your 2D Hamiltonian matrix
    J = 1.0 # hard-coded

    for Ket = 0:Dim-1  #Loop over Hilbert Space
        Diagonal = 0.
        # bond term
        for SpinIndex = 0:N-2  #Loop over spin index (base zero, stop one spin before the end of the chain)
            Spin1 = 2*((Ket>>SpinIndex)&1) - 1
            NextIndex = SpinIndex + 1
            Spin2 = 2*((Ket>>NextIndex)&1) - 1
            Diagonal = Diagonal - J*Spin1*Spin2 #spins are +1 and -1
        end

        # PBC term
        if PBC
            Spin1 = 2*((Ket>>0)&1) - 1
            SpinN = 2*((Ket>>(N-1))&1) - 1
            Diagonal = Diagonal - J*Spin1*SpinN #spins are +1 and -1
        end
    
        # long. field term
        for SpinIndex = 0:N-1
            Spin = 2*((Ket>>SpinIndex)&1) - 1
            Diagonal = Diagonal - hz*Spin
        end
    
        H_[Ket+1,Ket+1] = Diagonal
    
        for SpinIndex = 0:N-1
            bit = 2^SpinIndex   #The "label" of the bit to be flipped
            Bra = Ket ⊻ bit    #Binary XOR flips the bit
            H_[Bra+1,Ket+1] = -hx
        end
    
    end

    return H_
end

function ThermalObservables(β::Float64, H::Array{Float64,2})
    
    N = convert(Int64, log2(size(H, 1))) 

    expH = exp(-β*H)
    Z = tr(expH)
    ρ = expH / Z

    ThermalEnergy = tr(ρ*H) / N

    sigmaZ1 = [1 0; 0 -1]
    sigmaZN = kron(sigmaZ1, sigmaZ1)
    for i = 3:N
        sigmaZN = kron(sigmaZN, sigmaZ1)
    end

    ThermalMag = tr(ρ*sigmaZN) / N

    return ThermalEnergy, ThermalMag

end


function LTFIM_1D(N::Int, hx::Float64, hz::Float64, β::Float64)

    # TODO: figure out how to make PBC an option with default=False

    Hamiltonian = H(N, hx, hz, PBC=true)
    ThermalEnergy, ThermalMagnetization = ThermalObservables(β, Hamiltonian)

    return LTFIM_1D(
        N, 
        hx, 
        hz, 
        β, 
        Hamiltonian, 
        ThermalEnergy, 
        ThermalMagnetization
    )
end
