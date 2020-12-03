using LinearAlgebra

# LATTICE DEFINITIONS

abstract type Lattice end

struct Rectangle <: Lattice
    nX::Int
    nY::Int
    aX::Float64
    aY::Float64
    distance_matrix::Array{Array{Float64,1},1}
end

function Rectangle(nX::Int, nY::Int, aX::Float64, aY::Float64, PBC::Bool)
    N = nX*nY
    distance_matrix = [zeros(Float64, N) for _ in 1:N]

    for i in 1:(N-1)
        x1 = rem(i, nX) > 0 ? rem(i, nX) : nX
        y1 = rem(i, nX) > 0 ? div(i, nX) + 1 : div(i, nX)

        for j in (i+1):N
            x2 = rem(j, nX) > 0 ? rem(j, nX) : nX
            y2 = rem(j, nX) > 0 ? div(j, nX) + 1 : div(j, nX)

            if PBC
                dy = abs(y1 - y2) > 0.5*nY ? 0.5*nY - rem(abs(y1-y2), 0.5*nY) : abs(y1-y2)
                dx = abs(x1 - x2) > 0.5*nX ? 0.5*nX - rem(abs(x1-x2), 0.5*nX) : abs(x1-x2)
            else
                dy = abs(y1 - y2)
                dx = abs(x1 - x2)
            end

            dx *= aX
            dy *= aY

            distance_matrix[i][j] = sqrt(dy^2 + dx^2)

        end
    end
    return Rectangle(nX, nY, aX, aY, distance_matrix)
end


struct Chain <: Lattice
    nX::Int
    aX::Float64
    distance_matrix::Array{Array{Float64,1},1}
end

function Chain(nX::Int, aX::Float64, PBC::Bool)
    distance_matrix = [zeros(Float64, nX) for _ in 1:nX]

    for i in 1:(nX-1)
        x1 = rem(i, nX) > 0 ? rem(i, nX) : nX
        
        for j in (i+1):nX
            x2 = rem(j, nX) > 0 ? rem(j, nX) : nX
            
            if PBC
                dx = abs(x1-x2) > 0.5*nX ? 0.5*nX - rem(abs(x1-x2), 0.5*nX) : abs(x1-x2)
            else
                dx = abs(x1-x2)
            end

            dx *= aX
            distance_matrix[i][j] = dx 
        end
    end
    return Chain(nX, aX, distance_matrix)
end

nspins(lattice::Rectangle) = lattice.nX * lattice.nY
nspins(lattice::Chain) = lattice.nX

################################################################################

function groundstate(H::Array{Float64,2})
    N = convert(Int64, log2(size(H, 1))) 
    Dim = 2^N
    
    Diag = eigen(H)
    gs = Diag.vectors[:,1]
    ρ = reshape(kron(gs, gs), Dim, Dim)
    
    energy = Diag.values[1] / N

    magnetization = 0
    for Ket = 0:Dim-1  
        SumSz = 0.
        for SpinIndex = 0:N-1  
            Spin1 = 2*((Ket>>SpinIndex)&1) - 1
            SumSz += Spin1 
        end
        magnetization += abs(SumSz)*gs[Ket+1]^2  
    end

    return ρ, energy, magnetization / N
end

function mixedstate(H::Array{Float64,2})
    N = convert(Int64, log2(size(H, 1))) 

    expH = exp(-β*H)
    Z = tr(expH)
    ρ = expH / Z

    energy = tr(ρ*H) / N

    sigmaZ1 = [1 0; 0 -1]
    sigmaZN = kron(sigmaZ1, sigmaZ1)
    for i = 3:N
        sigmaZN = kron(sigmaZN, sigmaZ1)
    end

    magnetization = tr(ρ*sigmaZN) / N

    return ρ, energy, magnetization
end

################################################################################

struct Rydberg
    energy::Float64
    magnetization::Float64
end

function Rydberg(lattice::Lattice, δ, Ω, C; β=Nothing)

    N = nspins(lattice)
    V = [zeros(Float64, N) for _ in 1:N]
    
    for i in 1:(N-1)
        for j in (i+1):N
            V[i][j] = C / lattice.distance_matrix[i][j]^2
        end
    end

    Dim = 2^N
    hamiltonian = zeros(Dim, Dim)

    for Ket = 0:Dim-1  
        diag = 0.
        # bond term
        for i in 0:N-2
            for j in (i+1):N-1
                # NOT the following since the Vij term only has the |1> projector.
                # So, just keep in 0,1 basis so that |0> doesn't contribute
                # Spin = 2*((Ket>>SpinIndex)&1) - 1 << NOT THIS
                Spin1 = (Ket>>i)&1
                Spin2 = (Ket>>j)&1
                diag += V[i+1][j+1]*Spin1*Spin2 
            end
        end

        # long. field term
        for i = 0:N-1
            # NOT the following since the δ term only has the |1> projector.
            # So, just keep in 0,1 basis so that |0> doesn't contribute
            # Spin = 2*((Ket>>SpinIndex)&1) - 1 << NOT THIS
            Spin = (Ket>>i)&1
            diag -= δ*Spin
        end
    
        hamiltonian[Ket+1,Ket+1] = diag
   
        # off-diagonal term
        for SpinIndex = 0:N-1
            bit = 2^SpinIndex   
            Bra = Ket ⊻ bit    
            hamiltonian[Bra+1,Ket+1] = Ω
        end
    end

    if β == Nothing
        ρ, energy, magnetization = groundstate(hamiltonian)
    else
        ρ, energy, magnetization = mixedstate(hamiltonian)
    end

    return Rydberg(energy, magnetization) 

end

################################################################################

# Change these parameters below to do different things
# too lazy to do an argparse script >:)

nX = 3
nY = 3
aX = 1.
aY = 1.
PBC = true

lattice = Rectangle(nX, nY, aX, aY, PBC)
δ = 1.
C = 1.
Ω = 1.
β = Nothing

ryd = Rydberg(lattice, δ, Ω, C, β=β) 

if β==Nothing
    println("Ground state: ")
else
    println("Thermal state: ")
end

@show ryd.energy
@show ryd.magnetization
