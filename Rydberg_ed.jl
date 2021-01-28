using LinearAlgebra

# LATTICE DEFINITIONS

abstract type Lattice end

struct Rectangle <: Lattice
    nX::Int
    nY::Int
    aX::Float64
    aY::Float64
    distance_matrix::Array{Float64, 2}
end

function Rectangle(nX::Int, nY::Int, aX::Float64, aY::Float64, PBC::NTuple{2, Bool})
    N = nX*nY
    distance_matrix = zeros(Float64, N, N)

    for i in 1:(N-1)
        x1 = rem(i, nX) > 0 ? rem(i, nX) : nX
        y1 = rem(i, nX) > 0 ? div(i, nX) + 1 : div(i, nX)

        for j in (i+1):N
            x2 = rem(j, nX) > 0 ? rem(j, nX) : nX
            y2 = rem(j, nX) > 0 ? div(j, nX) + 1 : div(j, nX)

            if PBC[1]
                dx = abs(x1 - x2) > 0.5*nX ? 0.5*nX - rem(abs(x1-x2), 0.5*nX) : abs(x1-x2)
            else
                dx = abs(x1 - x2)
            end

            if PBC[2]
                dy = abs(y1 - y2) > 0.5*nY ? 0.5*nY - rem(abs(y1-y2), 0.5*nY) : abs(y1-y2)
            else
                dy = abs(y1 - y2)
            end

            dx *= aX
            dy *= aY

            distance_matrix[i, j] = sqrt(dy^2 + dx^2)

        end
    end
    return Rectangle(nX, nY, aX, aY, distance_matrix)
end
Rectangle(nX, nY, aX, aY, PBC::Bool) = Rectangle(nX, nY, aX, aY, (PBC, PBC))


struct Chain <: Lattice
    nX::Int
    aX::Float64
    distance_matrix::Array{Float64, 2}
end

function Chain(nX::Int, aX::Float64, PBC::Bool)
    distance_matrix = zeros(Float64, nX, nX)

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
            distance_matrix[i, j] = dx
        end
    end
    return Chain(nX, aX, distance_matrix)
end

nspins(lattice::Rectangle) = lattice.nX * lattice.nY
nspins(lattice::Chain) = lattice.nX

################################################################################

function groundstate(H::Array{Float64,2})
    N = convert(Int64, log2(size(H, 1)))
    Dim = size(H, 1)

    Diag = eigen(H)
    gs = Diag.vectors[:,1]
    ρ = gs * gs'

    energy = Diag.values[1] / N

    magnetization = zeros(Dim)
    abs_mag = zeros(Dim)
    mag_squared = zeros(Dim)

    SumSz = dropdims(sum(@. (2 * (((0:Dim-1) >> (0:N-1)') & 1) - 1); dims=2); dims=2)
    AbsSumSz = abs.(SumSz)
    SumSzSq = abs2.(SumSz)

    magnetization = SumSz' * abs2.(Diag.vectors)
    abs_mag = AbsSumSz' * abs2.(Diag.vectors)
    mag_squared = SumSzSq' * abs2.(Diag.vectors)
    M4 = abs2.(SumSzSq)' * abs2.(Diag.vectors)

    U = (3 - M4[1] / (mag_squared[1]^2)) / 2
    @show U

    return ρ, energy, magnetization[1] / N, abs_mag[1] / N, mag_squared[1] / (N*N)
end

function mixedstate(H::Array{Float64,2})
    N = convert(Int64, log2(size(H, 1)))
    Dim = size(H, 1)

    Diag = eigen(H)

    magnetization = zeros(Dim)
    abs_mag = zeros(Dim)
    mag_squared = zeros(Dim)

    SumSz = dropdims(sum(@. (2 * (((0:Dim-1) >> (0:N-1)') & 1) - 1); dims=2); dims=2)
    AbsSumSz = abs.(SumSz)
    SumSzSq = abs2.(SumSz)

    magnetization = SumSz' * abs2.(Diag.vectors)
    abs_mag = AbsSumSz' * abs2.(Diag.vectors)
    mag_squared = SumSzSq' * abs2.(Diag.vectors)

    ρ = exp(-β * H)

    weights = exp.(-β * Diag.values)
    Z = sum(weights)
    ρ /= Z

    E = dot(Diag.values, weights) / (N*Z)

    # magnetization of thermal state
    M = dot(weights, magnetization) / (N*Z)
    M_abs = dot(weights, abs_mag) / (N*Z)
    M2 = dot(weights, mag_squared) / (N*N*Z)

    return ρ, E, M, M_abs, M2
end

################################################################################

struct Rydberg
    energy::Float64
    magnetization::Float64
end

function Rydberg(lattice::Lattice, δ, Ω, R_b; β=nothing)
    N = nspins(lattice)
    V = zeros(Float64, N, N)

    for i in 1:(N-1)
        for j in (i+1):N
            V[i, j] = Ω * (R_b / lattice.distance_matrix[i, j])^6
        end
    end

    Dim = 2^N
    hamiltonian = zeros(Dim, Dim)

    for Ket = 0:Dim-1
        diag = 0.
        # bond term
        for i in 0:(N-2)
            for j in (i+1):(N-1)
                # since the Vij term only has the |1> projector.
                # we just stay with 0,1 so that |0> doesn't contribute
                # Spin = 2*((Ket>>SpinIndex)&1) - 1 << NOT THIS
                Spin1 = ((Ket>>i)&1)
                Spin2 = ((Ket>>j)&1)
                diag += V[i+1, j+1]*Spin1*Spin2
            end
        end

        # long. field term
        for i = 0:N-1
            # since the δ term only has the |1> projector.
            # we just stay with 0,1 so that |0> doesn't contribute
            # Spin = 2*((Ket>>SpinIndex)&1) - 1 << NOT THIS
            Spin = ((Ket>>i)&1)
            diag -= δ*Spin
        end

        hamiltonian[Ket+1,Ket+1] = diag

        # off-diagonal term
        for SpinIndex = 0:N-1
            bit = 2^SpinIndex
            Bra = Ket ⊻ bit
            hamiltonian[Bra+1,Ket+1] = -Ω/2
        end
    end

    if β === nothing
        res = groundstate(hamiltonian)
    else
        res = mixedstate(hamiltonian)
    end

    @show res[2:end]
    energy, magnetization = res[2], res[3]
    return Rydberg(energy, magnetization)

end

################################################################################

# Change these parameters below to do different things
# too lazy to do an argparse script >:)

nX = 4
nY = 3
aX = 1.
aY = 1.
PBC = false


struct NearestNeighborChain <: Lattice
    nX::Int
    aX::Float64
    distance_matrix::Array{Float64, 2}
end

function NearestNeighborChain(nX::Int, aX::Float64)
    distance_matrix = zeros(Float64, nX, nX)

    for i in 1:nX
        j = i + 1
        j > nX && (j = 1)

        i, j = minmax(i, j)
        distance_matrix[i, j] = aX
    end
    return NearestNeighborChain(nX, aX, distance_matrix)
end

nspins(lattice::NearestNeighborChain) = lattice.nX


lattice = Rectangle(nX, nY, aX, aY, (false, true))
# lattice = Chain(nX, aX, PBC)
# lattice = NearestNeighborChain(4, 1.0)
δ = 1.5
Ω = 1.

R_b = 1.2

β = nothing

ryd = Rydberg(lattice, δ, Ω, R_b, β=β)

if β === nothing
    println("Ground state: ")
else
    println("Thermal state: ")
end

@show ryd.energy
@show ryd.magnetization
