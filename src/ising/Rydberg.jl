using Base.Iterators

###############################################################################

#=
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


struct Kagome <: Lattice
    # parameter that defines the equilateral triangle side length
    t::Float64

    # translation vectors
    a1::Array{Float64,1}
    a2::Array{Float64,1}

    # number of repititions in directions of a1 and a2
    n1::Int # a1
    n2::Int # a2

    # coordinates of sites inside a unit cell
    r::Array{Array{Float64,1},1}

    PBC::NTuple{2, Bool}

    distance_matrix::Array{Float64, 2}
end

function Kagome(t::Float64, n1::Int, n2::Int, PBC::NTuple{2, Bool}; trunc::Float64 = Inf)
    a = 2. * t
    a1 = [a, 0.]
    a2 = [a*0.5, a*sqrt(3)*0.5]

    # coordinates of each site in the unit cell
    r1 = [0., 0.]
    r2 = 0.5 * a2
    r3 = 0.5 * a1
    r = [r1, r2, r3]

    distance_matrix = dist_matrix(a1, a2, n1, n2, r, PBC; trunc=trunc)

    return Kagome(t, a1, a2, n1, n2, r, PBC, distance_matrix)
end


function dist_matrix(a1::Array{Float64,1}, a2::Array{Float64,1}, n1::Int, n2::Int, r::Array{Array{Float64,1},1}, PBC::NTuple{2, Bool}; trunc::Float64 = Inf)
    # this is currently only for non-trivial lattices (i.e. not Rectangle or Chain)
    # the plotting flag is strictly for saving a picture of the lattice on a graph
    PBC1, PBC2 = PBC[1], PBC[2]

    a2_length = sqrt(a2[1]^2 + a2[2]^2) # length of a2 vector
    θ = acos(a2[1] / a2_length) # a2 angle from horizontal

    N = n1*n2*length(r) # total number of sites
    dij = zeros(Float64, N, N) # distance matrix
    num_cells = n1*n2 # number of repitions of the unit cell

    a2_length = sqrt(a2[1]^2 + a2[2]^2) # length of a2 vector
    θ = acos(a2[1] / a2_length) # a2 angle from horizontal

    # NOT ENDING ON num_cells-1 BECAUSE WE NEED INTER-CELL BONDS
    for i in 1:num_cells
        cellp1_x = rem(i, n1) > 0 ? rem(i, n1) - 1 : n1 - 1
        cellp1_y = rem(i, n1) > 0 ? div(i, n1) : div(i, n1) - 1

        cell1_x = cellp1_x*a1[1] + cellp1_y*a2_length*cos(θ)
        cell1_y = cellp1_y*a2_length*sin(θ)
        cell1   = [cell1_x, cell1_y]

        # NOT STARTING FROM i+1 BECAUSE WE NEED INTER-CELL BONDS
        for j in i:num_cells
            cellp2_x = rem(j, n1) > 0 ? rem(j, n1) - 1 : n1 - 1
            cellp2_y = rem(j, n1) > 0 ? div(j, n1) : div(j, n1) - 1

            cell2_x = cellp2_x*a1[1] + cellp2_y*a2_length*cos(θ)
            cell2_y = cellp2_y*a2_length*sin(θ)
            cell2   = [cell2_x, cell2_y]

            for site_i in 1:length(r)
                site_num_i = site_i + length(r)*(i-1)
                ri = r[site_i] + cell1

                for site_j in 1:length(r)
                    # ensure that there are no diagonal entries
                    if i == j & site_i == site_j
                        continue
                    end

                    site_num_j = site_j + length(r)*(j-1)
                    rj = r[site_j] + cell2

                    Δ = ri - rj
                    Δx, Δy = Δ[1], Δ[2]

                    # checks for PBCs
                    if PBC1 & PBC2
                        # non-periodic
                        d_np = sqrt(Δx^2 + Δy^2)

                        # periodic in a1 only
                        rj[1] -= a1[1]*n1
                        Δ = ri - rj
                        Δx, Δy = Δ[1], Δ[2]
                        d_p1 = sqrt(Δx^2 + Δy^2)

                        # periodic in a1 and a2
                        rj[1] -= a2_length*n2*cos(θ)
                        rj[2] -= a2_length*n2*sin(θ)
                        Δ = ri - rj
                        Δx, Δy = Δ[1], Δ[2]
                        d_p12 = sqrt(Δx^2 + Δy^2)

                        # periodic in a2 only
                        rj[1] += a1[1]*n1
                        Δ = ri .- rj
                        Δx, Δy = Δ[1], Δ[2]
                        d_p2 = sqrt(Δx^2 + Δy^2)

                        # take the minimum distance
                        d = min(d_np, d_p1, d_p2, d_p12)

                    elseif PBC1 & !PBC2
                        if abs(Δx) > 0.5*n1*a1[1]
                            rj[1] -= a1[1]*n1
                            Δ = ri - rj
                            Δx, Δy = Δ[1], Δ[2]
                            d = sqrt(Δx^2 + Δy^2)
                        end

                    elseif PBC2 & !PBC1
                        # rotate lattice by -θ to lay along a2
                        yi = ri[1]*cos(θ) + ri[2]*sin(θ)
                        yj = rj[1]*cos(θ) + rj[2]*sin(θ)

                        if abs(yi - yj) > 0.5*n2*a2_length
                            rj[1] -= a2_length*n2*cos(θ)
                            rj[2] -= a2_length*n2*sin(θ)
                            Δ = ri - rj
                            Δx, Δy = Δ[1], Δ[2]
                            d = sqrt(Δx^2 + Δy^2)
                        end

                    else
                        d = sqrt(Δx^2 + Δy^2)
                    end

                    dij[site_num_i, site_num_j] = d <= trunc ? d : 0.0

                end
            end
        end
    end

    return dij
end
=#


###############################################################################


abstract type AbstractRydberg{O <: AbstractOperatorSampler} <: AbstractLTFIM{O} end

struct Rydberg{O,M <: UpperTriangular{Float64},UΩ <: AbstractVector{Float64}, Uδ <: AbstractVector{Float64}, L <: Lattice} <: AbstractRydberg{O}
    op_sampler::O
    V::M
    Ω::UΩ
    δ::Uδ
    lattice::L
    energy_shift::Float64
end
nspins(H::Rydberg) = nspins(H.lattice)


function make_prob_vector(::Type{<:AbstractRydberg}, V::UpperTriangular{T}, Ω::AbstractVector{T}, δ::AbstractVector{T}; epsilon=0.0) where T
    @assert length(Ω) == length(δ) == size(V, 1) == size(V, 2)

    ops = Vector{NTuple{3, Int}}()
    p = Vector{T}()
    energy_shift = zero(T)

    for i in eachindex(Ω)
        if !iszero(Ω[i])
            push!(ops, makediagonalsiteop(AbstractLTFIM, i))
            push!(p, Ω[i] / 2)
            energy_shift += Ω[i] / 2
        end
    end

    Ns = length(Ω)
    bond_spins = Set{NTuple{2,Int}}()
    coordination_numbers = zeros(Int, Ns)
    for j in axes(V, 2), i in axes(V, 1)
        if i < j && !iszero(V[i, j])
            push!(bond_spins, (i, j))
            coordination_numbers[i] += 1
            coordination_numbers[j] += 1
        end
    end

    n = [0 0; 0 1]
    I = [1 0; 0 1]

    # TODO: add fictitious bonds if there's a z-field on an "unbonded" site
    for (site1, site2) in bond_spins
        # by this point we can assume site1 < site2
        δb1 = δ[site1] / coordination_numbers[site1]
        δb2 = δ[site2] / coordination_numbers[site2]
        local_H = V[site1, site2]*kron(n, n) - δb1*kron(n, I) - δb2*kron(I, n)

        p_spins = -diag(local_H)
        C = abs(min(0, minimum(p_spins))) + epsilon
        p_spins .+= C
        energy_shift += C

        for (t, p_t) in enumerate(p_spins)
            if !iszero(p_t)
                push!(ops, (t, site1, site2))
                push!(p, p_t)
            end
        end
    end

    return ops, p, energy_shift
end


###############################################################################

# function BlockadeRydberg(dims::NTuple{N, Int}, J::Float64, hx::Float64, hz::Float64, pbc=true) where N
# end

function Rydberg(dims::NTuple{D, Int}, R_b, Ω, δ, pbc=true) where D
    if D == 1
        lat = Chain(dims[1], 1.0, pbc)
    elseif D == 2
        lat = Rectangle(dims[1], dims[2], 1.0, 1.0, pbc)
    else
        error("Unsupported number of dimensions. 1- and 2-dimensional lattices are supported.")
    end
    return Rydberg(lat, R_b, Ω, δ)
end


function Rydberg(lat::Lattice, R_b::Float64, Ω::Float64, δ::Float64)
    Ns = nspins(lat)
    V = zeros(Float64, Ns, Ns)

    @inbounds for i in 1:(Ns-1)
        for j in (i+1):Ns
            # a zero entry in distance_matrix means there should be no bond
            V[i, j] = lat.distance_matrix[i, j] != 0.0 ? Ω * (R_b / lat.distance_matrix[i, j])^6 : 0.0
        end
    end
    V = UpperTriangular(triu!(V, 1))

    return Rydberg(V, Ω*ones(Ns), δ*ones(Ns), lat)
end

function Rydberg(V::AbstractMatrix{T}, Ω::AbstractVector{T}, δ::AbstractVector{T}, lattice::Lattice; epsilon=zero(T)) where T
    ops, p, energy_shift = make_prob_vector(AbstractRydberg, V, Ω, δ, epsilon=epsilon)
    op_sampler = ImprovedOperatorSampler(AbstractLTFIM, ops, p)
    return Rydberg{typeof(op_sampler), typeof(V), typeof(Ω), typeof(δ), typeof(lattice)}(op_sampler, V, Ω, δ, lattice, energy_shift)
end

total_hx(H::Rydberg)::Float64 = sum(H.Ω) / 2
haslongitudinalfield(H::AbstractRydberg) = !iszero(H.δ)
