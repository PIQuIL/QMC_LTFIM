# for plotting lattice geometries
using PyPlot

abstract type AbstractLattice end
abstract type PolyLattice <: AbstractLattice end
abstract type SimpleLattice end

# AbstractLattice: Completely custom
# PolyLattice: A lattice with a multi-site basis, but is common (e.g. Kagome)
# SimpleLattice: Single-site basis lattices

# TODO: honeycomb, kagome, 1D chains with multi-site basis

struct Custom <: AbstractLattice
    # translation vectors
    a1::Array{Float64,1} 
    a2::Array{Float64,1} 

    # number of repititions in directions of a1 and a2
    n1::Int # a1
    n2::Int # a2
    
    # coordinates of sites inside a unit cell
    r::Array{Array{Float64,1},1}

    # Periodic boundaries in directions of a1, a2
    # these are conventional PBCs
    PBC::Tuple{Bool,Bool}
end

function Custom(a::Float64, a2::Tuple{Float64,Float64}, n1::Int, n2::Int, r::Array{Array{Float64,1},1}, PBC::Tuple{Bool, Bool})
    a1 = [a, 0.]
    return Custom(a1, a2, n1, n2, r, PBC)
end


struct Ruby <: PolyLattice
    # parameter that defines spacing
    ρ::Float64
    
    # translation vectors
    a1::Array{Float64,1} 
    a2::Array{Float64,1} 

    # number of repititions in directions of a1 and a2
    n1::Int # a1
    n2::Int # a2
    
    # coordinates of sites inside a unit cell
    r::Array{Array{Float64,1},1}

    # Periodic boundaries in directions of a1, a2
    # these are conventional PBCs
    PBC::Tuple{Bool,Bool}
end

function Ruby(ρ::Float64, n1::Int, n2::Int, PBC::Tuple{Bool, Bool})
    a = 4*ρ/sqrt(3)
    a1 = [a, 0.]
    a2 = [a*0.5, a*sqrt(3)*0.5]

    # coordinates of each site in the unit cell
    r1 = [0., 0.]
    r2 = 0.75 * a2
    r3 = 0.25 * (a1 + a2)
    r4 = 0.5 * a1
    r5 = 0.25*a1 + 0.75*a2
    r6 = 0.5*a1 + 0.25*a2
    r = [r1, r2, r3, r4, r5, r6]

    return Ruby(ρ, a1, a2, n1, n2, r, PBC)
end


struct Triangle <: SimpleLattice
    a1::Array{Float64,1} 
    a2::Array{Float64,1} 
    n1::Int
    n2::Int
    PBC::Tuple{Bool, Bool}
end

function Triangle(a::Float64, a2::Array{Float64, 1}, n1::Int, n2::Int, PBC::Tuple{Bool, Bool})
    a1 = [a, 0.]
    return Triangle(a1, a2, n1, n2, PBC)
end


struct Rectangle <: SimpleLattice
    a1::Array{Float64,1} 
    a2::Array{Float64,1} 
    n1::Int
    n2::Int
    PBC::Tuple{Bool, Bool}
end

function Rectangle(a::Float64, a2::Array{Float64, 1}, n1::Int, n2::Int, PBC::Tuple{Bool, Bool})
    a1 = [a, 0.]
    return Rectangle(a1, a2, n1, n2, PBC)
end


function distance_matrix(lattice::AbstractLattice)
    # TODO: Periodic boundaries

    a1 = lattice.a1
    a2 = lattice.a2
    n1 = lattice.n1
    n2 = lattice.n2
    r = lattice.r
    PBC1, PBC2 = lattice.PBC
    
    a2_length = sqrt(a2[1]^2 + a2[2]^2) # length of a2 vector
    θ = acos(a2[1] / a2_length) # a2 angle from horizontal

    N = n1*n2*length(r) # total number of sites
    dij = [zeros(Float64, N) for _ in 1:N] # distance matrix
    num_cells = n1*n2 # number of repitions of the unit cell

    a2_length = sqrt(a2[1]^2 + a2[2]^2) # length of a2 vector
    θ = acos(a2[1] / a2_length) # a2 angle from horizontal

    # for plotting
    sites_x = zeros(Float64, N)
    sites_y = zeros(Float64, N)

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
    
                # for plotting the ruby lattice as a scatter plot
                sites_x[site_num_i] = ri[1]
                sites_y[site_num_i] = ri[2]

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

                    dij[site_num_i][site_num_j] = d 
                end
            end
        end
    end
    scatter(sites_x, sites_y, color="blue")
    savefig("custom_lattice.png", format="png", dpi=500)
    return dij
end


function distance_matrix(lattice::SimpleLattice)
    a1 = lattice.a1
    a2 = lattice.a2
    n1 = lattice.n1
    n2 = lattice.n2
    PBC1, PBC2 = lattice.PBC

    a2_length = sqrt(a2[1]^2 + a2[2]^2) # length of a2 vector
    θ = acos(a2[1] / a2_length) # a2 angle from horizontal
    
    # total number of sites
    N = n1*n2
    dij = [zeros(Float64, N) for _ in 1:N]

    # (xpi, ypi) are lattice index coordinates
    # need to keep track of these for PBC enforcement
    # subtract one from xpi and ypi to center everything at (0,0)
    for i in 1:(N-1)
        xp1 = rem(i, n1) > 0 ? rem(i, n1) - 1 : n1 - 1
        yp1 = rem(i, n1) > 0 ? div(i, n1) : div(i, n1) - 1

        x1 = xp1*a1[1] + yp1*a2_length*cos(θ)
        y1 = yp1*a2_length*sin(θ)

        for j in (i+1):N
            xp2 = rem(j, n1) > 0 ? rem(j, n1) - 1 : n1 - 1
            yp2 = rem(j, n1) > 0 ? div(j, n1) : div(j, n1) - 1

            x2 = xp2*a1[1] + yp2*a2_length*cos(θ)
            y2 = yp2*a2_length*sin(θ)

            # check x and y directions to enforce PBCs
            if PBC1 & PBC2
                # calculate minimum distance
                
                # non-periodic
                d_np = sqrt( (y1 - y2)^2 + (x1 - x2)^2 )

                # periodic in x only
                x2 -= a1[1]*n1*cos(θ)
                d_px = sqrt( (y1 - y2)^2 + (x1 - x2)^2 )

                # periodic in x and y
                # x2 has already been taken care of
                y2 -= a2_length*n2*sin(θ)
                d_pxy = sqrt( (y1 - y2)^2 + (x1 - x2)^2 )

                # periodic in y only
                # undo the x2 periodicity
                x2 += a1[1]*n1*cos(θ) 
                d_py = sqrt( (y1 - y2)^2 + (x1 - x2)^2 )

                d = min(d_py, d_px, d_pxy, d_np)

            elseif PBC1 & !PBC2
                if abs(x1 - x2) > 0.5*n1*a1[1]
                    x2 -= a1[1]*n1*cos(θ)
                end

                d = sqrt( (y1 - y2)^2 + (x1 - x2)^2 )
            
            elseif PBC2 & !PBC1
                yi = x1*cos(θ) + y1*sin(θ)
                yj = x2*cos(θ) + y2*sin(θ)
                
                if abs(yi - yj) > 0.5*n2*a2_length
                    x2 -= a2_length*n2*cos(θ)
                    y2 -= a2_length*n2*sin(θ)
                end

                d = sqrt( (y1 - y2)^2 + (x1 - x2)^2 )
            
            else # no PBCs anywhere
                d = sqrt( (y1 - y2)^2 + (x1 - x2)^2 )
            
            end

            dij[i][j] = d
        end
    end

    return dij 
end

# TODO: standard chains, and then multi-site bases
#struct Chain <: OneDLattice
#    a::Float64
#    n::Int
#    PBC::Bool
#end
#
#
#function Chain(a::Float64, n::Int, PBC::Bool)
#    return Chain(a, n, PBC)
#end
#
#function
#    distance_matrix = [zeros(Float64, n1) for _ in 1:n1]
#
#    for i in 1:(n1-1)
#        x1 = rem(i, n1) > 0 ? rem(i, n1) : n1
#        
#        for j in (i+1):n1
#            x2 = rem(j, n1) > 0 ? rem(j, n1) : n1
#            
#            if PBC
#                dx = abs(x1-x2) > 0.5*n1 ? 0.5*n1 - rem(abs(x1-x2), 0.5*n1) : abs(x1-x2)
#            else
#                dx = abs(x1-x2)
#            end
#
#            dx *= aX
#            distance_matrix[i][j] = dx 
#        end
#    end
#    return Chain(n1, aX, distance_matrix)
#end


#nspins(lattice::Custom) = lattice.num_cells*length(lattice.r)
#nspins(lattice::Ruby) = lattice.num_cells*6
#nspins(lattice::Rectangle) = lattice.n1 * lattice.n2
#nspins(lattice::Triangle) = lattice.n1 * lattice.n2
#nspins(lattice::Chain) = lattice.n1
