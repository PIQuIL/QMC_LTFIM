# for plotting lattice geometries
#using PyPlot

abstract type Lattice end


struct Ruby <: Lattice
    ρ::Float64
    num_cells::Int
    distance_matrix::Array{Array{Float64, 1},1}
end


function Ruby(ρ::Float64, nX::Int, nY::Int)
    # nX and nY specify the number of repitions of the unit cell in the
    # x and y directions. NOT the number of sites
    # TODO: Periodic boundary conditions

    # define the primitive lattice translation vectors in terms of ρ
    a = 4*ρ/sqrt(3)
    a1 = (a, 0)
    a2 = (a*0.5, a*sqrt(3)*0.5)

    # coordinates of each site in the unit cell
    r1 = (0, 0)
    r2 = 0.75 .* a2
    r3 = 0.25 .* (a1 .+ a2)
    r4 = 0.5 .* a1 
    r5 = (0.25 .* a1) .+ (0.75 .* a2) 
    r6 = (0.5 .* a1) .+ (0.25 .* a2) 
    r  = [r1, r2, r3, r4, r5, r6]

    N = nX*nY*6 # 6-site basis
    distance_matrix = [zeros(Float64, N) for _ in 1:N]
    num_cells = nX*nY # number of repitions of the unit cell

    sites_x = zeros(Float64, N)
    sites_y = zeros(Float64, N)

    # NOT ENDING ON num_cells-1 BECAUSE WE NEED INTER-CELL BONDS
    for i in 1:num_cells
        cellp1_x = rem(i, nX) > 0 ? rem(i, nX) - 1 : nX - 1
        cellp1_y = rem(i, nX) > 0 ? div(i, nX) : div(i, nX) - 1

        # factor of 1/2 or sqrt(3)/2 in the expressions below come from cos/sin pi/3
        cell1_x = cellp1_x*a + cellp1_y*a*0.5
        cell1_y = cellp1_y*a*sqrt(3)*0.5
        cell1   = (cell1_x, cell1_y)
        
        # NOT STARTING FROM i+1 BECAUSE WE NEED INTER-CELL BONDS
        for j in i:num_cells
            cellp2_x = rem(j, nX) > 0 ? rem(j, nX) - 1 : nX - 1
            cellp2_y = rem(j, nX) > 0 ? div(j, nX) : div(j, nX) - 1

            # factor of 1/2 or sqrt(3)/2 in the expressions below come from cos/sin pi/3
            cell2_x = cellp2_x*a + cellp2_y*a*0.5
            cell2_y = cellp2_y*a*sqrt(3)*0.5
            cell2   = (cell2_x, cell2_y)
        
            for site_i in 1:6
                site_num_i = site_i + 6*(i-1) 
                ri = r[site_i] .+ cell1
    
                # for plotting the ruby lattice as a scatter plot
                #sites_x[site_num_i] = ri[1]
                #sites_y[site_num_i] = ri[2]

                for site_j in 1:6
                    # ensure that there are no diagonal entries 
                    if i == j & site_i == site_j
                        continue
                    end
                
                    site_num_j = site_j + 6*(j-1) 
                    rj = r[site_j] .+ cell2

                    Δ = ri .- rj
                    Δx, Δy = Δ[1], Δ[2]

                    distance_matrix[site_num_i][site_num_j] = sqrt(Δx^2 + Δy^2)
                end
            end
        end
    end
    #scatter(sites_x, sites_y, color="blue")
    #savefig("ruby_lattice.png", format="png", dpi=500)
    return Ruby(a, num_cells, distance_matrix) 
end


struct Triangle <: Lattice
    nX::Int
    nY::Int
    # primitive vectors are (magnitude, angle from horizontal)
    # a1 is just a scalar since it is along the horizontal
    a1::Float64
    a2::Tuple{Float64,Float64}
    distance_matrix::Array{Array{Float64,1},1}
end


function Triangle(nX::Int, nY::Int, a1::Float64, a2::Tuple{Float64, Float64}, PBC::Tuple{Bool,Bool})
    N = nX*nY
    PBCx = PBC[1]
    PBCy = PBC[2]

    distance_matrix = [zeros(Float64, N) for _ in 1:N]

    # (xpi, ypi) are lattice index coordinates
    # need to keep track of these for PBC enforcement
    # subtract one from xpi and ypi to center everything at (0,0)
    for i in 1:(N-1)
        xp1 = rem(i, nX) > 0 ? rem(i, nX) - 1 : nX - 1
        yp1 = rem(i, nX) > 0 ? div(i, nX) : div(i, nX) - 1

        x1 = xp1*a1 + yp1*a2[1]*cos(a2[2])
        y1 = yp1*a2[1]*sin(a2[2])

        for j in (i+1):N
            xp2 = rem(j, nX) > 0 ? rem(j, nX) - 1 : nX - 1
            yp2 = rem(j, nX) > 0 ? div(j, nX) : div(j, nX) - 1

            x2 = xp2*a1 + yp2*a2[1]*cos(a2[2])
            y2 = yp2*a2[1]*sin(a2[2])

            # check x and y directions to enforce PBCs
            if PBCx & PBCy
                # calculate minimum distance
                
                # non-periodic
                d_np = sqrt( (y1 - y2)^2 + (x1 - x2)^2 )

                # periodic in x only
                x2 -= a1*nX*cos(a2[2])
                d_px = sqrt( (y1 - y2)^2 + (x1 - x2)^2 )

                # periodic in x and y
                # x2 has already been taken care of
                y2 -= a2[1]*nY*sin(a2[2])
                d_pxy = sqrt( (y1 - y2)^2 + (x1 - x2)^2 )

                # periodic in y only
                # undo the x2 periodicity
                x2 += a1*nX*cos(a2[2]) 
                d_py = sqrt( (y1 - y2)^2 + (x1 - x2)^2 )

                d = min(d_py, d_px, d_pxy, d_np)

            elseif PBCx & !PBCy
                if abs(xp1 - xp2) > 0.5*nX
                    x2 -= a1*nX*cos(a2[2])
                end

                d = sqrt( (y1 - y2)^2 + (x1 - x2)^2 )
            
            elseif PBCy & !PBCx
                if abs(yp1 - yp2) > 0.5*nY
                    y2 -= a2[1]*nY*sin(a2[2])
                end

                d = sqrt( (y1 - y2)^2 + (x1 - x2)^2 )
            
            else # no PBCs anywhere
                d = sqrt( (y1 - y2)^2 + (x1 - x2)^2 )
            
            end

            distance_matrix[i][j] = d
        end
    end

    return Triangle(nX, nY, a1, a2, distance_matrix)
end


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


nspins(lattice::Ruby) = lattice.num_cells*6
nspins(lattice::Rectangle) = lattice.nX * lattice.nY
nspins(lattice::Triangle) = lattice.nX * lattice.nY
nspins(lattice::Chain) = lattice.nX
