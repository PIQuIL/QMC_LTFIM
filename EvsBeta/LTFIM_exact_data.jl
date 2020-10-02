using Plots
using ProgressMeter
using DelimitedFiles

include("LTFIM_thermal.jl")

header = ["N" "β" "B" "Ω" "E/N" "M/N"]

global data_full = header

for N = 2:10
    @show N
    for β = 0.5:0.5:5
        for B = 0.0:1.0:3.0
            for Ω = 0.0:1.0:3.0

                ltfim = LTFIM_1D_OBC(N, B, Ω, β)
                energy = ltfim.Energy
                magnetization = ltfim.Magnetization

                data = [N β B Ω energy magnetization]
                global data_full = vcat(data_full, data)

            end
        end
    end                
end


open("energy_magnetization.txt", "w") do io
    writedlm(io, data_full)
end
