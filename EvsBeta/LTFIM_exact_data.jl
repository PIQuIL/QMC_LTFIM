using Plots
using ProgressMeter
using DelimitedFiles

include("LTFIM_thermal.jl")

header = ["N" "β" "B" "Ω" "E/N" "M/N"]

global data_full = header

for N = 2:2:10
    @show N
    for β = 0.2:0.2:2
        for hx = 0.0:0.5:2
            for hz = 0.0:0.5:2

                ltfim = LTFIM_1D(N, hx, hz, β)
                energy = ltfim.Energy
                magnetization = ltfim.Magnetization

                data = [N β hx hz energy magnetization]
                global data_full = vcat(data_full, data)

            end
        end
    end                
end


open("energy_magnetization.txt", "w") do io
    writedlm(io, data_full)
end
