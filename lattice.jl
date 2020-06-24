# 1D boundary conditions only for now
# assign site indices to bonds

N = 4
Nb = N - 1

bond_spin = zeros(Nb, 2)

for i = 1:Nb
    bond_spin[i,1] = i
    bond_spin[i,2] = i + 1
end

bond_spin = convert(Array{Int64}, bond_spin)
