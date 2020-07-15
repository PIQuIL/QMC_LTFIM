using Distributions

include("lattice.jl")
include("updates.jl")
include("measurements.jl")

M = 1000
J = 1.0 # interaction strength
Ω = 0.0 # long. field strength
h = 1.0 # transverse field strength
β = 1.0
MCS = 10000

# ~~~ PRE-SIMULATION CALCULATIONS ~~~

# constant shift in Hamiltonian
constant = 2*N*Ω + N*h + Nb*J
println("Constant shift to Hamiltonian: ", constant/N)

# used in denominator / numerator for probabilities in diag update
c1 = 2*N*Ω + 4*N*h + 4*Nb*J

P_H0a = 4*N*h / c1
P_H1a = 2*N*Ω / c1
P_H2a = 4*Nb*J / c1
diagProbs = [P_H0a, P_H1a, P_H2a]

# for choosing which diag operator to insert in diag update
d = Multinomial(1, diagProbs)
labels = [1,2,3]

spin_left = rand(0:1, N)

# operator list contains the operator types (integers) at every time slice
# also contains the position of the operator (i,j)
# if i = j, site operator. Else, bond operator
operator_list = zeros(Int, M, 3)

# first index = op. type
# 2nd and 3rd (bonds only) indices: location
# this information goes in the operator_list
# (-1, i)   : off-diag operator (transverse field) - site
# (0, 0)    : identity operator
# (1, i)    : diag operator (h - transverse field) - site
# (2, i)    : diag operator (Ω - long. field) - site 
# (3, i, j) : diagonal operator (J - interactions) - bond

# equilibrate
for i = 1:5000
    DiagonalUpdate()
    LinkedList()
    ClusterUpdate()
end

#DiagonalUpdate()
#LinkedList()
#@show LinkList
#@show FlagForH1a
#@show in_cluster
#ClusterUpdate()
#@show operator_list

ener = 0
for i = 1:MCS
    DiagonalUpdate()
    LinkedList()
    if i%100 == 0
        global ener += Measure()
    end
    ClusterUpdate()
end 

#println(ener/(N*MCS/100) + constant/N)
println(ener/(N*MCS/100))
