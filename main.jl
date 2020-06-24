include("lattice.jl")
include("updates.jl")
include("measurements.jl")

M = 300  
J = 1.0 # interaction strength
Omega = 1.0 # long. field strength
h = 1.0 # transverse field strength
β = 4.0

N = 4
Nb = N - 1 # number of bonds

# ~~~ PRE-SIMULATION CALCULATIONS ~~~

# constant shift in Hamiltonian
constant = 2*N*Ω + N*h + Nb*J

# used in denominator / numerator for probabilities in diag update
c1 = β*(2*N*Ω + 4*N*h + 4*Nb*J)

P_H1a = 2*N*Ω / constant
P_H2a = 4*N*h / constant
P_H3a = 4*Nb*J / constant
diagProbs = [P_H1a, P_H2a, P_H3a]

# for choosing which diag operator to insert in diag update
d = Multinomial(1, diagProbs)
labels = [1,2,3]

spin_left = rand(0:1, N)
LinkList = []
LegType = []
Associates = []

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
for i = 1:2000
    DiagonalUpdate()
    LinkedList()
    ClusterUpdate()
end

M2 = 0
for i = 1:MCS
    DiagonalUpdate()
    LinkedList()
    global M2 += Measure()
    ClusterUpdate()
end

println(M2)
