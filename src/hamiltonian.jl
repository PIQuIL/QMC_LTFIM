abstract type Hamiltonian{D,N,L<:BoundedLattice{N}} end

localdim(::Hamiltonian{D}) where {D} = D
dim(::Hamiltonian{D,N}) where {D,N} = N

# Ω: longitudinal field strength
# h: transverse field strength
# J: interaction strength

struct LTFIM{N,L} <: Hamiltonian{2,N,L}
    lattice::L
    bond_spin::Vector{NTuple{2, Int}}
    h::Float64
    J::Float64
    Ω::Float64
    P_h::Float64
    P_J::Float64
    P_Ω::Float64
    P_normalization::Float64
    Ns::Int
    Nb::Int
end

function LTFIM(lattice::L, h::Float64, J::Float64, Ω::Float64) where {L<:BoundedLattice{N}} where {N}
    bond_spin = lattice_bond_spins(lattice)

    Ns, Nb = length(lattice), length(bond_spin)

    P_Ω = 4 * Ω * Ns
    P_J = 2 * J * Nb
    P_h = h * Ns

    P_normalization = P_h + P_J + P_Ω
    P_h /= P_normalization
    P_J /= P_normalization
    P_Ω /= P_normalization

    return LTFIM{N,L}(
        lattice, 
        bond_spin, 
        h, 
        J,
        Ω, 
        P_h, 
        P_J, 
        P_Ω, 
        P_normalization, 
        Ns, 
        Nb
    )
end

zero(H::Hamiltonian{2}) = falses(size(H.lattice)...)
zero(H::Hamiltonian) = zeros(size(H.lattice)...)
one(H::Hamiltonian{2}) = trues(size(H.lattice)...)
one(H::Hamiltonian) = ones(size(H.lattice)...)

nspins(H::LTFIM) = H.Ns
nbonds(H::LTFIM) = H.Nb
