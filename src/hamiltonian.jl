abstract type Hamiltonian{D,N,O<:AbstractOperatorSampler} end

localdim(::Hamiltonian{D}) where {D} = D
dim(::Hamiltonian{D,N}) where {D,N} = N

zero(H::Hamiltonian{2}) = falses(nspins(H))
zero(H::Hamiltonian) = zeros(nspins(H))
one(H::Hamiltonian{2}) = trues(nspins(H))
one(H::Hamiltonian) = ones(nspins(H))

nspins(H::Hamiltonian) = H.Ns
nbonds(H::Hamiltonian) = H.Nb
