function make_hamiltonian_matrix(H::Union{AbstractLTFIM, AbstractTFIM})
    if !haslongitudinalfield(H)
        J, hx, hz = Symmetric(H.J), H.hx, zero(H.hx)
    else
        J, hx, hz = Symmetric(H.J), H.hx, H.hz
    end

    N = nspins(H)
    dim = 2^N

    Hamiltonian = zeros(dim,dim)   #This is your 2D Hamiltonian matrix

    for Ket = 0:dim-1  #Loop over Hilbert Space
        Diagonal = 0.0
        # interaction piece
        for SpinIndex = 0:N-1  #Loop over spin index (zero indexing)
            Spin1 = 2*((Ket>>SpinIndex)&1) - 1

            for NextIndex = SpinIndex+1:N-1
                Spin2 = 2*((Ket>>NextIndex)&1) - 1
                Diagonal += J[SpinIndex+1, NextIndex+1]*Spin1*Spin2
            end

            if hz isa AbstractArray
                Diagonal -= hz[SpinIndex+1]*Spin1
            else
                Diagonal -= hz*Spin1
            end

            bit = 2^SpinIndex   #The "label" of the bit to be flipped
            Bra = Ket ⊻ bit    #Binary XOR flips the bit
            Hamiltonian[Bra+1,Ket+1] = -hx[SpinIndex+1]
        end

        Hamiltonian[Ket+1,Ket+1] = Diagonal
    end

    return Hermitian(Hamiltonian)
end

function make_hamiltonian_matrix(H::AbstractRydberg)
    V, Ω, δ = H.V, H.Ω, H.δ

    N = nspins(H)
    dim = 2^N

    Hamiltonian = zeros(dim,dim)   #This is your 2D Hamiltonian matrix

    for Ket = 0:dim-1  #Loop over Hilbert Space
        Diagonal = 0.0
        # interaction piece
        for SpinIndex = 0:N-1  #Loop over spin index (zero indexing)
            Spin1 = ((Ket>>SpinIndex)&1)

            for NextIndex = SpinIndex+1:N-1
                Spin2 = ((Ket>>NextIndex)&1)
                Diagonal += V[SpinIndex+1, NextIndex+1]*Spin1*Spin2
            end

            if δ isa AbstractArray
                Diagonal -= δ[SpinIndex+1]*Spin1
            else
                Diagonal -= δ*Spin1
            end

            bit = 2^SpinIndex   #The "label" of the bit to be flipped
            Bra = Ket ⊻ bit    #Binary XOR flips the bit
            Hamiltonian[Bra+1,Ket+1] = -Ω[SpinIndex+1]
        end

        Hamiltonian[Ket+1,Ket+1] = Diagonal
    end

    return Hermitian(Hamiltonian)
end
