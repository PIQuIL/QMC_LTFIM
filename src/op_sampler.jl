# (-3, i) = long field
# (-2, i) = transverse field off-diag
# (-1, i) = transverse field diag
# (0, 0) = id
# (i, j) = diag bond op

using Base.Iterators

function make_prob_vector(J::AbstractMatrix{T}, hx::AbstractVector{T}) where T
    ops = Vector{NTuple{2, Int}}(undef, 0)
    p = Vector{T}(undef, 0)

    k = 0
    for i in eachindex(hx)
        if hx[i] != 0
            push!(ops, (-1, i))
            push!(p, hx[i])
        end
    end

    # only take J_ij terms from upper diagonal since it's symmetric
    for j in axes(J, 2), i in axes(J, 1)
        if i < j
            if J[i, j] != 0
                push!(ops, op)
                push!(p, 2*abs(J[i, j]))
            end
        end
    end

    return ops, p
end

function make_prob_vector(bond_spins::Vector{NTuple{2,Int}}, Ns::Int, J::T, h::T) where T
    ops = Vector{NTuple{2, Int}}(undef, 0)
    p = Vector{T}(undef, 0)

    if !iszero(h)
        for i in 1:Ns
            push!(ops, (-1, i))
            push!(p, h)
        end
    end

    if !iszero(J)
        for op in bond_spins
            push!(ops, op)
            push!(p, 2*abs(J))
        end
    end

    return ops, p
end

function make_prob_vector(dims::NTuple{N, Int}, J::T, hx::T, hz::T, pbc=true) where {N, T}
    bond_spins, Ns, Nb = lattice_bond_spins(dims, pbc)
    bond_spins = Set(bond_spins)
    edge_sites = Set{Int}()

    if !pbc
        pbc_s = Set(lattice_bond_spins(dims, true)[1])
        edge_bonds = setdiff(pbc_s, bond_spins)
        edge_sites = Set(unique(flatten(edge_bonds)))
    end

    ops = Vector{NTuple{3, Int}}(undef, 0)
    p = Vector{T}(undef, 0)
    energy_shift = Ns*float(hx)

    if !iszero(hx)
        for i in 1:Ns
            push!(ops, (-1, i, 0))
            push!(p, hx)
        end
    end

    if !(iszero(J) && iszero(hz))
        hzb = hz * Ns / (2*Nb)
        #   order:   DD,        DU,       UD,       UU
        p_spins   = [J - 2*hzb, -J,       -J,       J + 2*hzb]
        p_spins_l = [J - 3*hzb, -J - hzb, -J + hzb, J + 3*hzb]
        p_spins_r = [J - 3*hzb, -J + hzb, -J - hzb, J + 3*hzb]
        C   = abs(min(0, minimum(p_spins)))
        C_l = abs(min(0, minimum(p_spins_l)))
        C_r = abs(min(0, minimum(p_spins_r)))
        p_spins .+= C
        p_spins_l .+= C_l
        p_spins_r .+= C_r

        for t in eachindex(p_spins), (site1, site2) in bond_spins
            if !(site1 in edge_sites || site2 in edge_sites)
                p_t = p_spins[t]
                energy_shift += C/4
            elseif site1 in edge_sites
                p_t = p_spins_l[t]
                energy_shift += C_l/4
            else  # site2 in edge_sites
                p_t = p_spins_r[t]
                energy_shift += C_r/4
            end

            if !iszero(p_t)
                push!(ops, (t, site1, site2))
                push!(p, p_t)
            end
        end

    end

    return ops, p, Ns, Nb, energy_shift
end



###################


abstract type AbstractOperatorSampler{K, T, P <: AbstractProbabilityVector{T}} end
firstindex(::AbstractOperatorSampler) = 1
lastindex(os::AbstractOperatorSampler) = length(os)

getweight(os::AbstractOperatorSampler{K}, op::NTuple{K, Int}) where K = haskey(os.op_indices, op) ? getweight(os.pvec, os.op_indices[op]) : 0


struct OperatorSampler{K, T, P} <: AbstractOperatorSampler{K, T, P}
    operators::Vector{NTuple{K, Int}}
    pvec::P
    op_indices::Dict{NTuple{K, Int}, Int}
end


# samples operators using a heap
# based on a blog post by Tim Vieira
# https://timvieira.github.io/blog/post/2016/11/21/heaps-for-incremental-computation/
function OperatorSampler(operators::Vector{NTuple{K, Int}}, p::Vector{T}) where {K, T <: Real}
    @assert length(operators) == length(p) "Given vectors must have the same length!"
    pvec = probability_vector(p)
    op_indices = Dict{NTuple{K, Int}, Int}(op => i for (i, op) in enumerate(operators))
    return OperatorSampler{K, T, typeof(pvec)}(operators, pvec, op_indices)
end
rand(os::OperatorSampler{K}) where K = @inbounds os.operators[rand(os.pvec)]
@inline length(os::OperatorSampler) = length(os.operators)




function cluster_probs_vec(operators::Vector{NTuple{K, Int}}, p::AbstractVector{T}) where {K, T <: Real}
    perm = sortperm(p)
    p = p[perm]
    operators = operators[perm]

    uniq_p = T[]
    uniq_ops = Vector{Vector{NTuple{K, Int}}}(undef, 0)
    op_bins = Dict{NTuple{K, Int}, Int}()

    for i in axes(p, 1)
        if length(uniq_p) == 0
            push!(uniq_p, p[i])
            push!(uniq_ops, [operators[i]])
        elseif !(last(uniq_p) ≈ p[i])
            push!(uniq_p, p[i])
            push!(uniq_ops, [operators[i]])
        else
            push!(uniq_ops[end], operators[i])
        end
        push!(op_bins, operators[i] => length(uniq_p))
    end

    # rescale uniq_p
    for i in eachindex(uniq_p, uniq_ops)
        uniq_p[i] *= length(uniq_ops[i])
    end
    return uniq_ops, uniq_p, op_bins
end



struct HierarchicalOperatorSampler{K, T, P} <: AbstractOperatorSampler{K, T, P}
    operator_bins::Vector{Vector{NTuple{K, Int}}}
    pvec::P
    op_indices::Dict{NTuple{K, Int}, Int}
end

function HierarchicalOperatorSampler(operators::Vector{NTuple{K, Int}}, p::AbstractVector{T}) where {K, T <: Real}
    @assert length(operators) == length(p) "Given vectors must have the same length!"
    operator_bins, p, op_indices = cluster_probs_vec(operators, p)
    pvec = probability_vector(p)
    return HierarchicalOperatorSampler{K, T, typeof(pvec)}(operator_bins, pvec, op_indices)
end
length(os::HierarchicalOperatorSampler) = sum(length, os.operator_bins)

function rand(os::HierarchicalOperatorSampler{K})::NTuple{K, Int} where K
    @inbounds ops_list = os.operator_bins[rand(os.pvec)]
    l = rand(1:length(ops_list))

    return @inbounds ops_list[l]
end

function getweight(os::HierarchicalOperatorSampler{K}, op::NTuple{K, Int}) where K
    if !haskey(os.op_indices, op)
        return 0
    end
    idx = os.op_indices[op]
    return getweight(os.pvec, idx) / length(os.operator_bins[idx])
end
