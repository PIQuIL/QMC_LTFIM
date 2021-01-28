# Using HashTuple as dict keys seems to work fine, it's still slower than
# the array scheme though
struct HashTuple{K}
    tup::NTuple{K, Int}
end
Base.hash(op::HashTuple{K}) where K = foldl((r,i) -> 31r + i, op.tup, init=1)
Base.:(==)(a::HashTuple{K}, b::HashTuple{K}) where K = (a.tup === b.tup)
Base.:(==)(a::HashTuple, b::HashTuple) = false

##############################################################################

function conv_op_to_idx(op::NTuple{K, Int}, strides::NTuple{K, Int}, shifts::NTuple{K, Int}) where K
    idx = 0
    @inbounds if K > 1
        # compute packed storage format for last two indices (i, j),
        #  assuming i <= j which we've enforced in all of our constructors
        i, j = op[end-1] - shifts[end-1], op[end] - shifts[end]
        p_ij = i + ((j * (j-1)) ÷ 2)
        idx += p_ij
        idx *= strides[end-1]

        for i in reverse(eachindex(op[1:end-2]))
            idx += (op[i] - shifts[i])
            idx *= strides[i]
        end
    elseif K == 1
        idx += (op[1] - shifts[1])
        idx *= strides[1]
    end
    return idx + 1
end


struct OperatorDict{K, V}
    values::Vector{V}
    strides::NTuple{K, Int}
    shifts::NTuple{K, Int}
    axes::NTuple{K, UnitRange{Int}}

    function OperatorDict(operators::Vector{NTuple{K, Int}}, values::Vector{T}; default=zero(T)) where {K, T}
        axs = ntuple(i -> extrema(x->x[i], operators), K)
        axes_ = ntuple(i -> UnitRange(axs[i]...), K)
        strides = tuple(1, length.(axes_[1:end-1])...)
        shifts = first.(axes_)

        l = conv_op_to_idx(last.(axes_), strides, shifts)

        vals = fill!(zeros(T, l), default)
        for (i, op) in enumerate(operators)
            idx = conv_op_to_idx(op, strides, shifts)
            @assert !iszero(vals[idx]) "Collision in operator dictionary! $(op => idx)"
            vals[idx] = values[i]
        end

        new{K, T}(vals, strides, shifts, axes_)
    end
end

@inline function Base.haskey(op_dict::OperatorDict{K}, op::NTuple{K, Int}) where K
    @inbounds all(i -> op[i] in op_dict.axes[i], 1:K)
end

@inline function Base.get(op_dict::OperatorDict{K}, op::NTuple{K, Int}) where K
    @boundscheck haskey(op_dict, op)
    strides, shifts = op_dict.strides, op_dict.shifts
    idx = conv_op_to_idx(op, strides, shifts)
    return @inbounds op_dict.values[idx]
end
