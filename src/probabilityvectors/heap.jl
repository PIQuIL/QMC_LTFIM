# draw samples using a heap
# based on a blog post by Tim Vieira
# https://timvieira.github.io/blog/post/2016/11/21/heaps-for-incremental-computation/

# more efficient for larger probability vectors
struct ProbabilityHeap{T} <: AbstractProbabilityVector{T}
    prob_heap::Vector{T}
    length::Int

    function ProbabilityHeap{T}(p::Vector{T}) where {T <: Real}
        if length(p) == 0
            throw(ArgumentError("probability vector must have non-zero length!"))
        end
        if any(x -> x < zero(T), p)
            throw(ArgumentError("weights must be non-negative!"))
        end

        L = length(p)
        d = nextpow(2, L)

        heap = zeros(T, 2*d)
        @inbounds @views heap[d : d + L - 1] = p

        @inbounds for i in (d-1):-1:1
            heap[i] = heap[2*i] + heap[2*i + 1]
        end

        new{T}(heap, L)
    end
end
ProbabilityHeap(p::Vector{T}) where T = ProbabilityHeap{T}(p)
@inline length(pvec::ProbabilityHeap) = pvec.length
normalization(pvec::ProbabilityHeap) = @inbounds pvec.prob_heap[1]

function show(io::IO, p::ProbabilityHeap{T}) where T
    heap, L = p.prob_heap, p.length

    d = nextpow(2, L)

    pvec = heap[d : d + L - 1]
    r = repr(pvec; context=IOContext(io, :limit=>true))

    print(io, "ProbabilityHeap{$T}($r)")
end


function rand(rng::AbstractRNG, pvec::ProbabilityHeap{T})::Int where T
    heap = pvec.prob_heap
    l = length(heap) ÷ 2
    @inbounds r = rand(rng) * heap[1]

    i = 1
    while i < l
        i *= 2
        @inbounds left = heap[i]
        if r > left
            r -= left
            i += 1
        end
    end
    return (i - l + 1)
end

@inline function setweight!(pvec::ProbabilityHeap{T}, w::T, i::Int) where T
    @boundscheck checkbounds(pvec, i)

    heap = pvec.prob_heap
    j = (length(heap) ÷ 2) - 1 + i
    @inbounds heap[j] = w
    @inbounds while j > 1
        j ÷= 2
        heap[j] = heap[2*j] + heap[2*j + 1]
    end
end

@inline function getweight(pvec::ProbabilityHeap, i::Int)
    @boundscheck checkbounds(pvec, i)

    heap = pvec.prob_heap
    d = (length(heap) ÷ 2) - 1
    @inbounds x = heap[d + i]
    return x
end

@inline function getweight(pvec::ProbabilityHeap, r::AbstractArray{Int})
    @boundscheck checkbounds(pvec, r)

    heap = pvec.prob_heap
    d = (length(heap) ÷ 2) - 1
    @inbounds x = heap[d .+ r]
    return x
end

