mutable struct Bootstrap{F <: Function, T} <: AbstractVarianceAccumulator{T}
    f::F

    mean::T
    count::Int

    replicate_means::Vector{T}
    rep_counts::Vector{Int}
end


Bootstrap(f::Function, zero_element::T; nreps::Int = 500) where T =
    Bootstrap{typeof(f), T}(f, zero(zero_element), zero(Int),
                            [zero(zero_element) for _ in 1:nreps],
                            zeros(Int, nreps))

Bootstrap(f::Function, T::Type; nreps::Int = 500) =
    Bootstrap{typeof(f), T}(f, zero(T), zero(Int),
                            zeros(T, nreps),
                            zeros(Int, nreps))

Bootstrap(f::Function; kw...) = Bootstrap(f, Float64; kw...)
Bootstrap(T::Type; kw...) = Bootstrap(identity, T; kw...)
Bootstrap(; kw...) = Bootstrap(identity; kw...)



nreplicates(B::Bootstrap) = length(B.replicate_means)


function _combine_means(a_val::T, a_count::Int, b_val::T, b_count::Int) where T
    if iszero(a_count) && iszero(b_count)
        return zero(a_val), zero(a_count)
    end

    n = a_count + b_count
    new_mean = (a_val*a_count + b_val*b_count) / n
    return new_mean, n
end


function combine(a::Bootstrap{F, T}, b::Bootstrap{F, T}; method::Symbol=:merge) where {F, T}
    if method == :merge
        @assert nreplicates(a) == nreplicates(b)

        rep_counts = Vector{Int}(undef, nreplicates(a))
        reps = Vector{T}(undef, nreplicates(a))
        @inbounds for i in 1:nreplicates(a)
            reps[i], rep_counts[i] = _combine_means(
                a.replicate_means[i], a.rep_counts[i],
                b.replicate_means[i], b.rep_counts[i]
            )
        end
    elseif method == :concat
        rep_counts = vcat(a.rep_counts, b.rep_counts)
        reps = vcat(a.replicate_means, b.replicate_means)
    else
        throw(ArgumentError("Unrecognized combining method: $(method)!"))
    end

    new_mean, new_count = _combine_means(a.mean, a.count, b.mean, b.count)
    return Bootstrap{F, T}(f, new_mean, new_count, reps, rep_counts)
end



function value(B::Bootstrap; corrected::Bool=true, replicate_vals=map(B.f, B.replicate_means))
    replicate_mean = mean(replicate_vals)

    if corrected
        return 2*B.f(B.mean) - replicate_mean
    else
        return replicate_mean
    end
end

# abuse of terminology
mean(B::Bootstrap; kw...) = value(B; kw...)


"""Bootstrap estimate of var / N"""
function varN(B::Bootstrap; replicate_vals=map(B.f, B.replicate_means))
    replicate_std = var(replicate_vals; corrected=false)
    prefactor = B.count / (B.count - 1)
    return prefactor * replicate_std
end
var(B::Bootstrap; kw...) = varN(B; kw...) * B.count
std_error(B::Bootstrap; kw...) = sqrt(varN(B; kw...))


function estimate(B::Bootstrap)
    replicate_vals = map(B.f, B.replicate_means)
    (value(B; replicate_vals=replicate_vals)
        ± std_error(B; replicate_vals=replicate_vals))
end


function push!(B::Bootstrap{F, T}, val::T, rng::AbstractRNG=Random.GLOBAL_RNG) where {F, T}
    ns = rand(rng, Poisson(1), nreplicates(B))

    @inbounds for r in 1:nreplicates(B)
        B.replicate_means[r], B.rep_counts[r] = _combine_means(
            val, ns[r], B.replicate_means[r], B.rep_counts[r]
        )
    end

    B.mean, B.count = _combine_means(B.mean, B.count, val, 1)

    return B
end

function append!(B::Bootstrap{F, T}, vals::Vector{T}, rng::AbstractRNG=Random.GLOBAL_RNG) where {F, T}
    for val in vals
        push!(B, val, rng)
    end
    return B
end