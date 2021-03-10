mutable struct Mean{T}
    mean::T
    count::Int

    Mean{T}(mean::T, count::Int) where T = new{T}(mean, count)
    Mean(mean::T, count::Int) where T = new{T}(mean, count)

    Mean{T}(zero_element::T=zero(T)) where T = new{T}(zero_element, zero(Int))
end
Mean(zero_element::T) where T = Mean{T}(zero_element)
Mean(T::Type) = Mean{T}()
Mean() = Mean{Float64}()

mean(m::Mean) = m.mean
count(m::Mean) = m.count

zero(m::Mean{T}) where T = Mean{T}(zero(mean(m)), count(m))

function push!(m::Mean{T}, val::T, k::Int=1) where T
    m.count += k
    m.mean += k * (val - m.mean) / m.count
    return m
end
function append!(m::Mean{T}, vals::AbstractVector{T}) where T
    vm = Mean{T}(mean(vals), length(vals))
    combine!(m, vm)
end

isempty(m::Mean) = iszero(m.count)
function empty!(m::Mean)
    m.mean = zero(m.mean)
    m.count = zero(m.count)
    return m
end

function combine!(a::Mean{T}, b::Mean{T}) where T
    isempty(a) && isempty(b) && return a
    s = a.mean*a.count + b.mean*b.count
    a.count += b.count
    a.mean = s / a.count
    return a
end
function combine(a::Mean{T}, b::Mean{T}) where T
    isempty(a) && isempty(b) && return deepcopy(a)
    n = a.count + b.count
    new_mean = (a.mean*a.count + b.mean*b.count) / n
    return Mean{T}(new_mean, n)
end




struct Bootstrap{F <: Function, T, R <: AbstractRNG} <: BinningAnalysis.AbstractVarianceAccumulator{T}
    f::F

    mean::Mean{T}
    replicate_means::Vector{Mean{T}}

    buffer::Vector{T}

    rng_buffer::Vector{Int}
    rng::R

    function Bootstrap{F, T, R}(f::F, mean::Mean{T}, replicate_means::Vector{Mean{T}},
                                buffer::Vector{T}, rng_buffer::Vector{Int},
                                rng::R=Random.GLOBAL_RNG) where {F, T, R}
        @assert length(replicate_means) == length(buffer) == length(rng_buffer)
        return new{F, T, R}(f, mean, replicate_means, buffer, rng_buffer, rng)
    end

    function Bootstrap{F, T}(f::F; rng::AbstractRNG=Random.GLOBAL_RNG, nreps=500) where {F, T}
        return new{typeof(f), T, typeof(rng)}(
            f,
            Mean{T}(), [Mean{T}() for _ in 1:nreps],
            zeros(T, nreps),
            zeros(Int, nreps), rng
        )
    end

    function Bootstrap{F, T}(f::F, zero_element::T; rng::AbstractRNG=Random.GLOBAL_RNG, nreps=500) where {F, T}
        return new{typeof(f), T, typeof(rng)}(
            f,
            Mean{T}(copy(zero_element)),
            [Mean{T}(copy(zero_element)) for _ in 1:nreps],
            [copy(zero_element) for _ in 1:nreps],
            zeros(Int, nreps), rng
        )
    end
end

Bootstrap{F}(f::F, zero_element::T; kwargs...) where {F, T} =
    Bootstrap{F, T}(f, zero_element; kwargs...)
Bootstrap(f::F, zero_element::T; kwargs...) where {F, T} =
    Bootstrap{F, T}(f, zero_element; kwargs...)
Bootstrap(f::F, T::Type; kwargs...) where F = Bootstrap{F, T}(f; kwargs...)
Bootstrap(f::F; kwargs...) where F = Bootstrap{F, Float64}(f; kwargs...)
Bootstrap(T::Type; kwargs...) = Bootstrap(identity, T; kwargs...)
Bootstrap(; kwargs...) = Bootstrap(identity; kwargs...)


count(B::Bootstrap) = count(B.mean)
nreplicates(B::Bootstrap) = length(B.replicate_means)
isempty(B::Bootstrap) = iszero(count(B))

function Base.getproperty(B::Bootstrap, property::Symbol)
    if property === :count
        return count(B)
    else
        return getfield(B, property)
    end
end

streamingmode(B::Bootstrap) = count(B) >= nreplicates(B)


function combine!(a::Bootstrap{F, T}, b::Bootstrap{F, T}; method::Symbol=:merge) where {F, T}
    if method === :merge
        @assert nreplicates(a) == nreplicates(b)

        if streamingmode(b)
            make_bins!(a)
            @inbounds for i in 1:nreplicates(a)
                combine!(a.replicate_means[i], b.replicate_means[i])
            end
            combine!(a.mean, b.mean)
        elseif !streamingmode(b)
            append!(a, b.buffer)
        end
    elseif method === :concat
        append!(a.replicate_means, b.replicate_means)
        append!(a.buffer, b.buffer)
        append!(a.rng_buffer, b.rng_buffer)

        combine!(a.mean, b.mean)
    else
        throw(ArgumentError("Unrecognized combining method: $(method)!"))
    end

    return a
end



function value(B::Bootstrap; corrected::Bool=true, replicate_vals=get_replicates!(B))
    replicate_mean = mean(replicate_vals)

    if corrected
        return 2*B.f(mean(B.mean)) - replicate_mean
    else
        return replicate_mean
    end
end

# abuse of terminology
mean(B::Bootstrap; kw...) = value(B; kw...)


"""Bootstrap estimate of var / N"""
varN(B::Bootstrap; replicate_vals=get_replicates!(B)) = var(replicate_vals)
var(B::Bootstrap; kw...) = varN(B; kw...) * count(B)
std_error(B::Bootstrap; kw...) = sqrt(varN(B; kw...))



function _make_bins!(B::Bootstrap)
    L = count(B)
    @inbounds for i in 1:nreplicates(B)
        rand!(B.rng, @view(B.rng_buffer[1:L]), 1:L)
        B.replicate_means[i] = Mean(mean(@view B.buffer[B.rng_buffer[1:L]]), L)
    end
end
make_bins!(B::Bootstrap) = streamingmode(B) ? nothing : _make_bins!(B)

function get_replicates!(B::Bootstrap{F, T})::Vector{T} where {F, T}
    isempty(B) && throw(ArgumentError("Can't perform bootstrapping without data!"))
    make_bins!(B)
    return @. B.f(mean(B.replicate_means))
end


function push!(B::Bootstrap{F, T}, val::T) where {F, T}
    if streamingmode(B)
        for replicate in B.replicate_means
            push!(replicate, val, rand(B.rng, Poisson(1)))
        end
        push!(B.mean, val)
        return B
    else
        push!(B.mean, val)
        B.buffer[count(B)] = val
        if count(B) == nreplicates(B)
            _make_bins!(B)
            empty!(B.buffer)
            empty!(B.rng_buffer)
        end
        return B
    end
end

append!(B::Bootstrap{F, T}, vals::Vector{T}) where {F, T} =
    (foreach(v -> push!(B, v), vals); B)


function BinningAnalysis.LogBinner(f::Function, x::T;
        capacity::Int64 = BinningAnalysis._nlvls2capacity(32), kw...
    ) where {T <: Union{Number, AbstractArray}}

    # check keyword args
    capacity <= 0 && throw(ArgumentError("`capacity` must be finite and positive."))

    # got_timeseries = didn't receive a zero && is a vector
    got_timeseries = count(!iszero, x) > 0 && ndims(T) == 1

    if got_timeseries
        # x = time series
        N = BinningAnalysis._capacity2nlvls(length(x))
        S = BinningAnalysis._sum_type_heuristic(eltype(T), ndims(x[1]))
        el = zero(x[1])
    else
        # x = zero_element
        N = BinningAnalysis._capacity2nlvls(capacity)
        S = BinningAnalysis._sum_type_heuristic(T, ndims(x))
        el = x
    end

    boot = Bootstrap{typeof(f), S}(f, zero(el); kw...)
    B = LogBinner{S, N}(
        tuple([BinningAnalysis.Compressor{S}(copy(el), false) for i in 1:N]...),
        tuple([deepcopy(boot) for _ in 1:N]...)
    )

    got_timeseries && append!(B, x)

    return B
end
