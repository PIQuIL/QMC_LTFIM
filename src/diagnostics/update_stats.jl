using OnlineStats

std_error(v::Variance) = sqrt(var(v) / nobs(v))
summarize(V::Variance) = (mean=mean(V), error=std_error(V))

#########################################################################################

# Construct a Frequency Histogram using a Vector
#   The index of the vector will be the values, hence we require that inputs be non-negative.
#   We shift the val index up by one to handle events with val=0.
#   As a result, we need to subtract 1 from the index when computing statistics
struct VectorHistogram
    hist::Vector{Int}
    VectorHistogram(size::Int) = new(zeros(Int, size))
end

function OnlineStats.fit!(V::VectorHistogram, val::Int)
    hist = V.hist
    size_diff = val+1 - length(hist)
    if size_diff > 0
        append!(hist, zeros(Int, 2size_diff))
    end
    hist[val+1] += 1

    return V
end

OnlineStats.nobs(V::VectorHistogram) = sum(V.hist)
mean(V::VectorHistogram) = sum(w*(val-1) for (val, w) in enumerate(V.hist)) / nobs(V)
function var(V::VectorHistogram)
    m = mean(V)
    va = sum(w*((val-1 - m)^2) for (val, w) in enumerate(V.hist))
    
    return va / (nobs(V) - 1)
end
std(V::VectorHistogram) = sqrt(var(V))
std_error(V::VectorHistogram) = sqrt(var(V) / nobs(V))
summarize(V::VectorHistogram) = (mean=mean(V), error=std_error(V))

# update V_a with data from V_b
function OnlineStats.merge!(V_a::VectorHistogram, V_b::VectorHistogram)
    size_diff = length(V_b.hist) - length(V_a.hist)
    if size_diff > 0
        append!(V_a.hist, zeros(Int, size_diff))
    end
    for i in eachindex(V_a.hist, V_b.hist)
        V_a.hist[i] += V_b.hist[i]
    end
    return V_a
end

#########################################################################################

std_error(h::Hist) = sqrt(var(h) / nobs(h))
summarize(h::Hist) = (mean=mean(h), error=std_error(h))

#########################################################################################

mutable struct SuccessRate{T <: Real}
    success_count::Int
    fail_count::Int

    success_rate::T
    fail_rate::T

    SuccessRate{T}() where T = new{T}(0, 0, zero(T), zero(T))
end
SuccessRate(T::Type) = SuccessRate{T}()

@inline add_success!(s::SuccessRate) = (s.success_count += 1; s) 
@inline add_fail!(s::SuccessRate) = (s.fail_count += 1; s)
function update_rate!(s::SuccessRate)
    total_count = s.success_count + s.fail_count
    if total_count > 0
        s.success_rate = s.success_count/total_count
        s.fail_rate = s.fail_count/total_count
    end
    return s
end
function empty!(s::SuccessRate{T}) where T
    s.success_count, s.fail_count = 0, 0
    s.success_rate, s.fail_rate = zero(T), zero(T)
    return s
end

# update s with data from r
function OnlineStats.merge!(s::SuccessRate, r::SuccessRate)
    s.success_count += r.success_count
    s.fail_count += r.fail_count
    update_rate!(s)
end

function summarize(s::SuccessRate)
    update_rate!(s)
    return (
        success_count = s.success_count,
        fail_count = s.fail_count,
        
        success_rate = s.success_rate,
        fail_rate = s.fail_rate,
    )
end


#########################################################################################

abstract type AbstractUpdateStat; end

struct NoUpdateStat <: AbstractUpdateStat; end
@inline add_prob!(stat::NoUpdateStat, p::T) where {T <: Real} = stat
@inline add_success!(stat::NoUpdateStat) = stat
@inline add_fail!(stat::NoUpdateStat) = stat
@inline add_unit_prob!(stat::NoUpdateStat) = stat
@inline end_step!(stat::NoUpdateStat) = stat
summarize(::NoUpdateStat) = NamedTuple()

OnlineStats.merge!(S_a::NoUpdateStat, S_b::NoUpdateStat) = S_a
OnlineStats.merge!(S_a::NoUpdateStat, S_b::AbstractUpdateStat) = deepcopy(S_b)
OnlineStats.merge!(S_a::AbstractUpdateStat, S_b::NoUpdateStat) = S_a

#########################################################################################

struct UpdateStat{T} <: AbstractUpdateStat
    prob::Variance{T}

    success_rate_step::Variance{T}
    fail_rate_step::Variance{T}

    nsuccess_step::Variance{T}
    nfail_step::Variance{T}

    step_success_rate::SuccessRate{T}
    total_success_rate::SuccessRate{T}
    
    UpdateStat{T}() where T = new{T}(
        Variance(T),
        Variance(T), Variance(T),
        Variance(T), Variance(T),
        SuccessRate(T), SuccessRate(T))
end
UpdateStat() = UpdateStat{Float64}()
UpdateStat(T::Type) = UpdateStat{T}()

eltype(::Type{UpdateStat{T}}) where T = T

#########################################################################################

struct UpdateHistogram{T, R <: StepRangeLen} <: AbstractUpdateStat
    prob::Hist{T, R}

    success_rate_step::Hist{T, R}
    fail_rate_step::Hist{T, R}

    nsuccess_step::VectorHistogram
    nfail_step::VectorHistogram
    step_success_rate::SuccessRate{T}
    total_success_rate::SuccessRate{T}
    
    UpdateHistogram{T}(
            ph::Hist{T, R}, sh::Hist{T, R}, fh::Hist{T, R}, 
            ns::VectorHistogram, nf::VectorHistogram, 
            ssr::SuccessRate{T}, tsr::SuccessRate{T}) where {T, R} = new{T, R}(
        ph, sh, fh, 
        ns, nf,
        ssr, tsr        
    )
end
UpdateHistogram{T}(size::Int) where T = UpdateHistogram{T}(
    Hist(0:(1/size):1, T),
    Hist(0:(1/size):1, T), Hist(0:(1/size):1, T),
    VectorHistogram(size), VectorHistogram(size),
    SuccessRate(T), SuccessRate(T))
UpdateHistogram(size::Int) = UpdateHistogram{Float64}(size::Int)
UpdateHistogram(T::Type, size::Int) = UpdateHistogram{T}(size::Int)

eltype(::Type{UpdateHistogram{T, R}}) where {T, R} = T


#########################################################################################

# update U_a with data from U_b
function OnlineStats.merge!(U_a::U, U_b::U) where {U <: AbstractUpdateStat}
    merge!(U_a.prob, U_b.prob)
    
    merge!(U_a.success_rate_step, U_b.success_rate_step)
    merge!(U_a.fail_rate_step, U_b.fail_rate_step)

    merge!(U_a.nsuccess_step, U_b.nsuccess_step)
    merge!(U_a.nfail_step, U_b.nfail_step)
    merge!(U_a.step_success_rate, U_b.step_success_rate)
    merge!(U_a.total_success_rate, U_b.total_success_rate)

    return U_a
end


@inline add_prob!(stat::AbstractUpdateStat, p::T) where {T <: Real} = (fit!(stat.prob, p); stat)
@inline add_success!(stat::AbstractUpdateStat) = (add_success!(stat.step_success_rate); stat)
@inline add_fail!(stat::AbstractUpdateStat) = (add_fail!(stat.step_success_rate); stat)
@inline add_unit_prob!(stat::AbstractUpdateStat) = (add_prob!(stat, one(eltype(stat))); add_success!(stat))

@inline function end_step!(stat::AbstractUpdateStat)
    rate = update_rate!(stat.step_success_rate)
    
    fit!(stat.nsuccess_step, rate.success_count)
    fit!(stat.nfail_step, rate.fail_count)
    
    fit!(stat.success_rate_step, rate.success_rate)
    fit!(stat.fail_rate_step, rate.fail_rate)

    merge!(stat.total_success_rate, rate)
    empty!(rate)
    return stat
end

summarize(stat::AbstractUpdateStat) = (
    success_probability = summarize(stat.prob),

    single_step_success_rate = summarize(stat.success_rate_step),
    single_step_fail_rate = summarize(stat.fail_rate_step),
    
    single_step_success_count = summarize(stat.nsuccess_step),
    single_step_fail_count = summarize(stat.nfail_step),

    NamedTuple(Symbol("overall_", k) => v for (k,v) in pairs(summarize(stat.total_success_rate)))...
)


function check_prob(rng::AbstractRNG, prob::T, stat::AbstractUpdateStat) where {T <: Real}
    if prob >= one(T)
        add_unit_prob!(stat)
        return true
    else
        add_prob!(stat, prob)
        if rand(rng) < prob
            add_success!(stat)
            return true
        else
            add_fail!(stat) 
            return false
        end
    end
end

function check_logprob(rng::AbstractRNG, logprob::T, stat::AbstractUpdateStat) where {T <: Real}
    if logprob >= zero(T)
        add_unit_prob!(stat)
        return true
    else
        prob = exp(logprob)
        add_prob!(stat, prob)
        if rand(rng) < prob
            add_success!(stat)
            return true
        else
            add_fail!(stat)
            return false
        end
    end
end

