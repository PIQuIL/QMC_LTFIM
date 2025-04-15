abstract type AbstractDiagonalUpdateStats; end


struct NoDiagonalUpdateStats <: AbstractDiagonalUpdateStats; end
@inline end_step!(D::NoDiagonalUpdateStats) = D
Base.getproperty(::NoDiagonalUpdateStats, ::Symbol) = NoUpdateStat()
summarize(::NoDiagonalUpdateStats) = NamedTuple()


struct DiagonalUpdateStats{T <: Real} <: AbstractDiagonalUpdateStats
    operator_insertion::UpdateStat{T}
    matelem_insertion::UpdateStat{T}
    removal::UpdateStat{T}
    replace::UpdateStat{T}
    replace_steps::Variance{T}

    DiagonalUpdateStats{T}() where T = new{T}(
        UpdateStat(T), UpdateStat(T), UpdateStat(T), UpdateStat(T), Variance(T)
    )
end
DiagonalUpdateStats(T::Type=Float64) = DiagonalUpdateStats{T}()

#########################################################################################

struct DiagonalUpdateHistograms{T <: Real, R <: StepRangeLen} <: AbstractDiagonalUpdateStats
    operator_insertion::UpdateHistogram{T, R}
    matelem_insertion::UpdateHistogram{T, R}
    removal::UpdateHistogram{T, R}
    replace::UpdateHistogram{T, R}
    replace_steps::VectorHistogram

    DiagonalUpdateHistograms{T}(
        oi::UpdateHistogram{T, R}, mi::UpdateHistogram{T, R}, 
        rm::UpdateHistogram{T, R}, rp::UpdateHistogram{T, R}, 
        rps::VectorHistogram) where {T, R} = new{T, R}(oi, mi, rm, rp, rps)
end
DiagonalUpdateHistograms{T}(size::Int) where T = DiagonalUpdateHistograms{T}(
    UpdateHistogram{T}(size), UpdateHistogram{T}(size), 
    UpdateHistogram{T}(size), UpdateHistogram{T}(size), 
    VectorHistogram(size)
)
DiagonalUpdateHistograms(T::Type, size::Int) = DiagonalUpdateHistograms{T}(size)
DiagonalUpdateHistograms(size::Int) = DiagonalUpdateHistograms{Float64}(size)

#########################################################################################

@inline function end_step!(D::AbstractDiagonalUpdateStats)
    end_step!(D.operator_insertion)
    end_step!(D.matelem_insertion)
    end_step!(D.removal)
    end_step!(D.replace)
    return D
end

summarize(D::AbstractDiagonalUpdateStats) = (
    operator_insertion = summarize(D.operator_insertion),
    matelem_insertion = summarize(D.matelem_insertion),
    removal = summarize(D.removal),
    replace = summarize(D.replace),
    replace_steps = summarize(D.replace_steps),
)
