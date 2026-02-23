# Datei: eval.jl
# Evaluation-Pipeline für ψ-FNO
# Getrennt vom Training – arbeitet auf gespeicherten Modellen.
#
# Funktion              Was sie tut
# predict_sample        Ein Sample: X → model → ψ̂_norm (+ optional denorm, v)
# compute_eval_metrics  ψ-Metriken + optional v-Metriken + Divergenz
# evaluate_dataset      Loop über Dataset → Vector{NamedTuple}
# aggregate_results     Groupby n_crystals → Mittelwerte/Std
# save_results_csv      Ergebnis-Tabelle als CSV
# print_eval_summary    Kompakte Ausgabe: best/worst/overall

module EvalPsi

using Statistics, Random, Printf
using Flux
using JLD2

using ..DatasetPsi: PsiDataset, get_sample
using ..Normalization: denormalize_psi
using ..GridFDUtils: velocity_from_streamfunction, divergence, interior_mask
using ..Losses: mse_loss, rel_l2, max_abs_error, compute_metrics

export predict_sample,
       compute_eval_metrics,
       evaluate_dataset,
       aggregate_results,
       save_results_csv,
       print_eval_summary

# =============================================================================
# E2 – Prediction für ein Sample
# =============================================================================

"""
    predict_sample(model, X; use_gpu=false)

Forward-Pass für ein einzelnes Sample.

- `X`: (nx, nz, Cin) oder (nx, nz, Cin, 1)
- Returns: `ψ̂_norm` als (nx, nz) Matrix
"""
function predict_sample(model, X::AbstractArray; use_gpu::Bool = false)
    # Batch-Dimension hinzufügen falls nötig
    if ndims(X) == 3
        Xb = reshape(X, size(X)..., 1)
    else
        Xb = X
    end

    # Daten aufs selbe Device wie das Modell
    if use_gpu
        Xb = gpu(Xb)
    end

    Flux.testmode!(model)
    ŷ = model(Xb)
    Flux.trainmode!(model)

    # Zurück auf CPU für weitere Verarbeitung
    ŷ_cpu = cpu(ŷ)

    # (nx, nz, 1, 1) → (nx, nz)
    return dropdims(ŷ_cpu; dims=(3, 4))
end

# =============================================================================
# E3 – Eval-Metriken (ψ + optional v)
# =============================================================================

"""
    compute_eval_metrics(ψ̂_norm, ψ_norm, sample_meta;
                         dx=nothing, dz=nothing, mask_width=2)

Berechnet ψ-Metriken im normierten Raum und optional
physikalische v-Metriken + Divergenz.

Returns: flaches NamedTuple (CSV-freundlich).
"""
function compute_eval_metrics(ψ̂_norm::AbstractMatrix,
                              ψ_norm::AbstractMatrix,
                              sample_meta::NamedTuple;
                              dx::Union{Real, Nothing} = nothing,
                              dz::Union{Real, Nothing} = nothing,
                              mask_width::Int = 2)
    nx, nz = size(ψ̂_norm)

    # ── ψ-Metriken (normiert) ──
    ψ̂_4d = reshape(Float32.(ψ̂_norm), nx, nz, 1, 1)
    ψ_4d  = reshape(Float32.(ψ_norm), nx, nz, 1, 1)

    psi_mse   = Float64(mse_loss(ψ̂_4d, ψ_4d))
    psi_rel   = Float64(rel_l2(ψ̂_4d, ψ_4d))
    psi_max   = Float64(max_abs_error(ψ̂_4d, ψ_4d))

    # ── n_crystals (aus Meta extrahieren) ──
    n_cryst = _infer_n_crystals(sample_meta)

    # ── Physikalische Metriken (optional) ──
    v_rel_l2  = NaN
    div_rms   = NaN

    if dx !== nothing && dz !== nothing
        scale  = hasproperty(sample_meta, :scale) ? sample_meta.scale : 1.0
        offset = _get_offset(sample_meta)

        ψ̂_phys = Float64.(ψ̂_norm) ./ scale .+ offset
        ψ_phys  = Float64.(ψ_norm) ./ scale .+ offset

        Vx̂, Vẑ = velocity_from_streamfunction(ψ̂_phys, Float64(dx), Float64(dz))
        Vx,  Vz  = velocity_from_streamfunction(ψ_phys, Float64(dx), Float64(dz))

        mask = interior_mask(nx, nz; width=mask_width)

        speed_true = sqrt.(Vx .^ 2 .+ Vz .^ 2)
        speed_pred = sqrt.(Vx̂ .^ 2 .+ Vẑ .^ 2)
        denom = sqrt(mean(speed_true[mask] .^ 2))
        v_rel_l2 = denom > 0 ? sqrt(mean((speed_pred[mask] .- speed_true[mask]) .^ 2)) / denom : NaN

        div_pred = divergence(Vx̂, Vẑ, Float64(dx), Float64(dz))
        div_rms = sqrt(mean(div_pred[mask] .^ 2))
    end

    return (
        psi_mse     = psi_mse,
        psi_rel_l2  = psi_rel,
        psi_max_err = psi_max,
        v_rel_l2    = v_rel_l2,
        div_rms     = div_rms,
        n_crystals  = n_cryst,
        pred_min    = Float64(minimum(ψ̂_norm)),
        pred_max    = Float64(maximum(ψ̂_norm)),
        target_min  = Float64(minimum(ψ_norm)),
        target_max  = Float64(maximum(ψ_norm)),
    )
end

function _infer_n_crystals(meta::NamedTuple)
    # Versuche n_crystals direkt oder aus centers_2D
    inner = hasproperty(meta, :meta) ? meta.meta : meta
    if hasproperty(inner, :n_crystals)
        return Int(inner.n_crystals)
    elseif hasproperty(inner, :centers_2D)
        return length(inner.centers_2D)
    else
        return -1
    end
end

function _get_offset(meta::NamedTuple)
    inner = hasproperty(meta, :meta) ? meta.meta : meta
    if hasproperty(inner, :psi_gauge_offset)
        return Float64(inner.psi_gauge_offset)
    end
    return 0.0
end

# =============================================================================
# E4 – Dataset Evaluation
# =============================================================================

"""
    evaluate_dataset(model, ds; dx=nothing, dz=nothing, mask_width=2,
                     use_gpu=false, save_dir=nothing, verbose=true)

Evaluiert jedes Sample einzeln. Gibt `Vector{NamedTuple}` zurück.
Optional: speichert Predictions als JLD2 in `save_dir`.
"""
function evaluate_dataset(model, ds::PsiDataset;
                          dx::Union{Real, Nothing} = nothing,
                          dz::Union{Real, Nothing} = nothing,
                          mask_width::Int = 2,
                          use_gpu::Bool = false,
                          save_dir::Union{AbstractString, Nothing} = nothing,
                          verbose::Bool = true)
    save_dir !== nothing && mkpath(save_dir)
    results = NamedTuple[]

    n = length(ds)
    verbose && @info "Evaluiere $n Samples..."

    for i in 1:n
        X, Y, smeta = get_sample(ds, i)
        ψ_norm = dropdims(Y; dims=3)  # (nx, nz)

        ψ̂_norm = predict_sample(model, X; use_gpu=use_gpu)

        metrics = compute_eval_metrics(ψ̂_norm, ψ_norm, smeta;
                                       dx=dx, dz=dz, mask_width=mask_width)

        row = merge(metrics, (sample_idx=i, filepath=smeta.filepath))
        push!(results, row)

        # Optional: Prediction speichern
        if save_dir !== nothing
            fname = "pred_$(lpad(i, 6, '0')).jld2"
            jldsave(joinpath(save_dir, fname);
                    ψ_pred_norm=ψ̂_norm, ψ_true_norm=ψ_norm, metrics=metrics,
                    sample_meta=smeta)
        end

        if verbose && i % max(1, n ÷ 10) == 0
            @info @sprintf("  [%d/%d] rel_l2=%.4f max_err=%.4e",
                           i, n, metrics.psi_rel_l2, metrics.psi_max_err)
        end
    end

    return results
end

# =============================================================================
# E5 – Aggregation nach Kristallanzahl
# =============================================================================

"""
    aggregate_results(results; group_key=:n_crystals)

Gruppiert Ergebnisse nach `group_key` und berechnet Mittelwerte + Std.
Gibt `Vector{NamedTuple}` zurück (eine Zeile pro Gruppe).
"""
function aggregate_results(results::Vector{<:NamedTuple};
                           group_key::Symbol = :n_crystals)
    # Gruppieren
    groups = Dict{Any, Vector{NamedTuple}}()
    for r in results
        key = get(r, group_key, :unknown)
        push!(get!(groups, key, NamedTuple[]), r)
    end

    # Metriken aggregieren
    agg = NamedTuple[]
    metric_keys = [:psi_mse, :psi_rel_l2, :psi_max_err, :v_rel_l2, :div_rms]

    for gk in sort(collect(keys(groups)))
        group = groups[gk]
        n = length(group)

        row = Dict{Symbol, Any}(group_key => gk, :n_samples => n)
        for mk in metric_keys
            vals = [Float64(get(r, mk, NaN)) for r in group]
            vals_clean = filter(!isnan, vals)
            if !isempty(vals_clean)
                row[Symbol(mk, :_mean)] = mean(vals_clean)
                row[Symbol(mk, :_std)]  = length(vals_clean) > 1 ? std(vals_clean) : 0.0
            else
                row[Symbol(mk, :_mean)] = NaN
                row[Symbol(mk, :_std)]  = NaN
            end
        end

        push!(agg, NamedTuple{Tuple(keys(row))}(values(row)))
    end

    return agg
end

# =============================================================================
# E6 – CSV Export
# =============================================================================

"""
    save_results_csv(path, results)

Speichert `Vector{NamedTuple}` als CSV.
"""
function save_results_csv(path::AbstractString, results::Vector{<:NamedTuple})
    isempty(results) && return

    open(path, "w") do io
        # Header
        ks = keys(results[1])
        println(io, join(string.(ks), ","))

        # Rows
        for r in results
            vals = [_fmt(v) for v in values(r)]
            println(io, join(vals, ","))
        end
    end
    @info "Ergebnisse gespeichert: $path ($(length(results)) Zeilen)"
end

_fmt(v::AbstractFloat) = @sprintf("%.8e", v)
_fmt(v::Integer) = string(v)
_fmt(v) = string(v)

# =============================================================================
# E7 – Zusammenfassung
# =============================================================================

"""
    print_eval_summary(results; n_show=5)

Gibt eine kompakte Zusammenfassung aus: Overall + Best/Worst Samples.
"""
function print_eval_summary(results::Vector{<:NamedTuple}; n_show::Int = 5)
    n = length(results)
    n == 0 && return @warn "Keine Ergebnisse"

    rel_l2s = [Float64(r.psi_rel_l2) for r in results]

    println("=" ^ 60)
    @printf("Evaluation Summary (%d Samples)\n", n)
    println("=" ^ 60)
    @printf("  ψ rel_l2:   mean=%.4f  std=%.4f  min=%.4f  max=%.4f\n",
            mean(rel_l2s), std(rel_l2s), minimum(rel_l2s), maximum(rel_l2s))

    mses = [Float64(r.psi_mse) for r in results]
    @printf("  ψ MSE:      mean=%.4e  max=%.4e\n", mean(mses), maximum(mses))

    max_errs = [Float64(r.psi_max_err) for r in results]
    @printf("  ψ max_err:  mean=%.4e  max=%.4e\n", mean(max_errs), maximum(max_errs))

    v_l2s = filter(!isnan, [Float64(r.v_rel_l2) for r in results])
    if !isempty(v_l2s)
        @printf("  v rel_l2:   mean=%.4f  std=%.4f\n", mean(v_l2s), std(v_l2s))
    end

    # Best / Worst
    sorted = sortperm(rel_l2s)
    n_s = min(n_show, n)

    println("\n  Best $n_s:")
    for i in sorted[1:n_s]
        r = results[i]
        @printf("    #%d  rel_l2=%.4f  n_cryst=%d  %s\n",
                i, r.psi_rel_l2, r.n_crystals, basename(string(r.filepath)))
    end

    println("  Worst $n_s:")
    for i in sorted[end-n_s+1:end]
        r = results[i]
        @printf("    #%d  rel_l2=%.4f  n_cryst=%d  %s\n",
                i, r.psi_rel_l2, r.n_crystals, basename(string(r.filepath)))
    end
    println("=" ^ 60)
end

end # module
