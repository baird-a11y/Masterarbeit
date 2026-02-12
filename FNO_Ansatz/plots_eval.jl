# Datei: plots_eval.jl
# Visualisierung für ψ-FNO Evaluation
# Getrennt von Training/Eval – arbeitet auf gespeicherten Ergebnissen.
#
# Funktion                  Was sie tut
# plot_training_history     Training-CSV → Loss-Kurven (MSE, rel_l2, grad_mse)
# plot_psi_comparison       3-Panel: ψ_true | ψ_pred | Error
# plot_error_map            Fehlerkarte mit Statistik-Overlay
# plot_velocity_comparison  |v| true | |v| pred | |v| error
# plot_divergence_map       Divergenzkarte mit RMS
# plot_metrics_vs_crystals  Metriken vs. Kristallanzahl (Generalisierung)
# make_eval_gallery         Batch: n Beispiele → Plots automatisch erzeugen
# common_clim               Robuste gemeinsame Farbskala (Quantile)
# speed                     |v| = √(Vx² + Vz²)
# safe_title                Titelstring aus Meta-Daten

module PlotsEval

using Plots
using Plots.Measures
using Statistics
using Printf
using DelimitedFiles   # CSV lesen (stdlib, kein Extra-Paket)
using JLD2

using ..GridFDUtils: velocity_from_streamfunction, divergence, interior_mask

export plot_training_history,
       plot_psi_comparison,
       plot_velocity_comparison,
       plot_divergence_map,
       plot_metrics_vs_crystals,
       make_eval_gallery,
       common_clim,
       speed,
       safe_title

# =============================================================================
# P8 – Helper-Funktionen
# =============================================================================

"""
    common_clim(A, B; q=0.995)

Bestimmt gemeinsame Farbskala robust via Quantile (statt min/max).
Gibt `(-cmax, cmax)` für symmetrische Skala zurück.
"""
function common_clim(A::AbstractMatrix, B::AbstractMatrix; q::Real = 0.995)
    all_vals = vcat(vec(A), vec(B))
    all_vals = filter(!isnan, all_vals)
    isempty(all_vals) && return (-1.0, 1.0)
    sorted = sort(abs.(all_vals))
    idx = min(length(sorted), max(1, round(Int, q * length(sorted))))
    cmax = sorted[idx]
    cmax = cmax > 0 ? cmax : 1.0
    return (-cmax, cmax)
end

"""
    speed(Vx, Vz)

Geschwindigkeitsbetrag: |v| = √(Vx² + Vz²).
"""
speed(Vx, Vz) = sqrt.(Vx .^ 2 .+ Vz .^ 2)

"""
    safe_title(meta; max_len=60)

Baut einen kompakten Titelstring aus Meta-Daten.
"""
function safe_title(meta::NamedTuple; max_len::Int = 60)
    parts = String[]
    inner = hasproperty(meta, :meta) ? meta.meta : meta

    if hasproperty(inner, :n_crystals)
        push!(parts, "n=$(inner.n_crystals)")
    end
    if hasproperty(inner, :filepath)
        push!(parts, basename(string(inner.filepath)))
    end

    title = join(parts, " | ")
    return length(title) > max_len ? title[1:max_len] * "…" : title
end

# =============================================================================
# P1 – Training History Plot
# =============================================================================

"""
    plot_training_history(history_csv; out_path="loss_plot.png")

Liest Training-CSV und erzeugt Loss-Kurven.
Plots: train_mse/val_mse (log-scale), val_rel_l2, grad_norm.
"""
function plot_training_history(history_csv::AbstractString;
                               out_path::AbstractString = "loss_plot.png")
    # CSV manuell lesen
    lines = readlines(history_csv)
    isempty(lines) && error("CSV ist leer: $history_csv")

    header = Symbol.(strip.(split(lines[1], ",")))
    n_rows = length(lines) - 1
    data = Dict{Symbol, Vector{Float64}}()
    for h in header
        data[h] = Float64[]
    end

    for i in 2:length(lines)
        vals = strip.(split(lines[i], ","))
        for (j, h) in enumerate(header)
            v = tryparse(Float64, vals[j])
            push!(data[h], v === nothing ? NaN : v)
        end
    end

    epochs = haskey(data, :epoch) ? data[:epoch] : collect(1.0:n_rows)

    # Nur den letzten Trainingslauf nehmen (falls CSV mehrere Runs enthält)
    if haskey(data, :epoch)
        # Finde letzten Neustart: wo epoch[i] <= epoch[i-1]
        last_start = 1
        for i in 2:length(epochs)
            if epochs[i] <= epochs[i-1]
                last_start = i
            end
        end
        if last_start > 1
            for h in header
                data[h] = data[h][last_start:end]
            end
            epochs = data[:epoch]
            n_rows = length(epochs)
        end
    end

    # ── Plot 1: MSE (log scale) ──
    p1 = plot(; title="Loss (MSE)", xlabel="Epoch", ylabel="MSE",
              yscale=:log10, legend=:topright, grid=true)
    if haskey(data, :train_mse)
        plot!(p1, epochs, data[:train_mse]; label="train", lw=2)
    end
    if haskey(data, :val_mse)
        plot!(p1, epochs, data[:val_mse]; label="val", lw=2, ls=:dash)
    end

    # ── Plot 2: rel_l2 ──
    p2 = plot(; title="Relative L2", xlabel="Epoch", ylabel="rel_l2",
              legend=:topright, grid=true)
    if haskey(data, :val_rel_l2)
        plot!(p2, epochs, data[:val_rel_l2]; label="val rel_l2", lw=2, color=:red)
    end

    # ── Plot 3: Grad Norm ──
    p3 = plot(; title="Gradient Norm", xlabel="Epoch", ylabel="‖∇‖",
              legend=:topright, grid=true)
    if haskey(data, :grad_norm)
        plot!(p3, epochs, data[:grad_norm]; label="grad_norm", lw=1.5, color=:green)
    end

    # ── Plot 4: Alpha Schedule ──
    p4 = plot(; title="α_grad Schedule", xlabel="Epoch", ylabel="α",
              legend=:topright, grid=true)
    if haskey(data, :alpha)
        plot!(p4, epochs, data[:alpha]; label="α_grad", lw=1.5, color=:purple)
    end

    fig = plot(p1, p2, p3, p4; layout=(2, 2), size=(1200, 800))
    savefig(fig, out_path)
    @info "Training-History gespeichert: $out_path"
    return fig
end

# =============================================================================
# P2 – ψ Comparison (True | Pred | Error)
# =============================================================================

"""
    plot_psi_comparison(ψ_true, ψ_pred; out_path="psi_comparison.png",
                        title="", clim=nothing)

3-Panel Vergleich: ψ_true | ψ_pred | Error.
Gleiche Farbskala für true/pred, eigene für Error.
"""
function plot_psi_comparison(ψ_true::AbstractMatrix, ψ_pred::AbstractMatrix;
                             out_path::AbstractString = "psi_comparison.png",
                             title::AbstractString = "",
                             clim = nothing)
    err = ψ_pred .- ψ_true

    # Farbskala
    cl = clim !== nothing ? clim : common_clim(ψ_true, ψ_pred)

    # Error-Skala (symmetrisch)
    emax = maximum(abs.(filter(!isnan, vec(err))))
    emax = emax > 0 ? emax : 1.0
    ecl = (-emax, emax)

    p1 = heatmap(ψ_true'; title="ψ true", clim=cl, color=:RdBu,
                 aspect_ratio=:equal, xlabel="x", ylabel="z")
    p2 = heatmap(ψ_pred'; title="ψ pred", clim=cl, color=:RdBu,
                 aspect_ratio=:equal, xlabel="x", ylabel="z")
    p3 = heatmap(err'; title="Error (pred−true)", clim=ecl, color=:RdBu,
                 aspect_ratio=:equal, xlabel="x", ylabel="z")

    sup_title = isempty(title) ? "ψ Comparison" : title
    fig = plot(p1, p2, p3; layout=(1, 3), size=(1500, 400),
              plot_title=sup_title)
    savefig(fig, out_path)
    @info "ψ-Comparison gespeichert: $out_path"
    return fig
end

# =============================================================================
# P4 – Velocity Comparison
# =============================================================================

"""
    plot_velocity_comparison(Vx_true, Vz_true, Vx_pred, Vz_pred;
                             out_path="velocity_comparison.png", title="")

3-Panel Vergleich der Geschwindigkeitsbeträge: |v| true | |v| pred | |v| error.
"""
function plot_velocity_comparison(Vx_true::AbstractMatrix, Vz_true::AbstractMatrix,
                                  Vx_pred::AbstractMatrix, Vz_pred::AbstractMatrix;
                                  out_path::AbstractString = "velocity_comparison.png",
                                  title::AbstractString = "")
    s_true = speed(Vx_true, Vz_true)
    s_pred = speed(Vx_pred, Vz_pred)
    s_err  = s_pred .- s_true

    vmax = max(maximum(filter(!isnan, vec(s_true))),
               maximum(filter(!isnan, vec(s_pred))))
    vmax = vmax > 0 ? vmax : 1.0

    p1 = heatmap(s_true'; title="|v| true", clim=(0, vmax), color=:viridis,
                 aspect_ratio=:equal, xlabel="x", ylabel="z")
    p2 = heatmap(s_pred'; title="|v| pred", clim=(0, vmax), color=:viridis,
                 aspect_ratio=:equal, xlabel="x", ylabel="z")

    emax = maximum(abs.(filter(!isnan, vec(s_err))))
    emax = emax > 0 ? emax : 1.0
    p3 = heatmap(s_err'; title="|v| error", clim=(-emax, emax), color=:RdBu,
                 aspect_ratio=:equal, xlabel="x", ylabel="z")

    sup_title = isempty(title) ? "Velocity Comparison" : title
    fig = plot(p1, p2, p3; layout=(1, 3), size=(1500, 400),
              plot_title=sup_title)
    savefig(fig, out_path)
    @info "Velocity-Comparison gespeichert: $out_path"
    return fig
end

# =============================================================================
# P5 – Divergence Map
# =============================================================================

"""
    plot_divergence_map(divv; out_path="divergence_map.png", title="",
                        mask_width=2)

Heatmap der Divergenz mit RMS im Inneren.
"""
function plot_divergence_map(divv::AbstractMatrix;
                             out_path::AbstractString = "divergence_map.png",
                             title::AbstractString = "",
                             mask_width::Int = 2)
    nx, nz = size(divv)
    mask = interior_mask(nx, nz; width=mask_width)
    div_clean = filter(!isnan, vec(divv[mask]))

    rms = isempty(div_clean) ? NaN : sqrt(mean(div_clean .^ 2))

    t = isempty(title) ? "Divergence" : title
    t = @sprintf("%s | RMS_interior=%.3e", t, rms)

    dmax = maximum(abs.(filter(!isnan, vec(divv))))
    dmax = dmax > 0 ? dmax : 1.0

    fig = heatmap(divv'; title=t, clim=(-dmax, dmax), color=:RdBu,
                  aspect_ratio=:equal, xlabel="x", ylabel="z",
                  size=(600, 500))
    savefig(fig, out_path)
    @info "Divergence-Map gespeichert: $out_path"
    return fig
end

# =============================================================================
# P6 – Metrics vs. Crystals
# =============================================================================

"""
    plot_metrics_vs_crystals(agg_csv; out_path="metrics_vs_crystals.png")

Plottet aggregierte Metriken vs. Kristallanzahl mit Fehlerbalken.
"""
function plot_metrics_vs_crystals(agg_csv::AbstractString;
                                  out_path::AbstractString = "metrics_vs_crystals.png")
    lines = readlines(agg_csv)
    length(lines) < 2 && error("CSV zu kurz: $agg_csv")

    header = Symbol.(strip.(split(lines[1], ",")))
    data = Dict{Symbol, Vector{Float64}}()
    for h in header
        data[h] = Float64[]
    end
    for i in 2:length(lines)
        vals = strip.(split(lines[i], ","))
        for (j, h) in enumerate(header)
            v = tryparse(Float64, vals[j])
            push!(data[h], v === nothing ? NaN : v)
        end
    end

    n_cryst = haskey(data, :n_crystals) ? data[:n_crystals] : collect(1.0:length(lines)-1)

    # ── Plot 1: rel_l2 ──
    p1 = plot(; title="ψ rel_l2 vs. Crystals", xlabel="n_crystals",
              ylabel="rel_l2 (mean ± std)", legend=:topleft, grid=true)
    if haskey(data, :psi_rel_l2_mean)
        y = data[:psi_rel_l2_mean]
        yerr = haskey(data, :psi_rel_l2_std) ? data[:psi_rel_l2_std] : zeros(length(y))
        scatter!(p1, n_cryst, y; yerror=yerr, label="rel_l2", ms=6, color=:blue)
        plot!(p1, n_cryst, y; lw=1.5, color=:blue, label="")
    end

    # ── Plot 2: MSE (log scale, keine yerror weil log+symmetric error bars buggy) ──
    p2 = plot(; title="ψ MSE vs. Crystals", xlabel="n_crystals",
              ylabel="MSE (mean)", yscale=:log10, legend=:topleft, grid=true)
    if haskey(data, :psi_mse_mean)
        y = data[:psi_mse_mean]
        scatter!(p2, n_cryst, y; label="MSE", ms=6, color=:red)
        plot!(p2, n_cryst, y; lw=1.5, color=:red, label="")
        # Std als Band auf log-Skala: obere/untere Grenze
        if haskey(data, :psi_mse_std)
            y_lo = max.(y .- data[:psi_mse_std], y .* 0.1)  # min 10% vom Mittelwert
            y_hi = y .+ data[:psi_mse_std]
            plot!(p2, n_cryst, y_lo; fillrange=y_hi, fillalpha=0.2,
                  color=:red, label="±1σ", lw=0)
        end
    end

    # ── Plot 3: v rel_l2 ──
    p3 = plot(; title="v rel_l2 vs. Crystals", xlabel="n_crystals",
              ylabel="v_rel_l2 (mean)", legend=:topleft, grid=true)
    if haskey(data, :v_rel_l2_mean)
        y = data[:v_rel_l2_mean]
        valid = .!isnan.(y)
        if any(valid)
            yerr = haskey(data, :v_rel_l2_std) ? data[:v_rel_l2_std] : zeros(length(y))
            scatter!(p3, n_cryst[valid], y[valid]; yerror=yerr[valid],
                     label="v_rel_l2", ms=6, color=:green)
            plot!(p3, n_cryst[valid], y[valid]; lw=1.5, color=:green, label="")
        end
    end

    # ── Plot 4: Samples pro Gruppe ──
    p4 = plot(; title="Samples per Group", xlabel="n_crystals",
              ylabel="n_samples", legend=false, grid=true)
    if haskey(data, :n_samples)
        bar!(p4, n_cryst, data[:n_samples]; color=:gray, alpha=0.7)
        ylims!(p4, (0, maximum(data[:n_samples]) * 1.3))
    end

    fig = plot(p1, p2, p3, p4; layout=(2, 2), size=(1200, 900),
              margin=5mm)
    savefig(fig, out_path)
    @info "Metrics vs. Crystals gespeichert: $out_path"
    return fig
end

# =============================================================================
# P7 – Gallery / Report-Bilder automatisch erzeugen
# =============================================================================

"""
    make_eval_gallery(pred_dir, out_dir; n=0, pick=:worst,
                      dx=nothing, dz=nothing, mask_width=2)

Lädt gespeicherte Predictions (JLD2 aus eval.jl) und erzeugt
pro Sample: ψ-comparison + optional velocity + divergence.

Plots werden in Unterordner nach Kristallanzahl aufgeteilt:
  out_dir/n01/..., out_dir/n02/..., etc.

`n`: Anzahl Samples pro Kristallgruppe (0 = alle).
`pick`: `:worst` (höchstes rel_l2), `:best`, oder `:random`.
"""
function make_eval_gallery(pred_dir::AbstractString, out_dir::AbstractString;
                           n::Int = 0, pick::Symbol = :worst,
                           dx::Union{Real, Nothing} = nothing,
                           dz::Union{Real, Nothing} = nothing,
                           mask_width::Int = 2)
    mkpath(out_dir)

    # Alle pred_*.jld2 finden und laden
    files = sort(filter(f -> startswith(basename(f), "pred_") && endswith(f, ".jld2"),
                        readdir(pred_dir; join=true)))
    isempty(files) && error("Keine pred_*.jld2 in $pred_dir gefunden")

    # Metriken + n_crystals extrahieren
    entries = []
    for f in files
        d = load(f)
        m = d["metrics"]
        nc = haskey(m, :n_crystals) ? Int(m.n_crystals) : -1
        push!(entries, (file=f, rel_l2=Float64(m.psi_rel_l2), n_crystals=nc,
                        ψ_pred=d["ψ_pred_norm"], ψ_true=d["ψ_true_norm"]))
    end

    # Nach Kristallanzahl gruppieren
    groups = Dict{Int, Vector{eltype(entries)}}()
    for e in entries
        push!(get!(groups, e.n_crystals, eltype(entries)[]), e)
    end

    total_plotted = 0

    for nc in sort(collect(keys(groups)))
        group = groups[nc]

        # Sortieren nach Auswahlkriterium
        if pick == :worst
            sort!(group; by=e -> e.rel_l2, rev=true)
        elseif pick == :best
            sort!(group; by=e -> e.rel_l2)
        else  # :random
            shuffle!(group)
        end

        n_sel = n <= 0 ? length(group) : min(n, length(group))

        # Unterordner: n01, n02, ... oder "unknown"
        sub = nc > 0 ? @sprintf("n%02d", nc) : "unknown"
        sub_dir = joinpath(out_dir, sub)
        mkpath(sub_dir)

        @info "Gallery $sub: $n_sel / $(length(group)) Samples ($pick)"

        for (idx, entry) in enumerate(group[1:n_sel])
            prefix = @sprintf("%s/%03d_rel%.4f", sub_dir, idx, entry.rel_l2)

            # ψ Comparison
            plot_psi_comparison(entry.ψ_true, entry.ψ_pred;
                                out_path="$(prefix)_psi.png",
                                title=@sprintf("n=%d #%d rel_l2=%.4f", nc, idx, entry.rel_l2))

            # Velocity + Divergence (falls dx/dz gegeben)
            if dx !== nothing && dz !== nothing
                Vx_true, Vz_true = velocity_from_streamfunction(
                    Float64.(entry.ψ_true), Float64(dx), Float64(dz))
                Vx_pred, Vz_pred = velocity_from_streamfunction(
                    Float64.(entry.ψ_pred), Float64(dx), Float64(dz))

                plot_velocity_comparison(Vx_true, Vz_true, Vx_pred, Vz_pred;
                                         out_path="$(prefix)_vel.png",
                                         title=@sprintf("n=%d #%d velocity", nc, idx))

                div_pred = divergence(Vx_pred, Vz_pred, Float64(dx), Float64(dz))
                plot_divergence_map(div_pred;
                                    out_path="$(prefix)_div.png",
                                    title=@sprintf("n=%d #%d divergence", nc, idx),
                                    mask_width=mask_width)
            end

            total_plotted += 1
        end
    end

    @info "Gallery fertig: $total_plotted Plots in $out_dir ($(length(groups)) Kristallgruppen)"
end

end # module
