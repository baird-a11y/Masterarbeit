#!/usr/bin/env julia

# ============================================================
# error_vs_center_distance.jl
#
# Analysiert den Zusammenhang zwischen rel. L²-Fehler (ψ) und
# Kristallposition (Abstand vom Domänenmittelpunkt) für
# Experiment 1 (n = 1, Ein-Kristall).
#
# Nutzt vorberechnete eval_results.csv + JLD2-Metadaten –
# kein Modell laden notwendig.
#
# Ausgabe: Bilder_Arbeit/error_vs_center_distance_exp1_bs*.pdf/.png
# ============================================================

using CSV, DataFrames, JLD2, Statistics, LinearAlgebra, Printf, CairoMakie
using Base.Filesystem: mkpath

const SCRIPT_DIR = @__DIR__
const RES = dirname(SCRIPT_DIR)   # Ergebnisse/
const OUT = SCRIPT_DIR             # Ergebnisse/Bilder_Arbeit/

const C = Makie.wong_colors()

# ─── LR-Mapping (lt. Memory/Experiment-Struktur) ─────────────────────────────
const LR_LABEL = Dict(
    1 => "lr = 1×10⁻³",
    2 => "lr = 5×10⁻³",
    3 => "lr = 1×10⁻⁴",
    4 => "lr = 5×10⁻⁴",
)

# ─── Pfad-Hilfsfunktionen ────────────────────────────────────────────────────
function unet_exp1_sample_csv(bs::Int, cfg::Int)
    suffix = cfg == 1 ? "" : "_$cfg"
    joinpath(RES, "UNET_Ergebnisse", "One_Crystal", "exp1_$(bs)$suffix",
             "eval_psi_indist_samples.csv")
end

fno_exp1_dir(bs::Int, cfg::Int) =
    joinpath(RES, "FNO_Ergebnisse", "One_Crystal", "eval_output_exp1_$(bs)_$(cfg)")

# ─── Ergebnisse laden + Kristallpositionen aus JLD2 ergänzen ─────────────────

"""
FNO: Lädt alle pred_*.jld2 aus dem predictions/-Unterordner.
Jede Datei enthält metrics (psi_rel_l2) + sample_meta mit Kristallpositionen.
Funktioniert lokal ohne Zugriff auf die originalen Datendateien.
"""
function load_fno_predictions(eval_dir::String)
    pred_dir = joinpath(eval_dir, "predictions")
    isdir(pred_dir) || (println("  Kein predictions/-Ordner: $pred_dir"); return NamedTuple[])

    rows = NamedTuple[]
    for f in sort(readdir(pred_dir))
        endswith(f, ".jld2") || continue
        d    = JLD2.load(joinpath(pred_dir, f))
        meta = d["sample_meta"].meta
        m    = d["metrics"]

        x_ctr = (meta.x_vec_1D[1] + meta.x_vec_1D[end]) / 2
        z_ctr = (meta.z_vec_1D[1] + meta.z_vec_1D[end]) / 2

        for c in meta.centers_2D
            cx, cz = c
            push!(rows, (;
                rel_l2_psi  = Float64(m.psi_rel_l2),
                cx, cz,
                dist_center = sqrt((cx - x_ctr)^2 + (cz - z_ctr)^2),
            ))
        end
    end
    println("  FNO: $(length(rows)) Samples aus $pred_dir geladen.")
    return rows
end

"""
U-Net: Kombiniert Fehler aus eval-CSV mit Kristallpositionen aus den lokal
vorhandenen FNO-Prediction-JLD2s (gleiche Eval-Samples, gleiche Reihenfolge).
fno_pred_dir: Pfad zum predictions/-Ordner eines FNO-Experiments mit gleichen Samples.
"""
function load_unet_predictions(csv_path::String, fno_pred_dir::String)
    isfile(csv_path) || (println("  Nicht gefunden: $csv_path"); return NamedTuple[])
    isdir(fno_pred_dir) || (println("  Kein FNO-predictions/-Ordner: $fno_pred_dir"); return NamedTuple[])

    df        = CSV.read(csv_path, DataFrame)
    pred_files = sort(filter(f -> endswith(f, ".jld2"), readdir(fno_pred_dir)))
    rows      = NamedTuple[]

    for row in eachrow(df)
        idx = Int(row.sample_idx)
        fname = @sprintf("pred_%06d.jld2", idx)
        fp = joinpath(fno_pred_dir, fname)
        isfile(fp) || continue

        d    = JLD2.load(fp)
        meta = d["sample_meta"].meta

        x_ctr = (meta.x_vec_1D[1] + meta.x_vec_1D[end]) / 2
        z_ctr = (meta.z_vec_1D[1] + meta.z_vec_1D[end]) / 2

        for c in meta.centers_2D
            cx, cz = c
            push!(rows, (;
                rel_l2_psi  = Float64(row.rel_l2_psi),
                cx, cz,
                dist_center = sqrt((cx - x_ctr)^2 + (cz - z_ctr)^2),
            ))
        end
    end
    println("  U-Net: $(length(rows)) Samples aus $csv_path geladen.")
    return rows
end

# ─── Scatter-Plot: Fehler vs. Abstand ────────────────────────────────────────
"""
Erzeugt einen Scatter-Plot (Fehler vs. Abstand vom Domänenmittelpunkt)
für FNO und U-Net gemeinsam.

fno_entries / unet_entries: Vector{NamedTuple} aus load_results_with_positions
"""
function plot_error_vs_distance(fno_entries, unet_entries;
        title::String,
        outfile::String)

    mkpath(dirname(outfile))
    fig = Figure(size=(800, 500), fontsize=14)
    ax  = Axis(fig[1, 1];
        xlabel = "Abstand vom Domänenmittelpunkt (km)",
        ylabel = "Rel. L²-Fehler ψ",
        title  = title,
    )

    any_plotted = false
    for (entries, col, marker, lbl) in [
        (fno_entries,  C[1], :circle,  "FNO"),
        (unet_entries, C[2], :diamond, "U-Net"),
    ]
        isempty(entries) && continue
        any_plotted = true
        d   = [r.dist_center for r in entries]
        err = [r.rel_l2_psi  for r in entries]
        scatter!(ax, d, err; color=col, marker=marker, markersize=7, label=lbl)
        # Regressionsgerade
        A = hcat(ones(length(d)), d)
        coef = A \ err
        x_reg = range(minimum(d), maximum(d); length=100)
        lines!(ax, collect(x_reg), coef[1] .+ coef[2] .* collect(x_reg);
               color=col, linewidth=2, linestyle=:dash, label="$lbl (Regression)")
    end

    any_plotted && axislegend(ax; position=:rt, framevisible=true)

    base = splitext(outfile)[1]
    save(outfile, fig)
    save(base * ".png", fig, px_per_unit=2)
    println("  Gespeichert: $(basename(outfile))")
    return fig
end

# ─── Subplot-Übersicht: alle Configs einer Batch-Größe ───────────────────────
"""
Erzeugt ein 2×2-Gitter (eine Zelle pro lr-Config) mit FNO und U-Net
überlagert, für eine feste Batch-Größe.
"""
function plot_error_vs_distance_all_configs(bs::Int)
    fig = Figure(size=(1200, 900), fontsize=13)

    for cfg in 1:4
        row_idx = (cfg - 1) ÷ 2 + 1
        col_idx = (cfg - 1) % 2 + 1

        ax = Axis(fig[row_idx, col_idx];
            xlabel = "Abstand vom Domänenmittelpunkt (km)",
            ylabel = "Rel. L²-Fehler ψ",
            title  = "$(LR_LABEL[cfg]), bs = $bs",
        )

        fno_entries  = load_fno_predictions(fno_exp1_dir(bs, cfg))
        unet_entries = load_unet_predictions(unet_exp1_sample_csv(bs, cfg),
                           joinpath(fno_exp1_dir(bs, cfg), "predictions"))

        any_plotted = false
        for (entries, col, marker, lbl) in [
            (fno_entries,  C[1], :circle,  "FNO"),
            (unet_entries, C[2], :diamond, "U-Net"),
        ]
            isempty(entries) && continue
            any_plotted = true
            d   = [r.dist_center for r in entries]
            err = [r.rel_l2_psi  for r in entries]
            scatter!(ax, d, err; color=col, marker=marker, markersize=6, label=lbl)
            # Regressionsgerade
            A = hcat(ones(length(d)), d)
            coef = A \ err
            x_reg = range(minimum(d), maximum(d); length=100)
            lines!(ax, collect(x_reg), coef[1] .+ coef[2] .* collect(x_reg);
                   color=col, linewidth=2, linestyle=:dash, label="$lbl (Regression)")
        end

        any_plotted && axislegend(ax; position=:rt, framevisible=true)
    end

    fname = "error_vs_center_distance_exp1_bs$(bs)_allconfigs"
    save(joinpath(OUT, fname * ".pdf"), fig)
    save(joinpath(OUT, fname * ".png"), fig, px_per_unit=2)
    println("  Gespeichert: $fname")
    return fig
end

# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
println("=" ^ 60)
println("Fehler vs. Kristallposition – Experiment 1 (n = 1)")
println("Ausgabe: $OUT")
println("=" ^ 60)

# ── 1) Beste Einzelkonfigs: bs=16 (FNO: cfg=4 / U-Net: cfg=1) ──
fno_best_16  = load_fno_predictions(fno_exp1_dir(16, 4))
unet_best_16 = load_unet_predictions(unet_exp1_sample_csv(16, 1),
                   joinpath(fno_exp1_dir(16, 1), "predictions"))

plot_error_vs_distance(fno_best_16, unet_best_16;
    title   = "Fehler vs. Kristallposition – Exp. 1, bs = 16 (beste Konfigs)",
    outfile = joinpath(OUT, "error_vs_center_distance_exp1_bs16.pdf"))

# ── 2) Beste Einzelkonfigs: bs=8 (FNO: cfg=2 / U-Net: cfg=1) ──
fno_best_8  = load_fno_predictions(fno_exp1_dir(8, 2))
unet_best_8 = load_unet_predictions(unet_exp1_sample_csv(8, 1),
                  joinpath(fno_exp1_dir(8, 1), "predictions"))

plot_error_vs_distance(fno_best_8, unet_best_8;
    title   = "Fehler vs. Kristallposition – Exp. 1, bs = 8 (beste Konfigs)",
    outfile = joinpath(OUT, "error_vs_center_distance_exp1_bs8.pdf"))

# ── 3) Alle Configs als 2×2-Übersicht ──
plot_error_vs_distance_all_configs(16)
plot_error_vs_distance_all_configs(8)

println("=" ^ 60)
println("Fertig! Alle Abbildungen gespeichert in:")
println("  $OUT")
println("=" ^ 60)
