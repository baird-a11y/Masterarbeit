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

using CSV, DataFrames, JLD2, Statistics, CairoMakie
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
    eval_base = (bs == 16 && cfg == 4) ? "One_Crystall" : "One_Crystal"
    joinpath(RES, "UNET_Ergebnisse", eval_base, "exp1_$(bs)$suffix",
             "eval_psi_indist_samples.csv")
end

fno_exp1_sample_csv(bs::Int, cfg::Int) =
    joinpath(RES, "FNO_Ergebnisse", "One_Crystal",
             "eval_output_exp1_$(bs)_$(cfg)", "eval_results.csv")

# ─── Ergebnisse laden + Kristallpositionen aus JLD2 ergänzen ─────────────────
"""
Lädt ein eval-CSV (FNO oder U-Net) und ergänzt die Kristallposition
(Abstand vom Domänenmittelpunkt) aus den verlinkten JLD2-Dateien.

Gibt einen Vector{NamedTuple} mit Feldern:
  sample_idx, rel_l2_psi, cx, cz, dist_center
zurück. Samples deren JLD2-Datei nicht gefunden wird, werden übersprungen.
"""
function load_results_with_positions(csv_path::String)
    isfile(csv_path) || (println("  Nicht gefunden: $csv_path"); return NamedTuple[])

    df = CSV.read(csv_path, DataFrame)

    # Spaltenname ist zwischen FNO und U-Net verschieden
    err_col = hasproperty(df, :psi_rel_l2) ? :psi_rel_l2 : :rel_l2_psi

    rows = NamedTuple[]
    for row in eachrow(df)
        fp = String(row.filepath)
        isfile(fp) || continue

        d    = JLD2.load(fp)
        meta = d["meta"]

        centers_2D = haskey(meta, :centers_2D) ? meta[:centers_2D] : meta.centers_2D
        xcoords    = haskey(meta, :x_vec_1D)   ? meta[:x_vec_1D]   : Float64[]
        zcoords    = haskey(meta, :z_vec_1D)   ? meta[:z_vec_1D]   : Float64[]

        x_ctr = isempty(xcoords) ? 0.0 : (xcoords[1] + xcoords[end]) / 2
        z_ctr = isempty(zcoords) ? 0.0 : (zcoords[1] + zcoords[end]) / 2

        for c in centers_2D
            cx, cz = c
            dist = sqrt((cx - x_ctr)^2 + (cz - z_ctr)^2)
            push!(rows, (;
                sample_idx  = Int(row.sample_idx),
                rel_l2_psi  = Float64(row[err_col]),
                cx, cz,
                dist_center = dist,
            ))
        end
    end
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

    for (entries, col, marker, lbl) in [
        (fno_entries,  C[1], :circle,  "FNO"),
        (unet_entries, C[2], :diamond, "U-Net"),
    ]
        isempty(entries) && continue
        d   = [r.dist_center for r in entries]
        err = [r.rel_l2_psi  for r in entries]
        scatter!(ax, d, err; color=col, marker=marker, markersize=7, label=lbl)
        ord = sortperm(d)
        lines!(ax, d[ord], err[ord]; color=col, linewidth=1.5, alpha=0.35)
    end

    axislegend(ax; position=:rt, framevisible=true)

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

        fno_csv  = fno_exp1_sample_csv(bs, cfg)
        unet_csv = unet_exp1_sample_csv(bs, cfg)

        for (csv, col, marker, lbl) in [
            (fno_csv,  C[1], :circle,  "FNO"),
            (unet_csv, C[2], :diamond, "U-Net"),
        ]
            entries = load_results_with_positions(csv)
            isempty(entries) && continue
            d   = [r.dist_center for r in entries]
            err = [r.rel_l2_psi  for r in entries]
            scatter!(ax, d, err; color=col, marker=marker, markersize=6, label=lbl)
            ord = sortperm(d)
            lines!(ax, d[ord], err[ord]; color=col, linewidth=1.5, alpha=0.35)
        end

        axislegend(ax; position=:rt, framevisible=true)
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
fno_best_16  = load_results_with_positions(fno_exp1_sample_csv(16, 4))
unet_best_16 = load_results_with_positions(unet_exp1_sample_csv(16, 1))

plot_error_vs_distance(fno_best_16, unet_best_16;
    title   = "Fehler vs. Kristallposition – Exp. 1, bs = 16 (beste Konfigs)",
    outfile = joinpath(OUT, "error_vs_center_distance_exp1_bs16.pdf"))

# ── 2) Beste Einzelkonfigs: bs=8 (FNO: cfg=2 / U-Net: cfg=1) ──
fno_best_8  = load_results_with_positions(fno_exp1_sample_csv(8, 2))
unet_best_8 = load_results_with_positions(unet_exp1_sample_csv(8, 1))

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
