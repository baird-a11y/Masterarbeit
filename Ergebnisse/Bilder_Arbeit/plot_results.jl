# ============================================================
# plot_results.jl
# Erzeugt publikationsreife Abbildungen aus den Experiment-CSVs
#
# Verwendung: julia plot_results.jl
# Pakete:     CSV, DataFrames, CairoMakie
# ============================================================
using CSV, DataFrames, CairoMakie

const SCRIPT_DIR = @__DIR__
const RES = dirname(SCRIPT_DIR)   # Ergebnisse/
const OUT = SCRIPT_DIR             # Ergebnisse/Bilder_Arbeit/

# ─── LR-Mapping (lt. Memory/Experiment-Struktur) ─────────────
const LR_LABEL = Dict(
    1 => "lr = 1×10⁻³",
    2 => "lr = 5×10⁻³",
    3 => "lr = 1×10⁻⁴",
    4 => "lr = 5×10⁻⁴",
)

# Farb-Palette (Wong 2011, farbenblindfreundlich)
const C = Makie.wong_colors()

# ─── Hilfsfunktionen ─────────────────────────────────────────
safe_csv(p) = isfile(p) ? CSV.read(p, DataFrame) : nothing

"""Gibt Pfade für U-Net Exp-1 zurück (beachtet den 'One_Crystal'-Tippfehler)."""
function unet_exp1_dirs(bs::Int, cfg::Int)
    suffix = cfg == 1 ? "" : "_$cfg"
    if bs == 16
        hist_base = "One_Crystal"
        eval_base = cfg == 4 ? "One_Crystall" : "One_Crystal"
        hist_dir  = joinpath(RES, "UNET_Ergebnisse", hist_base, "exp1_16$suffix")
        eval_dir  = joinpath(RES, "UNET_Ergebnisse", eval_base, "exp1_16$suffix")
    else  # bs = 8
        hist_dir = joinpath(RES, "UNET_Ergebnisse", "One_Crystal", "exp1_8$suffix")
        eval_dir = hist_dir
    end
    return hist_dir, eval_dir
end

"""Gibt den Pfad für FNO Exp-1 zurück."""
fno_exp1_dir(bs::Int, cfg::Int) =
    joinpath(RES, "FNO_Ergebnisse", "One_Crystal", "eval_output_exp1_$(bs)_$(cfg)")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 1 & 2: Trainingskurven (FNO und U-Net)
# ═══════════════════════════════════════════════════════════════════════════════
function plot_training_curves(arch::String, fig_nr::Int)
    fig = Figure(size=(1100, 450), fontsize=14)

    for (col, bs) in enumerate([16, 8])
        ax = Axis(fig[1, col],
            xlabel = "Epoche",
            ylabel = col == 1 ? "Validierungsfehler (rel. L²)" : "",
            title  = "$arch – Experiment 1, Batch-Größe $bs",
            yscale = log10,
        )

        n_plotted = 0
        for cfg in 1:4
            df = if arch == "FNO"
                safe_csv(joinpath(fno_exp1_dir(bs, cfg),
                                  "history_exp1_$(bs)_$(cfg).csv"))
            else
                hist_dir, _ = unet_exp1_dirs(bs, cfg)
                safe_csv(joinpath(hist_dir, "training_history.csv"))
            end
            df === nothing && continue
            lines!(ax, df.epoch, df.val_rel_l2;
                   color=C[cfg], linewidth=2, label=LR_LABEL[cfg])
            n_plotted += 1
        end
        n_plotted > 0 && axislegend(ax, position=:rt, framevisible=true)
    end

    fname = "fig$(fig_nr)_$(lowercase(arch))_trainingskurven"
    save(joinpath(OUT, fname * ".pdf"), fig)
    save(joinpath(OUT, fname * ".png"), fig, px_per_unit=2)
    println("  Gespeichert: $fname")
end


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 3: Eval-Metriken Exp-1 – Balkendiagramm (FNO vs. U-Net, bs=16 | bs=8)
# ═══════════════════════════════════════════════════════════════════════════════
function plot_eval_bar_chart()
    fig = Figure(size=(1200, 500), fontsize=13)

    for (col, bs) in enumerate([16, 8])
        ax = Axis(fig[1, col],
            xlabel = "Lernrate",
            ylabel = col == 1 ? "Relativer L²-Fehler (n = 1)" : "",
            title  = "Exp. 1 – Evaluation, Batch-Größe $bs",
            xticks = (1:4, [LR_LABEL[i] for i in 1:4]),
            xticklabelrotation = π / 5,
        )

        psi_fno  = fill(NaN, 4)
        vel_fno  = fill(NaN, 4)
        psi_unet = fill(NaN, 4)
        vel_unet = fill(NaN, 4)

        for cfg in 1:4
            # FNO
            df = safe_csv(joinpath(fno_exp1_dir(bs, cfg), "eval_aggregated.csv"))
            if df !== nothing
                row = df[df.n_crystals .== 1, :]
                if nrow(row) > 0
                    psi_fno[cfg] = row.psi_rel_l2_mean[1]
                    vel_fno[cfg] = row.v_rel_l2_mean[1]
                end
            end

            # U-Net
            _, eval_dir = unet_exp1_dirs(bs, cfg)
            df = safe_csv(joinpath(eval_dir, "eval_psi_indist_by_n.csv"))
            if df !== nothing
                row = df[df.n_crystals .== 1, :]
                if nrow(row) > 0
                    psi_unet[cfg] = row.psi_rel_l2_mean[1]
                    vel_unet[cfg] = row.v_rel_l2_mean[1]
                end
            end
        end

        xs = 1:4
        w  = 0.18

        # Nur Balken mit gültigen Werten plotten
        valid(v) = .!isnan.(v)
        any(valid(psi_fno))  && barplot!(ax, xs[valid(psi_fno)]  .- 1.5w,
            psi_fno[valid(psi_fno)];  width=w, color=C[1], label="FNO – ψ")
        any(valid(vel_fno))  && barplot!(ax, xs[valid(vel_fno)]  .- 0.5w,
            vel_fno[valid(vel_fno)];  width=w, color=C[2], label="FNO – v")
        any(valid(psi_unet)) && barplot!(ax, xs[valid(psi_unet)] .+ 0.5w,
            psi_unet[valid(psi_unet)]; width=w, color=C[3], label="U-Net – ψ")
        any(valid(vel_unet)) && barplot!(ax, xs[valid(vel_unet)] .+ 1.5w,
            vel_unet[valid(vel_unet)]; width=w, color=C[4], label="U-Net – v")

        axislegend(ax, position=:rt, framevisible=true)
    end

    fname = "fig3_exp1_eval_balken"
    save(joinpath(OUT, fname * ".pdf"), fig)
    save(joinpath(OUT, fname * ".png"), fig, px_per_unit=2)
    println("  Gespeichert: $fname")
end


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 4: FNO Exp-2 & Exp-3 – Generalisierung über Kristallanzahl
# ═══════════════════════════════════════════════════════════════════════════════
function plot_exp23_scaling()
    df2 = safe_csv(joinpath(RES, "FNO_Ergebnisse", "Experiment_2",
                             "eval_output_exp2", "eval_aggregated.csv"))
    df3 = safe_csv(joinpath(RES, "FNO_Ergebnisse", "Experiment_3",
                             "eval_output_exp3", "eval_aggregated.csv"))

    fig = Figure(size=(1150, 470), fontsize=14)

    for (col, (df, title, max_train)) in enumerate([
        (df2, "FNO – Exp. 2 (Training: n = 1–10)",  10),
        (df3, "FNO – Exp. 3 (Training: n = 1–25)",  25),
    ])
        df === nothing && (println("  Warnung: Daten für $(title) nicht gefunden."); continue)

        ax = Axis(fig[1, col],
            xlabel = "Anzahl Kristalle n",
            ylabel = col == 1 ? "Relativer L²-Fehler" : "",
            title  = title,
        )

        id  = df.n_crystals .<= max_train
        ood = .!id

        # In-distribution
        scatter!(ax, df.n_crystals[id], df.psi_rel_l2_mean[id];
                 color=C[1], markersize=8)
        lines!(ax, df.n_crystals[id], df.psi_rel_l2_mean[id];
               color=C[1], linewidth=2, label="ψ (in-dist)")
        scatter!(ax, df.n_crystals[id], df.v_rel_l2_mean[id];
                 color=C[2], markersize=8)
        lines!(ax, df.n_crystals[id], df.v_rel_l2_mean[id];
               color=C[2], linewidth=2, label="v (in-dist)")

        # Out-of-distribution
        if any(ood)
            scatter!(ax, df.n_crystals[ood], df.psi_rel_l2_mean[ood];
                     color=C[1], marker=:diamond, markersize=9)
            lines!(ax, df.n_crystals[ood], df.psi_rel_l2_mean[ood];
                   color=C[1], linewidth=2, linestyle=:dash, label="ψ (OOD)")
            scatter!(ax, df.n_crystals[ood], df.v_rel_l2_mean[ood];
                     color=C[2], marker=:diamond, markersize=9)
            lines!(ax, df.n_crystals[ood], df.v_rel_l2_mean[ood];
                   color=C[2], linewidth=2, linestyle=:dash, label="v (OOD)")
            vlines!(ax, [max_train + 0.5]; color=:gray60, linestyle=:dot, linewidth=1.5)
        end

        axislegend(ax, position=:lt, framevisible=true)
    end

    fname = "fig4_fno_exp23_generalisierung"
    save(joinpath(OUT, fname * ".pdf"), fig)
    save(joinpath(OUT, fname * ".png"), fig, px_per_unit=2)
    println("  Gespeichert: $fname")
end


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 5: FNO vs. U-Net – Beste Konfigurationen Trainingsverlauf (Übersicht)
# ═══════════════════════════════════════════════════════════════════════════════
function plot_best_config_comparison()
    # Beste Configs laut Memory: FNO bs=8 cfg=2 (5e-3), U-Net bs=8 cfg=1 (1e-3)
    # Hier zeigen wir bs=16 als Vergleich (vollständige Daten)
    fig = Figure(size=(900, 430), fontsize=14)
    ax = Axis(fig[1, 1],
        xlabel = "Epoche",
        ylabel = "Validierungsfehler (rel. L²)",
        title  = "Vergleich FNO vs. U-Net – Batch-Größe 16",
        yscale = log10,
    )

    styles = [:solid, :dash, :dot, :dashdot]
    arch_cfgs = [
        ("FNO",   16, 1, "FNO $(LR_LABEL[1])"),
        ("FNO",   16, 2, "FNO $(LR_LABEL[2])"),
        ("FNO",   16, 4, "FNO $(LR_LABEL[4])"),
        ("UNet",  16, 1, "U-Net $(LR_LABEL[1])"),
        ("UNet",  16, 3, "U-Net $(LR_LABEL[3])"),
    ]

    for (i, (arch, bs, cfg, lbl)) in enumerate(arch_cfgs)
        df = if arch == "FNO"
            safe_csv(joinpath(fno_exp1_dir(bs, cfg),
                              "history_exp1_$(bs)_$(cfg).csv"))
        else
            hist_dir, _ = unet_exp1_dirs(bs, cfg)
            safe_csv(joinpath(hist_dir, "training_history.csv"))
        end
        df === nothing && continue
        col = i <= 3 ? C[1] : C[3]
        lst = arch == "FNO" ? styles[cfg] : styles[cfg > 2 ? 2 : cfg]
        lines!(ax, df.epoch, df.val_rel_l2;
               color=col, linestyle=lst, linewidth=2, label=lbl)
    end
    axislegend(ax, position=:rt, framevisible=true)

    fname = "fig5_fno_unet_vergleich_bs16"
    save(joinpath(OUT, fname * ".pdf"), fig)
    save(joinpath(OUT, fname * ".png"), fig, px_per_unit=2)
    println("  Gespeichert: $fname")
end


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 6 & 7: FNO Exp-2 & Exp-3 – Trainingskurven
# ═══════════════════════════════════════════════════════════════════════════════
function plot_fno_exp23_training()
    paths = [
        (joinpath(RES, "FNO_Ergebnisse", "Experiment_2", "eval_output_exp2", "history_exp2.csv"),
         "FNO – Experiment 2 (n = 1–10)", 6),
        (joinpath(RES, "FNO_Ergebnisse", "Experiment_3", "eval_output_exp3", "history_exp3.csv"),
         "FNO – Experiment 3 (n = 1–25)", 7),
    ]

    for (p, title, fig_nr) in paths
        df = safe_csv(p)
        df === nothing && (println("  Warnung: $p nicht gefunden."); continue)

        fig = Figure(size=(700, 430), fontsize=14)
        ax  = Axis(fig[1, 1],
            xlabel = "Epoche",
            ylabel = "Validierungsfehler (rel. L²)",
            title  = title,
            yscale = log10,
        )
        lines!(ax, df.epoch, df.val_rel_l2; color=C[1], linewidth=2, label="val rel. L²")
        lines!(ax, df.epoch, df.train_mse ./ maximum(df.train_mse);
               color=C[2], linewidth=1.5, linestyle=:dash, label="train MSE (norm.)")
        axislegend(ax, position=:rt, framevisible=true)

        exp_str = fig_nr == 6 ? "exp2" : "exp3"
        fname   = "fig$(fig_nr)_fno_$(exp_str)_training"
        save(joinpath(OUT, fname * ".pdf"), fig)
        save(joinpath(OUT, fname * ".png"), fig, px_per_unit=2)
        println("  Gespeichert: $fname")
    end
end


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
println("=" ^ 60)
println("Erzeuge Abbildungen für die Masterarbeit...")
println("Ausgabe: $OUT")
println("=" ^ 60)

plot_training_curves("FNO",  1)
plot_training_curves("UNet", 2)
plot_eval_bar_chart()
plot_exp23_scaling()
plot_best_config_comparison()
plot_fno_exp23_training()

println("=" ^ 60)
println("Fertig! Alle Abbildungen gespeichert in:")
println("  $OUT")
println("=" ^ 60)
