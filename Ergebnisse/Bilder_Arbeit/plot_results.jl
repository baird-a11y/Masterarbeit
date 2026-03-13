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
        hist_dir = joinpath(RES, "UNET_Ergebnisse", "One_Crystal", "exp1_16$suffix")
        eval_dir = hist_dir
    else  # bs = 8
        hist_dir = joinpath(RES, "UNET_Ergebnisse", "One_Crystal", "exp1_8$suffix")
        eval_dir = hist_dir
    end
    return hist_dir, eval_dir
end

"""Gibt den Pfad für FNO Exp-1 zurück."""
fno_exp1_dir(bs::Int, cfg::Int) =
    joinpath(RES, "FNO_Ergebnisse", "One_Crystal", "eval_output_exp1_$(bs)_$(cfg)")

"""Gibt den Pfad für FNO Exp-2 / Exp-3 zurück."""
fno_exp23_dir(exp::Int, bs::Int) =
    joinpath(RES, "FNO_Ergebnisse", "Experiment_$exp", "eval_output_exp$(exp)_$(bs)")

"""Gibt den Pfad für U-Net Exp-2 / Exp-3 zurück."""
unet_exp23_dir(exp::Int, bs::Int) =
    joinpath(RES, "UNET_Ergebnisse", "Experiment_$exp", "exp$(exp)_$(bs)")

"""Gibt den Pfad für FNO Exp-4 (Kristallgröße) zurück. size ∈ ("big", "small")"""
fno_exp4_dir(size::String) =
    joinpath(RES, "FNO_Ergebnisse", "Experiment_4", "eval_output_exp1_16_$size")

"""Gibt den Pfad für U-Net Exp-4 (Kristallgröße) zurück. size ∈ ("big", "small")"""
unet_exp4_dir(size::String) =
    joinpath(RES, "UNET_Ergebnisse", "Experiment_4", "exp2_8_$size")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 1 & 2: Trainingskurven (FNO und U-Net)
# ═══════════════════════════════════════════════════════════════════════════════
function plot_training_curves(arch::String, fig_nr::Int)
    fig = Figure(size=(1100, 450), fontsize=14)

    for (col, bs) in enumerate([16, 8])
        ax = Axis(fig[1, col],
            xlabel = "Epoch",
            ylabel = col == 1 ? "Validation error (rel. L²)" : "",
            title  = "$arch – Experiment 1, batch size $bs",
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
            xlabel = "Learning rate",
            ylabel = col == 1 ? "Relative L² error (n = 1)" : "",
            title  = "Exp. 1 – Evaluation, batch size $bs",
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
            vel_unet[valid(vel_unet)]; width=w, color=C[4], label="U-Net – v")  # labels already English

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
    fig = Figure(size=(1150, 470), fontsize=14)

    for (col, (exp_nr, title, max_train)) in enumerate([
        (2, "FNO – Exp. 2 (training: n = 1–10)", 10),
        (3, "FNO – Exp. 3 (training: n = 1–25)", 25),
    ])
        ax = Axis(fig[1, col],
            xlabel = "Number of crystals n",
            ylabel = col == 1 ? "Relative L² error" : "",
            title  = title,
        )

        any_data = false
        for (bs_idx, (bs, lstyle)) in enumerate([(16, :solid), (8, :dash)])
            df = safe_csv(joinpath(fno_exp23_dir(exp_nr, bs), "eval_aggregated.csv"))
            df === nothing && continue
            any_data = true

            id  = df.n_crystals .<= max_train
            ood = .!id
            bs_lbl = "bs=$bs"

            scatter!(ax, df.n_crystals[id], df.psi_rel_l2_mean[id];
                     color=C[1], markersize=8)
            lines!(ax, df.n_crystals[id], df.psi_rel_l2_mean[id];
                   color=C[1], linewidth=2, linestyle=lstyle, label="ψ ($bs_lbl, in-dist)")
            scatter!(ax, df.n_crystals[id], df.v_rel_l2_mean[id];
                     color=C[2], markersize=8)
            lines!(ax, df.n_crystals[id], df.v_rel_l2_mean[id];
                   color=C[2], linewidth=2, linestyle=lstyle, label="v ($bs_lbl, in-dist)")

            if any(ood)
                scatter!(ax, df.n_crystals[ood], df.psi_rel_l2_mean[ood];
                         color=C[1], marker=:diamond, markersize=9)
                lines!(ax, df.n_crystals[ood], df.psi_rel_l2_mean[ood];
                       color=C[1], linewidth=2, linestyle=lstyle, label="ψ ($bs_lbl, OOD)")
                scatter!(ax, df.n_crystals[ood], df.v_rel_l2_mean[ood];
                         color=C[2], marker=:diamond, markersize=9)
                lines!(ax, df.n_crystals[ood], df.v_rel_l2_mean[ood];
                       color=C[2], linewidth=2, linestyle=lstyle, label="v ($bs_lbl, OOD)")
                bs_idx == 1 && vlines!(ax, [max_train + 0.5];
                                       color=:gray60, linestyle=:dot, linewidth=1.5)
            end
        end

        any_data || println("  Warnung: keine Daten für Exp. $exp_nr gefunden.")
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
        ylabel = "Validation error (rel. L²)",
        title  = "FNO vs. U-Net comparison – batch size 16",
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
    for (exp_nr, fig_nr, exp_str, title) in [
        (2, 6, "exp2", "FNO – Experiment 2 (n = 1–10)"),
        (3, 7, "exp3", "FNO – Experiment 3 (n = 1–25)"),
    ]
        fig = Figure(size=(700, 430), fontsize=14)
        ax  = Axis(fig[1, 1],
            xlabel = "Epoch",
            ylabel = "Validation error (rel. L²)",
            title  = title,
            yscale = log10,
        )

        any_data = false
        for (bs, col_idx, lstyle) in [(16, 1, :solid), (8, 2, :dash)]
            csv_name = "history_$(exp_str)_$(bs).csv"
            p  = joinpath(fno_exp23_dir(exp_nr, bs), csv_name)
            df = safe_csv(p)
            df === nothing && (println("  Warnung: $p nicht gefunden."); continue)
            any_data = true
            lines!(ax, df.epoch, df.val_rel_l2;
                   color=C[col_idx], linewidth=2, linestyle=lstyle, label="val. rel. L² (bs=$bs)")
        end
        any_data || println("  Warnung: keine History für Exp. $exp_nr gefunden.")
        axislegend(ax, position=:rt, framevisible=true)

        fname = "fig$(fig_nr)_fno_$(exp_str)_training"
        save(joinpath(OUT, fname * ".pdf"), fig)
        save(joinpath(OUT, fname * ".png"), fig, px_per_unit=2)
        println("  Gespeichert: $fname")
    end
end


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 8 & 9: U-Net Exp-2 & Exp-3 – Trainingskurven
# ═══════════════════════════════════════════════════════════════════════════════
function plot_unet_exp23_training()
    for (exp_nr, fig_nr, exp_str, title) in [
        (2, 8, "exp2", "U-Net – Experiment 2 (n = 1–10)"),
        (3, 9, "exp3", "U-Net – Experiment 3 (n = 1–25)"),
    ]
        fig = Figure(size=(700, 430), fontsize=14)
        ax  = Axis(fig[1, 1],
            xlabel = "Epoch",
            ylabel = "Validation error (rel. L²)",
            title  = title,
            yscale = log10,
        )

        any_data = false
        for (bs, col_idx, lstyle) in [(16, 1, :solid), (8, 2, :dash)]
            p  = joinpath(unet_exp23_dir(exp_nr, bs), "training_history.csv")
            df = safe_csv(p)
            df === nothing && (println("  Warnung: $p nicht gefunden."); continue)
            any_data = true
            lines!(ax, df.epoch, df.val_rel_l2;
                   color=C[col_idx], linewidth=2, linestyle=lstyle, label="val. rel. L² (bs=$bs)")
        end
        any_data || println("  Warnung: keine History für U-Net Exp. $exp_nr gefunden.")
        axislegend(ax, position=:rt, framevisible=true)

        fname = "fig$(fig_nr)_unet_$(exp_str)_training"
        save(joinpath(OUT, fname * ".pdf"), fig)
        save(joinpath(OUT, fname * ".png"), fig, px_per_unit=2)
        println("  Gespeichert: $fname")
    end
end


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 10 & 11: U-Net Exp-2 & Exp-3 – Generalisierung über Kristallanzahl
# ═══════════════════════════════════════════════════════════════════════════════
function plot_unet_exp23_scaling()
    fig = Figure(size=(1150, 470), fontsize=14)

    for (col, (exp_nr, title, max_train)) in enumerate([
        (2, "U-Net – Exp. 2 (training: n = 1–10)", 10),
        (3, "U-Net – Exp. 3 (training: n = 1–25)", 25),
    ])
        ax = Axis(fig[1, col],
            xlabel = "Number of crystals n",
            ylabel = col == 1 ? "Relative L² error" : "",
            title  = title,
        )

        any_data = false
        for (bs_idx, (bs, lstyle)) in enumerate([(16, :solid), (8, :dash)])
            p  = joinpath(unet_exp23_dir(exp_nr, bs), "eval_psi_indist_by_n.csv")
            df = safe_csv(p)
            df === nothing && continue
            any_data = true

            id  = df.n_crystals .<= max_train
            ood = .!id
            bs_lbl = "bs=$bs"

            scatter!(ax, df.n_crystals[id], df.psi_rel_l2_mean[id];
                     color=C[1], markersize=8)
            lines!(ax, df.n_crystals[id], df.psi_rel_l2_mean[id];
                   color=C[1], linewidth=2, linestyle=lstyle, label="ψ ($bs_lbl, in-dist)")
            scatter!(ax, df.n_crystals[id], df.v_rel_l2_mean[id];
                     color=C[2], markersize=8)
            lines!(ax, df.n_crystals[id], df.v_rel_l2_mean[id];
                   color=C[2], linewidth=2, linestyle=lstyle, label="v ($bs_lbl, in-dist)")

            if any(ood)
                scatter!(ax, df.n_crystals[ood], df.psi_rel_l2_mean[ood];
                         color=C[1], marker=:diamond, markersize=9)
                lines!(ax, df.n_crystals[ood], df.psi_rel_l2_mean[ood];
                       color=C[1], linewidth=2, linestyle=lstyle, label="ψ ($bs_lbl, OOD)")
                scatter!(ax, df.n_crystals[ood], df.v_rel_l2_mean[ood];
                         color=C[2], marker=:diamond, markersize=9)
                lines!(ax, df.n_crystals[ood], df.v_rel_l2_mean[ood];
                       color=C[2], linewidth=2, linestyle=lstyle, label="v ($bs_lbl, OOD)")
                bs_idx == 1 && vlines!(ax, [max_train + 0.5];
                                       color=:gray60, linestyle=:dot, linewidth=1.5)
            end
        end

        any_data || println("  Warnung: keine Daten für U-Net Exp. $exp_nr gefunden.")
        axislegend(ax, position=:lt, framevisible=true)
    end

    fname = "fig10_unet_exp23_generalisierung"
    save(joinpath(OUT, fname * ".pdf"), fig)
    save(joinpath(OUT, fname * ".png"), fig, px_per_unit=2)
    println("  Gespeichert: $fname")
end


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 11: Exp-4 – Kristallgrößen-Generalisierung (FNO vs. U-Net, big / small)
# ═══════════════════════════════════════════════════════════════════════════════
function plot_exp4_size_generalization()
    fig = Figure(size=(700, 480), fontsize=14)
    ax  = Axis(fig[1, 1],
        ylabel = "Relative L² error",
        title  = "Exp. 4 – Crystal size generalization",
        xticks = ([1, 2, 3, 4], ["FNO – large", "FNO – small", "U-Net – large", "U-Net – small"]),
        xticklabelrotation = π / 6,
    )

    # Werte einlesen: FNO nutzt eval_aggregated.csv, U-Net nutzt eval_psi_indist_by_n.csv
    vals = Dict{Tuple{String,String,String}, Float64}()  # (arch, size, metric) => value

    for size in ("big", "small")
        df = safe_csv(joinpath(fno_exp4_dir(size), "eval_aggregated.csv"))
        if df !== nothing && nrow(df) > 0
            vals[("fno", size, "psi")] = df.psi_rel_l2_mean[1]
            vals[("fno", size, "v")]   = df.v_rel_l2_mean[1]
        end

        df = safe_csv(joinpath(unet_exp4_dir(size), "eval_psi_indist_by_n.csv"))
        if df !== nothing && nrow(df) > 0
            vals[("unet", size, "psi")] = df.psi_rel_l2_mean[1]
            vals[("unet", size, "v")]   = df.v_rel_l2_mean[1]
        end
    end

    if isempty(vals)
        println("  Warnung: keine Daten für Exp. 4 gefunden.")
        return
    end

    # Positionen: 1=FNO-big, 2=FNO-small, 3=UNet-big, 4=UNet-small
    configs = [("fno","big"), ("fno","small"), ("unet","big"), ("unet","small")]
    w = 0.3

    psi_vals = [get(vals, (a, s, "psi"), NaN) for (a,s) in configs]
    v_vals   = [get(vals, (a, s, "v"),   NaN) for (a,s) in configs]

    xs = 1:4
    valid(v) = .!isnan.(v)

    any(valid(psi_vals)) && barplot!(ax, xs[valid(psi_vals)] .- w/2,
        psi_vals[valid(psi_vals)]; width=w, color=C[1], label="ψ")
    any(valid(v_vals))   && barplot!(ax, xs[valid(v_vals)]   .+ w/2,
        v_vals[valid(v_vals)];   width=w, color=C[2], label="v")

    # Trennlinie zwischen FNO und U-Net
    vlines!(ax, [2.5]; color=:gray60, linestyle=:dot, linewidth=1.5)

    axislegend(ax, position=:rt, framevisible=true)

    fname = "fig11_exp4_kristallgroesse"
    save(joinpath(OUT, fname * ".pdf"), fig)
    save(joinpath(OUT, fname * ".png"), fig, px_per_unit=2)
    println("  Gespeichert: $fname")
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
plot_unet_exp23_training()
plot_unet_exp23_scaling()
plot_exp4_size_generalization()

println("=" ^ 60)
println("Fertig! Alle Abbildungen gespeichert in:")
println("  $OUT")
println("=" ^ 60)
