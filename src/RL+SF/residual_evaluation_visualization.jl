# =============================================================================
# RESIDUAL LEARNING EVALUATION - VISUALISIERUNG
# =============================================================================
# Spezialisierte Plots für Residual Learning Analyse

using Plots
using Statistics
using Colors
using Printf

# Server-Modus für Plots (falls kein Display)
ENV["GKSwstype"] = "100"

# =============================================================================
# 4-PANEL VISUALISIERUNG: Stokes | Residuum | Prediction | Ground Truth
# =============================================================================

"""
Erstellt 4-Panel Vergleichs-Plot für Residual Learning
KERNVISUALISIERUNG für Residual Learning Ansatz!

Layout:
┌─────────────┬─────────────┐
│ 1. Stokes   │ 2. Residuum │
│  (Baseline) │  (Δv)       │
├─────────────┼─────────────┤
│ 3. Total    │ 4. LaMEM    │
│  (Stokes+Δv)│  (Ground T) │
└─────────────┴─────────────┘
"""
function plot_residual_decomposition(phase_2d, 
                                    v_stokes_vz, residual_vz, 
                                    prediction_vz, ground_truth_vz;
                                    title_prefix="",
                                    save_path=nothing,
                                    crystal_centers=nothing,
                                    pred_minima=nothing)
    
    # Farbskala basierend auf Ground Truth
    vmin = minimum(ground_truth_vz)
    vmax = maximum(ground_truth_vz)
    
    # Layout: 2×2 Grid
    p1 = heatmap(v_stokes_vz', 
                 c=:balance, clim=(vmin, vmax),
                 title="Stokes Baseline (analytisch)",
                 xlabel="x", ylabel="z",
                 aspect_ratio=:equal, colorbar=true)
    
    # Residuum-spezifische Farbskala
    res_max = maximum(abs, residual_vz)
    p2 = heatmap(residual_vz', 
                 c=:RdBu, clim=(-res_max, res_max),
                 title="Gelerntes Residuum (Δv)",
                 xlabel="x", ylabel="z",
                 aspect_ratio=:equal, colorbar=true)
    
    # Kristall-Zentren in Residuum-Plot
    if !isnothing(crystal_centers)
        for (x, z) in crystal_centers
            scatter!(p2, [x], [z], marker=:circle, ms=8, 
                    color=:white, markerstrokewidth=2, 
                    markerstrokecolor=:black, label="")
        end
    end
    
    p3 = heatmap(prediction_vz', 
                 c=:balance, clim=(vmin, vmax),
                 title="Total = Stokes + Δv",
                 xlabel="x", ylabel="z",
                 aspect_ratio=:equal, colorbar=true)
    
    # Predicted Minima
    if !isnothing(pred_minima)
        for (x, z) in pred_minima
            scatter!(p3, [x], [z], marker=:star5, ms=10, 
                    color=:yellow, markerstrokewidth=1, 
                    markerstrokecolor=:black, label="")
        end
    end
    
    p4 = heatmap(ground_truth_vz', 
                 c=:balance, clim=(vmin, vmax),
                 title="LaMEM Ground Truth",
                 xlabel="x", ylabel="z",
                 aspect_ratio=:equal, colorbar=true)
    
    # GT Minima
    if !isnothing(crystal_centers)
        for (x, z) in crystal_centers
            scatter!(p4, [x], [z], marker=:star5, ms=10, 
                    color=:red, markerstrokewidth=1, 
                    markerstrokecolor=:black, label="")
        end
    end
    
    # Kombiniere
    main_title = isempty(title_prefix) ? "Residual Learning Decomposition" : 
                                         "$title_prefix - Residual Learning Decomposition"
    final_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 1000),
                     plot_title=main_title)
    
    # Speichern
    if !isnothing(save_path)
        savefig(final_plot, save_path)
        println("  ✓ Gespeichert: $save_path")
    end
    
    return final_plot
end

# =============================================================================
# SKALIERUNGS-PLOTS
# =============================================================================

"""
Plot: MAE vs. Kristallanzahl mit Stokes-Baseline-Vergleich
"""
function plot_mae_scaling(results::ResidualBatchResults; save_path=nothing)
    crystal_counts = sort(collect(keys(results.aggregated_stats)))
    
    mae_means = [results.aggregated_stats[n].mae_mean for n in crystal_counts]
    mae_stds = [results.aggregated_stats[n].mae_std for n in crystal_counts]
    
    # Plot mit Fehlerbalken
    p = plot(crystal_counts, mae_means,
             ribbon=mae_stds,
             xlabel="Anzahl Kristalle",
             ylabel="MAE (Mean ± Std)",
             title="Skalierung: MAE vs. Kristallanzahl",
             label="ResidualUNet",
             lw=2, marker=:circle, ms=6,
             legend=:topleft,
             size=(800, 600))
    
    # Optional: Trend-Linie
    if length(crystal_counts) >= 3
        # Lineare Regression
        A = hcat(ones(length(crystal_counts)), float.(crystal_counts))
        coeffs = A \ mae_means
        trend_line = coeffs[1] .+ coeffs[2] .* crystal_counts
        plot!(p, crystal_counts, trend_line, 
              label="Linear Trend", ls=:dash, lw=2, color=:red)
    end
    
    if !isnothing(save_path)
        savefig(p, save_path)
        println("  ✓ Gespeichert: $save_path")
    end
    
    return p
end

"""
Plot: Verbesserung über Stokes vs. Kristallanzahl
KRITISCHE METRIK für Residual Learning!
"""
function plot_stokes_improvement(results::ResidualBatchResults; save_path=nothing)
    crystal_counts = sort(collect(keys(results.aggregated_stats)))
    
    improvements = [results.aggregated_stats[n].improvement_mean for n in crystal_counts]
    
    p = plot(crystal_counts, improvements,
             xlabel="Anzahl Kristalle",
             ylabel="Verbesserung über Stokes (%)",
             title="Residual Learning: Nutzen über analytischer Baseline",
             label="Improvement",
             lw=3, marker=:circle, ms=8, color=:green,
             legend=:topright,
             size=(800, 600))
    
    # Referenzlinie bei 0%
    hline!(p, [0.0], label="Stokes-Baseline", ls=:dash, lw=2, color=:black)
    
    # Zielbereich (z.B. >10% Verbesserung als Erfolg)
    hspan!(p, [10.0, maximum(improvements)*1.1], alpha=0.1, color=:green, label="Ziel: >10%")
    
    if !isnothing(save_path)
        savefig(p, save_path)
        println("  ✓ Gespeichert: $save_path")
    end
    
    return p
end

"""
Plot: Divergenz-Verletzung vs. Kristallanzahl
KRITISCH für Nicht-Stream-Function Ansatz!
"""
function plot_continuity_violation(results::ResidualBatchResults; save_path=nothing)
    crystal_counts = sort(collect(keys(results.aggregated_stats)))
    
    cont_means = [results.aggregated_stats[n].continuity_violation_mean for n in crystal_counts]
    
    p = plot(crystal_counts, cont_means,
             xlabel="Anzahl Kristalle",
             ylabel="Kontinuitätsverletzung (|∇·v|)",
             title="Physikalische Konsistenz: Divergenz",
             label="Mean Divergence",
             lw=2, marker=:circle, ms=6, color=:red,
             yscale=:log10,
             legend=:topleft,
             size=(800, 600))
    
    # Akzeptable Schwellen (anpassen nach Bedarf)
    hline!(p, [1e-3], label="Akzeptabel (<1e-3)", ls=:dash, lw=2, color=:orange)
    hline!(p, [1e-5], label="Exzellent (<1e-5)", ls=:dash, lw=2, color=:green)
    
    if !isnothing(save_path)
        savefig(p, save_path)
        println("  ✓ Gespeichert: $save_path")
    end
    
    return p
end

"""
Plot: Residuum-Magnitude vs. Kristallanzahl
"""
function plot_residual_magnitude(results::ResidualBatchResults; save_path=nothing)
    crystal_counts = sort(collect(keys(results.aggregated_stats)))
    
    res_mags = [results.aggregated_stats[n].residual_magnitude_mean for n in crystal_counts]
    
    p = plot(crystal_counts, res_mags,
             xlabel="Anzahl Kristalle",
             ylabel="Residuum-Magnitude (Mean |Δv|)",
             title="Residual Learning: Korrektur-Größe",
             label="Mean |Δv|",
             lw=2, marker=:circle, ms=6, color=:blue,
             legend=:topleft,
             size=(800, 600))
    
    if !isnothing(save_path)
        savefig(p, save_path)
        println("  ✓ Gespeichert: $save_path")
    end
    
    return p
end

# =============================================================================
# MULTI-PANEL DASHBOARD
# =============================================================================

"""
Erstellt komplettes Dashboard mit allen wichtigen Metriken
"""
function create_residual_dashboard(results::ResidualBatchResults; save_path=nothing)
    p1 = plot_mae_scaling(results)
    p2 = plot_stokes_improvement(results)
    p3 = plot_continuity_violation(results)
    p4 = plot_residual_magnitude(results)
    
    dashboard = plot(p1, p2, p3, p4, 
                    layout=(2,2), 
                    size=(1600, 1200),
                    plot_title="Residual Learning: Vollständige Evaluierung")
    
    if !isnothing(save_path)
        savefig(dashboard, save_path)
        println("  ✓ Dashboard gespeichert: $save_path")
    end
    
    return dashboard
end

# =============================================================================
# VERGLEICHS-VISUALISIERUNG
# =============================================================================

"""
Vergleicht Standard UNet vs. Residual UNet
Benötigt Ergebnisse von beiden Modellen
"""
function compare_standard_vs_residual(results_standard, results_residual; save_path=nothing)
    crystal_counts = sort(collect(keys(results_residual.aggregated_stats)))
    
    # MAE-Vergleich
    mae_standard = [results_standard.aggregated_stats[n].mae_mean for n in crystal_counts]
    mae_residual = [results_residual.aggregated_stats[n].mae_mean for n in crystal_counts]
    
    p = plot(crystal_counts, mae_standard,
             xlabel="Anzahl Kristalle",
             ylabel="MAE",
             title="Vergleich: Standard UNet vs. Residual UNet",
             label="Standard UNet",
             lw=2, marker=:circle, ms=6,
             legend=:topleft,
             size=(1000, 600))
    
    plot!(p, crystal_counts, mae_residual,
          label="Residual UNet",
          lw=2, marker=:square, ms=6)
    
    # Verbesserung annotieren
    for (i, n) in enumerate(crystal_counts)
        improvement = ((mae_standard[i] - mae_residual[i]) / mae_standard[i]) * 100
        annotate!(p, n, mae_residual[i], 
                 text(@sprintf("%.1f%%", improvement), 8, :bottom, :green))
    end
    
    if !isnothing(save_path)
        savefig(p, save_path)
        println("  ✓ Gespeichert: $save_path")
    end
    
    return p
end

println("✓ Residual Evaluation Visualisierung geladen")
println("  - plot_residual_decomposition(): 4-Panel Stokes|Residuum|Total|GT")
println("  - plot_mae_scaling(): MAE vs. Kristallanzahl")
println("  - plot_stokes_improvement(): Verbesserung über Baseline")
println("  - plot_continuity_violation(): Divergenz-Monitoring")
println("  - create_residual_dashboard(): Komplettes Dashboard")
println("  - compare_standard_vs_residual(): Vergleich mit Standard UNet")