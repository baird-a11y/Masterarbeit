# =============================================================================
# ADVANCED VISUALIZATION MODULE
# =============================================================================


using Plots
using Colors
using ColorSchemes
using Statistics
using StatsPlots
using PlotlyJS
using Printf, JSON3, Statistics, LinearAlgebra, Random, Dates, BSON, CSV, DataFrames, Plots, Colors, Serialization, StatsBase, Distributions, HypothesisTests, Flux, CUDA, LaMEM, GeophysicalModelGenerator
plotlyjs()

# Konfiguration für konsistente Visualisierung
const VIZ_CONFIG = (
    figure_size = (1200, 400),
    dpi = 300,
    font_size = 12,
    colormap_velocity = :RdBu_r,
    colormap_phase = :grays,
    marker_size_crystal = 8,
    marker_size_minima = 10,
    save_format = "png"
)

"""
Erstellt erweiterte 3-Panel-Visualisierung mit verbesserter wissenschaftlicher Darstellung
"""
function create_advanced_three_panel_plot(model, sample; 
                                        target_resolution=256, 
                                        save_path=nothing,
                                        title_prefix="",
                                        show_metrics=true,
                                        custom_colormap=nothing)
    
    println("Erstelle erweiterte 3-Panel Visualisierung...")
    
    try
        # 1. Sample verarbeiten und Evaluierung
        evaluation_result = evaluate_model_comprehensive(model, sample, 
                                                       target_resolution=target_resolution)
        
        # Daten extrahieren für Visualisierung
        x, z, phase, vx, vz, exx, ezz, v_stokes = sample
        
        phase_tensor, velocity_tensor = preprocess_lamem_sample(
            x, z, phase, vx, vz, v_stokes, target_resolution=target_resolution
        )
        
        prediction = cpu(model(phase_tensor))
        
        # 2D Arrays extrahieren
        phase_2d = phase_tensor[:,:,1,1]
        gt_vz = velocity_tensor[:,:,2,1]
        pred_vz = prediction[:,:,2,1]
        
        # 2. Kristall-Analyse
        crystal_centers = find_crystal_centers(phase_2d)
        gt_minima = find_velocity_minima(gt_vz, length(crystal_centers))
        pred_minima = find_velocity_minima(pred_vz, length(crystal_centers))
        
        # 3. Bestimme globale Farbskala für konsistente Darstellung
        vz_max = max(maximum(abs.(gt_vz)), maximum(abs.(pred_vz)))
        colormap = custom_colormap !== nothing ? custom_colormap : VIZ_CONFIG.colormap_velocity
        
        # 4. Panel 1: Phasenfeld mit Kristall-Zentren
        p1 = create_enhanced_phase_plot(phase_2d, crystal_centers, target_resolution, evaluation_result)
        
        # 5. Panel 2: LaMEM Ground Truth
        p2 = create_enhanced_velocity_plot(gt_vz, gt_minima, target_resolution, 
                                         "LaMEM Ground Truth v_z", vz_max, colormap)
        
        # 6. Panel 3: UNet Vorhersage  
        p3 = create_enhanced_velocity_plot(pred_vz, pred_minima, target_resolution, 
                                         "UNet Vorhersage v_z", vz_max, colormap)
        
        # 7. Layout mit erweiterten Metriken
        layout_title = create_comprehensive_title(title_prefix, evaluation_result, show_metrics)
        
        final_plot = plot(p1, p2, p3, 
                         layout=(1, 3), 
                         size=VIZ_CONFIG.figure_size,
                         plot_title=layout_title,
                         titlefontsize=VIZ_CONFIG.font_size+2,
                         dpi=VIZ_CONFIG.dpi)
        
        # 8. Speichern mit Metadaten
        if save_path !== nothing
            full_path = ensure_output_path(save_path)
            savefig(final_plot, full_path)
            
            # Zusätzlich Metadaten speichern
            save_visualization_metadata(full_path, evaluation_result, sample)
            
            println("Erweiterte Visualisierung gespeichert: $full_path")
        end
        
        return final_plot, evaluation_result
        
    catch e
        println("Fehler bei erweiterter Visualisierung: $e")
        return nothing, nothing
    end
end

"""
Erstellt verbessertes Phasenfeld-Panel mit detaillierter Kristall-Information
"""
function create_enhanced_phase_plot(phase_2d, crystal_centers, resolution, eval_result)
    # Basis Heatmap mit verbesserter Farbpalette
    p = heatmap(1:resolution, 1:resolution, phase_2d,
                c=VIZ_CONFIG.colormap_phase, 
                aspect_ratio=:equal,
                title="Phasenfeld ($(length(crystal_centers)) Kristalle)",
                xlabel="x [Pixel]", ylabel="z [Pixel]",
                xlims=(1, resolution), ylims=(1, resolution),
                titlefontsize=VIZ_CONFIG.font_size,
                guidefontsize=VIZ_CONFIG.font_size-1)
    
    # Kristall-Zentren mit verbesserter Markierung
    for (i, center) in enumerate(crystal_centers)
        x_coord, y_coord = center
        
        # Hauptmarkierung: Weiße Kreise mit schwarzem Rand
        scatter!(p, [x_coord], [y_coord], 
                markersize=VIZ_CONFIG.marker_size_crystal, 
                markercolor=:white, 
                markerstrokecolor=:black,
                markerstrokewidth=2,
                label=i==1 ? "Kristall-Zentren" : "")
        
        # Nummerierung für Multi-Kristall-Systeme
        if length(crystal_centers) > 3
            annotate!(p, x_coord+10, y_coord+10, text("$i", :black, 8))
        end
    end
    
    # Zusätzliche Informationen
    if eval_result !== nothing
        detection_rate = round(eval_result.crystal_detection_rate * 100, digits=1)
        complexity = round(eval_result.interaction_complexity_index, digits=2)
        
        annotate!(p, 0.05*resolution, 0.95*resolution, 
                 text("Erkennungsrate: $(detection_rate)%\nKomplexität: $complexity", 
                      :black, 8, :left))
    end
    
    return p
end

"""
Erstellt verbessertes Geschwindigkeitsfeld-Panel mit erweiterten Features
"""
function create_enhanced_velocity_plot(vz_field, velocity_minima, resolution, plot_title, vz_max, colormap)
    # Erweiterte Heatmap mit wissenschaftlicher Farbskala
    p = heatmap(1:resolution, 1:resolution, vz_field,
                c=colormap,
                aspect_ratio=:equal,
                title=plot_title,
                xlabel="x [Pixel]", ylabel="z [Pixel]",
                xlims=(1, resolution), ylims=(1, resolution),
                clims=(-vz_max, vz_max),
                titlefontsize=VIZ_CONFIG.font_size,
                guidefontsize=VIZ_CONFIG.font_size-1)
    
    # Geschwindigkeits-Minima mit verbesserter Sichtbarkeit
    for (i, minimum) in enumerate(velocity_minima)
        x_coord, y_coord = minimum
        
        # Sterne mit Kontrast-optimierter Darstellung
        scatter!(p, [x_coord], [y_coord], 
                markersize=VIZ_CONFIG.marker_size_minima, 
                markershape=:star5,
                markercolor=:yellow, 
                markerstrokecolor=:black,
                markerstrokewidth=2,
                label=i==1 ? "v_z Minima" : "")
    end
    
    # Geschwindigkeits-Statistiken
    vz_min, vz_max_val = extrema(vz_field)
    vz_mean = mean(vz_field)
    
    annotate!(p, 0.02*resolution, 0.95*resolution, 
             text("Min: $(round(vz_min, digits=3))\nMax: $(round(vz_max_val, digits=3))\nMean: $(round(vz_mean, digits=3))", 
                  :white, 7, :left))
    
    return p
end

"""
Erstellt umfassenden Titel mit Schlüsselmetriken
"""
function create_comprehensive_title(title_prefix, eval_result, show_metrics)
    base_title = isempty(title_prefix) ? "UNet Geschwindigkeitsfeld-Vorhersage" : title_prefix
    
    if !show_metrics || eval_result === nothing
        return base_title
    end
    
    # Kernmetriken für Titel
    mae = round(eval_result.mae_total, digits=4)
    correlation = round(eval_result.pearson_correlation_vz, digits=3)
    alignment = round(eval_result.alignment_error_mean, digits=1)
    
    metrics_text = " | MAE: $mae | Korr: $correlation | Align: $(alignment)px"
    
    return base_title * metrics_text
end

"""
Speichert Visualisierungs-Metadaten für Nachverfolgbarkeit
"""
function save_visualization_metadata(image_path, eval_result, sample)
    metadata_path = replace(image_path, ".png" => "_metadata.json")
    
    metadata = Dict(
        "timestamp" => string(now()),
        "image_path" => image_path,
        "crystal_count" => eval_result.crystal_count,
        "mae_total" => eval_result.mae_total,
        "correlation_vz" => eval_result.pearson_correlation_vz,
        "alignment_error" => eval_result.alignment_error_mean,
        "sample_info" => Dict(
            "v_stokes" => length(sample) >= 8 ? sample[8] : "unknown"
        )
    )
    
    open(metadata_path, "w") do f
        JSON3.pretty(f, metadata)
    end
end

"""
Batch-Visualisierung für systematischen Vergleich
"""
function create_systematic_crystal_comparison(model_path, crystal_counts=[1, 3, 5, 8, 10, 15];
                                            samples_per_count=3, output_dir="systematic_comparison")
    
    println("=== SYSTEMATISCHER KRISTALL-VERGLEICH ===")
    println("Kristallanzahlen: $crystal_counts")
    println("Samples pro Anzahl: $samples_per_count")
    
    mkpath(output_dir)
    
    # Modell laden
    model = load_trained_model(model_path)
    
    # Für jede Kristallanzahl
    for n_crystals in crystal_counts
        println("\nErstelle Visualisierungen für $n_crystals Kristalle...")
        
        crystal_dir = joinpath(output_dir, "$(n_crystals)_crystals")
        mkpath(crystal_dir)
        
        for sample_id in 1:samples_per_count
            try
                # Sample generieren mit optimierter Platzierung
                sample = generate_optimized_crystal_sample(n_crystals, 256)
                
                # Visualisierung erstellen
                save_name = "$(n_crystals)_crystals_sample_$(sample_id).png"
                save_path = joinpath(crystal_dir, save_name)
                
                plot_result, eval_result = create_advanced_three_panel_plot(
                    model, sample,
                    save_path=save_path,
                    title_prefix="$n_crystals Kristalle (Sample $sample_id)",
                    show_metrics=true
                )
                
                if plot_result !== nothing && eval_result !== nothing
                    println("  Sample $sample_id: MAE=$(round(eval_result.mae_total, digits=4))")
                else
                    println("  Sample $sample_id: Fehler bei Erstellung")
                end
                
            catch e
                println("  Sample $sample_id: Fehler - $e")
                continue
            end
        end
    end
    
    println("\nSystematischer Vergleich abgeschlossen in: $output_dir")
end

"""
Erstellt Skalierungs-Analyse-Plots
"""
function create_scaling_analysis_plots(batch_results::BatchEvaluationResults, output_dir="scaling_analysis")
    
    println("=== SKALIERUNGS-ANALYSE PLOTS ===")
    mkpath(output_dir)
    
    # 1. Performance vs. Kristallanzahl
    p1 = create_performance_scaling_plot(batch_results)
    savefig(p1, joinpath(output_dir, "performance_scaling.png"))
    
    # 2. Korrelations-Heatmap zwischen Metriken
    p2 = create_metrics_correlation_heatmap(batch_results)
    savefig(p2, joinpath(output_dir, "metrics_correlation.png"))
    
    # 3. Fehler-Verteilungs-Histogramme
    p3 = create_error_distribution_histograms(batch_results)
    savefig(p3, joinpath(output_dir, "error_distributions.png"))
    
    # 4. Komplexitäts-Index vs. Performance
    p4 = create_complexity_performance_scatter(batch_results)
    savefig(p4, joinpath(output_dir, "complexity_vs_performance.png"))
    
    println("Skalierungs-Analyse-Plots erstellt in: $output_dir")
    
    return [p1, p2, p3, p4]
end

"""
Performance-Skalierungs-Plot
"""
function create_performance_scaling_plot(batch_results)
    crystal_counts = batch_results.scaling_metrics["crystal_counts"]
    mae_progression = batch_results.scaling_metrics["mae_progression"]
    correlation_progression = batch_results.scaling_metrics["correlation_progression"]
    
    # Dual-Achsen Plot
    p = plot(crystal_counts, mae_progression, 
             label="MAE", 
             linewidth=3,
             marker=:circle,
             markersize=6,
             color=:red,
             ylabel="Mean Absolute Error",
             xlabel="Anzahl Kristalle",
             title="Performance-Skalierung mit Kristallanzahl",
             legend=:topleft)
    
    # Zweite Y-Achse für Korrelation
    p2 = twinx(p)
    plot!(p2, crystal_counts, correlation_progression,
          label="Korrelation v_z",
          linewidth=3,
          marker=:square,
          markersize=6,
          color=:blue,
          ylabel="Pearson Korrelation",
          legend=:topright)
    
    return p
end

"""
Metriken-Korrelations-Heatmap
"""
function create_metrics_correlation_heatmap(batch_results)
    # Sammle alle Metriken aus allen Kristallanzahlen
    all_results = []
    for (n_crystals, results) in batch_results.results_per_crystal
        append!(all_results, results)
    end
    
    if isempty(all_results)
        return plot(title="Keine Daten für Korrelations-Heatmap")
    end
    
    # Extrahiere numerische Metriken
    metrics_data = hcat(
        [r.mae_total for r in all_results],
        [r.rmse_total for r in all_results],
        [r.pearson_correlation_vz for r in all_results],
        [r.alignment_error_mean for r in all_results if !isinf(r.alignment_error_mean)],
        [r.ssim_vz for r in all_results],
        [r.crystal_detection_rate for r in all_results]
    )
    
    metric_names = ["MAE", "RMSE", "Correlation", "Alignment", "SSIM", "Detection Rate"]
    
    # Korrelationsmatrix berechnen
    correlation_matrix = cor(metrics_data)
    
    # Heatmap erstellen
    return heatmap(correlation_matrix,
                  xticks=(1:length(metric_names), metric_names),
                  yticks=(1:length(metric_names), metric_names),
                  c=:RdBu,
                  aspect_ratio=:equal,
                  title="Metriken-Korrelations-Matrix",
                  clims=(-1, 1))
end

"""
Fehler-Verteilungs-Histogramme
"""
function create_error_distribution_histograms(batch_results)
    plots_array = []
    
    for (n_crystals, results) in sort(collect(batch_results.results_per_crystal))
        if length(results) < 3
            continue
        end
        
        mae_values = [r.mae_total for r in results]
        
        p = histogram(mae_values,
                     bins=10,
                     alpha=0.7,
                     title="$n_crystals Kristalle",
                     xlabel="MAE",
                     ylabel="Häufigkeit",
                     legend=false)
        
        push!(plots_array, p)
    end
    
    if isempty(plots_array)
        return plot(title="Keine Daten für Histogramme")
    end
    
    return plot(plots_array..., layout=(2, 3), size=(900, 600))
end

"""
Komplexitäts-Performance-Scatter
"""
function create_complexity_performance_scatter(batch_results)
    complexity_values = Float64[]
    mae_values = Float64[]
    crystal_counts = Int[]
    
    for (n_crystals, results) in batch_results.results_per_crystal
        for result in results
            push!(complexity_values, result.interaction_complexity_index)
            push!(mae_values, result.mae_total)
            push!(crystal_counts, n_crystals)
        end
    end
    
    return scatter(complexity_values, mae_values,
                  color=crystal_counts,
                  markersize=6,
                  alpha=0.7,
                  xlabel="Interaktions-Komplexitäts-Index",
                  ylabel="Mean Absolute Error",
                  title="Komplexität vs. Performance",
                  colorbar_title="Anzahl Kristalle")
end

"""
Hilfsfunktion: Optimierte Sample-Generierung für Visualisierung
"""
function generate_optimized_crystal_sample(n_crystals, resolution)
    # Adaptive Radius-Bestimmung
    base_radius = 0.06
    radius_factor = max(0.5, 1.0 - 0.05 * (n_crystals - 1))
    radius = base_radius * radius_factor
    
    radius_crystal = fill(radius, n_crystals)
    
    # Optimierte Positionierung basierend auf Kristallanzahl
    if n_crystals == 1
        centers = [(0.0, 0.0)]
    elseif n_crystals == 2
        centers = [(-0.4, 0.0), (0.4, 0.0)]
    elseif n_crystals <= 4
        # Quadrat-Formation
        positions = [(-0.3, -0.3), (0.3, -0.3), (-0.3, 0.3), (0.3, 0.3)]
        centers = positions[1:n_crystals]
    elseif n_crystals <= 9
        # 3x3 Grid
        centers = []
        for i in 1:n_crystals
            x_pos = -0.4 + (i-1) % 3 * 0.4
            z_pos = -0.4 + div(i-1, 3) * 0.4
            push!(centers, (x_pos, z_pos))
        end
    else
        # Dichteres Grid für > 9 Kristalle
        centers = []
        grid_size = ceil(Int, sqrt(n_crystals))
        spacing = 1.6 / (grid_size + 1)
        
        for i in 1:n_crystals
            x_pos = -0.8 + (i-1) % grid_size * spacing
            z_pos = -0.8 + div(i-1, grid_size) * spacing
            push!(centers, (x_pos, z_pos))
        end
    end
    
    return LaMEM_Multi_crystal(
        resolution=(resolution, resolution),
        n_crystals=n_crystals,
        radius_crystal=radius_crystal,
        cen_2D=centers
    )
end

"""
Stellt sicheren Output-Pfad sicher
"""
function ensure_output_path(save_path)
    if isabspath(save_path)
        return save_path
    else
        # Standardverzeichnis verwenden
        output_dir = "advanced_visualizations"
        mkpath(output_dir)
        return joinpath(output_dir, save_path)
    end
end

"""
Interaktive Dashboard-Erstellung (für Jupyter/Pluto)
"""
function create_interactive_dashboard(batch_results::BatchEvaluationResults)
    println("=== INTERAKTIVES DASHBOARD ===")
    
    # Sammle alle Daten
    crystal_counts = []
    mae_values = []
    correlations = []
    
    for (n_crystals, results) in batch_results.results_per_crystal
        for result in results
            push!(crystal_counts, n_crystals)
            push!(mae_values, result.mae_total)
            push!(correlations, result.pearson_correlation_vz)
        end
    end
    
    # Interaktiver Scatter-Plot mit PlotlyJS
    p = plot(crystal_counts, mae_values,
             seriestype=:scatter,
             color=correlations,
             markersize=8,
             alpha=0.7,
             xlabel="Anzahl Kristalle",
             ylabel="Mean Absolute Error",
             title="Interaktive Performance-Analyse",
             colorbar_title="Korrelation v_z",
             hover=true)
    
    return p
end

println("Advanced Visualization Module geladen!")
println("Verfügbare Funktionen:")
println("  - create_advanced_three_panel_plot() - Erweiterte 3-Panel Visualisierung")
println("  - create_systematic_crystal_comparison() - Systematischer Batch-Vergleich")
println("  - create_scaling_analysis_plots() - Skalierungs-Analyse-Plots")
println("  - create_interactive_dashboard() - Interaktives Dashboard")
println("")
println("Haupteinstiegspunkt: create_systematic_crystal_comparison(model_path)")