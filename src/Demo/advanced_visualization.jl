# =============================================================================
# ADVANCED VISUALIZATION MODULE (KORRIGIERT)
# =============================================================================
# Speichern als: advanced_visualization.jl

using Plots
using Colors
using ColorSchemes
using Statistics
using JSON3
using Dates

# Konfiguration für konsistente Visualisierung
const VIZ_CONFIG = (
    figure_size = (1200, 400),
    dpi = 300,
    font_size = 12,
    colormap_velocity = :RdBu,  # Korrigiert: RdBu statt RdBu_r
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
        # 1. Sample verarbeiten
        x, z, phase, vx, vz, exx, ezz, v_stokes = sample
        
        phase_tensor, velocity_tensor = preprocess_lamem_sample(
            x, z, phase, vx, vz, v_stokes, target_resolution=target_resolution
        )
        
        prediction = cpu(model(phase_tensor))
        
        # 2D Arrays extrahieren
        phase_2d = phase_tensor[:,:,1,1]
        gt_vx = velocity_tensor[:,:,1,1]
        gt_vz = velocity_tensor[:,:,2,1]
        pred_vx = prediction[:,:,1,1]
        pred_vz = prediction[:,:,2,1]
        
        # 2. Evaluierung durchführen
        eval_result = evaluate_single_sample(phase_2d, gt_vx, gt_vz, pred_vx, pred_vz)
        
        # 3. Kristall-Analyse
        crystal_centers = find_crystal_centers(phase_2d)
        gt_minima = find_velocity_minima(gt_vz, length(crystal_centers))
        pred_minima = find_velocity_minima(pred_vz, length(crystal_centers))
        
        # 4. Bestimme globale Farbskala für konsistente Darstellung
        vz_max = max(maximum(abs.(gt_vz)), maximum(abs.(pred_vz)))
        colormap = custom_colormap !== nothing ? custom_colormap : VIZ_CONFIG.colormap_velocity
        
        # 5. Panel 1: Phasenfeld mit Kristall-Zentren
        p1 = create_enhanced_phase_plot(phase_2d, crystal_centers, target_resolution, eval_result)
        
        # 6. Panel 2: LaMEM Ground Truth
        p2 = create_enhanced_velocity_plot(gt_vz, gt_minima, target_resolution, 
                                         "LaMEM Ground Truth v_z", vz_max, colormap)
        
        # 7. Panel 3: UNet Vorhersage  
        p3 = create_enhanced_velocity_plot(pred_vz, pred_minima, target_resolution, 
                                         "UNet Vorhersage v_z", vz_max, colormap)
        
        # 8. Layout mit erweiterten Metriken
        layout_title = create_comprehensive_title(title_prefix, eval_result, show_metrics)
        
        final_plot = plot(p1, p2, p3, 
                         layout=(1, 3), 
                         size=VIZ_CONFIG.figure_size,
                         plot_title=layout_title,
                         titlefontsize=VIZ_CONFIG.font_size+2,
                         dpi=VIZ_CONFIG.dpi)
        
        # 9. Speichern mit Metadaten
        if save_path !== nothing
            full_path = ensure_output_path(save_path)
            savefig(final_plot, full_path)
            
            # Metadaten speichern
            save_visualization_metadata(full_path, eval_result, sample)
            
            println("Erweiterte Visualisierung gespeichert: $full_path")
        end
        
        return final_plot, eval_result
        
    catch e
        println("Fehler bei erweiterter Visualisierung: $e")
        return nothing, nothing
    end
end

"""
Vereinfachte Evaluierung für einzelnes Sample
"""
function evaluate_single_sample(phase_2d, gt_vx, gt_vz, pred_vx, pred_vz)
    # MAE und RMSE berechnen
    mae_vx = mean(abs.(pred_vx .- gt_vx))
    mae_vz = mean(abs.(pred_vz .- gt_vz))
    mae_total = (mae_vx + mae_vz) / 2
    
    rmse_vx = sqrt(mean((pred_vx .- gt_vx).^2))
    rmse_vz = sqrt(mean((pred_vz .- gt_vz).^2))
    rmse_total = (rmse_vx + rmse_vz) / 2
    
    # Pearson Korrelation
    correlation_vx = cor(vec(gt_vx), vec(pred_vx))
    correlation_vz = cor(vec(gt_vz), vec(pred_vz))
    
    # Kristall-Analyse
    crystal_centers = find_crystal_centers(phase_2d)
    gt_minima = find_velocity_minima(gt_vz, length(crystal_centers))
    pred_minima = find_velocity_minima(pred_vz, length(crystal_centers))
    
    alignment_error = calculate_alignment_error(crystal_centers, pred_minima)
    
    # Strukturelle Ähnlichkeit (vereinfacht)
    ssim_vz = calculate_simple_ssim(gt_vz, pred_vz)
    
    # Ergebnis-Dictionary
    return (
        mae_total = mae_total,
        mae_vx = mae_vx,
        mae_vz = mae_vz,
        rmse_total = rmse_total,
        rmse_vx = rmse_vx,
        rmse_vz = rmse_vz,
        pearson_correlation_vx = correlation_vx,
        pearson_correlation_vz = correlation_vz,
        alignment_error_mean = alignment_error,
        crystal_count = length(crystal_centers),
        crystal_detection_rate = length(pred_minima) / max(1, length(crystal_centers)),
        interaction_complexity_index = calculate_complexity_index(phase_2d, gt_vz),
        ssim_vz = ssim_vz
    )
end

"""
Berechnet vereinfachten SSIM (Structural Similarity Index)
"""
function calculate_simple_ssim(img1, img2; window_size=11)
    # Vereinfachte SSIM-Berechnung
    mu1 = mean(img1)
    mu2 = mean(img2)
    
    sigma1 = std(img1)
    sigma2 = std(img2)
    
    sigma12 = cov(vec(img1), vec(img2))
    
    C1 = (0.01 * 1.0)^2  # Kleine Konstante für Stabilität
    C2 = (0.03 * 1.0)^2
    
    ssim = ((2*mu1*mu2 + C1) * (2*sigma12 + C2)) / 
           ((mu1^2 + mu2^2 + C1) * (sigma1^2 + sigma2^2 + C2))
    
    return ssim
end

"""
Berechnet Komplexitäts-Index basierend auf Kristallinteraktionen
"""
function calculate_complexity_index(phase_field, velocity_field)
    # Anzahl der Kristalle
    crystal_centers = find_crystal_centers(phase_field)
    n_crystals = length(crystal_centers)
    
    if n_crystals <= 1
        return 0.0
    end
    
    # Mittlerer Abstand zwischen Kristallen
    mean_distance = 0.0
    count = 0
    for i in 1:n_crystals
        for j in i+1:n_crystals
            dist = sqrt((crystal_centers[i][1] - crystal_centers[j][1])^2 + 
                       (crystal_centers[i][2] - crystal_centers[j][2])^2)
            mean_distance += dist
            count += 1
        end
    end
    
    if count > 0
        mean_distance /= count
    end
    
    # Geschwindigkeitsfeld-Komplexität (Gradient-basiert)
    grad_x = diff(velocity_field, dims=1)
    grad_z = diff(velocity_field, dims=2)
    
    gradient_complexity = mean(abs.(grad_x)) + mean(abs.(grad_z))
    
    # Kombinierter Index
    complexity_index = n_crystals * gradient_complexity / max(mean_distance, 0.1)
    
    return complexity_index
end

"""
Erstellt verbessertes Phasenfeld-Panel mit detaillierter Kristall-Information
"""
function create_enhanced_phase_plot(phase_2d, crystal_centers, resolution, eval_result)
    # Basis Heatmap
    p = heatmap(1:resolution, 1:resolution, phase_2d,
                c=VIZ_CONFIG.colormap_phase, 
                aspect_ratio=:equal,
                title="Phasenfeld ($(length(crystal_centers)) Kristalle)",
                xlabel="x [Pixel]", ylabel="z [Pixel]",
                xlims=(1, resolution), ylims=(1, resolution),
                titlefontsize=VIZ_CONFIG.font_size,
                guidefontsize=VIZ_CONFIG.font_size-1)
    
    # Kristall-Zentren markieren
    for (i, center) in enumerate(crystal_centers)
        x_coord, y_coord = center
        
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
                 text("Erkennung: $(detection_rate)%\nKomplexität: $complexity", 
                      :black, 8, :left))
    end
    
    return p
end

"""
Erstellt verbessertes Geschwindigkeitsfeld-Panel
"""
function create_enhanced_velocity_plot(vz_field, velocity_minima, resolution, plot_title, vz_max, colormap)
    # Heatmap mit wissenschaftlicher Farbskala
    p = heatmap(1:resolution, 1:resolution, vz_field,
                c=colormap,
                aspect_ratio=:equal,
                title=plot_title,
                xlabel="x [Pixel]", ylabel="z [Pixel]",
                xlims=(1, resolution), ylims=(1, resolution),
                clims=(-vz_max, vz_max),
                titlefontsize=VIZ_CONFIG.font_size,
                guidefontsize=VIZ_CONFIG.font_size-1)
    
    # Geschwindigkeits-Minima markieren
    for (i, minimum) in enumerate(velocity_minima)
        x_coord, y_coord = minimum
        
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
Speichert Visualisierungs-Metadaten
"""
function save_visualization_metadata(image_path, eval_result, sample)
    metadata_path = replace(image_path, ".png" => "_metadata.txt")
    
    open(metadata_path, "w") do f
        println(f, "Visualisierungs-Metadaten")
        println(f, "========================")
        println(f, "Zeitstempel: $(Dates.now())")
        println(f, "Bildpfad: $image_path")
        println(f, "")
        println(f, "Evaluierungsergebnisse:")
        println(f, "  Kristallanzahl: $(eval_result.crystal_count)")
        println(f, "  MAE Total: $(round(eval_result.mae_total, digits=6))")
        println(f, "  Korrelation v_z: $(round(eval_result.pearson_correlation_vz, digits=4))")
        println(f, "  Alignment-Fehler: $(round(eval_result.alignment_error_mean, digits=2)) px")
        println(f, "  SSIM v_z: $(round(eval_result.ssim_vz, digits=4))")
        println(f, "  Komplexitäts-Index: $(round(eval_result.interaction_complexity_index, digits=3))")
        
        if length(sample) >= 8
            println(f, "")
            println(f, "Sample-Information:")
            println(f, "  Stokes-Geschwindigkeit: $(sample[8]) cm/Jahr")
        end
    end
    
    println("  Metadaten gespeichert: $metadata_path")
end

"""
Stellt sicheren Output-Pfad sicher
"""
function ensure_output_path(save_path)
    if isabspath(save_path)
        dir = dirname(save_path)
        if !isdir(dir)
            mkpath(dir)
        end
        return save_path
    else
        # Standardverzeichnis verwenden
        output_dir = "advanced_visualizations"
        mkpath(output_dir)
        return joinpath(output_dir, save_path)
    end
end

"""
Batch-Visualisierung für systematischen Vergleich
"""
function create_systematic_crystal_comparison(model_path; 
                                            crystal_counts=[1, 3, 5, 8, 10, 15],
                                            samples_per_count=3, 
                                            output_dir="systematic_comparison")
    
    println("=== SYSTEMATISCHER KRISTALL-VERGLEICH ===")
    println("Kristallanzahlen: $crystal_counts")
    println("Samples pro Anzahl: $samples_per_count")
    
    mkpath(output_dir)
    
    # Modell laden
    model = load_trained_model(model_path)
    
    all_results = []
    
    # Für jede Kristallanzahl
    for n_crystals in crystal_counts
        println("\nErstelle Visualisierungen für $n_crystals Kristalle...")
        
        crystal_dir = joinpath(output_dir, "$(n_crystals)_crystals")
        mkpath(crystal_dir)
        
        crystal_results = []
        
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
                    push!(crystal_results, eval_result)
                    println("  Sample $sample_id: MAE=$(round(eval_result.mae_total, digits=4)), Korr=$(round(eval_result.pearson_correlation_vz, digits=3))")
                else
                    println("  Sample $sample_id: Fehler bei Erstellung")
                end
                
            catch e
                println("  Sample $sample_id: Fehler - $e")
                continue
            end
        end
        
        # Statistiken für diese Kristallanzahl
        if !isempty(crystal_results)
            mean_mae = mean([r.mae_total for r in crystal_results])
            mean_corr = mean([r.pearson_correlation_vz for r in crystal_results])
            println("  Durchschnitt für $n_crystals Kristalle: MAE=$(round(mean_mae, digits=4)), Korr=$(round(mean_corr, digits=3))")
            
            push!(all_results, (n_crystals=n_crystals, results=crystal_results))
        end
    end
    
    # Zusammenfassungs-Plot erstellen
    if !isempty(all_results)
        summary_plot = create_scaling_summary_plot(all_results)
        savefig(summary_plot, joinpath(output_dir, "scaling_summary.png"))
        println("\nZusammenfassungs-Plot erstellt: $(joinpath(output_dir, "scaling_summary.png"))")
    end
    
    println("\nSystematischer Vergleich abgeschlossen in: $output_dir")
    
    return all_results
end

"""
Erstellt Zusammenfassungs-Plot für Skalierungsanalyse
"""
function create_scaling_summary_plot(all_results)
    crystal_counts = [r.n_crystals for r in all_results]
    mean_mae = [mean([res.mae_total for res in r.results]) for r in all_results]
    mean_corr = [mean([res.pearson_correlation_vz for res in r.results]) for r in all_results]
    
    # Zwei-Achsen-Plot
    p1 = plot(crystal_counts, mean_mae,
             label="MAE",
             linewidth=3,
             marker=:circle,
             markersize=8,
             color=:red,
             ylabel="Mean Absolute Error",
             xlabel="Anzahl Kristalle",
             title="Performance-Skalierung mit Kristallanzahl",
             legend=:topleft,
             grid=true,
             minorgrid=true)
    
    p2 = plot(crystal_counts, mean_corr,
             label="Korrelation v_z",
             linewidth=3,
             marker=:square,
             markersize=8,
             color=:blue,
             ylabel="Pearson Korrelation",
             xlabel="Anzahl Kristalle",
             legend=:topright,
             grid=true,
             minorgrid=true)
    
    # Kombinierter Plot
    final_plot = plot(p1, p2, layout=(2, 1), size=(800, 800))
    
    return final_plot
end

"""
Optimierte Sample-Generierung für Visualisierung
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
            x_pos = -0.8 + (i-1) % grid_size * spacing + spacing
            z_pos = -0.8 + div(i-1, grid_size) * spacing + spacing
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

println("Advanced Visualization Module geladen!")
println("Verfügbare Funktionen:")
println("  - create_advanced_three_panel_plot() - Erweiterte 3-Panel Visualisierung")
println("  - create_systematic_crystal_comparison() - Systematischer Batch-Vergleich")
println("")
println("Haupteinstiegspunkt: create_systematic_crystal_comparison(model_path)")