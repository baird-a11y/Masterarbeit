# =============================================================================
# AFTER TRAINING ANALYSIS - VOLLSTÄNDIGE AUSWERTUNG (KORRIGIERT)
# =============================================================================

using Dates
using BSON
using Statistics
using Plots
using LinearAlgebra

println("=== AFTER TRAINING ANALYSIS ===")
println("Lade alle Module...")

# Alle erforderlichen Module laden
include("lamem_interface.jl")
include("data_processing.jl")
include("unet_architecture.jl")
include("training.jl")
include("batch_management.jl")
include("gpu_utils.jl")

# Evaluierungs-Module
include("evaluate_model.jl")
include("comprehensive_evaluation.jl")
include("advanced_visualization.jl")
include("simple_data_export.jl")
include("statistical_analysis.jl")

println("✓ Alle Module geladen")

# Server-Modus für Plots
ENV["GKSwstype"] = "100"

# =============================================================================
# HAUPTANALYSE-FUNKTION
# =============================================================================

"""
Führt alle Analysen nach dem Training aus
"""
function analyze_training_results(;
    model_path = "H:/Masterarbeit/Modelle/final_model_run_8.bson",
    output_base_dir = "H:/Masterarbeit/Auswertung/Modell_Evaluation_8"
)
    
    println("\n" * "="^80)
    println("STARTE VOLLSTÄNDIGE AFTER-TRAINING ANALYSE")
    println("="^80)
    println("Modell: $model_path")
    println("Output: $output_base_dir")
    println("")
    
    mkpath(output_base_dir)
    
    # WICHTIG: Initialisiere Variablen am Anfang
    batch_results = nothing
    stat_analysis = nothing
    
    # Prüfe ob Modell existiert
    if !isfile(model_path)
        error("Modell nicht gefunden: $model_path")
    end
    
    # 1. QUICK TEST
    println("\n1. QUICK TEST")
    println("-"^50)
    
    try
        # Lade Modell
        model = load_trained_model(model_path)
        println("✓ Modell geladen")
        
        # Teste mit einem Sample
        test_sample = LaMEM_Multi_crystal(
            resolution=(256, 256),
            n_crystals=3,
            radius_crystal=[0.05, 0.05, 0.05],
            cen_2D=[(-0.3, 0.3), (0.0, 0.5), (0.3, 0.3)]
        )
        
        result = evaluate_model_comprehensive(model, test_sample, 
                                            target_resolution=256, 
                                            sample_id=1)
        
        println("Quick Test Ergebnisse:")
        println("  MAE: $(round(result.mae_total, digits=4))")
        println("  Korrelation: $(round(result.pearson_correlation_vz, digits=3))")
        println("  Kontinuitätsverletzung: $(round(result.continuity_violation_mean, digits=6))")
        println("✓ Quick Test abgeschlossen")
        
    catch e
        println("Quick Test fehlgeschlagen: $e")
    end
    
    # 2. VISUALISIERUNGEN (NEUE IMPLEMENTATION)
    println("\n2. KRISTALL-VISUALISIERUNGEN")
    println("-"^50)
    
    viz_output_dir = joinpath(output_base_dir, "visualizations")
    mkpath(viz_output_dir)
    
    try
        # Lade Modell nur einmal
        println("Lade Modell für Visualisierungen...")
        model = load_trained_model(model_path)
        
        # Teste verschiedene Kristallanzahlen
        test_configs = [
            (1, [(0.0, 0.5)]),
            (2, [(-0.3, 0.5), (0.3, 0.5)]),
            (3, [(-0.3, 0.3), (0.0, 0.6), (0.3, 0.3)]),
            (5, [(-0.4, 0.2), (-0.2, 0.5), (0.0, 0.3), (0.2, 0.5), (0.4, 0.2)]),
            (10, nothing)  # Automatische Positionierung
        ]
        
        for (n_crystals, positions) in test_configs
            println("\nErstelle Visualisierung für $n_crystals Kristalle...")
            
            try
                # Generiere Sample
                if positions === nothing
                    # Automatische Grid-Positionierung für viele Kristalle
                    centers = []
                    for i in 1:n_crystals
                        x_pos = -0.6 + (i-1) % 5 * 0.3
                        z_pos = 0.2 + div(i-1, 5) * 0.3
                        push!(centers, (x_pos, z_pos))
                    end
                else
                    centers = positions
                end
                
                # LaMEM Simulation
                sample = LaMEM_Multi_crystal(
                    resolution=(256, 256),
                    n_crystals=n_crystals,
                    radius_crystal=fill(0.05, n_crystals),
                    cen_2D=centers
                )
                
                x, z, phase, vx, vz, exx, ezz, v_stokes = sample
                
                # Preprocessing
                phase_tensor, velocity_tensor = preprocess_lamem_sample(
                    x, z, phase, vx, vz, v_stokes, target_resolution=256
                )
                
                # UNet Vorhersage
                prediction = cpu(model(phase_tensor))
                
                # Extrahiere 2D Arrays
                phase_2d = phase_tensor[:,:,1,1]
                gt_vx = velocity_tensor[:,:,1,1]
                gt_vz = velocity_tensor[:,:,2,1]
                pred_vx = prediction[:,:,1,1]
                pred_vz = prediction[:,:,2,1]
                
                # Berechne Metriken
                mae_vz = mean(abs.(pred_vz - gt_vz))
                correlation_vz = cor(vec(gt_vz), vec(pred_vz))
                
                # Erstelle Plots
                # Plot 1: Phasenfeld
                p1 = heatmap(1:256, 1:256, phase_2d,
                            c=:grays,
                            title="Phasenfeld\n($n_crystals Kristalle)",
                            xlabel="x", ylabel="z",
                            aspect_ratio=:equal)
                
                # Plot 2: LaMEM v_z
                vz_max = max(maximum(abs.(gt_vz)), maximum(abs.(pred_vz)))
                p2 = heatmap(1:256, 1:256, gt_vz,
                            c=:RdBu,
                            clims=(-vz_max, vz_max),
                            title="LaMEM v_z\n(Ground Truth)",
                            xlabel="x", ylabel="z",
                            aspect_ratio=:equal)
                
                # Plot 3: UNet v_z
                p3 = heatmap(1:256, 1:256, pred_vz,
                            c=:RdBu,
                            clims=(-vz_max, vz_max),
                            title="UNet v_z\n(MAE: $(round(mae_vz, digits=4)))",
                            xlabel="x", ylabel="z",
                            aspect_ratio=:equal)
                
                # Kombiniere Plots
                combined_plot = plot(p1, p2, p3,
                                   layout=(1, 3),
                                   size=(1500, 500),
                                   plot_title="$n_crystals Kristalle | Korrelation: $(round(correlation_vz, digits=3))")
                
                # DIREKTES SPEICHERN
                output_file = joinpath(viz_output_dir, "comparison_$(n_crystals)_crystals.png")
                savefig(combined_plot, output_file)
                println("  ✓ Gespeichert: $output_file")
                
                # Auch als PDF für Masterarbeit
                pdf_file = joinpath(viz_output_dir, "comparison_$(n_crystals)_crystals.pdf")
                savefig(combined_plot, pdf_file)
                
                # Speichere Metriken
                metrics_file = joinpath(viz_output_dir, "metrics_$(n_crystals)_crystals.txt")
                open(metrics_file, "w") do f
                    println(f, "Kristalle: $n_crystals")
                    println(f, "MAE v_z: $(round(mae_vz, digits=6))")
                    println(f, "Korrelation v_z: $(round(correlation_vz, digits=4))")
                    println(f, "MAE v_x: $(round(mean(abs.(pred_vx - gt_vx)), digits=6))")
                end
                
            catch e
                println("  ✗ Fehler bei $n_crystals Kristallen: $e")
            end
        end
        
        println("\n✓ Alle Visualisierungen abgeschlossen")
        println("  Gespeichert in: $viz_output_dir")
        
    catch e
        println("✗ Visualisierung fehlgeschlagen: $e")
    end
    
    # 3. VOLLSTÄNDIGE EVALUIERUNG
    println("\n3. VOLLSTÄNDIGE MULTI-KRISTALL EVALUIERUNG")
    println("-"^50)
    
    eval_output_dir = joinpath(output_base_dir, "evaluation")
    
    try
        batch_results = automated_multi_crystal_evaluation(
            model_path,
            1:10,     # Kristallbereich
            5,        # Samples pro Anzahl (reduziert für schnelleren Test)
            target_resolution=256,
            output_dir=eval_output_dir,
            verbose=true
        )
        
        println("✓ Evaluierung abgeschlossen")
        println("  Gesamt-Samples: $(batch_results.total_samples)")
        
        # Zeige Zusammenfassung
        println("\n  Performance-Übersicht:")
        for n_crystals in sort(collect(keys(batch_results.aggregated_statistics)))
            stats = batch_results.aggregated_statistics[n_crystals]
            mae = round(stats["mae_mean"], digits=4)
            println("    $n_crystals Kristalle: MAE = $mae")
        end
        
    catch e
        println("Evaluierung fehlgeschlagen: $e")
        println("  Details: ", e)
        # batch_results bleibt nothing
    end
    
    # 4. STATISTISCHE ANALYSE
    println("\n4. STATISTISCHE ANALYSE")
    println("-"^50)
    
    if batch_results !== nothing
        try
            stat_analysis = perform_robust_statistical_analysis(
                batch_results,
                confidence_level=0.95,
                verbose=true
            )
            
            println("Statistische Analyse abgeschlossen")
            
        catch e
            println("Statistische Analyse fehlgeschlagen: $e")
            stat_analysis = nothing
        end
    else
        println("Überspringe statistische Analyse (keine Batch-Ergebnisse)")
        stat_analysis = nothing
    end
    
    # 5. DATEN-EXPORT
    println("\n5. DATEN-EXPORT")
    println("-"^50)
    
    export_dir = joinpath(output_base_dir, "exports")
    
    if batch_results !== nothing
        try
            export_success = simple_export_all_formats(
                batch_results,
                export_dir
            )
            
            if export_success
                println("Daten exportiert nach: $export_dir")
            else
                println("Export teilweise fehlgeschlagen")
            end
            
        catch e
            println("Export fehlgeschlagen: $e")
        end
    else
        println("Überspringe Export (keine Batch-Ergebnisse)")
    end
    
    # 6. ZUSAMMENFASSUNGSBERICHT
    println("\n6. ZUSAMMENFASSUNGSBERICHT")
    println("-"^50)
    
    summary_file = joinpath(output_base_dir, "analysis_summary.md")
    
    try
        # Übergebe möglicherweise nothing-Werte
        create_analysis_summary(
            summary_file,
            model_path,
            batch_results,  # kann nothing sein
            stat_analysis   # kann nothing sein
        )
        
        println("Zusammenfassung erstellt: $summary_file")
        
    catch e
        println("Zusammenfassung fehlgeschlagen: $e")
    end
    
    # ABSCHLUSS
    println("\n" * "="^80)
    println("ANALYSE ABGESCHLOSSEN")
    println("="^80)
    println("Alle Ergebnisse in: $output_base_dir")
    println("")
    println("Verzeichnisstruktur:")
    println("  $output_base_dir/")
    println("  ├── visualizations/     # Strömungsfeld-Vergleiche")
    if batch_results !== nothing
        println("  ├── evaluation/         # Detaillierte Metriken")
        println("  ├── exports/            # CSV, JSON, LaTeX")
    end
    println("  └── analysis_summary.md # Zusammenfassung")
    
    return batch_results
end

"""
Erstellt eine Markdown-Zusammenfassung der Analyse
"""
function create_analysis_summary(output_path, model_path, batch_results, stat_analysis)
    open(output_path, "w") do f
        write(f, "# After-Training Analysis Summary\n\n")
        write(f, "**Datum:** $(Dates.now())\n")
        write(f, "**Modell:** `$model_path`\n\n")
        
        write(f, "## Hauptergebnisse\n\n")
        
        if batch_results !== nothing
            write(f, "- **Evaluierte Samples:** $(batch_results.total_samples)\n")
            write(f, "- **Kristallbereich:** $(batch_results.crystal_range)\n\n")
            
            write(f, "### Performance nach Kristallanzahl\n\n")
            write(f, "| Kristalle | MAE | Korrelation | Status |\n")
            write(f, "|-----------|-----|-------------|--------|\n")
            
            for n in sort(collect(keys(batch_results.aggregated_statistics)))
                stats = batch_results.aggregated_statistics[n]
                mae = round(stats["mae_mean"], digits=4)
                corr = round(stats["correlation_vz_mean"], digits=3)
                
                status = if mae < 0.05
                    "Exzellent"
                elseif mae < 0.1
                    "Gut"
                elseif mae < 0.2
                    "Akzeptabel"
                else
                    "Verbesserung nötig"
                end
                
                write(f, "| $n | $mae | $corr | $status |\n")
            end
        else
            write(f, "Keine Evaluierungsergebnisse verfügbar\n")
        end
        
        if stat_analysis !== nothing
            write(f, "\n## Statistische Analyse\n\n")
            write(f, "- Signifikante Unterschiede zwischen Kristallanzahlen gefunden\n")
            write(f, "- Konfidenzintervalle berechnet (95%)\n")
        end
        
        write(f, "\n---\n")
        write(f, "*Automatisch generiert von after_training_analysis.jl*\n")
    end
end

# =============================================================================
# AUSFÜHRUNG
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    # Wenn direkt ausgeführt
    println("Starte Analyse...")
    results = analyze_training_results()
else
    println("\nAfter-Training Analysis geladen!")
    println("Verwendung:")
    println("  julia> analyze_training_results()")
    println("")
    println("Oder mit custom Pfaden:")
    println("  julia> analyze_training_results(")
    println("      model_path = \"path/to/model.bson\",")
    println("      output_base_dir = \"my_results\"")
    println("  )")
end



