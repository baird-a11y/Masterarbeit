# =============================================================================
# MODELL-EVALUIERUNG MIT VISUALISIERUNG - 1 BIS 10 KRISTALLE
# =============================================================================

using Plots
using Statistics
using Printf
using BSON
using StatsBase
using Flux

# Lade alle benötigten Module
println("Lade benötigte Module...")
try
    include("lamem_interface.jl")
    include("data_processing.jl") 
    include("unet_architecture.jl")
    include("training.jl")
    include("batch_management.jl")
    println("Alle Module erfolgreich geladen")
catch e
    println("Warnung: Einige Module konnten nicht geladen werden: $e")
    println("Stelle sicher, dass alle .jl Dateien im gleichen Verzeichnis sind")
end

"""
Lädt ein gespeichertes Modell (falls nicht bereits verfügbar)
"""
function load_trained_model_safe(model_path::String)
    println("Lade Modell: $model_path")
    
    if !isfile(model_path)
        error("Modelldatei nicht gefunden: $model_path")
    end
    
    try
        model_dict = BSON.load(model_path)
        
        # Versuche verschiedene Schlüssel
        for key in [:model, :best_model, :final_model, :trained_model]
            if haskey(model_dict, key)
                model = model_dict[key]
                println("Modell unter Schlüssel '$key' gefunden")
                return model
            end
        end
        
        # Fallback: Nehme ersten Wert der ein Modell sein könnte
        for (key, value) in model_dict
            if isa(value, Flux.Chain) || hasproperty(value, :layers) || string(typeof(value)) |> x -> contains(x, "UNet")
                println("Modell unter Schlüssel '$key' gefunden")
                return value
            end
        end
        
        error("Kein Modell in der BSON-Datei gefunden")
        
    catch e
        error("Fehler beim Laden des Modells: $e")
    end
end

"""
Struktur für Evaluierungsmetriken pro Kristallanzahl
"""
struct CrystalEvaluationMetrics
    crystal_count::Int
    n_samples::Int
    
    # Absolute Fehler-Metriken
    mae_vx::Float64          # Mean Absolute Error v_x
    mae_vz::Float64          # Mean Absolute Error v_z  
    mae_total::Float64       # Gesamter MAE
    
    # Zusätzliche Metriken
    max_absolute_error::Float64      # Maximaler absoluter Fehler
    median_absolute_error::Float64   # Median absoluter Fehler
    std_absolute_error::Float64      # Standardabweichung absoluter Fehler
    
    # Physikalische Plausibilität
    continuity_error::Float64        # Kontinuitätsgleichungs-Verletzung
    
    # Performance-Indikatoren
    r2_total::Float64                # Bestimmtheitsmaß
    relative_error_percent::Float64  # Relativer Fehler in Prozent
end

"""
Berechnet umfassende Evaluierungsmetriken für eine Kristallanzahl
"""
function calculate_crystal_metrics(predictions, targets, crystal_count::Int)
    # Flatten arrays für Berechnungen
    pred_vx = vec(predictions[:,:,1,:])
    pred_vz = vec(predictions[:,:,2,:])
    true_vx = vec(targets[:,:,1,:])
    true_vz = vec(targets[:,:,2,:])
    
    n_samples = size(predictions, 4)
    
    # Absolute Fehler berechnen
    abs_errors_vx = abs.(pred_vx .- true_vx)
    abs_errors_vz = abs.(pred_vz .- true_vz)
    abs_errors_total = [abs_errors_vx; abs_errors_vz]
    
    # Mean Absolute Error
    mae_vx = mean(abs_errors_vx)
    mae_vz = mean(abs_errors_vz)
    mae_total = (mae_vx + mae_vz) / 2
    
    # Weitere statistische Metriken
    max_absolute_error = maximum(abs_errors_total)
    median_absolute_error = median(abs_errors_total)
    std_absolute_error = std(abs_errors_total)
    
    # R-squared
    r2_vx = calculate_r_squared(pred_vx, true_vx)
    r2_vz = calculate_r_squared(pred_vz, true_vz)
    r2_total = (r2_vx + r2_vz) / 2
    
    # Relativer Fehler
    true_magnitude = sqrt.(true_vx.^2 .+ true_vz.^2)
    mean_true_magnitude = mean(true_magnitude[true_magnitude .> 1e-8])  # Vermeide Division durch 0
    relative_error_percent = (mae_total / mean_true_magnitude) * 100
    
    # Kontinuitätsfehler
    continuity_error = calculate_continuity_error(predictions)
    
    return CrystalEvaluationMetrics(
        crystal_count, n_samples,
        mae_vx, mae_vz, mae_total,
        max_absolute_error, median_absolute_error, std_absolute_error,
        continuity_error, r2_total, relative_error_percent
    )
end

"""
Berechnet R² (Bestimmtheitsmaß)
"""
function calculate_r_squared(predicted, actual)
    if length(predicted) != length(actual) || length(predicted) == 0
        return 0.0
    end
    
    ss_res = sum((actual .- predicted).^2)
    ss_tot = sum((actual .- mean(actual)).^2)
    
    return ss_tot ≈ 0.0 ? 1.0 : max(0.0, 1.0 - ss_res / ss_tot)
end

"""
Berechnet Kontinuitätsgleichungs-Verletzung
"""
function calculate_continuity_error(velocity_field)
    h, w, channels, batch = size(velocity_field)
    continuity_errors = []
    
    for b in 1:batch
        vx = velocity_field[:, :, 1, b]
        vz = velocity_field[:, :, 2, b]
        
        # Numerische Ableitungen (zentrale Differenzen)
        dvx_dx = zeros(h, w)
        dvz_dz = zeros(h, w)
        
        # Innere Punkte
        for i in 2:h-1, j in 2:w-1
            dvx_dx[i, j] = (vx[i, j+1] - vx[i, j-1]) / 2
            dvz_dz[i, j] = (vz[i+1, j] - vz[i-1, j]) / 2
        end
        
        # Kontinuitätsfehler
        continuity = abs.(dvx_dx .+ dvz_dz)
        push!(continuity_errors, mean(continuity[2:end-1, 2:end-1]))
    end
    
    return mean(continuity_errors)
end

"""
Generiert Evaluierungsdatensatz für spezifische Kristallanzahl (falls nicht verfügbar)
"""
function generate_evaluation_dataset_safe(n_crystals::Int, n_samples::Int; resolution=256, verbose=true)
    if verbose
        println("Generiere Evaluierungsdatensatz: $n_crystals Kristalle, $n_samples Samples")
    end
    
    dataset = []
    
    for i in 1:n_samples
        try
            # Verwende verfügbare LaMEM-Funktion
            if isdefined(Main, :LaMEM_Variable_crystals)
                sample = LaMEM_Variable_crystals(
                    resolution=(resolution, resolution),
                    n_crystals=n_crystals
                )
            elseif isdefined(Main, :LaMEM_Multi_crystal)
                sample = LaMEM_Multi_crystal(
                    resolution=(resolution, resolution),
                    n_crystals=n_crystals,
                    radius_crystal=[0.05 for _ in 1:n_crystals],
                    cen_2D=[(rand(-0.5:0.1:0.5), rand(0.2:0.1:0.8)) for _ in 1:n_crystals]
                )
            else
                error("Keine LaMEM-Funktion verfügbar")
            end
            
            push!(dataset, sample)
            
            if verbose && i % max(1, n_samples ÷ 10) == 0
                println("  Evaluierungssample $i/$n_samples generiert")
            end
            
        catch e
            if verbose
                println("  Warnung: Sample $i fehlgeschlagen: $e")
            end
            continue
        end
        
        if i % 10 == 0
            GC.gc()
        end
    end
    
    if verbose
        println("Evaluierungsdatensatz: $(length(dataset))/$n_samples Samples erfolgreich")
    end
    
    return dataset
end

"""
Sichere Preprocessing-Funktion (falls nicht verfügbar)
"""
function preprocess_lamem_sample_safe(x, z, phase, vx, vz, v_stokes; target_resolution=256)
    try
        # Versuche die ursprüngliche Funktion
        if isdefined(Main, :preprocess_lamem_sample)
            return preprocess_lamem_sample(x, z, phase, vx, vz, v_stokes, target_resolution=target_resolution)
        else
            # Fallback: Einfache Implementierung
            # Größenanpassung
            current_size = size(phase, 1)
            if current_size != target_resolution
                # Einfache Interpolation
                scale_factor = target_resolution / current_size
                new_indices_x = round.(Int, range(1, stop=current_size, length=target_resolution))
                new_indices_z = round.(Int, range(1, stop=current_size, length=target_resolution))
                
                phase_resized = phase[new_indices_x, new_indices_z]
                vx_resized = vx[new_indices_x, new_indices_z]
                vz_resized = vz[new_indices_x, new_indices_z]
            else
                phase_resized = phase
                vx_resized = vx
                vz_resized = vz
            end
            
            # Normalisierung
            vx_norm = Float32.(vx_resized ./ v_stokes)
            vz_norm = Float32.(vz_resized ./ v_stokes)
            phase_float = Float32.(phase_resized)
            
            # Tensor-Format
            phase_tensor = reshape(phase_float, target_resolution, target_resolution, 1, 1)
            velocity_tensor = cat(vx_norm, vz_norm, dims=3)
            velocity_tensor = reshape(velocity_tensor, target_resolution, target_resolution, 2, 1)
            
            return phase_tensor, velocity_tensor
        end
    catch e
        error("Fehler bei Preprocessing: $e")
    end
end
"""
Evaluiert Modell auf einer spezifischen Kristallanzahl
"""
function evaluate_model_on_crystal_count(model, crystal_count::Int; 
    n_eval_samples=20, target_resolution=256, verbose=true)
    
    if verbose
        println("Evaluiere Modell auf $crystal_count Kristalle(n)...")
    end
    
    # Generiere Evaluierungsdaten
    eval_dataset = generate_evaluation_dataset_safe(crystal_count, n_eval_samples, 
                                                   resolution=target_resolution, verbose=false)
    
    if length(eval_dataset) == 0
        error("Keine Evaluierungsdaten für $crystal_count Kristalle generiert!")
    end
    
    predictions = []
    targets = []
    
    for (i, sample) in enumerate(eval_dataset)
        try
            # Sample verarbeiten
            if length(sample) >= 8
                x, z, phase, vx, vz, exx, ezz, v_stokes = sample[1:8]
            else
                error("Unvollständiges Sample")
            end
            
            # Preprocessing
            phase_tensor, velocity_tensor = preprocess_lamem_sample_safe(
                x, z, phase, vx, vz, v_stokes,
                target_resolution=target_resolution
            )
            
            # Vorhersage
            prediction = cpu(model(phase_tensor))
            
            push!(predictions, prediction)
            push!(targets, velocity_tensor)
            
        catch e
            if verbose
                println("  Warnung: Sample $i fehlgeschlagen: $e")
            end
            continue
        end
    end
    
    if length(predictions) == 0
        error("Keine erfolgreichen Evaluierungen!")
    end
    
    # Tensoren zusammenfügen
    predictions_batch = cat(predictions..., dims=4)
    targets_batch = cat(targets..., dims=4)
    
    # Metriken berechnen
    metrics = calculate_crystal_metrics(predictions_batch, targets_batch, crystal_count)
    
    if verbose
        println("  MAE Total: $(round(metrics.mae_total, digits=6))")
        println("  R²: $(round(metrics.r2_total, digits=3))")
        println("  Relativer Fehler: $(round(metrics.relative_error_percent, digits=2))%")
    end
    
    return metrics
end

"""
Vollständige Evaluierung auf 1-10 Kristalle
"""
function evaluate_model_comprehensive(model; 
    crystal_range=1:10, 
    n_eval_samples_per_count=20,
    target_resolution=256, 
    verbose=true)
    
    println("="^80)
    println("UMFASSENDE MODELL-EVALUIERUNG")
    println("="^80)
    println("Kristallbereich: $(collect(crystal_range))")
    println("Samples pro Kristallanzahl: $n_eval_samples_per_count")
    
    all_metrics = Dict{Int, CrystalEvaluationMetrics}()
    start_time = time()
    
    for crystal_count in crystal_range
        println("\n" * "-"^50)
        println("Evaluiere $crystal_count Kristalle")
        println("-"^50)
        
        try
            metrics = evaluate_model_on_crystal_count(
                model, crystal_count, 
                n_eval_samples=n_eval_samples_per_count,
                target_resolution=target_resolution,
                verbose=verbose
            )
            
            all_metrics[crystal_count] = metrics
            
        catch e
            println("FEHLER bei $crystal_count Kristallen: $e")
            continue
        end
        
        GC.gc()  # Memory cleanup
    end
    
    end_time = time()
    total_time = (end_time - start_time) / 60
    
    println("\n" * "="^80)
    println("EVALUIERUNG ABGESCHLOSSEN")
    println("="^80)
    println("Gesamtzeit: $(round(total_time, digits=1)) Minuten")
    println("Erfolgreich evaluiert: $(length(all_metrics)) Kristallanzahlen")
    
    return all_metrics
end

"""
Erstellt umfassende Visualisierungen der Evaluierungsergebnisse
"""
function create_evaluation_plots(metrics_dict::Dict{Int, CrystalEvaluationMetrics}; 
    save_plots=true, output_dir="evaluation_plots")
    
    if save_plots
        mkpath(output_dir)
    end
    
    crystal_counts = sort(collect(keys(metrics_dict)))
    
    if length(crystal_counts) == 0
        println("Keine Metriken für Visualisierung verfügbar!")
        return
    end
    
    # Daten extrahieren
    mae_total = [metrics_dict[k].mae_total for k in crystal_counts]
    mae_vx = [metrics_dict[k].mae_vx for k in crystal_counts]
    mae_vz = [metrics_dict[k].mae_vz for k in crystal_counts]
    r2_values = [metrics_dict[k].r2_total for k in crystal_counts]
    relative_errors = [metrics_dict[k].relative_error_percent for k in crystal_counts]
    continuity_errors = [metrics_dict[k].continuity_error for k in crystal_counts]
    
    # Plot 1: Mean Absolute Error (Hauptmetrik)
    p1 = plot(crystal_counts, mae_total, 
              marker=:circle, markersize=8, linewidth=3,
              xlabel="Anzahl Kristalle", ylabel="Mean Absolute Error (MAE)",
              title="Absoluter Fehler vs. Kristallanzahl",
              legend=false, grid=true, gridwidth=2,
              color=:red, markerstrokewidth=2, markerstrokecolor=:darkred)
    
    # Trendlinie hinzufügen
    if length(crystal_counts) > 2
        z = polyfit(crystal_counts, mae_total, 1)
        trend_y = [z[1] * x + z[2] for x in crystal_counts]
        plot!(p1, crystal_counts, trend_y, 
              linestyle=:dash, color=:gray, linewidth=2, alpha=0.7,
              label="Trend")
    end
    
    # Plot 2: MAE Komponenten (vx vs vz)
    p2 = plot(crystal_counts, [mae_vx mae_vz], 
              marker=[:circle :square], markersize=6, linewidth=2,
              xlabel="Anzahl Kristalle", ylabel="Mean Absolute Error",
              title="MAE Komponenten: vx vs vz",
              label=["MAE vx" "MAE vz"], grid=true,
              color=[:blue :green])
    
    # Plot 3: R² (Bestimmtheitsmaß)
    p3 = plot(crystal_counts, r2_values, 
              marker=:diamond, markersize=8, linewidth=3,
              xlabel="Anzahl Kristalle", ylabel="R² (Bestimmtheitsmaß)",
              title="Modell-Performance (R²)",
              legend=false, grid=true, color=:purple,
              ylims=(0, 1))
    
    # Horizontale Linie bei R² = 0.9 (gute Performance)
    hline!(p3, [0.9], linestyle=:dash, color=:gray, alpha=0.7, linewidth=2)
    annotate!(p3, [(maximum(crystal_counts)*0.8, 0.92, "Gute Performance", 10)])
    
    # Plot 4: Relativer Fehler (Prozent)
    p4 = plot(crystal_counts, relative_errors, 
              marker=:hexagon, markersize=8, linewidth=3,
              xlabel="Anzahl Kristalle", ylabel="Relativer Fehler (%)",
              title="Relativer Fehler in Prozent",
              legend=false, grid=true, color=:orange)
    
    # Plot 5: Kontinuitätsfehler (Physikalische Plausibilität)
    p5 = plot(crystal_counts, continuity_errors, 
              marker=:star, markersize=8, linewidth=3,
              xlabel="Anzahl Kristalle", ylabel="Kontinuitätsfehler",
              title="Physikalische Konsistenz (Kontinuitätsgleichung)",
              legend=false, grid=true, color=:brown)
    
    # Plot 6: Performance-Zusammenfassung (Normalisiert)
    normalized_mae = mae_total ./ maximum(mae_total)
    normalized_r2 = 1 .- r2_values  # Invertiert, so dass niedriger = besser
    
    p6 = plot(crystal_counts, [normalized_mae normalized_r2], 
              marker=[:circle :triangle], markersize=6, linewidth=2,
              xlabel="Anzahl Kristalle", ylabel="Normalisierte Metriken",
              title="Performance-Übersicht (normalisiert, niedriger = besser)",
              label=["MAE (norm.)" "1-R² (norm.)"], grid=true,
              color=[:red :blue])
    
    # Kombiniere alle Plots
    combined_plot = plot(p1, p2, p3, p4, p5, p6, 
                        layout=(3,2), size=(1200, 900),
                        plot_title="Modell-Evaluierung: 1-10 Kristalle")
    
    # Speichere Plots
    if save_plots
        savefig(combined_plot, joinpath(output_dir, "comprehensive_evaluation.png"))
        savefig(p1, joinpath(output_dir, "mae_main_metric.png"))
        savefig(p2, joinpath(output_dir, "mae_components.png"))
        savefig(p3, joinpath(output_dir, "r_squared.png"))
        savefig(p4, joinpath(output_dir, "relative_error.png"))
        savefig(p5, joinpath(output_dir, "continuity_error.png"))
        
        println("Plots gespeichert in: $output_dir")
    end
    
    # Zeige kombiniertes Plot
    display(combined_plot)
    
    return combined_plot
end

"""
Erstellt detaillierten Evaluierungsbericht
"""
function create_evaluation_report(metrics_dict::Dict{Int, CrystalEvaluationMetrics})
    println("\n" * "="^100)
    println("DETAILLIERTER EVALUIERUNGSBERICHT")
    println("="^100)
    
    # Header
    header = @sprintf("%-8s %-12s %-12s %-12s %-10s %-12s %-15s", 
                     "Kristalle", "MAE_total", "MAE_vx", "MAE_vz", "R²", "Rel.Fehler%", "Kontinuität")
    println(header)
    println("-"^100)
    
    # Daten
    crystal_counts = sort(collect(keys(metrics_dict)))
    
    for crystal_count in crystal_counts
        metrics = metrics_dict[crystal_count]
        
        row = @sprintf("%-8d %-12.6f %-12.6f %-12.6f %-10.3f %-12.2f %-15.6f",
                      crystal_count,
                      metrics.mae_total, metrics.mae_vx, metrics.mae_vz,
                      metrics.r2_total, metrics.relative_error_percent,
                      metrics.continuity_error)
        println(row)
    end
    
    println("-"^100)
    
    # Statistische Zusammenfassung
    if length(crystal_counts) > 1
        mae_values = [metrics_dict[k].mae_total for k in crystal_counts]
        r2_values = [metrics_dict[k].r2_total for k in crystal_counts]
        
        best_mae_idx = argmin(mae_values)
        worst_mae_idx = argmax(mae_values)
        best_r2_idx = argmax(r2_values)
        
        println("\nSTATISTISCHE ZUSAMMENFASSUNG:")
        println("Beste MAE-Performance: $(crystal_counts[best_mae_idx]) Kristalle (MAE = $(round(mae_values[best_mae_idx], digits=6)))")
        println("Schlechteste MAE-Performance: $(crystal_counts[worst_mae_idx]) Kristalle (MAE = $(round(mae_values[worst_mae_idx], digits=6)))")
        println("Beste R²-Performance: $(crystal_counts[best_r2_idx]) Kristalle (R² = $(round(r2_values[best_r2_idx], digits=3)))")
        
        # Performance-Gradient
        performance_range = mae_values[worst_mae_idx] - mae_values[best_mae_idx]
        relative_range = (performance_range / mae_values[best_mae_idx]) * 100
        
        println("MAE-Spannweite: $(round(performance_range, digits=6)) ($(round(relative_range, digits=1))% relativ)")
        
        # Durchschnittliche Performance
        avg_mae = mean(mae_values)
        avg_r2 = mean(r2_values)
        
        println("Durchschnittliche MAE: $(round(avg_mae, digits=6))")
        println("Durchschnittliches R²: $(round(avg_r2, digits=3))")
        
        # Konsistenz-Bewertung
        mae_std = std(mae_values)
        mae_cv = (mae_std / avg_mae) * 100  # Variationskoeffizient
        
        consistency_rating = if mae_cv < 10
            "Sehr konsistent"
        elseif mae_cv < 25
            "Konsistent"
        elseif mae_cv < 50
            "Mäßig konsistent"
        else
            "Inkonsistent"
        end
        
        println("MAE Variationskoeffizient: $(round(mae_cv, digits=1))% ($consistency_rating)")
    end
    
    println("="^100)
end

"""
Hauptfunktion für komplette Modell-Evaluierung mit Visualisierung
"""
function run_complete_model_evaluation(model_path::String; 
    crystal_range=1:10, 
    n_eval_samples=20,
    target_resolution=256,
    create_plots=true,
    save_results=true,
    output_dir="model_evaluation_results")
    
    println("="^80)
    println("STARTE KOMPLETTE MODELL-EVALUIERUNG")
    println("="^80)
    
    if save_results
        mkpath(output_dir)
    end
    
    # Modell laden
    println("Lade Modell: $model_path")
    model = load_trained_model_safe(model_path)
    println("✓ Modell erfolgreich geladen")
    
    # Evaluierung durchführen
    println("\nStarte umfassende Evaluierung...")
    metrics_dict = evaluate_model_comprehensive(
        model,
        crystal_range=crystal_range,
        n_eval_samples_per_count=n_eval_samples,
        target_resolution=target_resolution,
        verbose=true
    )
    
    if isempty(metrics_dict)
        error("Keine Evaluierungsmetriken erhalten!")
    end
    
    # Bericht erstellen
    create_evaluation_report(metrics_dict)
    
    # Visualisierungen erstellen
    if create_plots
        println("\nErstelle Visualisierungen...")
        plots_dir = joinpath(output_dir, "plots")
        create_evaluation_plots(metrics_dict, save_plots=true, output_dir=plots_dir)
    end
    
    # Ergebnisse speichern
    if save_results
        results_file = joinpath(output_dir, "evaluation_results.bson")
        
        results_data = Dict(
            "metrics" => metrics_dict,
            "evaluation_config" => Dict(
                "crystal_range" => collect(crystal_range),
                "n_eval_samples" => n_eval_samples,
                "target_resolution" => target_resolution,
                "model_path" => model_path,
                "evaluation_date" => now()
            )
        )
        
        BSON.@save results_file results_data
        println("✓ Ergebnisse gespeichert: $results_file")
    end
    
    println("\n" * "="^80)
    println("MODELL-EVALUIERUNG ABGESCHLOSSEN")
    println("="^80)
    
    return metrics_dict
end

# =============================================================================
# BEISPIEL-VERWENDUNG
# =============================================================================

"""
Beispiel für die Verwendung der Evaluierungsfunktionen
"""
function demo_evaluation()
    println("=== DEMO: MODELL-EVALUIERUNG ===")
    
    # Beispiel-Modellpfad (anpassen an Ihren tatsächlichen Pfad)
    model_path = "generalization_checkpoints/best_model.bson"
    
    if !isfile(model_path)
        println("Modell nicht gefunden: $model_path")
        println("Bitte Pfad anpassen oder Modell trainieren.")
        return
    end
    
    # Komplette Evaluierung ausführen
    metrics = run_complete_model_evaluation(
        model_path,
        crystal_range=1:5,        # Kleinerer Bereich für Demo
        n_eval_samples=10,        # Weniger Samples für Demo
        target_resolution=256,
        create_plots=true,
        save_results=true,
        output_dir="demo_evaluation_results"
    )
    
    println("✓ Demo-Evaluierung abgeschlossen!")
    return metrics
end

"""
Schnellstart-Funktion für direkte Verwendung
"""
function quick_evaluate_model(model_path::String; max_crystals=5, samples_per_crystal=10)
    println("=== SCHNELL-EVALUIERUNG ===")
    println("Modell: $model_path")
    println("Kristallbereich: 1-$max_crystals")
    println("Samples pro Kristallanzahl: $samples_per_crystal")
    
    # Prüfe ob Modell existiert
    if !isfile(model_path)
        available_models = filter(x -> endswith(x, ".bson"), readdir("."))
        if length(available_models) > 0
            println("Modell nicht gefunden. Verfügbare Modelle:")
            for model in available_models
                println("  - $model")
            end
        else
            println("Keine .bson Modelle im aktuellen Verzeichnis gefunden.")
        end
        return nothing
    end
    
    try
        # Modell laden
        model = load_trained_model_safe(model_path)
        
        # Schnelle Evaluierung
        metrics_dict = Dict{Int, CrystalEvaluationMetrics}()
        
        for crystal_count in 1:max_crystals
            println("\nEvaluiere $crystal_count Kristalle...")
            
            try
                metrics = evaluate_model_on_crystal_count(
                    model, crystal_count, 
                    n_eval_samples=samples_per_crystal,
                    target_resolution=256,
                    verbose=false
                )
                
                metrics_dict[crystal_count] = metrics
                println("  ✓ MAE: $(round(metrics.mae_total, digits=6)), R²: $(round(metrics.r2_total, digits=3))")
                
            catch e
                println("   Fehler: $e")
                continue
            end
        end
        
        if !isempty(metrics_dict)
            println("\n" * "="^60)
            println("SCHNELL-EVALUIERUNG ABGESCHLOSSEN")
            println("="^60)
            create_evaluation_report(metrics_dict)
            return metrics_dict
        else
            println(" Keine erfolgreichen Evaluierungen")
            return nothing
        end
        
    catch e
        println(" Fehler bei Evaluierung: $e")
        return nothing
    end
end


# Laden Sie Ihre Training-Losses:
training_data = BSON.load("final_model.bson")
train_losses = training_data["train_losses"]
val_losses = training_data["val_losses"]

println("Finale Training Loss: ", train_losses[end])
println("Finale Validation Loss: ", val_losses[end])
println("Beste Validation Loss: ", minimum(val_losses))

# Plot der Losses
using Plots
plot([train_losses val_losses], label=["Training" "Validation"], 
     title="Training History", xlabel="Epoch", ylabel="Loss")