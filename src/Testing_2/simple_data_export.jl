# =============================================================================
# SIMPLE DATA EXPORT MODULE
# =============================================================================


using BSON
using CSV
using DataFrames
using JSON3
using Dates
using Statistics

println("Lade Simple Data Export Module...")

"""
Vereinfachte Export-Konfiguration
"""
struct SimpleExportConfig
    decimal_precision::Int
    csv_delimiter::Char
    include_metadata::Bool
    
    SimpleExportConfig(; decimal_precision=4, csv_delimiter=',', include_metadata=true) = 
        new(decimal_precision, csv_delimiter, include_metadata)
end

"""
Hilfsfunktion: Rundet Werte sicher
"""
function safe_round(value, digits=4)
    if isnan(value) || isinf(value)
        return "N/A"
    else
        return string(round(value, digits=digits))
    end
end

"""
Konvertiert EvaluationResult zu Dictionary (vereinfacht)
"""
function simple_evaluation_result_to_dict(result)
    return Dict(
        "crystal_count" => result.crystal_count,
        "sample_id" => result.sample_id,
        "timestamp" => string(result.timestamp),
        
        # Hauptmetriken
        "mae_total" => safe_round(result.mae_total, 6),
        "rmse_total" => safe_round(result.rmse_total, 6), 
        "correlation_vz" => safe_round(result.pearson_correlation_vz, 4),
        "alignment_error" => safe_round(result.alignment_error_mean, 2),
        "detection_rate" => safe_round(result.crystal_detection_rate, 3),
        
        # Zusätzliche Metriken
        "mae_vx" => safe_round(result.mae_vx, 6),
        "mae_vz" => safe_round(result.mae_vz, 6),
        "ssim_vz" => safe_round(result.ssim_vz, 4),
        "processing_time" => safe_round(result.processing_time, 3),
        "continuity_violation" => safe_round(result.continuity_violation_mean, 8),
        "complexity_index" => safe_round(result.interaction_complexity_index, 3)
    )
end

"""
Einfacher CSV-Export
"""
function simple_export_to_csv(batch_results, output_path::String; 
                             config::SimpleExportConfig=SimpleExportConfig())
    
    println("Exportiere nach CSV: $output_path")
    
    # Sammle alle Daten
    all_data = []
    
    for (n_crystals, results) in batch_results.results_per_crystal
        for result in results
            result_dict = simple_evaluation_result_to_dict(result)
            push!(all_data, result_dict)
        end
    end
    
    if isempty(all_data)
        println("Warnung: Keine Daten zum Exportieren")
        return false
    end
    
    # DataFrame erstellen
    df = DataFrame(all_data)
    
    # CSV schreiben
    try
        CSV.write(output_path, df, delim=config.csv_delimiter)
        
        println("CSV-Export erfolgreich:")
        println("  Datei: $output_path")
        println("  Zeilen: $(nrow(df))")
        println("  Spalten: $(ncol(df))")
        
        return true
        
    catch e
        println("Fehler beim CSV-Export: $e")
        return false
    end
end

"""
Einfacher JSON-Export
"""
function simple_export_to_json(batch_results, output_path::String;
                              config::SimpleExportConfig=SimpleExportConfig())
    
    println("Exportiere nach JSON: $output_path")
    
    # Vereinfachte JSON-Struktur
    json_data = Dict(
        "export_timestamp" => string(now()),
        "evaluation_timestamp" => string(batch_results.evaluation_timestamp),
        "total_samples" => batch_results.total_samples,
        "crystal_range" => "$(batch_results.crystal_range.start):$(batch_results.crystal_range.stop)",
        "results_summary" => Dict()
    )
    
    # Aggregierte Ergebnisse pro Kristallanzahl
    for (n_crystals, results) in batch_results.results_per_crystal
        if !isempty(results)
            mae_values = [r.mae_total for r in results]
            corr_values = [r.pearson_correlation_vz for r in results]
            
            json_data["results_summary"][string(n_crystals)] = Dict(
                "samples" => length(results),
                "mae_mean" => safe_round(mean(mae_values), 6),
                "mae_std" => safe_round(std(mae_values), 6),
                "correlation_mean" => safe_round(mean(corr_values), 4),
                "correlation_std" => safe_round(std(corr_values), 4)
            )
        end
    end
    
    # JSON schreiben
    try
        open(output_path, "w") do f
            JSON3.pretty(f, json_data, indent=2)
        end
        
        println("JSON-Export erfolgreich: $output_path")
        return true
        
    catch e
        println("Fehler beim JSON-Export: $e")
        return false
    end
end

"""
Einfache Zusammenfassungs-Tabelle
"""
function create_simple_summary_table(batch_results, output_path::String)
    
    println("Erstelle einfache Zusammenfassung: $output_path")
    
    open(output_path, "w") do f
        write(f, "# UNet Multi-Kristall Performance Zusammenfassung\n\n")
        write(f, "Generiert am: $(now())\n\n")
        
        write(f, "| Kristalle | Samples | MAE (Mittel±Std) | Korrelation (Mittel±Std) | Status |\n")
        write(f, "|-----------|---------|------------------|--------------------------|--------|\n")
        
        for n_crystals in sort(collect(keys(batch_results.results_per_crystal)))
            results = batch_results.results_per_crystal[n_crystals]
            
            if !isempty(results)
                mae_values = [r.mae_total for r in results]
                corr_values = [r.pearson_correlation_vz for r in results]
                
                mae_mean = mean(mae_values)
                mae_std = std(mae_values)
                corr_mean = mean(corr_values)
                corr_std = std(corr_values)
                
                # Status basierend auf MAE
                status = if mae_mean < 0.05
                    "🥇 Exzellent"
                elseif mae_mean < 0.1
                    "🥈 Gut"
                elseif mae_mean < 0.2
                    "🥉 Akzeptabel"
                else
                    "⚠️ Verbesserung nötig"
                end
                
                mae_str = "$(safe_round(mae_mean, 4))±$(safe_round(mae_std, 4))"
                corr_str = "$(safe_round(corr_mean, 3))±$(safe_round(corr_std, 3))"
                
                write(f, "| $n_crystals | $(length(results)) | $mae_str | $corr_str | $status |\n")
            end
        end
        
        write(f, "\n## Bewertungsskala\n")
        write(f, "- **🥇 Exzellent:** MAE < 0.05\n")
        write(f, "- **🥈 Gut:** MAE < 0.1\n") 
        write(f, "- **🥉 Akzeptabel:** MAE < 0.2\n")
        write(f, "- **⚠️ Verbesserung nötig:** MAE ≥ 0.2\n\n")
        
        # Zusätzliche Statistiken
        write(f, "## Zusätzliche Statistiken\n\n")
        write(f, "- **Gesamt-Samples:** $(batch_results.total_samples)\n")
        write(f, "- **Kristallbereich:** $(batch_results.crystal_range)\n")
        write(f, "- **Erfolgreiche Kristallanzahlen:** $(length(batch_results.results_per_crystal))\n")
        write(f, "- **Evaluierungszeitraum:** $(batch_results.evaluation_timestamp)\n")
    end
    
    println("Zusammenfassung erstellt: $output_path")
end

"""
Vereinfachter Multi-Format-Export
"""
function simple_export_all_formats(batch_results, base_output_dir::String;
                                  config::SimpleExportConfig=SimpleExportConfig())
    
    println("=== VEREINFACHTER MULTI-FORMAT-EXPORT ===")
    println("Ausgabeverzeichnis: $base_output_dir")
    
    mkpath(base_output_dir)
    success_count = 0
    
    # 1. CSV-Export
    try
        csv_path = joinpath(base_output_dir, "evaluation_results.csv")
        if simple_export_to_csv(batch_results, csv_path, config=config)
            success_count += 1
        end
    catch e
        println("CSV-Export fehlgeschlagen: $e")
    end
    
    # 2. JSON-Export
    try
        json_path = joinpath(base_output_dir, "evaluation_results.json")
        if simple_export_to_json(batch_results, json_path, config=config)
            success_count += 1
        end
    catch e
        println("JSON-Export fehlgeschlagen: $e")
    end
    
    # 3. Einfache Zusammenfassung
    try
        summary_path = joinpath(base_output_dir, "summary_table.md")
        create_simple_summary_table(batch_results, summary_path)
        success_count += 1
    catch e
        println("Zusammenfassung fehlgeschlagen: $e")
    end
    
    # 4. BSON-Backup
    try
        bson_path = joinpath(base_output_dir, "batch_results_backup.bson")
        BSON.bson(bson_path, Dict("batch_results" => batch_results))
        success_count += 1
        println("BSON-Backup erstellt: $bson_path")
    catch e
        println("BSON-Backup fehlgeschlagen: $e")
    end
    
    println("\nVereinfachter Export abgeschlossen:")
    println("  Erfolgreiche Exporte: $success_count/4")
    println("  Ausgabeverzeichnis: $base_output_dir")
    
    return success_count >= 2  # Mindestens 2 von 4 erfolgreich
end

"""
Lädt Batch-Ergebnisse (vereinfacht)
"""
function simple_load_batch_results(file_path::String)
    println("Lade Batch-Ergebnisse: $file_path")
    
    if !isfile(file_path)
        error("Datei nicht gefunden: $file_path")
    end
    
    try
        data = BSON.load(file_path)
        
        # Versuche verschiedene Schlüssel
        for key in [:batch_results, :results, :data]
            if haskey(data, key)
                println("Batch-Ergebnisse gefunden unter Schlüssel: $key")
                return data[key]
            end
        end
        
        error("Keine Batch-Ergebnisse in BSON-Datei gefunden")
        
    catch e
        error("Fehler beim Laden: $e")
    end
end

"""
Schnelle Performance-Übersicht
"""
function quick_performance_overview(batch_results)
    println("=== SCHNELLE PERFORMANCE-ÜBERSICHT ===")
    
    for (n_crystals, results) in sort(collect(batch_results.results_per_crystal))
        if !isempty(results)
            mae_values = [r.mae_total for r in results]
            corr_values = [r.pearson_correlation_vz for r in results]
            
            mae_mean = round(mean(mae_values), digits=4)
            mae_std = round(std(mae_values), digits=4)
            corr_mean = round(mean(corr_values), digits=3)
            
            status = mae_mean < 0.05 ? "🥇 EXZELLENT" : 
                    mae_mean < 0.1 ? "🥈 GUT" : 
                    mae_mean < 0.2 ? "🥉 AKZEPTABEL" : "⚠️ VERBESSERUNG NÖTIG"
            
            println("📊 $n_crystals Kristalle: MAE=$mae_mean±$mae_std, Korr=$corr_mean ($(length(results)) Samples) - $status")
        end
    end
    
    println("\n📈 GESAMT-STATISTIKEN:")
    println("  Gesamt-Samples: $(batch_results.total_samples)")
    println("  Kristallbereich: $(batch_results.crystal_range)")
    println("  Erfolgreiche Kristallanzahlen: $(length(batch_results.results_per_crystal))")
    println("  Evaluierungszeitraum: $(batch_results.evaluation_timestamp)")
end

"""
Detaillierte Metriken-Analyse
"""
function detailed_metrics_analysis(batch_results)
    println("=== DETAILLIERTE METRIKEN-ANALYSE ===")
    
    for (n_crystals, results) in sort(collect(batch_results.results_per_crystal))
        if !isempty(results)
            println("\n🔍 $n_crystals KRISTALLE ($(length(results)) Samples):")
            
            # Hauptmetriken
            mae_vals = [r.mae_total for r in results]
            rmse_vals = [r.rmse_total for r in results]
            corr_vals = [r.pearson_correlation_vz for r in results]
            align_vals = [r.alignment_error_mean for r in results if !isinf(r.alignment_error_mean)]
            detect_vals = [r.crystal_detection_rate for r in results]
            
            println("  📊 MAE: $(round(mean(mae_vals), digits=4)) ± $(round(std(mae_vals), digits=4))")
            println("  📊 RMSE: $(round(mean(rmse_vals), digits=4)) ± $(round(std(rmse_vals), digits=4))")
            println("  📊 Korrelation: $(round(mean(corr_vals), digits=3)) ± $(round(std(corr_vals), digits=3))")
            
            if !isempty(align_vals)
                println("  📊 Alignment: $(round(mean(align_vals), digits=1)) ± $(round(std(align_vals), digits=1)) px")
            end
            
            println("  📊 Erkennungsrate: $(round(mean(detect_vals)*100, digits=1))% ± $(round(std(detect_vals)*100, digits=1))%")
        end
    end
end

println("Simple Data Export Module geladen!")
println("Verfügbare Funktionen:")
println("  - simple_export_all_formats() - Vollständiger Export")
println("  - simple_load_batch_results() - Lade gespeicherte Ergebnisse")
println("  - quick_performance_overview() - Schnelle Übersicht")
println("  - detailed_metrics_analysis() - Detaillierte Analyse")
println("")
println("Haupteinstiegspunkt: simple_export_all_formats(batch_results, output_dir)")