# =============================================================================
# DATA MANAGEMENT SYSTEM
# =============================================================================
# Speichern als: data_management_system.jl

using BSON
using CSV
using DataFrames
using JSON3
using Serialization
using Dates
using Statistics
using Printf, JSON3, Statistics, LinearAlgebra, Random, Dates, BSON, CSV, DataFrames, Plots, Colors, Serialization, StatsBase, Distributions, HypothesisTests, Flux, CUDA, LaMEM, GeophysicalModelGenerator

"""
Export-Konfiguration für verschiedene Formate
"""
struct ExportConfig
    include_metadata::Bool
    decimal_precision::Int
    csv_delimiter::Char
    json_pretty_print::Bool
    latex_float_format::String
    
    ExportConfig(; include_metadata=true, decimal_precision=6, csv_delimiter=',',
                json_pretty_print=true, latex_float_format="%.4f") = 
        new(include_metadata, decimal_precision, csv_delimiter, json_pretty_print, latex_float_format)
end

"""
Konvertiert EvaluationResult zu Dictionary für flexible Verarbeitung
"""
function evaluation_result_to_dict(result::EvaluationResult)
    return Dict(
        "crystal_count" => result.crystal_count,
        "sample_id" => result.sample_id,
        "timestamp" => string(result.timestamp),
        
        # Fehlermetriken
        "mae_total" => result.mae_total,
        "mae_vx" => result.mae_vx,
        "mae_vz" => result.mae_vz,
        "rmse_total" => result.rmse_total,
        "rmse_vx" => result.rmse_vx,
        "rmse_vz" => result.rmse_vz,
        "max_error_total" => result.max_error_total,
        "relative_error_stokes" => result.relative_error_stokes,
        
        # Physikalische Konsistenz
        "continuity_violation_mean" => result.continuity_violation_mean,
        "continuity_violation_max" => result.continuity_violation_max,
        "divergence_similarity" => result.divergence_similarity,
        "vorticity_preservation" => result.vorticity_preservation,
        
        # Strukturelle Ähnlichkeit
        "pearson_correlation_vx" => result.pearson_correlation_vx,
        "pearson_correlation_vz" => result.pearson_correlation_vz,
        "ssim_vx" => result.ssim_vx,
        "ssim_vz" => result.ssim_vz,
        "cross_correlation_max" => result.cross_correlation_max,
        
        # Kristall-spezifische Metriken
        "alignment_error_mean" => result.alignment_error_mean,
        "alignment_error_max" => result.alignment_error_max,
        "crystal_detection_rate" => result.crystal_detection_rate,
        "radial_profile_similarity" => result.radial_profile_similarity,
        
        # Multi-Kristall-Komplexität
        "interaction_complexity_index" => result.interaction_complexity_index,
        "density_robustness_score" => result.density_robustness_score,
        
        # Performance
        "processing_time" => result.processing_time,
        "memory_usage" => result.memory_usage
    )
end

"""
Exportiert BatchEvaluationResults nach CSV für externe Analyse
"""
function export_to_csv(batch_results::BatchEvaluationResults, output_path::String; 
                      config::ExportConfig=ExportConfig())
    
    println("Exportiere Batch-Ergebnisse nach CSV: $output_path")
    
    # Alle Einzelergebnisse sammeln
    all_data = []
    
    for (n_crystals, results) in batch_results.results_per_crystal
        for result in results
            result_dict = evaluation_result_to_dict(result)
            push!(all_data, result_dict)
        end
    end
    
    if isempty(all_data)
        println("Warnung: Keine Daten zum Exportieren vorhanden")
        return false
    end
    
    # DataFrame erstellen
    df = DataFrame(all_data)
    
    # Numerische Spalten runden
    numeric_columns = [col for col in names(df) if eltype(df[!, col]) <: AbstractFloat]
    for col in numeric_columns
        df[!, col] = round.(df[!, col], digits=config.decimal_precision)
    end
    
    # CSV schreiben
    try
        CSV.write(output_path, df, delim=config.csv_delimiter)
        
        # Zusätzlich aggregierte Statistiken
        stats_path = replace(output_path, ".csv" => "_summary_stats.csv")
        export_summary_statistics_csv(batch_results, stats_path, config)
        
        println("CSV-Export erfolgreich:")
        println("  Hauptdaten: $output_path")
        println("  Zusammenfassung: $stats_path")
        println("  Zeilen: $(nrow(df))")
        println("  Spalten: $(ncol(df))")
        
        return true
        
    catch e
        println("Fehler beim CSV-Export: $e")
        return false
    end
end

"""
Exportiert aggregierte Statistiken nach CSV
"""
function export_summary_statistics_csv(batch_results::BatchEvaluationResults, output_path::String,
                                     config::ExportConfig)
    
    summary_data = []
    
    for (n_crystals, stats) in batch_results.aggregated_statistics
        stats_row = Dict(
            "crystal_count" => n_crystals,
            "sample_count" => stats["sample_count"]
        )
        
        # Alle statistischen Metriken hinzufügen
        for (metric, value) in stats
            if metric != "sample_count"
                stats_row[metric] = round(value, digits=config.decimal_precision)
            end
        end
        
        push!(summary_data, stats_row)
    end
    
    df_summary = DataFrame(summary_data)
    CSV.write(output_path, df_summary, delim=config.csv_delimiter)
end

"""
Exportiert nach JSON für Webvisualisierungen
"""
function export_to_json(batch_results::BatchEvaluationResults, output_path::String;
                       config::ExportConfig=ExportConfig())
    
    println("Exportiere Batch-Ergebnisse nach JSON: $output_path")
    
    # Vollständige Datenstruktur für JSON
    json_data = Dict(
        "metadata" => Dict(
            "export_timestamp" => string(now()),
            "evaluation_timestamp" => string(batch_results.evaluation_timestamp),
            "total_samples" => batch_results.total_samples,
            "crystal_range" => Dict(
                "start" => batch_results.crystal_range.start,
                "stop" => batch_results.crystal_range.stop
            ),
            "configuration" => batch_results.configuration
        ),
        "results_per_crystal" => Dict(),
        "aggregated_statistics" => batch_results.aggregated_statistics,
        "scaling_metrics" => batch_results.scaling_metrics
    )
    
    # Einzelergebnisse konvertieren
    for (n_crystals, results) in batch_results.results_per_crystal
        json_data["results_per_crystal"][string(n_crystals)] = [
            evaluation_result_to_dict(result) for result in results
        ]
    end
    
    # JSON schreiben
    try
        if config.json_pretty_print
            open(output_path, "w") do f
                JSON3.pretty(f, json_data, indent=2)
            end
        else
            open(output_path, "w") do f
                JSON3.write(f, json_data)
            end
        end
        
        println("JSON-Export erfolgreich: $output_path")
        file_size = round(stat(output_path).size / 1024, digits=1)
        println("  Dateigröße: $(file_size) KB")
        
        return true
        
    catch e
        println("Fehler beim JSON-Export: $e")
        return false
    end
end

"""
Exportiert LaTeX-Tabellen für wissenschaftliche Publikationen
"""
function export_latex_tables(batch_results::BatchEvaluationResults, output_dir::String;
                            config::ExportConfig=ExportConfig())
    
    println("Exportiere LaTeX-Tabellen nach: $output_dir")
    mkpath(output_dir)
    
    # 1. Hauptergebnisse-Tabelle
    main_table_path = joinpath(output_dir, "main_results_table.tex")
    create_main_results_latex_table(batch_results, main_table_path, config)
    
    # 2. Detaillierte Metriken-Tabelle
    detailed_table_path = joinpath(output_dir, "detailed_metrics_table.tex")
    create_detailed_metrics_latex_table(batch_results, detailed_table_path, config)
    
    # 3. Skalierungs-Tabelle
    scaling_table_path = joinpath(output_dir, "scaling_analysis_table.tex")
    create_scaling_analysis_latex_table(batch_results, scaling_table_path, config)
    
    # 4. Zusätzlich: LaTeX-Include-Datei
    include_file_path = joinpath(output_dir, "include_tables.tex")
    create_latex_include_file(include_file_path, [
        "main_results_table.tex",
        "detailed_metrics_table.tex", 
        "scaling_analysis_table.tex"
    ])
    
    println("LaTeX-Tabellen erfolgreich erstellt:")
    println("  Hauptergebnisse: $main_table_path")
    println("  Detaillierte Metriken: $detailed_table_path")
    println("  Skalierungs-Analyse: $scaling_table_path")
    println("  Include-Datei: $include_file_path")
    
    return true
end

"""
Erstellt LaTeX-Tabelle für Hauptergebnisse
"""
function create_main_results_latex_table(batch_results::BatchEvaluationResults, 
                                       output_path::String, config::ExportConfig)
    
    open(output_path, "w") do f
        write(f, """
\\begin{table}[htbp]
\\centering
\\caption{UNet Performance-Übersicht für Multi-Kristall-Systeme}
\\label{tab:main_results}
\\begin{tabular}{|c|c|c|c|c|c|c|}
\\hline
\\textbf{Kristalle} & \\textbf{Samples} & \\textbf{MAE} & \\textbf{RMSE} & \\textbf{Korrelation} & \\textbf{Alignment} & \\textbf{Erkennung} \\\\
& & \\textbf{(Mittel ± Std)} & \\textbf{(Mittel)} & \\textbf{v_z (Mittel)} & \\textbf{Fehler [px]} & \\textbf{Rate [\\%]} \\\\
\\hline
""")
        
        # Datenzeilen
        for n_crystals in sort(collect(keys(batch_results.aggregated_statistics)))
            stats = batch_results.aggregated_statistics[n_crystals]
            
            mae_str = "$(Printf.@sprintf("%.4f", stats["mae_mean"])) ± $(Printf.@sprintf("%.3f", stats["mae_std"]))"
            rmse_str = Printf.@sprintf("%.4f", stats["rmse_mean"])
            corr_str = Printf.@sprintf("%.3f", stats["correlation_vz_mean"])
            align_str = Printf.@sprintf("%.1f", stats["alignment_mean"])
            detect_str = Printf.@sprintf("%.1f", stats["detection_rate_mean"] * 100)
            
            write(f, "$n_crystals & $(Int(stats["sample_count"])) & $mae_str & $rmse_str & $corr_str & $align_str & $detect_str \\\\\n")
        end
        
        write(f, """
\\hline
\\end{tabular}
\\end{table}

""")
    end
end

"""
Erstellt detaillierte LaTeX-Metriken-Tabelle
"""
function create_detailed_metrics_latex_table(batch_results::BatchEvaluationResults,
                                           output_path::String, config::ExportConfig)
    
    open(output_path, "w") do f
        write(f, """
\\begin{table}[htbp]
\\centering
\\caption{Detaillierte Evaluierungsmetriken für Multi-Kristall UNet}
\\label{tab:detailed_metrics}
\\begin{tabular}{|c|c|c|c|c|c|}
\\hline
\\textbf{Kristalle} & \\textbf{SSIM v_z} & \\textbf{Kontinuität} & \\textbf{Vortizität} & \\textbf{Komplexität} & \\textbf{Rechenzeit} \\\\
& \\textbf{(Mittel)} & \\textbf{Verletzung} & \\textbf{Erhaltung} & \\textbf{Index} & \\textbf{[s]} \\\\
\\hline
""")
        
        # Berechne erweiterte Statistiken aus Rohdaten
        for n_crystals in sort(collect(keys(batch_results.results_per_crystal)))
            results = batch_results.results_per_crystal[n_crystals]
            
            if isempty(results)
                continue
            end
            
            ssim_mean = mean([r.ssim_vz for r in results])
            continuity_mean = mean([r.continuity_violation_mean for r in results])
            vorticity_mean = mean([r.vorticity_preservation for r in results if !isnan(r.vorticity_preservation)])
            complexity_mean = mean([r.interaction_complexity_index for r in results])
            time_mean = mean([r.processing_time for r in results])
            
            ssim_str = Printf.@sprintf("%.3f", ssim_mean)
            cont_str = Printf.@sprintf("%.2e", continuity_mean)
            vort_str = Printf.@sprintf("%.3f", isnan(vorticity_mean) ? 0.0 : vorticity_mean)
            comp_str = Printf.@sprintf("%.2f", complexity_mean)
            time_str = Printf.@sprintf("%.2f", time_mean)
            
            write(f, "$n_crystals & $ssim_str & $cont_str & $vort_str & $comp_str & $time_str \\\\\n")
        end
        
        write(f, """
\\hline
\\end{tabular}
\\end{table}

""")
    end
end

"""
Erstellt Skalierungs-Analyse LaTeX-Tabelle
"""
function create_scaling_analysis_latex_table(batch_results::BatchEvaluationResults,
                                           output_path::String, config::ExportConfig)
    
    scaling = batch_results.scaling_metrics
    
    open(output_path, "w") do f
        write(f, """
\\begin{table}[htbp]
\\centering
\\caption{Skalierungsverhalten der UNet-Performance}
\\label{tab:scaling_analysis}
\\begin{tabular}{|c|c|c|c|c|}
\\hline
\\textbf{Kristalle} & \\textbf{MAE-Trend} & \\textbf{Korrelations-Trend} & \\textbf{Performance-} & \\textbf{Komplexitäts-} \\\\
& & & \\textbf{Degradation} & \\textbf{Zunahme} \\\\
\\hline
""")
        
        crystal_counts = scaling["crystal_counts"]
        mae_progression = scaling["mae_progression"]
        corr_progression = scaling["correlation_progression"]
        complexity_progression = scaling["complexity_progression"]
        
        # Berechne Trends und relative Änderungen
        for i in 1:length(crystal_counts)
            n_crystals = Int(crystal_counts[i])
            mae_val = mae_progression[i]
            corr_val = corr_progression[i]
            complexity_val = complexity_progression[i]
            
            # Relative Änderung zur Baseline (1 Kristall)
            if i == 1
                mae_change = "Baseline"
                corr_change = "Baseline"
                perf_degradation = "—"
                complexity_increase = "—"
            else
                mae_baseline = mae_progression[1]
                corr_baseline = corr_progression[1]
                
                mae_change = Printf.@sprintf("%.1f%%", ((mae_val - mae_baseline) / mae_baseline) * 100)
                corr_change = Printf.@sprintf("%.1f%%", ((corr_val - corr_baseline) / corr_baseline) * 100)
                
                perf_degradation = Printf.@sprintf("%.2fx", mae_val / mae_baseline)
                complexity_increase = Printf.@sprintf("%.1fx", complexity_val / complexity_progression[1])
            end
            
            write(f, "$n_crystals & $mae_change & $corr_change & $perf_degradation & $complexity_increase \\\\\n")
        end
        
        write(f, """
\\hline
\\end{tabular}
\\end{table}

""")
    end
end

"""
Erstellt LaTeX-Include-Datei für einfache Integration
"""
function create_latex_include_file(output_path::String, table_files::Vector{String})
    open(output_path, "w") do f
        write(f, """
% Automatisch generierte LaTeX-Tabellen für UNet Multi-Kristall-Evaluierung
% Generiert am: $(now())

% Verwendung in LaTeX-Dokument:
% \\input{include_tables.tex}

""")
        
        for table_file in table_files
            write(f, "\\input{$table_file}\n")
        end
        
        write(f, """

% Ende der automatisch generierten Tabellen
""")
    end
end

"""
Lädt BatchEvaluationResults aus BSON-Datei
"""
function load_batch_results(file_path::String)
    println("Lade Batch-Ergebnisse: $file_path")
    
    if !isfile(file_path)
        error("Datei nicht gefunden: $file_path")
    end
    
    try
        data = BSON.load(file_path)
        
        # Versuche verschiedene Schlüssel
        for key in [:batch_results, :results, :data]
            if haskey(data, key)
                batch_results = data[key]
                println("Batch-Ergebnisse unter Schlüssel '$key' gefunden")
                return batch_results
            end
        end
        
        error("Keine Batch-Ergebnisse in der BSON-Datei gefunden")
        
    catch e
        error("Fehler beim Laden der Batch-Ergebnisse: $e")
    end
end

"""
Speichert BatchEvaluationResults in BSON-Format
"""
function save_batch_results(batch_results::BatchEvaluationResults, file_path::String)
    println("Speichere Batch-Ergebnisse: $file_path")
    
    try
        # Verzeichnis erstellen falls nötig
        mkpath(dirname(file_path))
        
        # BSON speichern
        BSON.bson(file_path, Dict("batch_results" => batch_results))
        
        file_size = round(stat(file_path).size / 1024^2, digits=2)
        println("Batch-Ergebnisse erfolgreich gespeichert")
        println("  Dateigröße: $(file_size) MB")
        println("  Gesamt-Samples: $(batch_results.total_samples)")
        
        return true
        
    catch e
        println("Fehler beim Speichern: $e")
        return false
    end
end

"""
Vollständiger Export in alle Formate
"""
function export_all_formats(batch_results::BatchEvaluationResults, base_output_dir::String;
                           config::ExportConfig=ExportConfig())
    
    println("=== VOLLSTÄNDIGER MULTI-FORMAT-EXPORT ===")
    println("Basis-Ausgabeverzeichnis: $base_output_dir")
    
    mkpath(base_output_dir)
    
    success_count = 0
    
    # 1. CSV-Export
    try
        csv_path = joinpath(base_output_dir, "evaluation_results.csv")
        if export_to_csv(batch_results, csv_path, config=config)
            success_count += 1
        end
    catch e
        println("CSV-Export fehlgeschlagen: $e")
    end
    
    # 2. JSON-Export
    try
        json_path = joinpath(base_output_dir, "evaluation_results.json")
        if export_to_json(batch_results, json_path, config=config)
            success_count += 1
        end
    catch e
        println("JSON-Export fehlgeschlagen: $e")
    end
    
    # 3. LaTeX-Export
    try
        latex_dir = joinpath(base_output_dir, "latex_tables")
        if export_latex_tables(batch_results, latex_dir, config=config)
            success_count += 1
        end
    catch e
        println("LaTeX-Export fehlgeschlagen: $e")
    end
    
    # 4. BSON-Backup
    try
        bson_path = joinpath(base_output_dir, "evaluation_results_backup.bson")
        if save_batch_results(batch_results, bson_path)
            success_count += 1
        end
    catch e
        println("BSON-Backup fehlgeschlagen: $e")
    end
    
    println("\nMulti-Format-Export abgeschlossen:")
    println("  Erfolgreiche Exporte: $success_count/4")
    println("  Ausgabeverzeichnis: $base_output_dir")
    
    return success_count == 4
end

"""
Erstellt Daten-Qualitätsbericht
"""
function create_data_quality_report(batch_results::BatchEvaluationResults, output_path::String)
    println("Erstelle Daten-Qualitätsbericht: $output_path")
    
    open(output_path, "w") do f
        write(f, "# Daten-Qualitätsbericht\n")
        write(f, "Generiert am: $(now())\n\n")
        
        write(f, "## Übersicht\n")
        write(f, "- Evaluierungszeitraum: $(batch_results.evaluation_timestamp)\n")
        write(f, "- Gesamt-Samples: $(batch_results.total_samples)\n")
        write(f, "- Kristallbereich: $(batch_results.crystal_range)\n\n")
        
        write(f, "## Datenqualität pro Kristallanzahl\n\n")
        
        total_successful = 0
        total_failed = 0
        
        for n_crystals in sort(collect(keys(batch_results.results_per_crystal)))
            results = batch_results.results_per_crystal[n_crystals]
            
            successful = length(results)
            failed_estimates = 0
            
            # Analysiere Fehlerwerte
            high_error_count = sum([r.mae_total > 1.0 for r in results])
            invalid_correlation_count = sum([isnan(r.pearson_correlation_vz) for r in results])
            infinite_alignment_count = sum([isinf(r.alignment_error_mean) for r in results])
            
            total_successful += successful
            
            write(f, "### $n_crystals Kristalle\n")
            write(f, "- Erfolgreiche Samples: $successful\n")
            write(f, "- Hohe Fehler (MAE > 1.0): $high_error_count\n")
            write(f, "- Ungültige Korrelationen: $invalid_correlation_count\n")
            write(f, "- Unendliche Alignment-Fehler: $infinite_alignment_count\n")
            
            if successful > 0
                mae_range = extrema([r.mae_total for r in results])
                write(f, "- MAE-Bereich: $(round(mae_range[1], digits=4)) - $(round(mae_range[2], digits=4))\n")
            end
            
            write(f, "\n")
        end
        
        write(f, "## Gesamtqualität\n")
        success_rate = round(total_successful / batch_results.total_samples * 100, digits=1)
        write(f, "- Gesamterfolgsrate: $success_rate%\n")
        write(f, "- Erfolgreich: $total_successful\n")
        write(f, "- Geschätzt fehlgeschlagen: $(batch_results.total_samples - total_successful)\n")
    end
    
    println("Daten-Qualitätsbericht erstellt: $output_path")
end

"""
Printf Import für LaTeX-Formatierung
"""
using Printf

println("Data Management System geladen!")
println("Verfügbare Funktionen:")
println("  - export_to_csv() - CSV-Export für externe Analyse")
println("  - export_to_json() - JSON-Export für Webvisualisierungen")
println("  - export_latex_tables() - LaTeX-Tabellen für Publikationen")
println("  - export_all_formats() - Vollständiger Multi-Format-Export")
println("  - load_batch_results() / save_batch_results() - BSON-Persistierung")
println("  - create_data_quality_report() - Qualitätsbericht")
println("")
println("Haupteinstiegspunkt: export_all_formats(batch_results, output_dir)")