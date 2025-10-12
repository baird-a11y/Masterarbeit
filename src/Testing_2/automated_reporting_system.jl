# =============================================================================
# AUTOMATED REPORTING SYSTEM
# ============================================================================= 

using Dates
using Plots
using Markdown
using Printf, JSON3, Statistics, LinearAlgebra, Random, Dates, BSON, CSV, DataFrames, Plots, Colors, Serialization, StatsBase, Distributions, HypothesisTests, Flux, CUDA, LaMEM, GeophysicalModelGenerator

"""
Konfiguration für Report-Generierung
"""
struct ReportConfig
    include_executive_summary::Bool
    include_detailed_analysis::Bool
    include_visualizations::Bool
    include_raw_data::Bool
    include_recommendations::Bool
    include_latex_export::Bool
    
    # Format-Einstellungen
    output_format::String  # "html", "markdown", "pdf"
    figure_format::String  # "png", "svg", "pdf"
    figure_dpi::Int
    
    # Sprache und Stil
    language::String       # "de", "en"
    academic_style::Bool
    
    ReportConfig(; include_executive_summary=true, include_detailed_analysis=true,
                include_visualizations=true, include_raw_data=false,
                include_recommendations=true, include_latex_export=false,
                output_format="html", figure_format="png", figure_dpi=300,
                language="de", academic_style=true) =
        new(include_executive_summary, include_detailed_analysis, include_visualizations,
            include_raw_data, include_recommendations, include_latex_export,
            output_format, figure_format, figure_dpi, language, academic_style)
end

"""
Generiert vollständigen wissenschaftlichen Bericht
"""
function generate_comprehensive_report(evaluation_results::BatchEvaluationResults,
                                     statistical_analysis::StatisticalAnalysisResult,
                                     output_dir::String;
                                     config::ReportConfig=ReportConfig())
    
    println("=== GENERIERE UMFASSENDEN WISSENSCHAFTLICHEN BERICHT ===")
    println("Ausgabeverzeichnis: $output_dir")
    println("Format: $(config.output_format)")
    println("Sprache: $(config.language)")
    
    # Ausgabe-Verzeichnis strukturieren
    mkpath(output_dir)
    figures_dir = joinpath(output_dir, "figures")
    mkpath(figures_dir)
    
    if config.include_raw_data
        data_dir = joinpath(output_dir, "data")
        mkpath(data_dir)
    end
    
    # 1. Visualisierungen erstellen
    figure_paths = Dict{String, String}()
    
    if config.include_visualizations
        println("Erstelle Visualisierungen...")
        figure_paths = create_all_report_figures(evaluation_results, statistical_analysis, 
                                                figures_dir, config)
    end
    
    # 2. Hauptbericht erstellen
    if config.output_format == "html"
        report_path = joinpath(output_dir, "comprehensive_report.html")
        create_html_report(evaluation_results, statistical_analysis, report_path, 
                          figure_paths, config)
    elseif config.output_format == "markdown"
        report_path = joinpath(output_dir, "comprehensive_report.md")
        create_markdown_report(evaluation_results, statistical_analysis, report_path,
                              figure_paths, config)
    else
        report_path = joinpath(output_dir, "comprehensive_report.md")
        create_markdown_report(evaluation_results, statistical_analysis, report_path,
                              figure_paths, config)
        println("Warnung: PDF-Output noch nicht implementiert, verwende Markdown")
    end
    
    # 3. Executive Summary separat
    if config.include_executive_summary
        summary_path = joinpath(output_dir, "executive_summary.md")
        create_performance_summary(evaluation_results, statistical_analysis, summary_path, config)
    end
    
    # 4. LaTeX-Export für Masterarbeit
    if config.include_latex_export
        latex_dir = joinpath(output_dir, "latex_export")
        export_latex_tables_and_figures(evaluation_results, statistical_analysis, 
                                       latex_dir, figure_paths, config)
    end
    
    # 5. Rohdaten-Export
    if config.include_raw_data
        data_dir = joinpath(output_dir, "data")
        export_all_formats(evaluation_results, data_dir)
    end
    
    # 6. Bericht-Metadaten
    create_report_metadata(output_dir, evaluation_results, statistical_analysis, config)
    
    println("\nUmfassender Bericht erfolgreich generiert:")
    println("  Hauptbericht: $report_path")
    if config.include_executive_summary
        println("  Executive Summary: $(joinpath(output_dir, "executive_summary.md"))")
    end
    if config.include_visualizations
        println("  Abbildungen: $figures_dir ($(length(figure_paths)) Dateien)")
    end
    
    return report_path
end

"""
Erstellt alle Report-Abbildungen
"""
function create_all_report_figures(evaluation_results::BatchEvaluationResults,
                                 statistical_analysis::StatisticalAnalysisResult,
                                 figures_dir::String, config::ReportConfig)
    
    figure_paths = Dict{String, String}()
    
    # 1. Performance-Skalierungs-Plot
    try
        p1 = create_performance_scaling_plot(evaluation_results)
        fig_path = joinpath(figures_dir, "performance_scaling.$(config.figure_format)")
        savefig(p1, fig_path)
        figure_paths["performance_scaling"] = fig_path
        println("  ✓ Performance-Skalierung")
    catch e
        println("  ✗ Performance-Skalierung: $e")
    end
    
    # 2. Skalierungs-Analyse-Plots
    try
        scaling_plots = create_scaling_analysis_plots(evaluation_results, figures_dir)
        figure_paths["scaling_analysis"] = figures_dir
        println("  ✓ Skalierungs-Analyse ($(length(scaling_plots)) Plots)")
    catch e
        println("  ✗ Skalierungs-Analyse: $e")
    end
    
    # 3. Statistische Plots
    try
        statistical_plots = create_statistical_plots(statistical_analysis, figures_dir)
        figure_paths["statistical_analysis"] = figures_dir
        println("  ✓ Statistische Analyse ($(length(statistical_plots)) Plots)")
    catch e
        println("  ✗ Statistische Analyse: $e")
    end
    
    # 4. Zusammenfassende Performance-Übersicht
    try
        p4 = create_performance_overview_plot(evaluation_results)
        fig_path = joinpath(figures_dir, "performance_overview.$(config.figure_format)")
        savefig(p4, fig_path)
        figure_paths["performance_overview"] = fig_path
        println("  ✓ Performance-Übersicht")
    catch e
        println("  ✗ Performance-Übersicht: $e")
    end
    
    # 5. Detaillierte Metriken-Heatmap
    try
        p5 = create_detailed_metrics_heatmap(evaluation_results)
        fig_path = joinpath(figures_dir, "metrics_heatmap.$(config.figure_format)")
        savefig(p5, fig_path)
        figure_paths["metrics_heatmap"] = fig_path
        println("  ✓ Metriken-Heatmap")
    catch e
        println("  ✗ Metriken-Heatmap: $e")
    end
    
    return figure_paths
end

"""
Erstellt HTML-Bericht mit eingebetteten Abbildungen
"""
function create_html_report(evaluation_results::BatchEvaluationResults,
                          statistical_analysis::StatisticalAnalysisResult,
                          output_path::String, figure_paths::Dict{String, String},
                          config::ReportConfig)
    
    open(output_path, "w") do f
        write(f, """
<!DOCTYPE html>
<html lang="$(config.language)">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UNet Multi-Kristall Performance-Analyse</title>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; line-height: 1.6; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; }
        h3 { color: #7f8c8d; }
        .executive-summary { background-color: #ecf0f1; padding: 20px; border-radius: 8px; margin: 20px 0; }
        .metric-highlight { background-color: #d5edda; padding: 5px 10px; border-radius: 4px; display: inline-block; margin: 2px; }
        .figure { text-align: center; margin: 30px 0; }
        .figure img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }
        .figure-caption { font-style: italic; color: #6c757d; margin-top: 10px; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: center; }
        th { background-color: #f8f9fa; font-weight: bold; }
        .recommendation { background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 10px 0; }
        .metadata { font-size: 0.9em; color: #6c757d; border-top: 1px solid #ddd; padding-top: 20px; margin-top: 40px; }
    </style>
</head>
<body>
    <div class="container">
""")
        
        # Titel und Metadaten
        write(f, """
        <h1>UNet Multi-Kristall Performance-Analyse</h1>
        <p><strong>Generiert am:</strong> $(now())</p>
        <p><strong>Evaluierungszeitraum:</strong> $(evaluation_results.evaluation_timestamp)</p>
        <p><strong>Gesamt-Samples:</strong> $(evaluation_results.total_samples)</p>
        <p><strong>Kristallbereich:</strong> $(evaluation_results.crystal_range)</p>
""")
        
        # Executive Summary
        if config.include_executive_summary
            write(f, create_html_executive_summary(evaluation_results, statistical_analysis))
        end
        
        # Hauptergebnisse-Tabelle
        write(f, create_html_results_table(evaluation_results))
        
        # Visualisierungen einbetten
        if config.include_visualizations && !isempty(figure_paths)
            write(f, create_html_figures_section(figure_paths))
        end
        
        # Detaillierte Analyse
        if config.include_detailed_analysis
            write(f, create_html_detailed_analysis(evaluation_results, statistical_analysis))
        end
        
        # Empfehlungen
        if config.include_recommendations && !isempty(statistical_analysis.recommendations)
            write(f, create_html_recommendations(statistical_analysis.recommendations))
        end
        
        # Abschluss
        write(f, """
        <div class="metadata">
            <p><em>Dieser Bericht wurde automatisch generiert durch das Automated Reporting System.</em></p>
            <p><em>Konfiguration: $(config.language), $(config.output_format), $(config.academic_style ? "Akademischer Stil" : "Standard")</em></p>
        </div>
    </div>
</body>
</html>
""")
    end
    
    println("HTML-Bericht erstellt: $output_path")
end

"""
Erstellt Markdown-Bericht
"""
function create_markdown_report(evaluation_results::BatchEvaluationResults,
                              statistical_analysis::StatisticalAnalysisResult,
                              output_path::String, figure_paths::Dict{String, String},
                              config::ReportConfig)
    
    open(output_path, "w") do f
        write(f, "# UNet Multi-Kristall Performance-Analyse\n\n")
        
        # Metadaten
        write(f, "**Generiert am:** $(now())\n")
        write(f, "**Evaluierungszeitraum:** $(evaluation_results.evaluation_timestamp)\n")
        write(f, "**Gesamt-Samples:** $(evaluation_results.total_samples)\n")
        write(f, "**Kristallbereich:** $(evaluation_results.crystal_range)\n\n")
        
        # Executive Summary
        if config.include_executive_summary
            write(f, create_markdown_executive_summary(evaluation_results, statistical_analysis))
        end
        
        # Hauptergebnisse
        write(f, create_markdown_results_table(evaluation_results))
        
        # Visualisierungen
        if config.include_visualizations && !isempty(figure_paths)
            write(f, create_markdown_figures_section(figure_paths))
        end
        
        # Detaillierte Analyse
        if config.include_detailed_analysis
            write(f, create_markdown_detailed_analysis(evaluation_results, statistical_analysis))
        end
        
        # Empfehlungen
        if config.include_recommendations && !isempty(statistical_analysis.recommendations)
            write(f, create_markdown_recommendations(statistical_analysis.recommendations))
        end
        
        # Anhang
        write(f, "\n---\n\n")
        write(f, "*Dieser Bericht wurde automatisch generiert durch das Automated Reporting System.*\n")
    end
    
    println("Markdown-Bericht erstellt: $output_path")
end

"""
HTML Executive Summary
"""
function create_html_executive_summary(evaluation_results::BatchEvaluationResults,
                                     statistical_analysis::StatisticalAnalysisResult)
    
    # Berechne Schlüsselmetriken
    crystal_counts = sort(collect(keys(evaluation_results.aggregated_statistics)))
    
    if isempty(crystal_counts)
        return "<div class=\"executive-summary\"><h2>Executive Summary</h2><p>Keine Daten verfügbar.</p></div>"
    end
    
    # Beste und schlechteste Performance
    best_crystal_count = crystal_counts[1]
    worst_crystal_count = crystal_counts[end]
    
    best_mae = evaluation_results.aggregated_statistics[best_crystal_count]["mae_mean"]
    worst_mae = evaluation_results.aggregated_statistics[worst_crystal_count]["mae_mean"]
    
    performance_degradation = ((worst_mae - best_mae) / best_mae) * 100
    
    summary_html = """
    <div class="executive-summary">
        <h2>Executive Summary</h2>
        
        <h3>Kernerkenntnisse</h3>
        <ul>
            <li><span class="metric-highlight">Performance-Bereich:</span> MAE von $(round(best_mae, digits=4)) ($best_crystal_count Kristalle) bis $(round(worst_mae, digits=4)) ($worst_crystal_count Kristalle)</li>
            <li><span class="metric-highlight">Skalierungs-Degradation:</span> $(round(performance_degradation, digits=1))% Verschlechterung über den evaluierten Bereich</li>
            <li><span class="metric-highlight">Gesamt-Samples:</span> $(evaluation_results.total_samples) evaluierte Konfigurationen</li>
        </ul>
        
        <h3>Statistische Signifikanz</h3>
        <p>
"""
    
    # Füge statistische Highlights hinzu
    if haskey(statistical_analysis.significance_tests, "mae_anova")
        anova = statistical_analysis.significance_tests["mae_anova"]
        if anova["significant"]
            summary_html *= "ANOVA bestätigt <strong>signifikante Unterschiede</strong> zwischen Kristallanzahlen (p = $(round(anova["p_value"], digits=4))). "
        else
            summary_html *= "ANOVA zeigt <strong>keine signifikanten Unterschiede</strong> zwischen Kristallanzahlen. "
        end
    end
    
    summary_html *= """
        </p>
        
        <h3>Wichtigste Empfehlungen</h3>
        <ul>
"""
    
    # Top 3 Empfehlungen
    for (i, rec) in enumerate(statistical_analysis.recommendations[1:min(3, length(statistical_analysis.recommendations))])
        summary_html *= "<li>$rec</li>"
    end
    
    summary_html *= """
        </ul>
    </div>
"""
    
    return summary_html
end

"""
HTML Hauptergebnisse-Tabelle
"""
function create_html_results_table(evaluation_results::BatchEvaluationResults)
    table_html = """
    <h2>Hauptergebnisse</h2>
    <table>
        <thead>
            <tr>
                <th>Kristalle</th>
                <th>Samples</th>
                <th>MAE (μ±σ)</th>
                <th>RMSE (μ)</th>
                <th>Korrelation v_z (μ±σ)</th>
                <th>Alignment Fehler [px]</th>
                <th>Erkennungsrate [%]</th>
            </tr>
        </thead>
        <tbody>
"""
    
    for n_crystals in sort(collect(keys(evaluation_results.aggregated_statistics)))
        stats = evaluation_results.aggregated_statistics[n_crystals]
        
        mae_str = "$(round(stats["mae_mean"], digits=4))±$(round(stats["mae_std"], digits=4))"
        rmse_str = "$(round(stats["rmse_mean"], digits=4))"
        corr_str = "$(round(stats["correlation_vz_mean"], digits=3))±$(round(stats["correlation_vz_std"], digits=3))"
        
        align_str = if stats["alignment_mean"] < 900  # Nicht-Inf Werte
            "$(round(stats["alignment_mean"], digits=1))±$(round(stats["alignment_std"], digits=1))"
        else
            "N/A"
        end
        
        detect_str = "$(round(stats["detection_rate_mean"] * 100, digits=1))"
        samples = Int(stats["sample_count"])
        
        # Farbkodierung für Performance
        mae_class = stats["mae_mean"] < 0.05 ? "style=\"background-color: #d4edda;\"" : 
                   stats["mae_mean"] < 0.1 ? "style=\"background-color: #fff3cd;\"" : 
                   "style=\"background-color: #f8d7da;\""
        
        table_html *= """
            <tr>
                <td>$n_crystals</td>
                <td>$samples</td>
                <td $mae_class>$mae_str</td>
                <td>$rmse_str</td>
                <td>$corr_str</td>
                <td>$align_str</td>
                <td>$detect_str</td>
            </tr>
"""
    end
    
    table_html *= """
        </tbody>
    </table>
"""
    
    return table_html
end

"""
Erstellt Performance-Übersichts-Plot
"""
function create_performance_overview_plot(evaluation_results::BatchEvaluationResults)
    crystal_counts = Float64[]
    mae_means = Float64[]
    mae_stds = Float64[]
    correlation_means = Float64[]
    sample_counts = Int[]
    
    for n_crystals in sort(collect(keys(evaluation_results.aggregated_statistics)))
        stats = evaluation_results.aggregated_statistics[n_crystals]
        
        push!(crystal_counts, Float64(n_crystals))
        push!(mae_means, stats["mae_mean"])
        push!(mae_stds, stats["mae_std"])
        push!(correlation_means, stats["correlation_vz_mean"])
        push!(sample_counts, Int(stats["sample_count"]))
    end
    
    # Subplot-Layout
    p1 = plot(crystal_counts, mae_means,
              ribbon=mae_stds,
              fillalpha=0.3,
              linewidth=3,
              marker=:circle,
              markersize=6,
              xlabel="Anzahl Kristalle",
              ylabel="MAE",
              title="MAE mit Standardabweichung",
              legend=false,
              color=:red)
    
    p2 = plot(crystal_counts, correlation_means,
              linewidth=3,
              marker=:square,
              markersize=6,
              xlabel="Anzahl Kristalle", 
              ylabel="Korrelation v_z",
              title="Strukturelle Ähnlichkeit",
              legend=false,
              color=:blue)
    
    p3 = bar(crystal_counts, sample_counts,
             xlabel="Anzahl Kristalle",
             ylabel="Anzahl Samples",
             title="Stichprobengrößen",
             legend=false,
             color=:green,
             alpha=0.7)
    
    p4 = scatter(mae_means, correlation_means,
                markersize=8,
                color=crystal_counts,
                xlabel="MAE",
                ylabel="Korrelation v_z",
                title="Performance-Korrelation",
                colorbar_title="Kristalle")
    
    return plot(p1, p2, p3, p4, layout=(2, 2), size=(800, 600))
end

"""
Executive Summary separat erstellen
"""
function create_performance_summary(evaluation_results::BatchEvaluationResults,
                                  statistical_analysis::StatisticalAnalysisResult,
                                  output_path::String, config::ReportConfig)
    
    open(output_path, "w") do f
        write(f, "# Executive Summary: UNet Multi-Kristall Performance\n\n")
        write(f, "**Datum:** $(now())\n")
        write(f, "**Evaluation:** $(evaluation_results.total_samples) Samples, $(evaluation_results.crystal_range)\n\n")
        
        # Schnelle Übersicht
        crystal_counts = sort(collect(keys(evaluation_results.aggregated_statistics)))
        if !isempty(crystal_counts)
            best_performance = evaluation_results.aggregated_statistics[crystal_counts[1]]
            worst_performance = evaluation_results.aggregated_statistics[crystal_counts[end]]
            
            write(f, "## Kernmetriken\n\n")
            write(f, "- **Beste Performance:** $(crystal_counts[1]) Kristalle, MAE = $(round(best_performance["mae_mean"], digits=4))\n")
            write(f, "- **Schlechteste Performance:** $(crystal_counts[end]) Kristalle, MAE = $(round(worst_performance["mae_mean"], digits=4))\n")
            write(f, "- **Performance-Degradation:** $(round(((worst_performance["mae_mean"] - best_performance["mae_mean"]) / best_performance["mae_mean"]) * 100, digits=1))%\n\n")
        end
        
        # Top-Empfehlungen
        write(f, "## Sofortige Handlungsempfehlungen\n\n")
        for (i, rec) in enumerate(statistical_analysis.recommendations[1:min(3, length(statistical_analysis.recommendations))])
            write(f, "$i. $rec\n\n")
        end
        
        # Kritische Erkenntnisse
        write(f, "## Kritische Erkenntnisse\n\n")
        if haskey(statistical_analysis.significance_tests, "mae_anova")
            anova = statistical_analysis.significance_tests["mae_anova"]
            if anova["significant"]
                write(f, "- **Statistisch signifikante** Unterschiede zwischen Kristallanzahlen bestätigt\n")
            else
                write(f, "- Keine statistisch signifikanten Unterschiede zwischen Kristallanzahlen\n")
            end
        end
        
        # Skalierungsverhalten
        total_effect = get(statistical_analysis.effect_sizes, "total_scaling_effect", 0.0)
        if total_effect > 0.5
            write(f, "- **Starke Skalierungseffekte:** $(round(total_effect*100, digits=1))% Verschlechterung\n")
        elseif total_effect > 0.2
            write(f, "- **Moderate Skalierungseffekte:** $(round(total_effect*100, digits=1))% Verschlechterung\n")
        else
            write(f, "- **Robuste Skalierung:** Geringe Performance-Degradation\n")
        end
        
        write(f, "\n---\n*Executive Summary automatisch generiert*\n")
    end
    
    println("Executive Summary erstellt: $output_path")
end

"""
LaTeX-Export für Masterarbeit
"""
function export_latex_tables_and_figures(evaluation_results::BatchEvaluationResults,
                                        statistical_analysis::StatisticalAnalysisResult,
                                        output_dir::String, figure_paths::Dict{String, String},
                                        config::ReportConfig)
    
    println("Exportiere LaTeX-Materialien für Masterarbeit...")
    mkpath(output_dir)
    
    # 1. Hauptergebnisse-Tabelle
    main_table_path = joinpath(output_dir, "main_results_table.tex")
    export_latex_tables(evaluation_results, dirname(main_table_path))
    
    # 2. Abbildungen für LaTeX
    figures_latex_dir = joinpath(output_dir, "figures")
    mkpath(figures_latex_dir)
    
    # Kopiere Abbildungen und erstelle LaTeX-Includes
    latex_figures = String[]
    for (fig_name, fig_path) in figure_paths
        if isfile(fig_path)
            target_path = joinpath(figures_latex_dir, basename(fig_path))
            cp(fig_path, target_path, force=true)
            push!(latex_figures, basename(fig_path))
        end
    end
    
    # 3. LaTeX-Include-Datei für Abbildungen
    figures_include_path = joinpath(output_dir, "include_figures.tex")
    open(figures_include_path, "w") do f
        write(f, "% LaTeX-Abbildungen für UNet Multi-Kristall Evaluierung\n")
        write(f, "% Generiert am: $(now())\n\n")
        
        for (i, fig_file) in enumerate(latex_figures)
            fig_name = replace(split(fig_file, ".")[1], "_" => " ")
            write(f, """
\\begin{figure}[htbp]
\\centering
\\includegraphics[width=0.8\\textwidth]{figures/$fig_file}
\\caption{$fig_name}
\\label{fig:$(replace(fig_name, " " => "_"))}
\\end{figure}

""")
        end
    end
    
    # 4. Zusammenfassende LaTeX-Datei
    master_include_path = joinpath(output_dir, "master_thesis_include.tex")
    open(master_include_path, "w") do f
        write(f, """
% Vollständiger LaTeX-Include für UNet Multi-Kristall Evaluierung
% Verwendung in Masterarbeit: \\input{master_thesis_include.tex}

\\section{UNet Multi-Kristall Performance Evaluierung}

\\subsection{Hauptergebnisse}
\\input{main_results_table.tex}

\\subsection{Abbildungen}
\\input{include_figures.tex}

% Ende der UNet-Evaluierung
""")
    end
    
    println("LaTeX-Export abgeschlossen:")
    println("  Tabellen: $(dirname(main_table_path))")
    println("  Abbildungen: $figures_latex_dir ($(length(latex_figures)) Dateien)")
    println("  Master-Include: $master_include_path")
end

"""
Erstellt Bericht-Metadaten für Nachverfolgbarkeit
"""
function create_report_metadata(output_dir::String, evaluation_results::BatchEvaluationResults,
                               statistical_analysis::StatisticalAnalysisResult, config::ReportConfig)
    
    metadata_path = joinpath(output_dir, "report_metadata.json")
    
    metadata = Dict(
        "generation_timestamp" => string(now()),
        "evaluation_timestamp" => string(evaluation_results.evaluation_timestamp),
        "total_samples" => evaluation_results.total_samples,
        "crystal_range" => Dict(
            "start" => evaluation_results.crystal_range.start,
            "stop" => evaluation_results.crystal_range.stop
        ),
        "report_config" => Dict(
            "output_format" => config.output_format,
            "language" => config.language,
            "academic_style" => config.academic_style,
            "include_visualizations" => config.include_visualizations,
            "include_raw_data" => config.include_raw_data
        ),
        "statistics_summary" => Dict(
            "total_recommendations" => length(statistical_analysis.recommendations),
            "significance_tests_performed" => length(statistical_analysis.significance_tests),
            "effect_sizes_calculated" => length(statistical_analysis.effect_sizes)
        ),
        "quality_indicators" => Dict(
            "successful_crystal_counts" => length(evaluation_results.results_per_crystal),
            "average_samples_per_crystal" => mean([length(results) for results in values(evaluation_results.results_per_crystal)])
        )
    )
    
    open(metadata_path, "w") do f
        JSON3.pretty(f, metadata, indent=2)
    end
    
    println("Bericht-Metadaten gespeichert: $metadata_path")
end

# Hilfsfunktionen für Markdown-Format (vereinfacht dargestellt)
function create_markdown_executive_summary(evaluation_results, statistical_analysis)
    return "\n## Executive Summary\n\n[Markdown-spezifische Implementierung]\n\n"
end

function create_markdown_results_table(evaluation_results)
    return "\n## Hauptergebnisse\n\n[Markdown-Tabelle]\n\n"
end

function create_markdown_figures_section(figure_paths)
    return "\n## Visualisierungen\n\n[Markdown-Abbildungen]\n\n"
end

function create_markdown_detailed_analysis(evaluation_results, statistical_analysis)
    return "\n## Detaillierte Analyse\n\n[Markdown-Analyse]\n\n"
end

function create_markdown_recommendations(recommendations)
    recommendations_text = "\n## Empfehlungen\n\n"
    for (i, rec) in enumerate(recommendations)
        recommendations_text *= "$i. $rec\n\n"
    end
    return recommendations_text
end

# Weitere HTML-Hilfsfunktionen
function create_html_figures_section(figure_paths)
    return "\n<h2>Visualisierungen</h2>\n[HTML-Abbildungen]\n\n"
end

function create_html_detailed_analysis(evaluation_results, statistical_analysis)
    return "\n<h2>Detaillierte Analyse</h2>\n[HTML-Analyse]\n\n"
end

function create_html_recommendations(recommendations)
    rec_html = "\n<h2>Empfehlungen</h2>\n"
    for rec in recommendations
        rec_html *= "<div class=\"recommendation\">$rec</div>\n"
    end
    return rec_html
end

function create_detailed_metrics_heatmap(evaluation_results)
    # Vereinfachte Heatmap-Erstellung
    return plot(title="Detaillierte Metriken-Heatmap [Implementierung erforderlich]")
end

println("Automated Reporting System geladen!")
println("Verfügbare Funktionen:")
println("  - generate_comprehensive_report() - Vollständiger wissenschaftlicher Bericht")
println("  - create_performance_summary() - Executive Summary")
println("  - export_latex_tables_and_figures() - LaTeX-Export für Masterarbeit") 
println("  - create_all_report_figures() - Alle Report-Visualisierungen")
println("")
println("Haupteinstiegspunkt: generate_comprehensive_report(evaluation_results, statistical_analysis, output_dir)")