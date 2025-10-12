# =============================================================================
# MASTER EVALUATION SCRIPT - VOLLSTÄNDIGE UNET MULTI-KRISTALL EVALUIERUNG (FIXED)
# =============================================================================
# Speichern als: master_evaluation_fixed.jl

using Dates
using Printf  # Importiere Printf am Anfang
using JSON3
using Statistics
using LinearAlgebra
using Random
using BSON
using CSV
using DataFrames
using Plots
using Colors
using Serialization
using StatsBase
using Distributions

println("=== UNET MULTI-KRISTALL EVALUIERUNG MASTER SCRIPT (FIXED) ===")
println("Gestartet am: $(now())")

# =============================================================================
# 1. ALLE ERFORDERLICHEN MODULE LADEN (OHNE PROBLEMATISCHE MODULE)
# =============================================================================

println("\n1. LADE ALLE MODULE...")

# Basis-Module (bestehende Pipeline)
println("  Lade Basis-Module...")
include("lamem_interface.jl")           # LaMEM-Integration
include("data_processing.jl")           # Datenvorverarbeitung  
include("unet_architecture.jl")         # UNet-Architektur
include("training.jl")                  # Training-Funktionen
include("batch_management.jl")          # Batch-Management

# Neue Evaluierungs-Module (ohne problematische Module)
println("  Lade Evaluierungs-Module...")
include("evaluate_model.jl")            # Kristall-Erkennung und Alignment
include("comprehensive_evaluation.jl")   # Umfassende Evaluierung
include("advanced_visualization.jl")     # Erweiterte Visualisierung

# Verwende vereinfachtes Export-Modul statt problematischem data_management_system.jl
include("simple_data_export.jl")        # Vereinfachtes Datenmanagement

# Lade statistische Analyse separat (falls vorhanden)
STATISTICAL_MODULE_LOADED = false  # Default-Wert setzen
try
    include("statistical_analysis.jl")
    println("  Statistisches Analyse-Modul geladen")
    global STATISTICAL_MODULE_LOADED = true
catch e
    println("  Warnung: Statistisches Analyse-Modul nicht geladen: $e")
    global STATISTICAL_MODULE_LOADED = false
end

println("✓ Alle verfügbaren Module erfolgreich geladen!")

# =============================================================================
# 2. SYSTEM-KONFIGURATION
# =============================================================================

println("\n2. KONFIGURATION...")

# Pfade
const MODEL_PATH = "H:/Masterarbeit/Modelle/ten_crystal_modells/final_model.bson"
const OUTPUT_BASE_DIR = "H:/Masterarbeit/Auswertung/Comprehensive_Evaluation_Fixed"

# Evaluierungs-Parameter
const EVALUATION_CONFIG = (
    crystal_range = 1:5,          # Reduziert für Stabilität
    samples_per_count = 10,       # Reduziert für schnellere Tests
    target_resolution = 256,
    confidence_level = 0.95,
    benchmark_baseline = "linear_interpolation"
)

println("✓ Konfiguration abgeschlossen")
println("  Modell: $MODEL_PATH")
println("  Kristallbereich: $(EVALUATION_CONFIG.crystal_range)")
println("  Samples pro Kristallanzahl: $(EVALUATION_CONFIG.samples_per_count)")
println("  Ausgabeverzeichnis: $OUTPUT_BASE_DIR")

# =============================================================================
# 3. SYSTEM-VALIDIERUNG
# =============================================================================

println("\n3. SYSTEM-VALIDIERUNG...")

# Prüfe Modell-Verfügbarkeit
if !isfile(MODEL_PATH)
    error("Modell nicht gefunden: $MODEL_PATH")
end
println("✓ Modell gefunden")

# Prüfe/Erstelle Ausgabeverzeichnisse
mkpath(OUTPUT_BASE_DIR)
println("✓ Ausgabeverzeichnisse bereit")

# Teste kritische Funktionen
try
    # Test: Modell laden
    test_model = load_trained_model(MODEL_PATH)
    println("✓ Modell erfolgreich geladen")
    
    # Test: Sample-Generierung
    test_sample = LaMEM_Multi_crystal(
        resolution=(64, 64),
        n_crystals=2,
        radius_crystal=[0.05, 0.05],
        cen_2D=[(0.0, 0.3), (0.0, 0.7)]
    )
    println("✓ LaMEM-Integration funktional")
    
    # Test: Evaluierung
    test_result = evaluate_model_comprehensive(test_model, test_sample, target_resolution=64, sample_id=999)
    println("✓ Evaluierungs-Pipeline funktional")
    
catch e
    error("System-Validierung fehlgeschlagen: $e")
end

println("✓ System-Validierung erfolgreich")

# =============================================================================
# 4. VEREINFACHTE HAUPTEVALUIERUNG
# =============================================================================

function run_simplified_evaluation()
    println("\n" * "="^80)
    println("STARTE VEREINFACHTE MULTI-KRISTALL EVALUIERUNG")
    println("="^80)
    
    start_time = time()
    
    try
        # Schritt 1: Automatisierte Multi-Kristall-Evaluierung
        println("\n📊 SCHRITT 1: AUTOMATISIERTE MULTI-KRISTALL-EVALUIERUNG")
        println("-"^60)
        
        evaluation_output_dir = joinpath(OUTPUT_BASE_DIR, "batch_evaluation")
        
        batch_results = automated_multi_crystal_evaluation(
            MODEL_PATH,
            EVALUATION_CONFIG.crystal_range,
            EVALUATION_CONFIG.samples_per_count,
            target_resolution = EVALUATION_CONFIG.target_resolution,
            output_dir = evaluation_output_dir,
            verbose = true
        )
        
        println("✓ Batch-Evaluierung abgeschlossen")
        println("  Gesamt-Samples: $(batch_results.total_samples)")
        println("  Erfolgreiche Kristallanzahlen: $(length(batch_results.results_per_crystal))")
        
        # Schritt 2: Statistische Analyse (falls verfügbar)
        if STATISTICAL_MODULE_LOADED
            println("\n📈 SCHRITT 2: STATISTISCHE ANALYSE")
            println("-"^60)
            
            try
                analysis = analyze_scaling_performance(
                    batch_results,
                    confidence_level = EVALUATION_CONFIG.confidence_level,
                    benchmark_baseline = EVALUATION_CONFIG.benchmark_baseline
                )
                
                println("✓ Statistische Analyse abgeschlossen")
                println("  Empfehlungen generiert: $(length(analysis.recommendations))")
                
            catch e
                println("⚠ Statistische Analyse fehlgeschlagen: $e")
                analysis = nothing
            end
        else
            println("\n⚠ SCHRITT 2: STATISTISCHE ANALYSE ÜBERSPRUNGEN (Modul nicht verfügbar)")
            analysis = nothing
        end
        
        # Schritt 3: Vereinfachte Visualisierungen
        println("\n🎨 SCHRITT 3: VISUALISIERUNGEN")
        println("-"^60)
        
        visualization_output_dir = joinpath(OUTPUT_BASE_DIR, "visualizations")
        
        try
            create_systematic_crystal_comparison(
                MODEL_PATH,
                crystal_counts = collect(EVALUATION_CONFIG.crystal_range),
                samples_per_count = 2,  # Weniger Samples für Visualisierung
                output_dir = visualization_output_dir
            )
            println("✓ Visualisierungen erstellt")
        catch e
            println("⚠ Visualisierung fehlgeschlagen: $e")
        end
        
        # Schritt 4: Vereinfachter Datenexport
        println("\n💾 SCHRITT 4: DATENEXPORT")
        println("-"^60)
        
        export_output_dir = joinpath(OUTPUT_BASE_DIR, "data_export")
        
        export_success = simple_export_all_formats(
            batch_results,
            export_output_dir
        )
        
        if export_success
            println("✓ Datenexport erfolgreich")
        else
            println("⚠ Datenexport teilweise fehlgeschlagen")
        end
        
        # Schritt 5: Einfacher Bericht
        println("\n📋 SCHRITT 5: EINFACHER BERICHT")
        println("-"^60)
        
        report_output_dir = joinpath(OUTPUT_BASE_DIR, "reports")
        mkpath(report_output_dir)
        
        # Einfache Zusammenfassung
        summary_path = joinpath(report_output_dir, "evaluation_summary.md")
        create_evaluation_summary(batch_results, analysis, summary_path)
        
        println("✓ Einfacher Bericht erstellt")
        
        # Erfolgreicher Abschluss
        end_time = time()
        total_duration = end_time - start_time
        
        println("\n" * "="^80)
        println("VEREINFACHTE EVALUIERUNG ERFOLGREICH ABGESCHLOSSEN")
        println("="^80)
        println("📊 Gesamt-Samples evaluiert: $(batch_results.total_samples)")
        println("📈 Kristallbereich: $(EVALUATION_CONFIG.crystal_range)")
        println("⏱️  Gesamtdauer: $(round(total_duration/60, digits=2)) Minuten")
        println("📁 Ausgabeverzeichnis: $OUTPUT_BASE_DIR")
        
        # Zeige Kernmetriken
        println("\n🎯 KERNMETRIKEN:")
        crystal_counts = sort(collect(keys(batch_results.aggregated_statistics)))
        if !isempty(crystal_counts)
            for n_crystals in crystal_counts
                stats = batch_results.aggregated_statistics[n_crystals]
                mae_mean = round(stats["mae_mean"], digits=4)
                corr_mean = round(stats["correlation_vz_mean"], digits=3)
                samples = Int(stats["sample_count"])
                
                println("  📊 $n_crystals Kristalle: MAE=$mae_mean, Korr=$corr_mean ($samples Samples)")
            end
        end
        
        # Generierte Ausgaben
        println("\n📂 GENERIERTE AUSGABEN:")
        println("  📊 Batch-Daten: $(joinpath(evaluation_output_dir, "data", "raw_results.bson"))")
        println("  💾 CSV-Export: $(joinpath(export_output_dir, "evaluation_results.csv"))")
        println("  📋 Zusammenfassung: $summary_path")
        println("  🎨 Visualisierungen: $visualization_output_dir")
        
        return true
        
    catch e
        end_time = time()
        total_duration = end_time - start_time
        
        println("\n" * "="^80)
        println("EVALUIERUNG FEHLGESCHLAGEN")
        println("="^80)
        println("❌ Fehler: $e")
        println("⏱️  Laufzeit bis Fehler: $(round(total_duration/60, digits=2)) Minuten")
        
        return false
    end
end

# =============================================================================
# 5. HILFSFUNKTIONEN
# =============================================================================

"""
Erstellt einfache Evaluierungs-Zusammenfassung
"""
function create_evaluation_summary(batch_results, analysis, output_path)
    open(output_path, "w") do f
        write(f, "# UNet Multi-Kristall Evaluierung - Zusammenfassung\n\n")
        write(f, "**Generiert am:** $(now())\n")
        write(f, "**Evaluierung:** $(batch_results.total_samples) Samples\n")
        write(f, "**Kristallbereich:** $(batch_results.crystal_range)\n\n")
        
        write(f, "## Ergebnisse pro Kristallanzahl\n\n")
        write(f, "| Kristalle | Samples | MAE (Mittel±Std) | Korrelation (Mittel) | Status |\n")
        write(f, "|-----------|---------|------------------|---------------------|--------|\n")
        
        for n_crystals in sort(collect(keys(batch_results.aggregated_statistics)))
            stats = batch_results.aggregated_statistics[n_crystals]
            
            mae_mean = round(stats["mae_mean"], digits=4)
            mae_std = round(stats["mae_std"], digits=4)
            corr_mean = round(stats["correlation_vz_mean"], digits=3)
            samples = Int(stats["sample_count"])
            
            status = if mae_mean < 0.05
                "Exzellent"
            elseif mae_mean < 0.1
                "Gut"
            else
                "Verbesserung nötig"
            end
            
            write(f, "| $n_crystals | $samples | $mae_mean±$mae_std | $corr_mean | $status |\n")
        end
        
        if analysis !== nothing && !isempty(analysis.recommendations)
            write(f, "\n## Empfehlungen\n\n")
            for (i, rec) in enumerate(analysis.recommendations[1:min(3, length(analysis.recommendations))])
                write(f, "$i. $rec\n\n")
            end
        end
        
        write(f, "\n---\n*Automatisch generierte Zusammenfassung*\n")
    end
    
    println("Evaluierungs-Zusammenfassung erstellt: $output_path")
end

"""
Schneller Test für einzelne Kristall-Evaluierung
"""
function run_quick_test()
    println("🔧 SCHNELLER TEST")
    
    try
        model = load_trained_model(MODEL_PATH)
        
        sample = LaMEM_Multi_crystal(
            resolution=(128, 128),
            n_crystals=2,
            radius_crystal=[0.05, 0.05],
            cen_2D=[(-0.3, 0.0), (0.3, 0.0)]
        )
        
        result = evaluate_model_comprehensive(model, sample, target_resolution=128, sample_id=1)
        
        println("✓ Schneller Test erfolgreich")
        println("  MAE: $(round(result.mae_total, digits=4))")
        println("  Korrelation: $(round(result.pearson_correlation_vz, digits=3))")
        
        return result
        
    catch e
        println("❌ Schneller Test fehlgeschlagen: $e")
        return nothing
    end
end

# =============================================================================
# 6. PROGRAMMSTART
# =============================================================================

println("\n" * "="^80)
println("MASTER EVALUATION SCRIPT (FIXED) BEREIT")
println("="^80)
println("📋 Verfügbare Funktionen:")
println("  🚀 run_simplified_evaluation() - Vereinfachte, stabile Evaluierung")
println("  🔧 run_quick_test() - Schneller Einzeltest")
println("")
println("💡 EMPFOHLENER START:")
println("   julia> run_simplified_evaluation()")
println("")
println("🔧 Für schnellen Test:")
println("   julia> run_quick_test()")
println("")
println("📁 Alle Ergebnisse werden gespeichert in: $OUTPUT_BASE_DIR")