# =============================================================================
# RESIDUAL LEARNING EVALUATION - KERNDATENSTRUKTUREN
# =============================================================================

using Dates
using Statistics: mean, std, median, quantile
"""
Erweiterte Evaluierungsstruktur für Residual Learning
NEUE FEATURES:
- Stokes-Baseline Metriken
- Residuum-Analyse
- Lambda-Gewichtungs-Tracking
"""
struct ResidualEvaluationResult
    # ============ GRUNDINFORMATIONEN ============
    crystal_count::Int
    sample_id::Int
    timestamp::DateTime
    
    # ============ STOKES-BASELINE (NEU) ============
    mae_stokes_only::Float64           # Fehler der reinen Stokes-Lösung
    mae_with_residual::Float64         # Fehler nach Residuum-Korrektur
    improvement_over_stokes_percent::Float64  # Relative Verbesserung
    
    # ============ RESIDUUM-ANALYSE (NEU) ============
    residual_mean::Float64             # Durchschnittliche Residuum-Magnitude
    residual_max::Float64              # Maximale Residuum-Magnitude
    residual_std::Float64              # Residuum-Streuung
    residual_sparsity::Float64         # Anteil ~0 Residuen (0-1)
    stokes_contribution_ratio::Float64 # v_stokes / v_total
    residual_to_stokes_ratio::Float64  # |Δv| / |v_stokes|
    
    # ============ KOMPONENTENWEISE FEHLER ============
    mae_vx::Float64
    mae_vz::Float64
    rmse_vx::Float64
    rmse_vz::Float64
    max_error_vx::Float64
    max_error_vz::Float64
    
    # ============ PHYSIKALISCHE KONSISTENZ (KRITISCH FÜR NICHT-STREAM-FUNCTION) ============
    # WICHTIG: Ohne Stream Function ist Massenerhaltung nicht automatisch erfüllt!
    continuity_violation_mean::Float64      # ∂vx/∂x + ∂vz/∂z (Durchschnitt)
    continuity_violation_max::Float64       # Maximum der Divergenz
    continuity_violation_std::Float64       # Streuung der Divergenz
    divergence_l2_norm::Float64             # L2-Norm der Divergenz
    
    # Vergleich mit Ground Truth Divergenz
    divergence_similarity::Float64          # Ähnlichkeit zu LaMEM-Divergenz
    vorticity_preservation::Float64         # Vortizitäts-Erhaltung
    
    # ============ STRUKTURELLE ÄHNLICHKEIT ============
    pearson_vx::Float64
    pearson_vz::Float64
    pearson_total::Float64           # Gesamt-Korrelation
    ssim_vx::Float64
    ssim_vz::Float64
    cross_correlation_max::Float64
    
    # ============ KRISTALL-SPEZIFISCHE METRIKEN ============
    alignment_error_mean::Float64    # Durchschnittliche Positions-Abweichung
    alignment_error_max::Float64     # Maximale Positions-Abweichung
    crystal_detection_rate::Float64  # Anteil korrekt erkannter Kristalle
    radial_profile_similarity::Float64
    
    # ============ KOMPLEXITÄTS-METRIKEN ============
    interaction_complexity_index::Float64
    density_robustness_score::Float64
    
    # ============ LAMBDA-GEWICHTE (NEU - für Hyperparameter-Tracking) ============
    lambda_velocity::Float64         # Gewicht für Velocity Loss
    lambda_divergence::Float64       # Gewicht für Divergenz-Penalty
    lambda_residual::Float64         # Gewicht für Residuum-Regularisierung
    
    # ============ PERFORMANCE-METRIKEN ============
    processing_time::Float64         # Inferenz-Zeit (Sekunden)
    memory_usage::Float64           # Speicherverbrauch (MB)
end

"""
Kompakte Statistik-Struktur für aggregierte Ergebnisse
Optimiert für schnelle Übersicht
"""
struct AggregatedStatistics
    crystal_count::Int
    n_samples::Int
    
    # Kernmetriken - Statistiken
    mae_mean::Float64
    mae_std::Float64
    mae_median::Float64
    mae_q25::Float64
    mae_q75::Float64
    
    # Verbesserung über Stokes
    improvement_mean::Float64
    improvement_std::Float64
    
    # Residuum-Charakteristika
    residual_magnitude_mean::Float64
    residual_sparsity_mean::Float64
    
    # Physik-Konsistenz
    continuity_violation_mean::Float64
    continuity_violation_max::Float64
    
    # Korrelation
    correlation_mean::Float64
    correlation_std::Float64
end

"""
Batch-Evaluierungsergebnisse für Multiple Kristallanzahlen
ERWEITERT für Residual Learning
"""
struct ResidualBatchResults
    crystal_range::UnitRange{Int}
    results_per_crystal::Dict{Int, Vector{ResidualEvaluationResult}}
    aggregated_stats::Dict{Int, AggregatedStatistics}
    
    # Skalierungs-Analysen
    mae_progression::Vector{Float64}
    improvement_progression::Vector{Float64}
    continuity_progression::Vector{Float64}
    residual_magnitude_progression::Vector{Float64}
    
    # Meta-Informationen
    total_samples::Int
    evaluation_timestamp::DateTime
    model_config::Dict{String, Any}
    lambda_config::Dict{String, Float64}  # NEU: Verwendete Lambda-Gewichte
end

"""
Hilfsfunktion: Erstelle leeres Ergebnis bei Fehler
"""
function create_error_result(crystal_count, sample_id)
    return ResidualEvaluationResult(
        crystal_count, sample_id, now(),
        999.0, 999.0, -999.0,  # Stokes-Baseline
        999.0, 999.0, 999.0, 0.0, 0.0, 999.0,  # Residuum
        999.0, 999.0, 999.0, 999.0, 999.0, 999.0,  # MAE/RMSE
        999.0, 999.0, 999.0, 999.0,  # Kontinuität
        0.0, 0.0,  # Divergenz/Vortizität
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # Strukturell
        999.0, 999.0, 0.0, 0.0,  # Kristall-Metriken
        0.0, 0.0,  # Komplexität
        0.0, 0.0, 0.0,  # Lambdas
        0.0, 0.0  # Performance
    )
end

"""
Hilfsfunktion: Berechne aggregierte Statistiken aus Ergebnisliste
OPTIMIERT: Verwendet vektorisierte Operationen
"""
function compute_aggregated_statistics(results::Vector{ResidualEvaluationResult})
    if isempty(results)
        return AggregatedStatistics(
            0, 0,
            999.0, 0.0, 999.0, 999.0, 999.0,
            0.0, 0.0,
            0.0, 0.0,
            999.0, 999.0,
            0.0, 0.0
        )
    end
    
    n = length(results)
    crystal_count = results[1].crystal_count
    
    # Filtere valide Ergebnisse (keine Fehler-Marker 999.0)
    valid_results = filter(r -> r.mae_with_residual < 900.0, results)
    
    if isempty(valid_results)
        return compute_aggregated_statistics([])  # Rekursion mit leerem Array
    end
    
    # Vektorisierte Extraktion
    maes = [r.mae_with_residual for r in valid_results]
    improvements = [r.improvement_over_stokes_percent for r in valid_results]
    residuals = [r.residual_mean for r in valid_results]
    sparsities = [r.residual_sparsity for r in valid_results]
    continuities_mean = [r.continuity_violation_mean for r in valid_results]
    continuities_max = [r.continuity_violation_max for r in valid_results]
    correlations = [r.pearson_total for r in valid_results]
    
  
    
    return AggregatedStatistics(
        crystal_count,
        length(valid_results),
        mean(maes), std(maes), median(maes), quantile(maes, 0.25), quantile(maes, 0.75),
        mean(improvements), std(improvements),
        mean(residuals), mean(sparsities),
        mean(continuities_mean), mean(continuities_max),
        mean(correlations), std(correlations)
    )
end

println("✓ Residual Evaluation Core Datenstrukturen geladen")
println("  - ResidualEvaluationResult: Erweiterte Metriken mit Stokes-Baseline & Residuum-Analyse")
println("  - AggregatedStatistics: Kompakte Übersichts-Statistiken")
println("  - ResidualBatchResults: Batch-Evaluierung mit Skalierungs-Tracking")