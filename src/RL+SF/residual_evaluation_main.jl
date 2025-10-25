# =============================================================================
# RESIDUAL LEARNING EVALUATION - HAUPTFUNKTIONEN
# =============================================================================

using Statistics
using Dates

# Benötigt die vorherigen Module:
include("residual_evaluation_core.jl")
include("residual_evaluation_metrics.jl")

# =============================================================================
# HAUPT-EVALUIERUNGSFUNKTION
# =============================================================================

"""
Evaluiert ein ResidualUNet Modell auf einem Sample
VOLLSTÄNDIGE RESIDUAL LEARNING EVALUIERUNG

# Arguments
- `model`: ResidualUNet Modell (muss v_total, v_stokes, Δv zurückgeben)
- `sample`: LaMEM Sample (x, z, phase, vx, vz, exx, ezz, v_stokes_scalar)
- `target_resolution`: Auflösung für Evaluation
- `sample_id`: ID für Tracking
- `lambda_config`: Dict mit verwendeten Lambda-Gewichten (optional)
- `sparsity_threshold`: Schwelle für Residuum-Sparsity (default: 1e-4)

# Returns
- `ResidualEvaluationResult` mit allen Metriken

# Workflow
1. Sample preprocessing (Phase → Tensor, LaMEM GT)
2. Model inference (v_total, v_stokes, Δv)
3. Kristall-Analyse (Zentren, Minima)
4. Stokes-Baseline Evaluation
5. Residuum-Analyse
6. Physikalische Konsistenz (Divergenz, Vortizität)
7. Strukturelle Ähnlichkeit
8. Kristall-spezifische Metriken
"""
function evaluate_residual_model(model, sample;
                                 target_resolution::Int=256,
                                 sample_id::Int=1,
                                 lambda_config::Dict{String, Float64}=Dict(
                                     "velocity" => 1.0,
                                     "divergence" => 0.0,
                                     "residual" => 0.0
                                 ),
                                 sparsity_threshold::Float64=1e-4)
    
    start_time = time()
    memory_before = Base.gc_num().allocd
    
    try
        # ============ 1. SAMPLE PREPROCESSING ============
        x, z, phase, vx, vz, exx, ezz, v_stokes_scalar = sample
        
        # ANNAHME: preprocess_lamem_sample existiert aus bestehendem Code
        phase_tensor, velocity_tensor = preprocess_lamem_sample(
            x, z, phase, vx, vz, v_stokes_scalar,
            target_resolution=target_resolution
        )
        
        # ============ 2. MODEL INFERENCE ============
        # Wichtig: ResidualUNet muss v_total, v_stokes, Δv zurückgeben!
        v_total, v_stokes, Δv = model(phase_tensor)  # [H, W, 2, batch]
        
        # CPU-Transfer für Analyse
        v_total = cpu(v_total)
        v_stokes = cpu(v_stokes)
        Δv = cpu(Δv)
        
        # Extrahiere 2D Arrays
        phase_2d = phase_tensor[:, :, 1, 1]
        gt_vx = velocity_tensor[:, :, 1, 1]
        gt_vz = velocity_tensor[:, :, 2, 1]
        
        pred_vx = v_total[:, :, 1, 1]
        pred_vz = v_total[:, :, 2, 1]
        
        stokes_vx = v_stokes[:, :, 1, 1]
        stokes_vz = v_stokes[:, :, 2, 1]
        
        residual_vx = Δv[:, :, 1, 1]
        residual_vz = Δv[:, :, 2, 1]
        
        # ============ 3. KRISTALL-ANALYSE ============
        # ANNAHME: Diese Funktionen existieren aus bestehendem Code
        crystal_centers = find_crystal_centers(phase_2d)
        n_crystals = length(crystal_centers)
        
        gt_minima = find_velocity_minima(gt_vz, n_crystals)
        pred_minima = find_velocity_minima(pred_vz, n_crystals)
        
        # ============ 4. STOKES-BASELINE EVALUATION ============
        mae_stokes, mae_residual, improvement = calculate_stokes_improvement(
            pred_vx, pred_vz,
            stokes_vx, stokes_vz,
            gt_vx, gt_vz
        )
        
        # ============ 5. RESIDUUM-ANALYSE ============
        res_mean, res_max, res_std, sparsity, stokes_ratio = 
            calculate_residual_statistics(
                residual_vx, residual_vz,
                stokes_vx, stokes_vz,
                sparsity_threshold=sparsity_threshold
            )
        
        # Stokes Contribution Ratio
        stokes_mag = sqrt.(stokes_vx.^2 .+ stokes_vz.^2)
        total_mag = sqrt.(pred_vx.^2 .+ pred_vz.^2)
        stokes_contribution = mean(stokes_mag) / (mean(total_mag) + 1e-8)
        
        # ============ 6. FEHLERMETRIKEN ============
        mae_vx, mae_vz, rmse_vx, rmse_vz, max_vx, max_vz = 
            calculate_error_metrics(pred_vx, pred_vz, gt_vx, gt_vz)
        
        # ============ 7. PHYSIKALISCHE KONSISTENZ (KRITISCH!) ============
        # Ohne Stream Function ist dies NICHT automatisch erfüllt!
        cont_mean, cont_max, cont_std, cont_l2 = 
            calculate_continuity_metrics(pred_vx, pred_vz)
        
        # Divergenz-Ähnlichkeit zu Ground Truth
        div_similarity = calculate_divergence_similarity(
            pred_vx, pred_vz, gt_vx, gt_vz
        )
        
        # Vortizität
        vort_preservation = calculate_vorticity_preservation(
            gt_vx, gt_vz, pred_vx, pred_vz
        )
        
        # ============ 8. STRUKTURELLE ÄHNLICHKEIT ============
        pearson_vx = safe_correlation(pred_vx, gt_vx)
        pearson_vz = safe_correlation(pred_vz, gt_vz)
        pearson_total = safe_correlation(
            sqrt.(pred_vx.^2 .+ pred_vz.^2),
            sqrt.(gt_vx.^2 .+ gt_vz.^2)
        )
        
        ssim_vx = calculate_ssim(gt_vx, pred_vx)
        ssim_vz = calculate_ssim(gt_vz, pred_vz)
        
        cross_corr = max(
            calculate_cross_correlation(gt_vx, pred_vx),
            calculate_cross_correlation(gt_vz, pred_vz)
        )
        
        # ============ 9. KRISTALL-SPEZIFISCHE METRIKEN ============
        alignment_mean = calculate_alignment_error(crystal_centers, pred_minima)
        if isinf(alignment_mean)
            alignment_mean = 999.0
        end
        
        alignment_max = 0.0
        if !isempty(crystal_centers) && !isempty(pred_minima)
            dists = [sqrt((c[1] - m[1])^2 + (c[2] - m[2])^2) 
                    for c in crystal_centers 
                    for m in pred_minima]
            alignment_max = isempty(dists) ? 999.0 : maximum(dists)
        end
        
        detection_rate = min(length(pred_minima), n_crystals) / max(n_crystals, 1)
        
        radial_similarity = calculate_radial_profile_similarity(
            phase_2d, gt_vz, pred_vz
        )
        
        # ============ 10. KOMPLEXITÄTS-METRIKEN ============
        complexity_index = calculate_interaction_complexity_index(
            crystal_centers, gt_vz
        )
        
        density_robustness = 1.0 / (1.0 + alignment_mean / 10.0)
        
        # ============ 11. PERFORMANCE-METRIKEN ============
        processing_time = time() - start_time
        memory_after = Base.gc_num().allocd
        memory_usage = (memory_after - memory_before) / 1024^2  # MB
        
        # ============ ERGEBNIS ERSTELLEN ============
        return ResidualEvaluationResult(
            # Grundinfo
            n_crystals, sample_id, now(),
            # Stokes-Baseline
            mae_stokes, mae_residual, improvement,
            # Residuum
            res_mean, res_max, res_std, sparsity, stokes_contribution, stokes_ratio,
            # Fehler
            mae_vx, mae_vz, rmse_vx, rmse_vz, max_vx, max_vz,
            # Physik
            cont_mean, cont_max, cont_std, cont_l2,
            div_similarity, vort_preservation,
            # Strukturell
            pearson_vx, pearson_vz, pearson_total,
            ssim_vx, ssim_vz, cross_corr,
            # Kristall
            alignment_mean, alignment_max, detection_rate, radial_similarity,
            # Komplexität
            complexity_index, density_robustness,
            # Lambda
            lambda_config["velocity"], lambda_config["divergence"], lambda_config["residual"],
            # Performance
            processing_time, memory_usage
        )
        
    catch e
        println("⚠ Fehler bei Evaluierung Sample $sample_id: $e")
        println("  Stacktrace: ", stacktrace(catch_backtrace()))
        return create_error_result(0, sample_id)
    end
end

# =============================================================================
# BATCH-EVALUIERUNG
# =============================================================================

"""
Evaluiert Modell über Multiple Kristallanzahlen
OPTIMIERT: Parallele Verarbeitung möglich, Memory-Management

# Arguments
- `model_path`: Pfad zum gespeicherten Modell
- `crystal_range`: Bereich der Kristallanzahlen (z.B. 1:10)
- `samples_per_count`: Anzahl Samples pro Kristallanzahl
- `target_resolution`: Auflösung
- `lambda_config`: Verwendete Lambda-Gewichte
- `output_dir`: Verzeichnis für Ergebnisse
- `verbose`: Fortschrittsmeldungen

# Returns
- `ResidualBatchResults`
"""
function evaluate_residual_batch(model_path::String;
                                 crystal_range::UnitRange{Int}=1:10,
                                 samples_per_count::Int=10,
                                 target_resolution::Int=256,
                                 lambda_config::Dict{String, Float64}=Dict(
                                     "velocity" => 1.0,
                                     "divergence" => 0.0,
                                     "residual" => 0.0
                                 ),
                                 output_dir::String="residual_evaluation",
                                 verbose::Bool=true)
    
    if verbose
        println("\n" * "="^80)
        println("RESIDUAL LEARNING BATCH EVALUATION")
        println("="^80)
        println("Modell: $model_path")
        println("Kristallbereich: $crystal_range")
        println("Samples pro Count: $samples_per_count")
        println("Auflösung: $(target_resolution)×$(target_resolution)")
        println("Lambda-Gewichte: $lambda_config")
    end
    
    # Verzeichnisse erstellen
    mkpath(output_dir)
    mkpath(joinpath(output_dir, "data"))
    
    # Modell laden
    if verbose
        println("\nLade Modell...")
    end
    model = load_trained_model(model_path)  # ANNAHME: Funktion existiert
    
    # Ergebnisse sammeln
    results_per_crystal = Dict{Int, Vector{ResidualEvaluationResult}}()
    total_samples = 0
    
    # ============ EVALUIERUNGS-LOOP ============
    for n_crystals in crystal_range
        if verbose
            println("\n" * "─"^60)
            println("Evaluiere $n_crystals Kristalle...")
        end
        
        crystal_results = ResidualEvaluationResult[]
        
        for sample_id in 1:samples_per_count
            if verbose && (sample_id % 5 == 1 || samples_per_count < 5)
                print("  Sample $sample_id/$samples_per_count... ")
            end
            
            try
                # Sample generieren
                sample = generate_crystal_sample(n_crystals, target_resolution)
                
                # Evaluieren
                result = evaluate_residual_model(
                    model, sample,
                    target_resolution=target_resolution,
                    sample_id=sample_id,
                    lambda_config=lambda_config
                )
                
                push!(crystal_results, result)
                total_samples += 1
                
                if verbose && (sample_id % 5 == 1 || samples_per_count < 5)
                    println("✓ (MAE: $(round(result.mae_with_residual, digits=4)), " *
                           "Improvement: $(round(result.improvement_over_stokes_percent, digits=1))%)")
                end
                
            catch e
                if verbose
                    println("✗ Fehler: $e")
                end
                continue
            end
            
            # Memory cleanup
            if sample_id % 5 == 0
                GC.gc()
            end
        end
        
        results_per_crystal[n_crystals] = crystal_results
        
        if verbose
            successful = length(crystal_results)
            avg_improvement = mean([r.improvement_over_stokes_percent 
                                   for r in crystal_results 
                                   if r.mae_with_residual < 900.0])
            println("  ✓ $successful/$samples_per_count erfolgreich")
            println("  Durchschnittliche Verbesserung: $(round(avg_improvement, digits=1))%")
        end
    end
    
    # ============ AGGREGIERTE STATISTIKEN ============
    if verbose
        println("\nBerechne aggregierte Statistiken...")
    end
    
    aggregated_stats = Dict{Int, AggregatedStatistics}()
    for (n_crystals, results) in results_per_crystal
        aggregated_stats[n_crystals] = compute_aggregated_statistics(results)
    end
    
    # ============ SKALIERUNGS-METRIKEN ============
    mae_prog = Float64[]
    improvement_prog = Float64[]
    continuity_prog = Float64[]
    residual_mag_prog = Float64[]
    
    for n_crystals in sort(collect(keys(aggregated_stats)))
        stats = aggregated_stats[n_crystals]
        push!(mae_prog, stats.mae_mean)
        push!(improvement_prog, stats.improvement_mean)
        push!(continuity_prog, stats.continuity_violation_mean)
        push!(residual_mag_prog, stats.residual_magnitude_mean)
    end
    
    # ============ BATCH-ERGEBNIS ERSTELLEN ============
    batch_results = ResidualBatchResults(
        crystal_range,
        results_per_crystal,
        aggregated_stats,
        mae_prog, improvement_prog, continuity_prog, residual_mag_prog,
        total_samples,
        now(),
        Dict(
            "model_path" => model_path,
            "target_resolution" => target_resolution,
            "samples_per_count" => samples_per_count
        ),
        lambda_config
    )
    
    # ============ ERGEBNISSE SPEICHERN ============
    results_path = joinpath(output_dir, "data", "residual_batch_results_1.bson")
    BSON.bson(results_path, Dict("batch_results" => batch_results))
    
    if verbose
        println("\n" * "="^80)
        println("EVALUIERUNG ABGESCHLOSSEN")
        println("="^80)
        println("Gesamt-Samples: $total_samples")
        println("Ergebnisse gespeichert: $results_path")
        println("Ausgabeverzeichnis: $output_dir")
    end
    
    return batch_results
end

# =============================================================================
# HILFSFUNKTIONEN (INTERFACE ZU BESTEHENDEM CODE)
# =============================================================================

"""
Wrapper für Kristallerkennung - Interface zu bestehendem Code
MUSS von bestehendem evaluate_model.jl bereitgestellt werden
"""
function find_crystal_centers(phase_field::AbstractMatrix)
    # Diese Funktion sollte aus dem bestehenden Code kommen
    # Placeholder: Nimm an sie existiert
    error("find_crystal_centers muss aus bestehendem Code geladen werden")
end

function find_velocity_minima(velocity_field::AbstractMatrix, n_expected::Int)
    error("find_velocity_minima muss aus bestehendem Code geladen werden")
end

function calculate_alignment_error(centers, minima)
    error("calculate_alignment_error muss aus bestehendem Code geladen werden")
end

function calculate_radial_profile_similarity(phase, vz_gt, vz_pred)
    error("calculate_radial_profile_similarity muss aus bestehendem Code geladen werden")
end

function calculate_interaction_complexity_index(centers, velocity_field)
    error("calculate_interaction_complexity_index muss aus bestehendem Code geladen werden")
end

function generate_crystal_sample(n_crystals::Int, resolution::Int)
    error("generate_crystal_sample muss aus bestehendem Code geladen werden")
end

println("✓ Residual Evaluation Hauptfunktionen geladen")
println("  - evaluate_residual_model(): Einzelne Sample-Evaluierung")
println("  - evaluate_residual_batch(): Multi-Kristall Batch-Evaluierung")
println("\n⚠ WICHTIG: Benötigt Interface-Funktionen aus bestehendem Code:")
println("  - find_crystal_centers, find_velocity_minima")
println("  - calculate_alignment_error, calculate_radial_profile_similarity")
println("  - calculate_interaction_complexity_index")
println("  - generate_crystal_sample, load_trained_model, preprocess_lamem_sample")