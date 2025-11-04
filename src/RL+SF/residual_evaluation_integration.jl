# =============================================================================
# RESIDUAL LEARNING EVALUATION - INTEGRATION MIT BESTEHENDEM CODE
# =============================================================================
# Verbindet neue Residual-Evaluation-Module mit bestehendem Code
# DIESES SCRIPT ANPASSEN AN DEINE DATEINAMEN/FUNKTIONEN!

using Dates
using Printf
using Statistics
using BSON
using Plots

println("="^80)
println("RESIDUAL LEARNING EVALUATION - INTEGRATION")
println("="^80)
println("Gestartet: $(now())")
println()

# =============================================================================
# 1. LADE BESTEHENDE MODULE (DEINE ORIGINALEN)
# =============================================================================

println("1. LADE BESTEHENDE PIPELINE-MODULE...")
println("-"^60)

# LaMEM Interface
try
    include("lamem_interface.jl")
    println("  ✓ LaMEM Interface geladen")
catch e
    println("  ✗ LaMEM Interface FEHLER: $e")
    error("Kann nicht ohne LaMEM Interface fortfahren!")
end

# Data Processing
try
    include("data_processing.jl")
    println("  ✓ Data Processing geladen")
catch e
    println("  ✗ Data Processing FEHLER: $e")
    error("Kann nicht ohne Data Processing fortfahren!")
end
include("stokes_analytical.jl")
# UNet Architecture
try
    include("unet_architecture.jl")
    println("  ✓ UNet Architecture geladen")
catch e
    println("  ✗ UNet Architecture FEHLER: $e")
end

# Evaluate Model (Kristall-Erkennung, etc.)
try
    include("evaluate_model.jl")
    println("  ✓ Evaluate Model geladen")
catch e
    println("  ⚠ Evaluate Model nicht gefunden (optional): $e")
end
# Im REPL:

println()

# =============================================================================
# 2. LADE NEUE RESIDUAL-EVALUATION MODULE
# =============================================================================

println("2. LADE NEUE RESIDUAL-EVALUATION MODULE...")
println("-"^60)

include("residual_evaluation_core.jl")
println("  ✓ Core (Datenstrukturen)")

include("residual_evaluation_metrics.jl")
println("  ✓ Metrics (Berechnungsfunktionen)")

include("residual_evaluation_main.jl")
println("  ✓ Main (Haupt-Evaluierung) - MIT PLACEHOLDERS")

include("residual_evaluation_visualization.jl")
println("  ✓ Visualization (Plots)")

println()

# =============================================================================
# 3. VERBINDE PLACEHOLDER-FUNKTIONEN MIT BESTEHENDEM CODE
# =============================================================================

println("3. VERBINDE PLACEHOLDER-FUNKTIONEN...")
println("-"^60)

# -----------------------------------------------------------------------------
# 3.1 KRISTALL-ERKENNUNG
# -----------------------------------------------------------------------------

"""
Wrapper: find_crystal_centers
ANPASSEN: Falls deine Funktion anders heißt!
"""
function find_crystal_centers(phase_field::AbstractMatrix)
    # OPTION A: Falls Funktion aus evaluate_model.jl existiert
    if isdefined(Main, :detect_crystal_centers)
        return Main.detect_crystal_centers(phase_field)
    end
    
    # OPTION B: Falls Funktion anders heißt - HIER ANPASSEN!
    # return deine_kristall_erkennungs_funktion(phase_field)
    
    # OPTION C: Minimale Fallback-Implementation (falls keine Funktion existiert)
    # Einfache Implementierung: Finde lokale Minima im Phasenfeld
    println("  ⚠ Verwende Fallback-Kristallerkennung (nicht optimal!)")
    
    # Finde Bereiche wo phase < 0.5 (Kristall = 0, Matrix = 1)
    H, W = size(phase_field)
    centers = []
    
    # Sehr einfach: Grid-basierte Suche
    for i in 20:40:H-20
        for j in 20:40:W-20
            if phase_field[i, j] < 0.5
                push!(centers, (j, i))  # (x, z) Format
            end
        end
    end
    
    return centers
end

println("  ✓ find_crystal_centers verbunden")

# -----------------------------------------------------------------------------
# 3.2 VELOCITY MINIMA
# -----------------------------------------------------------------------------

"""
Wrapper: find_velocity_minima
ANPASSEN: Falls deine Funktion anders heißt!
"""
function find_velocity_minima(velocity_field::AbstractMatrix, n_expected::Int)
    # OPTION A: Falls Funktion existiert
    if isdefined(Main, :find_minima_locations)
        return Main.find_minima_locations(velocity_field, n_expected)
    end
    
    # OPTION B: Fallback-Implementation
    println("  ⚠ Verwende Fallback-Minima-Erkennung")
    
    H, W = size(velocity_field)
    minima = []
    
    # Finde n_expected lokale Minima
    # Einfache Strategie: Sortiere alle Werte, nimm die kleinsten
    indices = sortperm(vec(velocity_field))
    
    for idx in indices[1:min(n_expected*2, length(indices))]
        i = ((idx - 1) ÷ W) + 1
        j = ((idx - 1) % W) + 1
        
        # Prüfe ob lokales Minimum (grob)
        is_local_min = true
        for di in -1:1, dj in -1:1
            ni, nj = i+di, j+dj
            if 1 <= ni <= H && 1 <= nj <= W
                if velocity_field[ni, nj] < velocity_field[i, j]
                    is_local_min = false
                    break
                end
            end
        end
        
        if is_local_min
            push!(minima, (j, i))  # (x, z) Format
            if length(minima) >= n_expected
                break
            end
        end
    end
    
    return minima
end

println("  ✓ find_velocity_minima verbunden")

# -----------------------------------------------------------------------------
# 3.3 ALIGNMENT ERROR
# -----------------------------------------------------------------------------

"""
Wrapper: calculate_alignment_error
"""
function calculate_alignment_error(centers::Vector, minima::Vector)
    if isempty(centers) || isempty(minima)
        return Inf
    end
    
    # OPTION A: Falls Funktion existiert
    if isdefined(Main, :compute_alignment_error)
        return Main.compute_alignment_error(centers, minima)
    end
    
    # OPTION B: Einfache Implementation
    # Berechne durchschnittliche Distanz zwischen nächsten Paaren
    n_pairs = min(length(centers), length(minima))
    total_dist = 0.0
    
    for i in 1:n_pairs
        c = centers[i]
        m = minima[i]
        dist = sqrt((c[1] - m[1])^2 + (c[2] - m[2])^2)
        total_dist += dist
    end
    
    return total_dist / n_pairs
end

println("  ✓ calculate_alignment_error verbunden")

# -----------------------------------------------------------------------------
# 3.4 RADIAL PROFILE SIMILARITY
# -----------------------------------------------------------------------------

"""
Wrapper: calculate_radial_profile_similarity
"""
function calculate_radial_profile_similarity(phase::AbstractMatrix, 
                                            vz_gt::AbstractMatrix, 
                                            vz_pred::AbstractMatrix)
    # OPTION A: Falls Funktion existiert
    if isdefined(Main, :compute_radial_similarity)
        return Main.compute_radial_similarity(phase, vz_gt, vz_pred)
    end
    
    # OPTION B: Vereinfachte Implementation - Korrelation
    # (Nicht perfekt, aber funktional)
    return cor(vec(vz_gt), vec(vz_pred))
end

println("  ✓ calculate_radial_profile_similarity verbunden")

# -----------------------------------------------------------------------------
# 3.5 INTERACTION COMPLEXITY INDEX
# -----------------------------------------------------------------------------

"""
Wrapper: calculate_interaction_complexity_index
"""
function calculate_interaction_complexity_index(centers::Vector, velocity_field::AbstractMatrix)
    if length(centers) <= 1
        return 0.0
    end
    
    # OPTION A: Falls Funktion existiert
    if isdefined(Main, :compute_complexity_index)
        return Main.compute_complexity_index(centers, velocity_field)
    end
    
    # OPTION B: Einfache Implementation
    # Basierend auf Kristallabständen und Geschwindigkeits-Varianz
    n_crystals = length(centers)
    
    # Paarweise Abstände
    distances = Float64[]
    for i in 1:n_crystals
        for j in (i+1):n_crystals
            dist = sqrt((centers[i][1] - centers[j][1])^2 + 
                       (centers[i][2] - centers[j][2])^2)
            push!(distances, dist)
        end
    end
    
    min_dist = isempty(distances) ? 1.0 : minimum(distances)
    velocity_var = var(velocity_field)
    
    complexity = (n_crystals^2) / (min_dist + 1e-6) * log(1 + velocity_var)
    
    return complexity
end

println("  ✓ calculate_interaction_complexity_index verbunden")

# -----------------------------------------------------------------------------
# 3.6 SAMPLE GENERATION
# -----------------------------------------------------------------------------

"""
Wrapper: generate_crystal_sample
WICHTIG: Diese Funktion MUSS angepasst werden!
"""
function generate_crystal_sample(n_crystals::Int, resolution::Int)
    # Radius für alle Kristalle
    radius_crystal = fill(0.04, n_crystals)
    
    # Grid-basierte Positionierung
    centers = []
    for i in 1:n_crystals
        x_pos = -0.6 + (i-1) % 4 * 0.4
        z_pos = 0.2 + div(i-1, 4) * 0.3
        push!(centers, (x_pos, z_pos))
    end
    
    # Rufe LaMEM_Multi_crystal aus lamem_interface.jl
    # ANPASSEN: Falls deine Funktion andere Parameter erwartet!
    return LaMEM_Multi_crystal(
        resolution=(resolution, resolution),
        n_crystals=n_crystals,
        radius_crystal=radius_crystal,
        cen_2D=centers
    )
end

println("  ✓ generate_crystal_sample verbunden")

# -----------------------------------------------------------------------------
# 3.7 MODEL LOADING
# -----------------------------------------------------------------------------

"""
Wrapper: load_trained_model
"""
# KORRIGIERTE Load-Funktion
function load_trained_model(model_path::String)
    println("Lade Modell: $model_path")
    
    if !isfile(model_path)
        error("Modelldatei nicht gefunden: $model_path")
    end
    
    data = BSON.load(model_path)
    
    # EXPLIZIT :model verwenden
    if haskey(data, :model)
        model = data[:model]
        println("✓ Modell geladen: $(typeof(model))")
        return model
    elseif haskey(data, "model")  # String-Key als Fallback
        model = data["model"]
        println("✓ Modell geladen: $(typeof(model))")
        return model
    end
    
    error("Key :model nicht in BSON gefunden! Keys: $(keys(data))")
end

println("  ✓ load_trained_model verbunden")

println()

# =============================================================================
# 4. KONFIGURATION
# =============================================================================

println("4. KONFIGURATION...")
println("-"^60)

# WICHTIG: DIESE PFADE ANPASSEN!
const MODEL_PATH = "H:/Masterarbeit/Modelle/final_model_rlsf_2.bson"  # ANPASSEN!
const OUTPUT_DIR = "H:/Masterarbeit/Auswertung/Residual_Evaluation_1"  # ANPASSEN!

# Lambda-Gewichte (sollte mit Training übereinstimmen!)
const LAMBDA_CONFIG = Dict(
    "velocity" => 1.0,      # Haupt-Loss
    "divergence" => 0.1,    # Divergenz-Penalty (WICHTIG!)
    "residual" => 0.01      # Residuum-Regularisierung
)

# Evaluierungs-Parameter
const EVAL_CONFIG = (
    crystal_range = 1:5,         # Start mit wenigen für Tests
    samples_per_count = 5,       # Wenige für Tests
    target_resolution = 256,
    sparsity_threshold = 1e-4
)

println("  Modell: $MODEL_PATH")
println("  Output: $OUTPUT_DIR")
println("  Kristallbereich: $(EVAL_CONFIG.crystal_range)")
println("  Samples pro Count: $(EVAL_CONFIG.samples_per_count)")
println("  Lambda-Gewichte:")
for (key, val) in LAMBDA_CONFIG
    println("    - $key: $val")
end

println()

# =============================================================================
# 5. QUICK TEST
# =============================================================================

println("5. QUICK TEST...")
println("-"^60)

function run_quick_test()
    try
        # Test 1: Sample-Generierung
        println("  Test 1: Sample-Generierung...")
        test_sample = generate_crystal_sample(2, 64)
        println("    ✓ Sample generiert (2 Kristalle, 64x64)")
        
        # Test 2: Preprocessing
        println("  Test 2: Preprocessing...")
        x, z, phase, vx, vz, exx, ezz, v_stokes = test_sample
        phase_tensor, velocity_tensor = preprocess_lamem_sample(
            x, z, phase, vx, vz, v_stokes,
            target_resolution=64
        )
        println("    ✓ Preprocessing funktioniert")
        
        # Test 3: Kristall-Erkennung
        println("  Test 3: Kristall-Erkennung...")
        phase_2d = phase_tensor[:, :, 1, 1]
        centers = find_crystal_centers(phase_2d)
        println("    ✓ Kristall-Zentren gefunden: $(length(centers))")
        
        # Test 4: Velocity Minima
        println("  Test 4: Velocity Minima...")
        gt_vz = velocity_tensor[:, :, 2, 1]
        minima = find_velocity_minima(gt_vz, 2)
        println("    ✓ Minima gefunden: $(length(minima))")
        
        # Test 5: Metriken
        println("  Test 5: Metriken-Funktionen...")
        gt_vx = velocity_tensor[:, :, 1, 1]
        
        # Kontinuitäts-Metriken
        cont_mean, cont_max, cont_std, cont_l2 = calculate_continuity_metrics(gt_vx, gt_vz)
        println("    ✓ Kontinuität: Mean=$(round(cont_mean, digits=6))")
        
        # Fehler-Metriken (Test mit identischen Feldern)
        mae_x, mae_z, rmse_x, rmse_z, max_x, max_z = calculate_error_metrics(
            gt_vx, gt_vz, gt_vx, gt_vz
        )
        println("    ✓ Fehlermetriken: MAE_x=$(round(mae_x, digits=6)) (sollte ~0 sein)")
        
        # SSIM
        ssim_val = calculate_ssim(gt_vz, gt_vz)
        println("    ✓ SSIM: $(round(ssim_val, digits=4)) (sollte ~1 sein)")
        
        println("\n  ✅ QUICK TEST ERFOLGREICH!")
        return true
        
    catch e
        println("\n  ❌ QUICK TEST FEHLGESCHLAGEN!")
        println("  Fehler: $e")
        println("\n  Stacktrace:")
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println()
        end
        return false
    end
end

TEST_SUCCESS = run_quick_test()

println()

# =============================================================================
# 6. HAUPTFUNKTIONEN DEFINIEREN
# =============================================================================

if TEST_SUCCESS
    println("6. HAUPTFUNKTIONEN BEREIT")
    println("-"^60)
    
    """
    Führt vollständige Residual Learning Evaluation durch
    """
    function run_full_residual_evaluation(;
        model_path::String=MODEL_PATH,
        output_dir::String=OUTPUT_DIR,
        crystal_range=EVAL_CONFIG.crystal_range,
        samples_per_count::Int=EVAL_CONFIG.samples_per_count)
        
        println("\n" * "="^80)
        println("STARTE VOLLSTÄNDIGE RESIDUAL LEARNING EVALUATION")
        println("="^80)
        
        # Prüfe Modell
        if !isfile(model_path)
            error("Modell nicht gefunden: $model_path")
        end
        
        # Evaluierung durchführen
        results = evaluate_residual_batch(
            model_path,
            crystal_range=crystal_range,
            samples_per_count=samples_per_count,
            target_resolution=EVAL_CONFIG.target_resolution,
            lambda_config=LAMBDA_CONFIG,
            output_dir=output_dir,
            verbose=true
        )
        
        # Zusammenfassung ausgeben
        print_evaluation_summary(results)
        
        return results
    end
    
    """
    Einzelnes Sample evaluieren (für Debugging)
    """
    function evaluate_single_residual_sample(model_path::String, n_crystals::Int=3)
        println("\nEvaluiere einzelnes Sample mit $n_crystals Kristallen...")
        
        # Modell laden
        model = load_trained_model(model_path)
        
        # Sample generieren
        sample = generate_crystal_sample(n_crystals, EVAL_CONFIG.target_resolution)
        
        # Evaluieren
        result = evaluate_residual_model(
            model, sample,
            target_resolution=EVAL_CONFIG.target_resolution,
            sample_id=1,
            lambda_config=LAMBDA_CONFIG
        )
        
        # Ausgabe
        println("\n📊 ERGEBNISSE:")
        println("  MAE (Stokes only): $(round(result.mae_stokes_only, digits=4))")
        println("  MAE (mit Residuum): $(round(result.mae_with_residual, digits=4))")
        println("  Verbesserung: $(round(result.improvement_over_stokes_percent, digits=2))%")
        println("  Residuum Mean: $(round(result.residual_mean, digits=6))")
        println("  Residuum Sparsity: $(round(result.residual_sparsity, digits=3))")
        println("  Kontinuitätsverletzung: $(round(result.continuity_violation_mean, digits=6))")
        println("  Pearson-Korrelation: $(round(result.pearson_total, digits=3))")
        
        return result
    end
    
    """
    Gibt Evaluierungs-Zusammenfassung aus
    """
    function print_evaluation_summary(results::ResidualBatchResults)
        println("\n" * "="^80)
        println("EVALUIERUNGS-ZUSAMMENFASSUNG")
        println("="^80)
        
        println("\n📊 GESAMTSTATISTIKEN")
        println("-"^60)
        println("  Gesamt-Samples: $(results.total_samples)")
        println("  Kristallbereich: $(results.crystal_range)")
        println("  Evaluiert am: $(results.evaluation_timestamp)")
        
        println("\n🎯 KERNMETRIKEN PRO KRISTALLANZAHL")
        println("-"^60)
        println(@sprintf("%-10s | %-10s | %-15s | %-15s | %-15s",
                        "Kristalle", "N Samples", "MAE (Mean)", "Improvement %", "Divergenz"))
        println("-"^90)
        
        for n_crystals in sort(collect(keys(results.aggregated_stats)))
            stats = results.aggregated_stats[n_crystals]
            println(@sprintf("%-10d | %-10d | %-15.5f | %-15.2f | %-15.6f",
                            n_crystals,
                            stats.n_samples,
                            stats.mae_mean,
                            stats.improvement_mean,
                            stats.continuity_violation_mean))
        end
        
        println("\n" * "="^80)
    end
    
    println("  ✓ run_full_residual_evaluation()")
    println("  ✓ evaluate_single_residual_sample()")
    println("  ✓ print_evaluation_summary()")
    
else
    println("⚠️  Quick Test fehlgeschlagen - Funktionen nicht verfügbar!")
    println("   Bitte Fehler beheben und neu laden.")
end

println()

# =============================================================================
# 7. USAGE GUIDE
# =============================================================================

println("7. USAGE GUIDE")
println("="^80)
println()
println("📚 VERFÜGBARE FUNKTIONEN:")
println()
println("1. EINZELNES SAMPLE (DEBUGGING):")
println("   julia> result = evaluate_single_residual_sample(MODEL_PATH, n_crystals=3)")
println()
println("2. VOLLSTÄNDIGE EVALUATION:")
println("   julia> results = run_full_residual_evaluation()")
println()
println("3. VISUALISIERUNG:")
println("   julia> dashboard = create_residual_dashboard(results,")
println("            save_path=joinpath(OUTPUT_DIR, \"dashboard.png\"))")
println()
println("4. 4-PANEL PLOT (nach Einzelevaluierung):")
println("   julia> # Benötigt v_stokes, residual, pred, gt aus Model-Output")
println()

if TEST_SUCCESS
    println("✅ SYSTEM BEREIT FÜR EVALUATION")
    println()
    println("🚀 NÄCHSTE SCHRITTE:")
    println("   1. Passe MODEL_PATH an (Zeile 242)")
    println("   2. Passe OUTPUT_DIR an (Zeile 243)")
    println("   3. Prüfe LAMBDA_CONFIG (Zeile 246-250)")
    println("   4. Führe aus: results = run_full_residual_evaluation()")
else
    println("⚠️  SYSTEM NICHT BEREIT")
    println()
    println("📝 ANPASSUNGEN NÖTIG:")
    println("   1. Prüfe dass alle Module korrekt laden")
    println("   2. Passe Wrapper-Funktionen an (Abschnitt 3)")
    println("   3. Teste erneut mit: include(\"residual_evaluation_integration.jl\")")
end

println()
println("="^80)