# =============================================================================
# EINFACHES SERVER-SCRIPT FÜR GENERALISIERUNGSSTUDIUM (KORRIGIERT)
# =============================================================================
# 

using Flux
using CUDA
using Statistics
using Random
using Serialization
using BSON
using BSON: @save, @load
using Printf

# Dates separat importieren um Namenskonflikt zu vermeiden
import Dates

println("=== GENERALISIERUNGSSTUDIUM: SERVER-VERSION ===")
println("Zeit: $(Dates.now())")

# Module laden - verwende die bestehenden
include("lamem_interface.jl")
include("data_processing.jl")
include("unet_config.jl")
include("unet_architecture.jl")
include("training.jl")
include("batch_management.jl")

# Einfache Server-Konfiguration
const SERVER_CONFIG = (
    target_crystal_count = 10,
    n_training_samples = 6,        # Mindestens 4 für Batch-Größe 2
    num_epochs = 20,               # Weniger Epochen für Test
    learning_rate = 0.00005f0,       # Etwas höhere Lernrate
    batch_size = 1,                # Kleinere Batch-Größe
    eval_crystal_range = 9:10,     # Nur 2 Kristallanzahlen
    n_eval_samples_per_count = 1,
    use_gpu = false,          # GPU verwenden
)

"""
EINFACHE Evaluierungsmetriken-Struktur
"""
struct SimpleMetrics
    crystal_count::Int
    mae_vx::Float64
    mae_vz::Float64
    mae_total::Float64
    r2_total::Float64
end

"""
EINFACHE Evaluierungsfunktion (ohne externe Abhängigkeiten)
"""
function simple_evaluate_on_crystal_count(model, crystal_count::Int; n_eval_samples=1, target_resolution=256)
    println("Evaluiere auf $crystal_count Kristalle(n) mit $n_eval_samples Samples...")
    
    # Generiere Evaluierungsdaten
    eval_dataset = generate_evaluation_dataset(crystal_count, n_eval_samples, 
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
                error("Unvollständiges Sample: $(length(sample)) Elemente")
            end
            
            # Preprocessing
            phase_tensor, velocity_tensor = preprocess_lamem_sample(
                x, z, phase, vx, vz, v_stokes,
                target_resolution=target_resolution
            )
            
            # Vorhersage
            prediction = cpu(model(phase_tensor))
            
            push!(predictions, prediction)
            push!(targets, velocity_tensor)
            
        catch e
            println("  Warnung: Evaluierungssample $i fehlgeschlagen: $e")
            continue
        end
    end
    
    if length(predictions) == 0
        error("Keine erfolgreichen Evaluierungen für $crystal_count Kristalle!")
    end
    
    # Tensoren zusammenfügen
    predictions_batch = cat(predictions..., dims=4)
    targets_batch = cat(targets..., dims=4)
    
    # Einfache Metriken berechnen
    pred_vx = vec(predictions_batch[:,:,1,:])
    pred_vz = vec(predictions_batch[:,:,2,:])
    true_vx = vec(targets_batch[:,:,1,:])
    true_vz = vec(targets_batch[:,:,2,:])
    
    # MAE berechnen
    mae_vx = mean(abs.(pred_vx .- true_vx))
    mae_vz = mean(abs.(pred_vz .- true_vz))
    mae_total = (mae_vx + mae_vz) / 2
    
    # R² berechnen
    ss_res_vx = sum((true_vx .- pred_vx).^2)
    ss_tot_vx = sum((true_vx .- mean(true_vx)).^2)
    r2_vx = max(0.0, 1.0 - ss_res_vx / max(ss_tot_vx, 1e-10))
    
    ss_res_vz = sum((true_vz .- pred_vz).^2)
    ss_tot_vz = sum((true_vz .- mean(true_vz)).^2)
    r2_vz = max(0.0, 1.0 - ss_res_vz / max(ss_tot_vz, 1e-10))
    
    r2_total = (r2_vx + r2_vz) / 2
    
    metrics = SimpleMetrics(crystal_count, mae_vx, mae_vz, mae_total, r2_total)
    
    println("  MAE Total: $(round(metrics.mae_total, digits=6))")
    println("  R² Total: $(round(metrics.r2_total, digits=3))")
    
    return metrics
end

"""
EINFACHE Generalisierungs-Evaluierung
"""
function evaluate_generalization(model; crystal_range=1:10, n_eval_samples_per_count=1, target_resolution=256, verbose=true)
    println("STARTE EINFACHE GENERALISIERUNGS-EVALUIERUNG")
    println("Kristallbereich: $(collect(crystal_range))")
    
    all_metrics = Dict{Int, SimpleMetrics}()
    
    for crystal_count in crystal_range
        try
            metrics = simple_evaluate_on_crystal_count(
                model, crystal_count, 
                n_eval_samples=n_eval_samples_per_count,
                target_resolution=target_resolution
            )
            
            all_metrics[crystal_count] = metrics
            
        catch e
            println("FEHLER bei $crystal_count Kristallen: $e")
            continue
        end
        
        GC.gc()
    end
    
    return all_metrics
end

"""
Hauptfunktion: Training + Evaluierung, speichere alles in einer BSON-Datei
"""
function run_server_study()
    println("Starte Generalisierungsstudie...")
    
    # 1. Training-Daten generieren
    println("\n1. Generiere Training-Daten...")
    training_dataset = generate_generalization_dataset(
        SERVER_CONFIG.n_training_samples,
        target_crystal_count=SERVER_CONFIG.target_crystal_count,
        verbose=true
    )
    
    # 2. Modell trainieren
    println("\n2. Trainiere UNet-Modell...")
    model = create_simplified_unet(1, 2, 64)
    
    train_config = create_training_config(
        learning_rate = SERVER_CONFIG.learning_rate,
        num_epochs = SERVER_CONFIG.num_epochs,
        batch_size = SERVER_CONFIG.batch_size,
        checkpoint_dir = "checkpoints",
        use_gpu = SERVER_CONFIG.use_gpu,
    )
    
    trained_model, train_losses, val_losses = train_velocity_unet_safe(
        model, training_dataset, 256, config=train_config
    )
    
    # Bestes Modell laden falls vorhanden
    if isfile("checkpoints/best_model.bson")
        trained_model = load_trained_model("checkpoints/best_model.bson")
    end
    
    # 3. Generalisierung evaluieren
    println("\n3. Evaluiere Generalisierung...")
    metrics = evaluate_generalization(
        trained_model,
        crystal_range=SERVER_CONFIG.eval_crystal_range,
        n_eval_samples_per_count=SERVER_CONFIG.n_eval_samples_per_count,
        verbose=true
    )
    
    # 4. Alles in eine Datei speichern
    println("\n4. Speichere Ergebnisse...")
    results = Dict(
        "model" => trained_model,
        "metrics" => metrics,
        "train_losses" => train_losses,
        "val_losses" => val_losses,
        "config" => SERVER_CONFIG,
        "date" => Dates.now()
    )
    
    @save "generalization_results.bson" results
    
    # Text-Zusammenfassung ausgeben
    println("\n" * "="^60)
    println("ERGEBNISSE:")
    println("="^60)
    
    crystal_counts = sort(collect(keys(metrics)))
    for cc in crystal_counts
        m = metrics[cc]
        println("$cc Kristalle: MAE = $(round(m.mae_total, digits=6)), R² = $(round(m.r2_total, digits=3))")
    end
    
    println("\nDatei gespeichert: generalization_results.bson")
    println("Training abgeschlossen!")
    
    return results
end

# Ausführen
results = run_server_study()