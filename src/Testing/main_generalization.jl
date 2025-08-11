# =============================================================================
# HAUPTSCRIPT FÜR GENERALISIERUNGSSTUDIUM - 10-KRISTALL-TRAINING
# =============================================================================
# Speichern als: main_generalization.jl

using Flux
using CUDA
using Statistics
using Random
using Dates
using Serialization
using BSON: @save, @load
using Printf

println("=== GENERALISIERUNGSSTUDIUM: 10-KRISTALL TRAINING ===")
# =============================================================================
# GENERALISIERUNGS-PIPELINE
# =============================================================================

"""
Vollständige Generalisierungsstudie: Training auf 10 Kristallen, Evaluierung auf 1-10
"""
function run_generalization_study()
    println("="^80)
    println("STARTE GENERALISIERUNGSSTUDIE")
    println("="^80)
    
    # Ausgabe-Verzeichnisse erstellen
    mkpath(GENERALIZATION_CONFIG.checkpoint_dir)
    mkpath(GENERALIZATION_CONFIG.results_dir)
    
    # Konfiguration ausgeben
    println("\nKonfiguration:")
    for (key, value) in pairs(GENERALIZATION_CONFIG)
        println("  $key: $value")
    end
    
    study_start_time = time()
    
    try
        # =================================================================
        # PHASE 1: SYSTEM-VALIDIERUNG
        # =================================================================
        println("\n" * "="^60)
        println("PHASE 1: SYSTEM-VALIDIERUNG")
        println("="^60)
        
        if !validate_generalization_system()
            error("System-Validierung fehlgeschlagen!")
        end
        println("✓ System-Validierung erfolgreich")
        
        # =================================================================
        # PHASE 2: TRAINING-DATEN GENERIERUNG
        # =================================================================
        println("\n" * "="^60)
        println("PHASE 2: TRAINING-DATEN GENERIERUNG")
        println("="^60)
        
        println("Generiere Training-Dataset mit Fokus auf $(GENERALIZATION_CONFIG.target_crystal_count) Kristalle...")
        
        training_dataset = generate_generalization_dataset(
            GENERALIZATION_CONFIG.n_training_samples,
            target_crystal_count=GENERALIZATION_CONFIG.target_crystal_count,
            crystal_distribution=GENERALIZATION_CONFIG.crystal_distribution,
            resolution=GENERALIZATION_CONFIG.target_resolution,
            verbose=true
        )
        
        if length(training_dataset) == 0
            error("Keine Trainingsdaten generiert!")
        end
        
        println("✓ Training-Dataset erstellt: $(length(training_dataset)) Samples")
        
        # Dataset speichern
        training_data_path = joinpath(GENERALIZATION_CONFIG.results_dir, "training_dataset.jls")
        serialize(training_data_path, training_dataset)
        println("✓ Training-Dataset gespeichert: $training_data_path")
        
        # =================================================================
        # PHASE 3: MODELL-TRAINING
        # =================================================================
        println("\n" * "="^60)
        println("PHASE 3: 10-KRISTALL-MODELL-TRAINING")
        println("="^60)
        
        # UNet erstellen
        println("Erstelle UNet für Generalisierungs-Training...")
        model = create_simplified_unet(1, 2, 64)  # Größere Base-Filter für Komplexität
        
        # Training-Konfiguration
        train_config = create_training_config(
            learning_rate = GENERALIZATION_CONFIG.learning_rate,
            num_epochs = GENERALIZATION_CONFIG.num_epochs,
            batch_size = GENERALIZATION_CONFIG.batch_size,
            checkpoint_dir = GENERALIZATION_CONFIG.checkpoint_dir,
            save_every_n_epochs = 5,
            use_gpu = GENERALIZATION_CONFIG.use_gpu,
            validation_split = GENERALIZATION_CONFIG.validation_split,
            early_stopping_patience = GENERALIZATION_CONFIG.early_stopping_patience
        )
        
        println("Starte Training mit $(train_config.num_epochs) Epochen...")
        
        # Training ausführen
        trained_model, train_losses, val_losses = train_velocity_unet_safe(
            model, training_dataset, GENERALIZATION_CONFIG.target_resolution,
            config=train_config
        )
        
        println("✓ Training abgeschlossen")
        
        # Bestes Modell laden
        best_model_path = joinpath(GENERALIZATION_CONFIG.checkpoint_dir, "best_model.bson")
        if isfile(best_model_path)
            println("Lade bestes Modell für Evaluierung...")
            trained_model = load_trained_model(best_model_path)
        end
        
        # =================================================================
        # PHASE 4: GENERALISIERUNGS-EVALUIERUNG
        # =================================================================
        println("\n" * "="^60)
        println("PHASE 4: GENERALISIERUNGS-EVALUIERUNG")
        println("="^60)
        
        println("Evaluiere Generalisierungsfähigkeit auf $(GENERALIZATION_CONFIG.eval_crystal_range)...")
        
        generalization_metrics = evaluate_generalization(
            trained_model,
            crystal_range=GENERALIZATION_CONFIG.eval_crystal_range,
            n_eval_samples_per_count=GENERALIZATION_CONFIG.n_eval_samples_per_count,
            target_resolution=GENERALIZATION_CONFIG.target_resolution,
            verbose=true,
            save_results=true,
            results_dir=GENERALIZATION_CONFIG.results_dir
        )
        
        if isempty(generalization_metrics)
            error("Keine Generalisierungs-Metriken erhalten!")
        end
        
        println("✓ Generalisierungs-Evaluierung abgeschlossen")
        
        # =================================================================
        # PHASE 5: ERGEBNISSE UND BERICHT
        # =================================================================
        println("\n" * "="^60)
        println("PHASE 5: ERGEBNISSE UND BERICHT")
        println("="^60)
        
        # Detaillierter Bericht
        create_generalization_report(generalization_metrics)
        
        # Kompakte Zusammenfassung
        create_generalization_summary(generalization_metrics, train_losses, val_losses)
        
        # Alles speichern
        final_results = Dict(
            "trained_model" => trained_model,
            "generalization_metrics" => generalization_metrics,
            "train_losses" => train_losses,
            "val_losses" => val_losses,
            "config" => GENERALIZATION_CONFIG,
            "study_date" => Dates.now()
        )
        
        final_results_path = joinpath(GENERALIZATION_CONFIG.results_dir, 
                                     "$(GENERALIZATION_CONFIG.model_name)_complete_results.bson")
        @save final_results_path final_results
        
        println("✓ Finale Ergebnisse gespeichert: $final_results_path")
        
        # =================================================================
        # ERFOLGREICHER ABSCHLUSS
        # =================================================================
        study_end_time = time()
        total_study_time = (study_end_time - study_start_time) / 60
        
        println("\n" * "="^80)
        println("GENERALISIERUNGSSTUDIE ERFOLGREICH ABGESCHLOSSEN")
        println("="^80)
        println("Gesamtzeit: $(round(total_study_time, digits=1)) Minuten")
        println("Trainierte Kristallanzahl: $(GENERALIZATION_CONFIG.target_crystal_count)")
        println("Evaluierte Kristallanzahlen: $(collect(GENERALIZATION_CONFIG.eval_crystal_range))")
        println("Ergebnisse gespeichert in: $(GENERALIZATION_CONFIG.results_dir)")
        
        return true, generalization_metrics
        
    catch e
        study_end_time = time()
        total_study_time = (study_end_time - study_start_time) / 60
        
        println("\n" * "="^80)
        println("GENERALISIERUNGSSTUDIE FEHLGESCHLAGEN")
        println("="^80)
        println("Fehler: $e")
        println("Laufzeit bis Fehler: $(round(total_study_time, digits=1)) Minuten")
        
        # Stacktrace für Debugging
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println()
        end
        
        return false, nothing
    end
end



"""
System-Validierung für Generalisierungsstudie
"""
function validate_generalization_system()
    try
        println("Teste erweiterte LaMEM-Funktionalität...")
        
        # Test verschiedener Kristallanzahlen
        for n_crystals in [1, 5, 10]
            test_sample = LaMEM_Variable_crystals(
                resolution=(64, 64),
                n_crystals=n_crystals
            )
            
            if length(test_sample) < 8
                error("Unvollständiges Sample für $n_crystals Kristalle")
            end
        end
        
        # Test UNet-Architektur
        model = create_simplified_unet(1, 2, 32)
        test_input = randn(Float32, 64, 64, 1, 2)
        output = model(test_input)
        
        if size(output) != (64, 64, 2, 2)
            error("UNet-Output-Dimension falsch: $(size(output))")
        end
        
        println("✓ System-Validierung abgeschlossen")
        return true
        
    catch e
        println("❌ System-Validierung fehlgeschlagen: $e")
        return false
    end
end

"""
Erstellt kompakte Generalisierungs-Zusammenfassung
"""
function create_generalization_summary(metrics_dict::Dict{Int, GeneralizationMetrics}, 
                                      train_losses, val_losses)
    println("\n" * "="^80)
    println("GENERALISIERUNGS-ZUSAMMENFASSUNG")
    println("="^80)
    
    # Training-Performance
    if length(train_losses) > 0 && length(val_losses) > 0
        println("TRAINING-PERFORMANCE:")
        println("  Finale Training Loss: $(round(train_losses[end], digits=6))")
        println("  Finale Validation Loss: $(round(val_losses[end], digits=6))")
        println("  Beste Validation Loss: $(round(minimum(val_losses), digits=6))")
        println()
    end
    
    # Generalisierung-Kernmetriken
    println("GENERALISIERUNG (MAE = Mean Absolute Error):")
    crystal_counts = sort(collect(keys(metrics_dict)))
    
    for crystal_count in crystal_counts
        metrics = metrics_dict[crystal_count]
        println("  $crystal_count Kristalle: MAE = $(round(metrics.mae_total, digits=6)), R² = $(round(metrics.r2_total, digits=3))")
    end
    
    # Generalisierungs-Analyse
    if length(crystal_counts) > 1
        mae_values = [metrics_dict[k].mae_total for k in crystal_counts]
        r2_values = [metrics_dict[k].r2_total for k in crystal_counts]
        
        println("\nGENERALISIERUNGS-ANALYSE:")
        println("  Trainingsziel: $(GENERALIZATION_CONFIG.target_crystal_count) Kristalle")
        
        target_idx = findfirst(x -> x == GENERALIZATION_CONFIG.target_crystal_count, crystal_counts)
        if target_idx !== nothing
            target_mae = mae_values[target_idx]
            println("  Performance auf Trainingsziel: MAE = $(round(target_mae, digits=6))")
            
            # Vergleich mit anderen Kristallanzahlen
            better_count = sum(mae_values .< target_mae)
            worse_count = sum(mae_values .> target_mae)
            
            println("  Bessere Performance bei $better_count anderen Kristallanzahlen")
            println("  Schlechtere Performance bei $worse_count anderen Kristallanzahlen")
        end
        
        # Beste und schlechteste Performance
        best_mae_idx = argmin(mae_values)
        worst_mae_idx = argmax(mae_values)
        
        println("  Beste Performance: $(crystal_counts[best_mae_idx]) Kristalle (MAE = $(round(mae_values[best_mae_idx], digits=6)))")
        println("  Schlechteste Performance: $(crystal_counts[worst_mae_idx]) Kristalle (MAE = $(round(mae_values[worst_mae_idx], digits=6)))")
        
        # Performance-Gradient
        if GENERALIZATION_CONFIG.target_crystal_count in crystal_counts
            single_crystal_mae = metrics_dict[1].mae_total
            target_crystal_mae = metrics_dict[GENERALIZATION_CONFIG.target_crystal_count].mae_total
            degradation = ((target_crystal_mae - single_crystal_mae) / single_crystal_mae) * 100
            
            println("  Performance-Degradation (1 → $(GENERALIZATION_CONFIG.target_crystal_count) Kristalle): $(round(degradation, digits=1))%")
        end
    end
    
    println("="^80)
end

"""
Demo für kleineres Generalisierungs-Experiment
"""
function demo_generalization_study(; n_samples=20, max_crystals=5, num_epochs=5)
    println("=== GENERALISIERUNGS-DEMO ===")
    
    # Reduzierte Konfiguration
    demo_config = merge(GENERALIZATION_CONFIG, (
        n_training_samples = n_samples,
        num_epochs = num_epochs,
        eval_crystal_range = 1:max_crystals,
        n_eval_samples_per_count = 5,
        checkpoint_dir = "demo_generalization_checkpoints",
        results_dir = "demo_generalization_results"
    ))
    
    # Temporär überschreiben
    global GENERALIZATION_CONFIG = demo_config
    
    success, metrics = run_generalization_study()
    
    if success && metrics !== nothing
        println("\n✓ Generalisierungs-Demo erfolgreich!")
        create_generalization_report(metrics)
    else
        println("\n❌ Generalisierungs-Demo fehlgeschlagen!")
    end
    
    return success, metrics
end

# =============================================================================
# HAUPTAUSFÜHRUNG
# =============================================================================

println("="^80)
println("GENERALISIERUNGSSTUDIE INITIALISIERUNG")
println("="^80)

# Führe vollständige Generalisierungsstudie aus
success, generalization_results = run_generalization_study()

if success
    println("\n" * "="^80)
    println(" GENERALISIERUNGSSTUDIE ERFOLGREICH ABGESCHLOSSEN!")
    println("="^80)
    exit(0)
else
    println("\n" * "="^80)
    println(" GENERALISIERUNGSSTUDIE FEHLGESCHLAGEN!")
    println("="^80)
    exit(1)
end
println("Zeit: $(Dates.now())")
println("Lade Module...")

# Module laden
include("lamem_interface.jl")        # Erweiterte LaMEM-Integration
include("data_processing.jl")                 # Datenverarbeitung  
include("unet_config.jl")                     # UNet-Konfiguration
include("unet_architecture.jl")               # Sichere UNet-Architektur
include("training.jl")                        # Sicheres Training
include("batch_management.jl")                # Batch-Management
include("generalization_evaluation.jl")      # Generalisierungs-Evaluierung

println("Alle Module erfolgreich geladen!")

# =============================================================================
# GENERALISIERUNGS-KONFIGURATION
# =============================================================================

const GENERALIZATION_CONFIG = (
    # Training-Parameter
    target_crystal_count = 10,                   # Hauptsächlich auf 10 Kristalle trainieren
    n_training_samples = 200,                    # Größeres Training-Dataset
    crystal_distribution = "weighted",           # 70% 10-Kristall, 30% andere
    target_resolution = 256,                     # Auflösung
    
    # Training-Hyperparameter
    num_epochs = 30,                            # Mehr Epochen für komplexere Aufgabe
    learning_rate = 0.0005f0,                   # Etwas niedrigere Lernrate
    batch_size = 4,                             # Mittlere Batch-Größe
    early_stopping_patience = 10,
    
    # Evaluierung-Parameter
    eval_crystal_range = 1:10,                  # Evaluiere auf allen Kristallanzahlen
    n_eval_samples_per_count = 25,              # Samples pro Kristallanzahl für Evaluierung
    
    # Output-Parameter
    checkpoint_dir = "generalization_checkpoints",
    results_dir = "generalization_results",
    model_name = "10crystal_generalization_model",
    
    # Hardware
    use_gpu = false,                            # CPU für Stabilität
    validation_split = 0.15f0,                  # Mehr Validierungsdaten
)