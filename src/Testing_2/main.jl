# =============================================================================
# HAUPTSCRIPT - 3D UNET TRAINING (ANGEPASST VON DEINEM 10-KRISTALL CODE)
# =============================================================================

using Flux
using CUDA
using Statistics
using Random
import Dates
using Serialization
using BSON
using BSON: @save, @load


println("=== 3D UNET TRAINING ===")
println("Zeit: $(Dates.now())")
println("Lade Module...")

# Module laden (jetzt 3D Versionen)
include("lamem_interface.jl")       # 3D Version
include("data_processing.jl")       # 3D Version  
include("unet_config.jl")           # Falls vorhanden
include("unet_architecture.jl")     # 3D Version
include("training.jl")              # 3D Version
include("batch_management.jl")      # Original (funktioniert mit 3D)
include("gpu_utils.jl")             # Original

println("Alle 3D Module erfolgreich geladen!")

# =============================================================================
# 3D KONFIGURATION - ANGEPASST VON DEINER 2D VERSION
# =============================================================================

const SERVER_CONFIG_3D = (
    # 3D spezifische Parameter
    target_crystal_count = 2,           # REDUZIERT für 3D: 2 statt 10
    
    # OPTIMIERT für 3D: Weniger Samples wegen längerer Rechenzeit
    n_training_samples = 20,            # REDUZIERT: 20 statt 750
    
    target_resolution = (64, 64, 64),   # 3D AUFLÖSUNG: (64,64,64) statt 64
    
    # OPTIMIERT: 3D Training-Parameter
    num_epochs = 15,                    # REDUZIERT: 15 statt 50
    learning_rate = 0.0005f0,           # Gleich (langsame Lernrate für 3D)
    batch_size = 1,                     # REDUZIERT: 1 statt 4 (wegen 3D Speicher!)
    early_stopping_patience = 8,        # REDUZIERT: 8 statt 15
    
    # Physics-Informed Parameter (angepasst für 3D)
    lambda_physics_initial = 0.01f0,    
    lambda_physics_final = 0.05f0,      # REDUZIERT: 0.05 statt 0.15
    physics_warmup_epochs = 5,          # REDUZIERT: 5 statt 15
    
    # Output-Parameter
    checkpoint_dir = "3d_crystal_checkpoints",
    results_dir = "3d_crystal_results",
    
    # Hardware (konservativ für 3D)
    use_gpu = false,                    # CPU für ersten 3D Test
    save_dataset = true,
    
    # 3D angepasste Parameter
    use_data_augmentation = false,      # Erstmal deaktiviert für 3D
    validation_split = 0.2f0,           # Mehr Validation für kleineres Dataset
    memory_efficient_mode = true,       # NEU für 3D
)

# =============================================================================
# 3D DATENGENERIERUNG - ANGEPASST VON DEINER 10-KRISTALL FUNKTION
# =============================================================================

"""
3D Version deiner generate_ten_crystal_dataset Funktion
"""
function generate_3d_crystal_dataset(n_samples; resolution=(64,64,64), verbose=true)
    if verbose
        println("=== 3D KRISTALL DATASET GENERATOR ===")
        println("Generiere $n_samples 3D Samples mit $(SERVER_CONFIG_3D.target_crystal_count) Kristallen")
        println("3D Auflösung: $(resolution[1])×$(resolution[2])×$(resolution[3])")
    end
    
    dataset = []
    successful_samples = 0
    
    for i in 1:n_samples
        if verbose && i % 3 == 1  # Häufigere Updates für langsamere 3D
            println("Generiere 3D Sample $i/$n_samples...")
        end
        
        # Fixe Kristall-Anzahl für 3D
        n_crystals = SERVER_CONFIG_3D.target_crystal_count
        
        # Größere Radien für 3D Sichtbarkeit
        radius_crystal = [rand(0.08:0.01:0.12) for _ in 1:n_crystals]
        
        # 3D Kristall-Positionen generieren
        centers = generate_3d_crystal_positions(n_crystals)
        
        try
            sample = LaMEM_Multi_crystal(
                resolution=resolution,
                n_crystals=n_crystals,
                radius_crystal=radius_crystal,
                cen_3D=centers,                     # 3D Zentren statt cen_2D
                η_magma=10^(rand() * 2 + 19),      # 1e19 bis 1e21
                Δρ=rand(150:50:300)                # Variable Dichtedifferenz
            )
            
            push!(dataset, sample)
            successful_samples += 1
            
        catch e
            if verbose
                println("  Warnung: 3D Sample $i fehlgeschlagen: $e")
            end
            
            # 3D Fallback: Einfaches System
            try
                fallback_centers = [(0.0, 0.0, 0.3), (0.0, 0.0, 0.7)]  # 3D Positionen
                
                fallback_sample = LaMEM_Multi_crystal(
                    resolution=resolution,
                    n_crystals=2,                    # Minimale Kristall-Anzahl
                    radius_crystal=[0.08, 0.08],     # Größere Radien für 3D
                    cen_3D=fallback_centers          # 3D Zentren
                )
                
                push!(dataset, fallback_sample)
                successful_samples += 1
                
                if verbose
                    println("  3D Fallback erfolgreich für Sample $i")
                end
                
            catch e2
                if verbose
                    println("  Auch 3D Fallback fehlgeschlagen: $e2")
                end
                continue
            end
        end
        
        # Häufigere Memory cleanup für 3D
        if i % 2 == 0
            GC.gc()
        end
    end
    
    if verbose
        success_rate = round(100 * successful_samples / n_samples, digits=1)
        println("\n3D Dataset-Generierung abgeschlossen:")
        println("  Erfolgreich: $successful_samples/$n_samples ($success_rate%)")
        println("  Alle Samples haben $(SERVER_CONFIG_3D.target_crystal_count) Kristalle in 3D")
    end
    
    return dataset
end

# =============================================================================
# HAUPTFUNKTION - ANGEPASST FÜR 3D
# =============================================================================

"""
Vollständiges 3D Training (angepasst von deiner run_ten_crystal_training)
"""
function run_3d_crystal_training()
    println("="^80)
    println("STARTE 3D UNET TRAINING")
    println("="^80)
    
    start_time = time()
    
    # Erstelle Output-Verzeichnisse
    mkpath(SERVER_CONFIG_3D.checkpoint_dir)
    mkpath(SERVER_CONFIG_3D.results_dir)
    
    try
        # 1. SYSTEM-CHECK für 3D
        println("\n1. 3D SYSTEM-CHECK")
        println("-"^50)
        
        if !quick_test_3d_safe()
            error("3D System-Test fehlgeschlagen!")
        end
        println("✓ System bereit für 3D Training")
        
        # 2. 3D DATENGENERIERUNG
        println("\n2. 3D DATENGENERIERUNG")
        println("-"^50)

        # 3D Dataset generieren
        dataset = generate_3d_crystal_dataset(
            SERVER_CONFIG_3D.n_training_samples,
            resolution=SERVER_CONFIG_3D.target_resolution,
            verbose=true
        )

        if length(dataset) == 0
            error("Keine 3D Trainingsdaten generiert!")
        end

        # Datenaugmentierung für 3D erstmal deaktiviert
        if SERVER_CONFIG_3D.use_data_augmentation
            println("\nWende 3D Datenaugmentierung an...")
            # dataset = augment_3d_dataset(dataset)  # Würde noch implementiert werden
        end

        println("Finales 3D Dataset: $(length(dataset)) Samples")

        # Dataset mischen
        dataset = dataset[randperm(length(dataset))]
        println("3D Dataset gemischt")
        
        println("✓ 3D Dataset erstellt: $(length(dataset)) Samples")
        
        # Dataset speichern
        if SERVER_CONFIG_3D.save_dataset
            dataset_path = joinpath(SERVER_CONFIG_3D.results_dir, "3d_crystal_dataset.jls")
            serialize(dataset_path, dataset)
            println("✓ 3D Dataset gespeichert: $dataset_path")
        end
        
        # 3. 3D MODELL-ERSTELLUNG
        println("\n3. 3D UNET-MODELL ERSTELLUNG")
        println("-"^50)
        
        # 3D UNet erstellen
        model = create_simplified_unet_3d(1, 3, 32)  # 1 input, 3 outputs (vx,vy,vz), 32 filters
        println("✓ 3D UNet-Modell erstellt")

        # Test des 3D Modells
        success = try
            test_input = randn(Float32, 64, 64, 64, 1, 1)  # 3D Test Input
            output = model(test_input)
            size(output) == (64, 64, 64, 3, 1)  # 3D Output mit 3 Geschwindigkeits-Komponenten
        catch e
            println("3D UNet-Test Fehler: $e")
            false
        end

        if !success
            error("3D UNet-Test fehlgeschlagen!")
        end
        println("✓ 3D UNet für Training bereit")
        
        # 4. 3D TRAINING
        println("\n4. 3D TRAINING")
        println("-"^50)
        
        train_config = create_training_config_3d(  # 3D Konfiguration
            learning_rate = SERVER_CONFIG_3D.learning_rate,
            num_epochs = SERVER_CONFIG_3D.num_epochs,
            batch_size = SERVER_CONFIG_3D.batch_size,
            checkpoint_dir = SERVER_CONFIG_3D.checkpoint_dir,
            save_every_n_epochs = 3,
            use_gpu = SERVER_CONFIG_3D.use_gpu,
            validation_split = SERVER_CONFIG_3D.validation_split,
            early_stopping_patience = SERVER_CONFIG_3D.early_stopping_patience,
            # 3D Physics-Informed Parameter
            lambda_physics_initial = SERVER_CONFIG_3D.lambda_physics_initial,
            lambda_physics_final = SERVER_CONFIG_3D.lambda_physics_final,
            physics_warmup_epochs = SERVER_CONFIG_3D.physics_warmup_epochs,
            memory_efficient_mode = SERVER_CONFIG_3D.memory_efficient_mode
        )
        
        println("Starte 3D Training...")
        
        trained_model, train_losses, val_losses, physics_losses = train_velocity_unet_3d_safe(
            model, dataset, SERVER_CONFIG_3D.target_resolution,
            config=train_config
        )   
        
        println("✓ 3D Training abgeschlossen")
        
        # 5. 3D TRAINING-ERGEBNISSE ANALYSE
        println("\n5. 3D TRAINING-ERGEBNISSE ANALYSE")
        println("-"^50)

        if length(train_losses) > 0 && length(val_losses) > 0
            # Convergence-Analyse
            initial_loss = train_losses[1]
            final_loss = train_losses[end]
            improvement = (initial_loss - final_loss) / initial_loss * 100
            
            println("3D Training Convergence:")
            println("  Initial Loss: $(round(initial_loss, digits=6))")
            println("  Final Loss: $(round(final_loss, digits=6))")
            println("  Verbesserung: $(round(improvement, digits=1))%")
            
            # Overfitting-Check
            train_val_gap = abs(train_losses[end] - val_losses[end])
            println("\n3D Overfitting-Analyse:")
            println("  Train-Val Gap: $(round(train_val_gap, digits=6))")
            println("  Status: $(train_val_gap < 0.01 ? "Kein Overfitting" : "Mögliches Overfitting")")

            # 3D Physics-Informed Analyse
            if length(physics_losses) > 0
                physics_improvement = (physics_losses[1] - physics_losses[end]) / physics_losses[1] * 100
                println("\n3D Physics Constraint:")
                println("  Initial 3D Physics Loss: $(round(physics_losses[1], digits=6))")
                println("  Final 3D Physics Loss: $(round(physics_losses[end], digits=6))")
                println("  Verbesserung: $(round(physics_improvement, digits=1))%")
                
                # 3D Physics Loss Bewertung
                if physics_losses[end] < 0.001
                    println("  Status: Exzellente 3D physikalische Konsistenz")
                elseif physics_losses[end] < 0.01
                    println("  Status: Gute 3D physikalische Konsistenz")
                else
                    println("  Status: 3D Physik benötigt weitere Optimierung")
                end
            end
            
            # Beste Performance
            best_val_loss = minimum(val_losses)
            best_epoch = argmin(val_losses)
            println("\nBeste 3D Performance:")
            println("  Beste Validation Loss: $(round(best_val_loss, digits=6)) (Epoche $best_epoch)")
            println("  Finale Validation Loss: $(round(val_losses[end], digits=6))")
        end

        # 6. 3D ERGEBNISSE SPEICHERN
        println("\n6. 3D ERGEBNISSE SPEICHERN")
        println("-"^50)
        
        results_data = Dict(
            "trained_model" => trained_model,
            "train_losses" => train_losses,
            "val_losses" => val_losses,
            "physics_losses" => physics_losses,
            "config" => SERVER_CONFIG_3D,
            "dataset_size" => length(dataset),
            "is_3d" => true,                    # NEU: 3D Flag
            "resolution" => SERVER_CONFIG_3D.target_resolution,
            "training_completed" => true
        )
        
        results_path = joinpath(SERVER_CONFIG_3D.results_dir, "3d_crystal_training_results.bson")
        @save results_path results_data
        
        println("✓ 3D Trainingsergebnisse gespeichert: $results_path")
        
        # Erfolgreicher 3D Abschluss
        end_time = time()
        total_time = end_time - start_time
        
        println("\n" * "="^80)
        println("3D TRAINING ERFOLGREICH ABGESCHLOSSEN")
        println("="^80)
        println("Gesamtzeit: $(round(total_time/60, digits=2)) Minuten")
        println("3D Modell: $(SERVER_CONFIG_3D.checkpoint_dir)/best_3d_model.bson")
        println("Ergebnisse: $results_path")
        
        return true
        
    catch e
        end_time = time()
        total_time = end_time - start_time
        
        println("\n" * "="^80)
        println("3D TRAINING FEHLGESCHLAGEN")
        println("="^80)
        println("Fehler: $e")
        println("Laufzeit bis Fehler: $(round(total_time/60, digits=1)) Minuten")
        
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println()
        end
        
        return false
    end
end

# =============================================================================
# 3D UTILITY FUNCTIONS
# =============================================================================

"""
3D Version deiner quick_test_safe Funktion
"""
function quick_test_3d_safe()
    try
        # Test mit 2 Kristallen in 3D
        test_sample = LaMEM_Multi_crystal(
            resolution=(32, 32, 32),            # Kleine 3D Auflösung für Test
            n_crystals=2,
            radius_crystal=[0.08, 0.08],
            cen_3D=[(0.0, 0.0, 0.3), (0.0, 0.0, 0.7)]  # 3D Zentren
        )
        
        x, y, z, phase, vx, vy, vz = test_sample[1:7]  # 3D Daten entpacken
        v_stokes = test_sample[end]
        
        phase_tensor, velocity_tensor = preprocess_lamem_sample_3d(
            x, y, z, phase, vx, vy, vz, v_stokes, target_resolution=(32,32,32)
        )
        
        # 3D UNet testen
        model = create_simplified_unet_3d(1, 3, 16)  # Kleines Modell für Test
        
        test_input = randn(Float32, 32, 32, 32, 1, 1)
        output = model(test_input)
        
        return size(output) == (32, 32, 32, 3, 1)  # 3D Output prüfen
        
    catch e
        println("3D System-Test Fehler: $e")
        return false
    end
end

# =============================================================================
# PROGRAMMSTART - 3D VERSION
# =============================================================================

println("="^80)
println("3D UNET TRAINING INITIALISIERUNG")
println("="^80)

# 3D Konfiguration anzeigen
println("3D KONFIGURATION:")
for (key, value) in pairs(SERVER_CONFIG_3D)
    println("  $key: $value")
end

# Führe 3D Training aus
success = run_3d_crystal_training()

if success
    println("\n" * "="^80)
    println("3D UNET TRAINING ERFOLGREICH!")
    println("="^80)
    exit(0)
else
    println("\n" * "="^80)
    println("3D UNET TRAINING FEHLGESCHLAGEN!")
    println("="^80)
    exit(1)
end