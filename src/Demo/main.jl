# =============================================================================
# HAUPTSCRIPT - 10-KRISTALL UNET TRAINING (EMPFOHLENER ANSATZ)
# =============================================================================
# Speichern als: main.jl

using Flux
using CUDA
using Statistics
using Random
import Dates
using Serialization
using BSON: @save

println("=== 10-KRISTALL UNET TRAINING ===")
println("Zeit: $(Dates.now())")
println("Lade Module...")

# Module laden
include("lamem_interface.jl")
include("data_processing.jl") 
include("unet_config.jl")
include("unet_architecture.jl")
include("training.jl")
include("batch_management.jl")

println("Alle Module erfolgreich geladen!")

# =============================================================================
# 10-KRISTALL KONFIGURATION
# =============================================================================

const SERVER_CONFIG = (
    # 10-Kristall spezifische Parameter
    target_crystal_count = 10,
    
    # OPTIMIERT: Deutlich mehr Trainingsdaten
    n_training_samples = 500,           # Von 200 auf 500 erhöht
    
    target_resolution = 256,
    
    # OPTIMIERT: Bessere Training-Parameter
    num_epochs = 50,                    # Von 30 auf 50 erhöht
    learning_rate = 0.001f0,            # Von 0.0005f0 auf 0.001f0 erhöht
    batch_size = 2,                     # Von 1 auf 2 erhöht
    early_stopping_patience = 15,       # Von 10 auf 15 erhöht
    
    # Physics-Informed Parameter
    lambda_physics_initial = 0.01f0,    
    lambda_physics_final = 0.1f0,       
    physics_warmup_epochs = 15,         # 15 Epochen Warm-up für Physics
    
    # Output-Parameter
    checkpoint_dir = "ten_crystal_checkpoints_optimized",
    results_dir = "ten_crystal_results_optimized",
    
    # Hardware
    use_gpu = false,
    save_dataset = true,
    
    # NEUE PARAMETER für besseres Training
    use_data_augmentation = true,       # Datenaugmentierung
    validation_split = 0.15f0,          # 15% Validation statt 20%
)

# =============================================================================
# 10-KRISTALL DATENGENERIERUNG (NEUE FUNKTION)
# =============================================================================

"""
Spezialisierte Datengenerierung für 10-Kristall Training
"""
function generate_ten_crystal_dataset(n_samples; resolution=256, verbose=true)
    if verbose
        println("=== 10-KRISTALL DATASET GENERATOR ===")
        println("Generiere $n_samples Samples mit 10 Kristallen")
        println("Auflösung: $(resolution)x$(resolution)")
    end
    
    dataset = []
    successful_samples = 0
    
    for i in 1:n_samples
        if verbose && i % 5 == 1
            println("Generiere Sample $i/$n_samples...")
        end
        
        # Fixe 10 Kristalle
        n_crystals = 10
        
        # Kleinere Radien für bessere Platzierung von 10 Kristallen
        radius_crystal = [rand(0.025:0.003:0.055) for _ in 1:n_crystals]
        
        # Intelligente Positionierung für 10 Kristalle
        centers = generate_ten_crystal_positions(n_crystals, radius_crystal)
        
        try
            sample = LaMEM_Multi_crystal(
                resolution=(resolution, resolution),
                n_crystals=n_crystals,
                radius_crystal=radius_crystal,
                cen_2D=centers,
                η_magma=10^(rand() * 2 + 19),  # 1e19 bis 1e21
                Δρ=rand(150:50:300)           # Variable Dichtedifferenz
            )
            
            push!(dataset, sample)
            successful_samples += 1
            
        catch e
            if verbose
                println("  Warnung: Sample $i fehlgeschlagen: $e")
            end
            
            # Fallback: Vereinfachtes 10-Kristall System
            try
                fallback_centers = generate_simple_ten_crystal_grid()
                
                fallback_sample = LaMEM_Multi_crystal(
                    resolution=(resolution, resolution),
                    n_crystals=10,
                    radius_crystal=fill(0.04, 10),  # Einheitliche kleine Radien
                    cen_2D=fallback_centers
                )
                
                push!(dataset, fallback_sample)
                successful_samples += 1
                
                if verbose
                    println("  Fallback erfolgreich für Sample $i")
                end
                
            catch e2
                if verbose
                    println("  Auch Fallback fehlgeschlagen: $e2")
                end
                continue
            end
        end
        
        # Memory cleanup alle 5 Samples
        if i % 5 == 0
            GC.gc()
        end
    end
    
    if verbose
        success_rate = round(100 * successful_samples / n_samples, digits=1)
        println("\nDataset-Generierung abgeschlossen:")
        println("  Erfolgreich: $successful_samples/$n_samples ($success_rate%)")
        println("  Alle Samples haben 10 Kristalle")
    end
    
    return dataset
end

"""
Datenaugmentierung für bessere Generalisierung
"""
function augment_dataset(dataset; augmentation_factor=2)
    println("=== DATENAUGMENTIERUNG ===")
    println("Original Dataset: $(length(dataset)) Samples")
    
    augmented_dataset = copy(dataset)
    
    for sample in dataset
        for aug in 1:augmentation_factor
            x, z, phase, vx, vz, exx, ezz, v_stokes = sample
            
            # Augmentierungs-Strategien
            if aug == 1
                # Horizontale Spiegelung
                phase_aug = reverse(phase, dims=1)
                vx_aug = -reverse(vx, dims=1)  # Vorzeichen umkehren
                vz_aug = reverse(vz, dims=1)
                augmented_sample = (x, z, phase_aug, vx_aug, vz_aug, exx, ezz, v_stokes)
                
            elseif aug == 2
                # Kleine Rotation (nur für robusteres Training)
                # Vereinfachte Version: Leichte Verschiebung
                shift_amount = rand(-5:5)
                phase_aug = circshift(phase, (shift_amount, 0))
                vx_aug = circshift(vx, (shift_amount, 0))
                vz_aug = circshift(vz, (shift_amount, 0))
                augmented_sample = (x, z, phase_aug, vx_aug, vz_aug, exx, ezz, v_stokes)
            end
            
            push!(augmented_dataset, augmented_sample)
        end
    end
    
    println("Augmentiertes Dataset: $(length(augmented_dataset)) Samples")
    println("Augmentierungsfaktor: $(length(augmented_dataset) / length(dataset))x")
    
    return augmented_dataset
end

"""
Intelligente Positionierung für 10 Kristalle mit Kollisionsvermeidung
"""
function generate_ten_crystal_positions(n_crystals, radius_crystal)
    centers = []
    max_attempts_per_crystal = 100
    min_distance = 0.1  # Mindestabstand zwischen Kristallen
    
    for i in 1:n_crystals
        placed = false
        attempts = 0
        
        while !placed && attempts < max_attempts_per_crystal
            # Zufällige Position in erweiterten Grenzen
            x_pos = rand(-0.85:0.05:0.85)
            z_pos = rand(0.05:0.05:0.95)
            new_center = (x_pos, z_pos)
            
            # Prüfe Kollision mit existierenden Kristallen
            collision = false
            current_radius = radius_crystal[i]
            
            for (j, existing_center) in enumerate(centers)
                existing_radius = radius_crystal[j]
                distance = sqrt((new_center[1] - existing_center[1])^2 + 
                               (new_center[2] - existing_center[2])^2)
                
                # Mindestabstand basierend auf Radien plus Puffer
                required_distance = current_radius + existing_radius + min_distance
                
                if distance < required_distance
                    collision = true
                    break
                end
            end
            
            if !collision
                push!(centers, new_center)
                placed = true
            end
            
            attempts += 1
        end
        
        # Fallback falls keine Position gefunden
        if !placed
            # Verwende Grid-basierte Position als Fallback
            grid_x = -0.6 + (i-1) % 5 * 0.3
            grid_z = 0.2 + div(i-1, 5) * 0.3
            push!(centers, (grid_x, grid_z))
        end
    end
    
    return centers
end

"""
Einfaches Grid-Layout als Fallback für 10 Kristalle
"""
function generate_simple_ten_crystal_grid()
    # 5x2 Grid Layout
    centers = []
    
    for row in 1:2
        for col in 1:5
            x_pos = -0.6 + (col-1) * 0.3
            z_pos = 0.25 + (row-1) * 0.5
            push!(centers, (x_pos, z_pos))
        end
    end
    
    return centers
end

# =============================================================================
# HAUPTFUNKTION
# =============================================================================

"""
Vollständiges 10-Kristall Training (ohne Evaluierung)
"""
function run_ten_crystal_training()
    println("="^80)
    println("STARTE 10-KRISTALL UNET TRAINING")
    println("="^80)
    
    start_time = time()
    
    # Erstelle Output-Verzeichnisse
    mkpath(SERVER_CONFIG.checkpoint_dir)
    mkpath(SERVER_CONFIG.results_dir)
    
    try
        # 1. SYSTEM-CHECK
        println("\n1. SYSTEM-CHECK")
        println("-"^50)
        
        if !quick_test_safe()
            error("System-Test fehlgeschlagen!")
        end
        println("✓ System bereit für 10-Kristall Training")
        
       # 2. 10-KRISTALL DATENGENERIERUNG
        println("\n2. 10-KRISTALL DATENGENERIERUNG")
        println("-"^50)

        # Basis-Dataset generieren
        dataset = generate_ten_crystal_dataset(
            SERVER_CONFIG.n_training_samples,
            resolution=SERVER_CONFIG.target_resolution,
            verbose=true
        )

        if length(dataset) == 0
            error("Keine Trainingsdaten generiert!")
        end

        # Datenaugmentierung anwenden
        if SERVER_CONFIG.use_data_augmentation
            println("\nWende Datenaugmentierung an...")
            dataset = augment_dataset(dataset, augmentation_factor=1)  # Verdoppelt Dataset
        end

        println("Finales Dataset: $(length(dataset)) Samples")

        # Dataset mischen für bessere Trainingsverteilung
        dataset = dataset[randperm(length(dataset))]
        println("Dataset gemischt")
        
        println("✓ 10-Kristall Dataset erstellt: $(length(dataset)) Samples")
        
        # Dataset speichern
        if SERVER_CONFIG.save_dataset
            dataset_path = joinpath(SERVER_CONFIG.results_dir, "ten_crystal_dataset.jls")
            serialize(dataset_path, dataset)
            println("✓ Dataset gespeichert: $dataset_path")
        end
        
        # 3. MODELL-ERSTELLUNG
        println("\n3. UNET-MODELL ERSTELLUNG")
        println("-"^50)
        
        model = create_simplified_unet(1, 2, 32)
        
        success, _ = test_simplified_unet(SERVER_CONFIG.target_resolution, batch_size=1)
        if !success
            error("UNet-Test fehlgeschlagen!")
        end
        println("✓ UNet für 10-Kristall Training bereit")
        
        # 4. TRAINING
        println("\n4. 10-KRISTALL TRAINING")
        println("-"^50)
        
        train_config = create_training_config(
            learning_rate = SERVER_CONFIG.learning_rate,
            num_epochs = SERVER_CONFIG.num_epochs,
            batch_size = SERVER_CONFIG.batch_size,
            checkpoint_dir = SERVER_CONFIG.checkpoint_dir,
            save_every_n_epochs = 5,
            use_gpu = SERVER_CONFIG.use_gpu,
            validation_split = SERVER_CONFIG.validation_split,
            early_stopping_patience = SERVER_CONFIG.early_stopping_patience,
            # Physics-Informed Parameter
            lambda_physics_initial = SERVER_CONFIG.lambda_physics_initial,
            lambda_physics_final = SERVER_CONFIG.lambda_physics_final,
            physics_warmup_epochs = SERVER_CONFIG.physics_warmup_epochs
            )
        
        println("Starte Training auf 10-Kristall Daten...")
        
        trained_model, train_losses, val_losses, physics_losses = train_velocity_unet_safe(
            model, dataset, SERVER_CONFIG.target_resolution,
            config=train_config
        )   
        
        println("✓ 10-Kristall Training abgeschlossen")
        
        # 5. TRAINING-ERGEBNISSE ANALYSE
        println("\n5. TRAINING-ERGEBNISSE ANALYSE")
        println("-"^50)

        if length(train_losses) > 0 && length(val_losses) > 0
            # Convergence-Analyse
            initial_loss = train_losses[1]
            final_loss = train_losses[end]
            improvement = (initial_loss - final_loss) / initial_loss * 100
            
            println("Training Convergence:")
            println("  Initial Loss: $(round(initial_loss, digits=6))")
            println("  Final Loss: $(round(final_loss, digits=6))")
            println("  Verbesserung: $(round(improvement, digits=1))%")
            
            # Overfitting-Check
            train_val_gap = abs(train_losses[end] - val_losses[end])
            println("\nOverfitting-Analyse:")
            println("  Train-Val Gap: $(round(train_val_gap, digits=6))")
            println("  Status: $(train_val_gap < 0.01 ? "Kein Overfitting" : "Mögliches Overfitting")")

            
            if @isdefined(physics_losses) && length(physics_losses) > 0
                physics_improvement = (physics_losses[1] - physics_losses[end]) / physics_losses[1] * 100
                println("\nPhysics Constraint:")
                println("  Initial Physics Loss: $(round(physics_losses[1], digits=6))")
                println("  Final Physics Loss: $(round(physics_losses[end], digits=6))")
                println("  Verbesserung: $(round(physics_improvement, digits=1))%")
                
                # Physics Loss sollte gegen 0 konvergieren
                if physics_losses[end] < 0.001
                    println("  Status: Exzellente physikalische Konsistenz")
                elseif physics_losses[end] < 0.01
                    println("  Status: Gute physikalische Konsistenz")
                else
                    println("  Status: Weitere Optimierung erforderlich")
                end
            end
            
            # Beste Performance
            best_val_loss = minimum(val_losses)
            best_epoch = argmin(val_losses)
            println("\nBeste Performance:")
            println("  Beste Validation Loss: $(round(best_val_loss, digits=6)) (Epoche $best_epoch)")
            println("  Finale Validation Loss: $(round(val_losses[end], digits=6))")
        end

        # Zeige Training-Verlauf
        if length(train_losses) > 5
            println("\nTraining-Verlauf (letzte 5 Epochen):")
            for i in (length(train_losses)-4):length(train_losses)
                println("  Epoche $i: Train = $(round(train_losses[i], digits=6)), Val = $(round(val_losses[i], digits=6))")
            end
        end
        
        # 6. ERGEBNISSE SPEICHERN
        println("\n6. ERGEBNISSE SPEICHERN")
        println("-"^50)
        
        results_data = Dict(
            "trained_model" => trained_model,
            "train_losses" => train_losses,
            "val_losses" => val_losses,
            "physics_losses" => @isdefined(physics_losses) ? physics_losses : Float32[],  # NEU
            "config" => SERVER_CONFIG,
            "dataset_size" => length(dataset),
            "training_completed" => true
        )
        
        results_path = joinpath(SERVER_CONFIG.results_dir, "ten_crystal_training_results.bson")
        @save results_path results_data
        
        println("✓ Trainingsergebnisse gespeichert: $results_path")
        
        # Erfolgreicher Abschluss
        end_time = time()
        total_time = end_time - start_time
        
        println("\n" * "="^80)
        println("10-KRISTALL TRAINING ERFOLGREICH ABGESCHLOSSEN")
        println("="^80)
        println("Gesamtzeit: $(round(total_time/60, digits=2)) Minuten")
        println("Trainiertes Modell: $(SERVER_CONFIG.checkpoint_dir)/best_model.bson")
        println("Ergebnisse: $results_path")
        
        return true
        
    catch e
        end_time = time()
        total_time = end_time - start_time
        
        println("\n" * "="^80)
        println("10-KRISTALL TRAINING FEHLGESCHLAGEN")
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

"""
Sicherer System-Test
"""
function quick_test_safe()
    try
        # Test mit 2 Kristallen (weniger komplex als 10)
        x, z, phase, vx, vz, exx, ezz, v_stokes = LaMEM_Multi_crystal(
            resolution=(64, 64),
            n_crystals=2,
            radius_crystal=[0.05, 0.05],
            cen_2D=[(0.0, 0.3), (0.0, 0.7)]
        )
        
        phase_tensor, velocity_tensor = preprocess_lamem_sample(
            x, z, phase, vx, vz, v_stokes, target_resolution=64
        )
        
        model = create_simplified_unet()
        test_input = randn(Float32, 64, 64, 1, 1)
        output = model(test_input)
        
        return size(output) == (64, 64, 2, 1)
        
    catch e
        println("System-Test Fehler: $e")
        return false
    end
end

# =============================================================================
# PROGRAMMSTART
# =============================================================================

println("="^80)
println("10-KRISTALL UNET TRAINING INITIALISIERUNG")
println("="^80)

# Konfiguration anzeigen
println("KONFIGURATION:")
for (key, value) in pairs(SERVER_CONFIG)
    println("  $key: $value")
end

# Führe 10-Kristall Training aus
success = run_ten_crystal_training()

if success
    println("\n" * "="^80)
    println("10-KRISTALL UNET TRAINING ERFOLGREICH!")
    println("="^80)
    exit(0)
else
    println("\n" * "="^80)
    println("10-KRISTALL UNET TRAINING FEHLGESCHLAGEN!")
    println("="^80)
    exit(1)
end