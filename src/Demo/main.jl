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
    target_crystal_count = 10,          # Hauptziel: 10 Kristalle
    n_training_samples = 20,            # Mehr Samples für komplexere Daten
    target_resolution = 256,            # Höhere Auflösung für 10 Kristalle
    
    # Training-Parameter  
    num_epochs = 30,                    # Mehr Epochen für komplexere Aufgabe
    learning_rate = 0.0005f0,           # Kleinere Lernrate für Stabilität
    batch_size = 1,                     # Kleine Batches für 10-Kristall Komplexität
    early_stopping_patience = 8,        # Mehr Patience
    
    # Evaluierung
    eval_crystal_range = 8:10,          # Teste 8, 9, 10 Kristalle
    n_eval_samples_per_count = 3,       # Mehr Evaluierungssamples
    
    # Output-Parameter
    checkpoint_dir = "ten_crystal_checkpoints",
    results_dir = "ten_crystal_results",
    
    # Hardware
    use_gpu = false,                    # CPU für Stabilität
    save_dataset = true,
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
# EVALUIERUNG AUF VERSCHIEDENE KRISTALLANZAHLEN
# =============================================================================

"""
Evaluiert trainiertes Modell auf verschiedene Kristallanzahlen
"""
function evaluate_generalization(model, crystal_counts; n_samples_per_count=3, resolution=256)
    println("\n=== GENERALISIERUNGS-EVALUIERUNG ===")
    println("Teste Kristallanzahlen: $crystal_counts")
    println("Samples pro Anzahl: $n_samples_per_count")
    
    results = Dict()
    
    for crystal_count in crystal_counts
        println("\n--- Teste $crystal_count Kristalle ---")
        
        mae_values = []
        r2_values = []
        
        for sample_idx in 1:n_samples_per_count
            try
                # Generiere Testsample
                test_sample = if crystal_count == 10
                    # Verwende 10-Kristall Generator
                    samples = generate_ten_crystal_dataset(1, resolution=resolution, verbose=false)
                    length(samples) > 0 ? samples[1] : nothing
                else
                    # Verwende Standard-Generator für andere Anzahlen
                    LaMEM_Multi_crystal(
                        resolution=(resolution, resolution),
                        n_crystals=crystal_count,
                        radius_crystal=fill(0.05, crystal_count),
                        cen_2D=[(rand(-0.5:0.1:0.5), rand(0.2:0.1:0.8)) for _ in 1:crystal_count]
                    )
                end
                
                if test_sample === nothing
                    continue
                end
                
                x, z, phase, vx, vz, exx, ezz, v_stokes = test_sample
                
                # Preprocessing
                phase_tensor, velocity_tensor = preprocess_lamem_sample(
                    x, z, phase, vx, vz, v_stokes,
                    target_resolution=resolution
                )
                
                # Vorhersage
                prediction = cpu(model(phase_tensor))
                
                # Metriken berechnen
                mae_vx = mean(abs.(prediction[:,:,1,1] .- velocity_tensor[:,:,1,1]))
                mae_vz = mean(abs.(prediction[:,:,2,1] .- velocity_tensor[:,:,2,1]))
                mae_total = (mae_vx + mae_vz) / 2
                
                # R² berechnen
                gt_flat = vcat(vec(velocity_tensor[:,:,1,1]), vec(velocity_tensor[:,:,2,1]))
                pred_flat = vcat(vec(prediction[:,:,1,1]), vec(prediction[:,:,2,1]))
                
                ss_res = sum((gt_flat .- pred_flat).^2)
                ss_tot = sum((gt_flat .- mean(gt_flat)).^2)
                r2 = 1 - ss_res / ss_tot
                
                push!(mae_values, mae_total)
                push!(r2_values, r2)
                
                println("  Sample $sample_idx: MAE = $(round(mae_total, digits=6)), R² = $(round(r2, digits=3))")
                
            catch e
                println("  Sample $sample_idx fehlgeschlagen: $e")
                continue
            end
        end
        
        if length(mae_values) > 0
            avg_mae = mean(mae_values)
            avg_r2 = mean(r2_values)
            
            results[crystal_count] = (
                mae = avg_mae,
                r2 = avg_r2,
                n_samples = length(mae_values)
            )
            
            println("  Durchschnitt: MAE = $(round(avg_mae, digits=6)), R² = $(round(avg_r2, digits=3))")
        else
            println("  Keine erfolgreichen Samples für $crystal_count Kristalle")
            results[crystal_count] = (mae = NaN, r2 = NaN, n_samples = 0)
        end
    end
    
    return results
end

# =============================================================================
# HAUPTFUNKTION
# =============================================================================

"""
Vollständiges 10-Kristall Training und Generalisierungstest
"""
function run_ten_crystal_study()
    println("="^80)
    println("STARTE 10-KRISTALL UNET STUDIE")
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
        
        dataset = generate_ten_crystal_dataset(
            SERVER_CONFIG.n_training_samples,
            resolution=SERVER_CONFIG.target_resolution,
            verbose=true
        )
        
        if length(dataset) == 0
            error("Keine Trainingsdaten generiert!")
        end
        
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
            validation_split = 0.2f0,
            early_stopping_patience = SERVER_CONFIG.early_stopping_patience
        )
        
        println("Starte Training auf 10-Kristall Daten...")
        
        trained_model, train_losses, val_losses = train_velocity_unet_safe(
            model, dataset, SERVER_CONFIG.target_resolution,
            config=train_config
        )
        
        println("✓ 10-Kristall Training abgeschlossen")
        
        # 5. GENERALISIERUNGS-EVALUIERUNG
        println("\n5. GENERALISIERUNGS-EVALUIERUNG")
        println("-"^50)
        
        generalization_results = evaluate_generalization(
            trained_model,
            collect(SERVER_CONFIG.eval_crystal_range),
            n_samples_per_count=SERVER_CONFIG.n_eval_samples_per_count,
            resolution=SERVER_CONFIG.target_resolution
        )
        
        # 6. ERGEBNISSE ZUSAMMENFASSUNG
        println("\n6. ERGEBNISSE")
        println("-"^50)
        
        if length(train_losses) > 0
            println("Training - Finale Loss: $(round(train_losses[end], digits=6))")
        end
        
        if length(val_losses) > 0
            best_val_loss = minimum(val_losses)
            println("Training - Beste Validation Loss: $(round(best_val_loss, digits=6))")
        end
        
        println("\nGeneralisierung:")
        for crystal_count in sort(collect(keys(generalization_results)))
            result = generalization_results[crystal_count]
            if result.n_samples > 0
                println("  $crystal_count Kristalle: MAE = $(round(result.mae, digits=6)), R² = $(round(result.r2, digits=3))")
            else
                println("  $crystal_count Kristalle: Keine erfolgreichen Tests")
            end
        end
        
        # 7. SPEICHERN
        println("\n7. ERGEBNISSE SPEICHERN")
        println("-"^50)
        
        results_data = Dict(
            "trained_model" => trained_model,
            "train_losses" => train_losses,
            "val_losses" => val_losses,
            "generalization_results" => generalization_results,
            "config" => SERVER_CONFIG,
            "dataset_size" => length(dataset)
        )
        
        results_path = joinpath(SERVER_CONFIG.results_dir, "ten_crystal_study_results.bson")
        @save results_path results_data
        
        println("✓ Alle Ergebnisse gespeichert: $results_path")
        
        # Erfolgreicher Abschluss
        end_time = time()
        total_time = end_time - start_time
        
        println("\n" * "="^80)
        println("10-KRISTALL STUDIE ERFOLGREICH ABGESCHLOSSEN")
        println("="^80)
        println("Gesamtzeit: $(round(total_time/60, digits=2)) Minuten")
        println("Checkpoints: $(SERVER_CONFIG.checkpoint_dir)")
        println("Ergebnisse: $(SERVER_CONFIG.results_dir)")
        
        return true
        
    catch e
        end_time = time()
        total_time = end_time - start_time
        
        println("\n" * "="^80)
        println("10-KRISTALL STUDIE FEHLGESCHLAGEN")
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

# Führe 10-Kristall Studie aus
success = run_ten_crystal_study()

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