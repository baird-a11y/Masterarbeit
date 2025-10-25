# =============================================================================
# HAUPTSCRIPT - 10-KRISTALL RESIDUAL UNET TRAINING
# =============================================================================

using Flux
using CUDA
using Statistics
using Random
import Dates
using Serialization
using BSON: @save

println("=== 10-KRISTALL RESIDUAL UNET TRAINING ===")
println("Zeit: $(Dates.now())")
println("Lade Module...")

# Module laden
include("lamem_interface.jl")
include("data_processing.jl") 
include("unet_config.jl")
include("unet_architecture.jl")
include("training.jl")
include("batch_management.jl")
include("gpu_utils.jl")
include("stokes_analytical.jl")

println("Alle Module erfolgreich geladen!")

# =============================================================================
# 10-KRISTALL KONFIGURATION
# =============================================================================

const SERVER_CONFIG = (
    target_crystal_count = 10,
    n_training_samples = 200,
    num_epochs = 50,
    target_resolution = 256,
    learning_rate = 0.0005f0,
    batch_size = 4,
    early_stopping_patience = 15,
    
    # Residual Learning Lambda-Werte
    lambda_residual = 0.01f0,
    lambda_sparsity = 0.001f0,
    lambda_physics = 0.1f0,
    
    # Physics-Informed Parameter
    lambda_physics_initial = 0.01f0,
    lambda_physics_final = 0.15f0,
    physics_warmup_epochs = 15,
    
    checkpoint_dir = "residual_checkpoints",
    results_dir = "residual_results",
    use_gpu = false,
    save_dataset = true,
    use_data_augmentation = true,
    validation_split = 0.15f0,
)

# =============================================================================
# 10-KRISTALL DATENGENERIERUNG
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
        
        # Kleinere Radien für bessere Platzierung
        radius_crystal = [rand(0.025:0.003:0.055) for _ in 1:n_crystals]
        
        # Intelligente Positionierung
        centers = generate_ten_crystal_positions(n_crystals, radius_crystal)
        
        try
            sample = LaMEM_Multi_crystal(
                resolution=(resolution, resolution),
                n_crystals=n_crystals,
                radius_crystal=radius_crystal,
                cen_2D=centers,
                η_magma=10^(rand() * 2 + 19),
                Δρ=rand(150:50:300)
            )
            
            push!(dataset, sample)
            successful_samples += 1
            
        catch e
            if verbose
                println("  Warnung: Sample $i fehlgeschlagen: $e")
            end
            
            # Fallback: Vereinfachtes System
            try
                fallback_centers = generate_simple_ten_crystal_grid()
                
                fallback_sample = LaMEM_Multi_crystal(
                    resolution=(resolution, resolution),
                    n_crystals=10,
                    radius_crystal=fill(0.04, 10),
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
        
        # Memory cleanup
        if i % 5 == 0
            GC.gc()
        end
    end
    
    if verbose
        success_rate = round(100 * successful_samples / n_samples, digits=1)
        println("\nDataset-Generierung abgeschlossen:")
        println("  Erfolgreich: $successful_samples/$n_samples ($success_rate%)")
    end
    
    return dataset
end

"""
Intelligente Positionierung für 10 Kristalle mit Kollisionsvermeidung
"""
function generate_ten_crystal_positions(n_crystals, radius_crystal)
    centers = []
    max_attempts_per_crystal = 100
    min_distance = 0.1
    
    for i in 1:n_crystals
        placed = false
        attempts = 0
        
        while !placed && attempts < max_attempts_per_crystal
            x_pos = rand(-0.85:0.05:0.85)
            z_pos = rand(0.05:0.05:0.95)
            new_center = (x_pos, z_pos)
            
            # Prüfe Kollision
            collision = false
            current_radius = radius_crystal[i]
            
            for (j, existing_center) in enumerate(centers)
                existing_radius = radius_crystal[j]
                distance = sqrt((new_center[1] - existing_center[1])^2 + 
                               (new_center[2] - existing_center[2])^2)
                
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
        
        # Fallback: Grid-basierte Position
        if !placed
            grid_x = -0.6 + (i-1) % 5 * 0.3
            grid_z = 0.2 + div(i-1, 5) * 0.3
            push!(centers, (grid_x, grid_z))
        end
    end
    
    return centers
end

"""
Einfaches Grid-Layout als Fallback
"""
function generate_simple_ten_crystal_grid()
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
# RESIDUAL UNET TRAINING PIPELINE
# =============================================================================

"""
Kompletter Residual UNet Workflow für 10-Kristall System
"""
function run_residual_10_crystal_training(; config_override=nothing)
    println("\n" * "="^80)
    println("RESIDUAL UNET 10-KRISTALL TRAINING PIPELINE")
    println("="^80)
    
    start_time = time()
    
    # Erstelle Output-Verzeichnisse
    mkpath(SERVER_CONFIG.checkpoint_dir)
    mkpath(SERVER_CONFIG.results_dir)
    
    try
        # =============================================================================
        # 1. SYSTEM-CHECK
        # =============================================================================
        println("\n1. SYSTEM-CHECK")
        println("-"^50)
        
        if !quick_test_safe()
            error("System-Test fehlgeschlagen!")
        end
        println("✓ System bereit für Residual Training")
        
        # =============================================================================
        # 2. DATENGENERIERUNG
        # =============================================================================
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
        
        # Augmentierung
        if SERVER_CONFIG.use_data_augmentation
            println("\nWende Datenaugmentierung an...")
            dataset = augment_lamem_dataset(dataset)
        end
        
        dataset = dataset[randperm(length(dataset))]
        println("✓ Dataset: $(length(dataset)) Samples")
        
        # Dataset speichern
        if SERVER_CONFIG.save_dataset
            dataset_path = joinpath(SERVER_CONFIG.results_dir, "residual_dataset.jls")
            serialize(dataset_path, dataset)
            println("✓ Dataset gespeichert: $dataset_path")
        end
        
        # =============================================================================
        # 3. RESIDUAL UNET ERSTELLEN
        # =============================================================================
        println("\n3. RESIDUAL UNET ERSTELLUNG")
        println("-"^50)
        
        model = create_residual_unet(
            input_channels=1,
            output_channels=2,
            base_filters=32,
            η_matrix=1e20,
            Δρ=200.0,
            g=9.81
        )
        
        println("✓ ResidualUNet erstellt (direktes Residuum-Lernen)")
        println("  Base UNet: SimplifiedUNetBN")
        println("  Ansatz: v_total = v_stokes + Δv")
        
        # Test des Modells
        success = try
            test_input = randn(Float32, SERVER_CONFIG.target_resolution, 
                              SERVER_CONFIG.target_resolution, 1, 1)
            output = model(test_input)
            v_total, v_stokes, Δv = forward_with_components(model, test_input)
            
            println("\nModell-Test erfolgreich:")
            println("  Output Shape: $(size(output))")
            println("  Stokes Magnitude: $(mean(abs.(v_stokes)))")
            println("  Residuum Magnitude: $(mean(abs.(Δv)))")
            
            size(output) == (SERVER_CONFIG.target_resolution, 
                            SERVER_CONFIG.target_resolution, 2, 1)
        catch e
            println("ResidualUNet-Test Fehler: $e")
            false
        end
        
        if !success
            error("ResidualUNet-Test fehlgeschlagen!")
        end
        println("✓ ResidualUNet funktioniert")
        
        # =============================================================================
        # 4. TRAINING KONFIGURATION
        # =============================================================================
        println("\n4. TRAINING KONFIGURATION")
        println("-"^50)
        
        if config_override === nothing
            train_config = create_training_config(
                learning_rate = SERVER_CONFIG.learning_rate,
                num_epochs = SERVER_CONFIG.num_epochs,
                batch_size = SERVER_CONFIG.batch_size,
                checkpoint_dir = SERVER_CONFIG.checkpoint_dir,
                save_every_n_epochs = 5,
                use_gpu = SERVER_CONFIG.use_gpu,
                validation_split = SERVER_CONFIG.validation_split,
                early_stopping_patience = SERVER_CONFIG.early_stopping_patience,
                lambda_physics_initial = SERVER_CONFIG.lambda_physics_initial,
                lambda_physics_final = SERVER_CONFIG.lambda_physics_final,
                physics_warmup_epochs = SERVER_CONFIG.physics_warmup_epochs
            )
        else
            train_config = config_override
        end
        
        println("✓ Training Config:")
        println("  Epochen: $(train_config.num_epochs)")
        println("  Batch Size: $(train_config.batch_size)")
        println("  Learning Rate: $(train_config.learning_rate)")
        println("  Residual Regularisierung: $(SERVER_CONFIG.lambda_residual)")
        println("  Sparsity Weight: $(SERVER_CONFIG.lambda_sparsity)")
        
        # =============================================================================
        # 5. RESIDUAL TRAINING
        # =============================================================================
        println("\n5. RESIDUAL TRAINING")
        println("-"^50)
        
        # Die Lambda-Werte werden in loss_residual() verwendet, nicht hier als Parameter
        println("Loss-Gewichte:")
        println("  lambda_residual: $(SERVER_CONFIG.lambda_residual)")
        println("  lambda_sparsity: $(SERVER_CONFIG.lambda_sparsity)")
        println("  lambda_physics: $(SERVER_CONFIG.lambda_physics)")
        
        trained_model, train_losses, val_losses, components_history = train_residual_unet(
        model, dataset, SERVER_CONFIG.target_resolution,
        config=train_config,
        use_adaptive=true,
        monitor_residuals=true,
        lambda_residual=SERVER_CONFIG.lambda_residual,
        lambda_sparsity=SERVER_CONFIG.lambda_sparsity,
        lambda_physics=SERVER_CONFIG.lambda_physics
        )
        
        println("✓ Training abgeschlossen")
        
        # =============================================================================
        # 6. ERGEBNISSE ANALYSE
        # =============================================================================
        println("\n6. TRAINING-ERGEBNISSE ANALYSE")
        println("-"^50)
        
        if length(train_losses) > 0 && length(val_losses) > 0
            # Convergence
            initial_loss = train_losses[1]
            final_loss = train_losses[end]
            improvement = (initial_loss - final_loss) / initial_loss * 100
            
            println("Training Convergence:")
            println("  Initial Loss: $(round(initial_loss, digits=6))")
            println("  Final Loss: $(round(final_loss, digits=6))")
            println("  Verbesserung: $(round(improvement, digits=1))%")
            
            # Overfitting
            train_val_gap = abs(train_losses[end] - val_losses[end])
            println("\nOverfitting-Analyse:")
            println("  Train-Val Gap: $(round(train_val_gap, digits=6))")
            println("  Status: $(train_val_gap < 0.01 ? "✓ Gut" : "⚠️  Overfitting möglich")")
            
            # Loss-Komponenten
            if !isempty(components_history)
                println("\nLoss-Komponenten (finale Epoche):")
                final_components = components_history[end]
                println("  Velocity Loss: $(round(final_components["velocity_loss"], digits=6))")
                println("  Residual Penalty: $(round(final_components["residual_penalty"], digits=6))")
                println("  Sparsity Loss: $(round(final_components["sparsity_loss"], digits=6))")
                
                if haskey(final_components, "physics_loss")
                    println("  Physics Loss: $(round(final_components["physics_loss"], digits=6))")
                end
                
                # Residuum-Analyse
                residual_ratio = final_components["residual_penalty"] / final_components["velocity_loss"]
                println("\nResiduum-Analyse:")
                println("  Residuum/Velocity Ratio: $(round(residual_ratio, digits=4))")
                println("  Interpretation: $(residual_ratio < 0.1 ? "✓ Residuen sind klein (gut!)" : "⚠️ Residuen relativ groß")")
            end
            
            # Beste Performance
            best_val_loss = minimum(val_losses)
            best_epoch = argmin(val_losses)
            println("\nBeste Performance:")
            println("  Beste Val Loss: $(round(best_val_loss, digits=6)) (Epoche $best_epoch)")
            println("  Finale Val Loss: $(round(val_losses[end], digits=6))")
        end
        
        # =============================================================================
        # 7. ERGEBNISSE SPEICHERN
        # =============================================================================
        println("\n7. ERGEBNISSE SPEICHERN")
        println("-"^50)
        
        # Sammle Normalisierungs-Statistiken
        println("Sammle Normalisierungs-Statistiken...")
        all_vx = []
        all_vz = []
        for sample in dataset[1:min(100, length(dataset))]
            _, _, _, vx, vz, _, _, _ = sample
            push!(all_vx, vec(vx))
            push!(all_vz, vec(vz))
        end
        vx_global_mean = mean(vcat(all_vx...))
        vx_global_std = std(vcat(all_vx...))
        vz_global_mean = mean(vcat(all_vz...))
        vz_global_std = std(vcat(all_vz...))
        
        # Training-Ergebnisse
        results_data = Dict(
            "trained_model" => trained_model,
            "train_losses" => train_losses,
            "val_losses" => val_losses,
            "components_history" => components_history,
            "config" => SERVER_CONFIG,
            "dataset_size" => length(dataset),
            "normalization_stats" => Dict(
                "vx_mean" => vx_global_mean,
                "vx_std" => vx_global_std,
                "vz_mean" => vz_global_mean,
                "vz_std" => vz_global_std
            ),
            "model_type" => "ResidualUNet",
            "approach" => "Residual Learning (v_total = v_stokes + Δv)",
            "training_completed" => true
        )
        
        results_path = joinpath(SERVER_CONFIG.results_dir, "residual_training_results.bson")
        @save results_path results_data
        println("✓ Ergebnisse gespeichert: $results_path")
        
        # Erfolgreicher Abschluss
        end_time = time()
        total_time = end_time - start_time
        
        println("\n" * "="^80)
        println("RESIDUAL UNET TRAINING ERFOLGREICH ABGESCHLOSSEN")
        println("="^80)
        println("Gesamtzeit: $(round(total_time/60, digits=2)) Minuten")
        println("Trainiertes Modell: $(SERVER_CONFIG.checkpoint_dir)/best_residual_model.bson")
        println("Ergebnisse: $results_path")
        println("\nAnsatz: Residual Learning (direktes Residuum)")
        println("  v_total = v_stokes(analytisch) + Δv(gelernt)")
        
        return trained_model, train_losses, val_losses, components_history
        
    catch e
        end_time = time()
        total_time = end_time - start_time
        
        println("\n" * "="^80)
        println("RESIDUAL UNET TRAINING FEHLGESCHLAGEN")
        println("="^80)
        println("Fehler: $e")
        println("Laufzeit bis Fehler: $(round(total_time/60, digits=1)) Minuten")
        
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println()
        end
        
        return nothing, Float32[], Float32[], []
    end
end

# =============================================================================
# VERGLEICHSFUNKTION: STANDARD VS RESIDUAL
# =============================================================================

"""
Vergleicht Standard-UNet mit Residual-UNet
"""
function compare_standard_vs_residual(;
    n_samples=20,
    n_epochs=10,
    batch_size=4
)
    println("\n" * "="^80)
    println("VERGLEICH: STANDARD UNET VS RESIDUAL UNET")
    println("="^80)
    
    # Generiere gemeinsames Test-Dataset
    println("\n1. Generiere Test-Dataset...")
    test_dataset = generate_ten_crystal_dataset(n_samples, resolution=256)
    
    if length(test_dataset) == 0
        error("Konnte kein Test-Dataset erstellen!")
    end
    
    # Training 1: Standard UNet
    println("\n2. Trainiere Standard UNet...")
    println("-"^50)
    
    model_standard = create_simplified_unet_bn(1, 2, 32)
    config_standard = create_training_config(
        num_epochs=n_epochs,
        batch_size=batch_size,
        checkpoint_dir="comparison_standard",
        use_gpu=false
    )
    
    _, train_std, val_std, _ = train_velocity_unet_safe(
        model_standard, test_dataset, 256,
        config=config_standard
    )
    
    println("✓ Standard UNet Training abgeschlossen")
    
    # Training 2: Residual UNet
    println("\n3. Trainiere Residual UNet...")
    println("-"^50)
    
    model_residual = create_residual_unet(
        base_filters=32,
        η_matrix=1e20,
        Δρ=200.0
    )
    config_residual = create_training_config(
        num_epochs=n_epochs,
        batch_size=batch_size,
        checkpoint_dir="comparison_residual",
        use_gpu=false
    )
    
    _, train_res, val_res, comp_res = train_residual_unet(
        model_residual, test_dataset, 256,
        config=config_residual,
        monitor_residuals=true
    )
    
    println("✓ Residual UNet Training abgeschlossen")
    
    # Vergleich
    println("\n" * "="^80)
    println("VERGLEICH ERGEBNISSE")
    println("="^80)
    
    println("\nFinale Validation Loss:")
    println("  Standard UNet:      $(round(val_std[end], digits=6))")
    println("  Residual UNet:      $(round(val_res[end], digits=6))")
    
    # Verbesserung
    improvement = (val_std[end] - val_res[end]) / val_std[end] * 100
    println("\nResidual UNet Verbesserung: $(round(improvement, digits=1))%")
    
    println("\nKonvergenz-Geschwindigkeit:")
    conv_std = (train_std[1] - train_std[end]) / train_std[1] * 100
    conv_res = (train_res[1] - train_res[end]) / train_res[1] * 100
    println("  Standard UNet:      $(round(conv_std, digits=1))%")
    println("  Residual UNet:      $(round(conv_res, digits=1))%")
    
    # Stabilität
    println("\nTraining-Stabilität (Train-Val Gap):")
    gap_std = abs(train_std[end] - val_std[end])
    gap_res = abs(train_res[end] - val_res[end])
    println("  Standard UNet:      $(round(gap_std, digits=6))")
    println("  Residual UNet:      $(round(gap_res, digits=6))")
    
    # Residuum-Analyse (nur für Residual UNet)
    if !isempty(comp_res)
        println("\nResiduum-Analyse (nur Residual UNet):")
        final_comp = comp_res[end]
        println("  Residual Penalty:   $(round(final_comp["residual_penalty"], digits=6))")
        println("  Sparsity Loss:      $(round(final_comp["sparsity_loss"], digits=6))")
        
        if haskey(final_comp, "physics_loss")
            println("  Physics Loss:       $(round(final_comp["physics_loss"], digits=6))")
        end
    end
    
    # Bester Ansatz
    winner = val_res[end] < val_std[end] ? "Residual UNet" : "Standard UNet"
    println("\n🏆 Bester Ansatz: $winner")
    
    # Empfehlung
    println("\nEmpfehlung:")
    if improvement > 10
        println("  ✓ Residual Learning zeigt deutliche Vorteile (>10% Verbesserung)")
        println("  → Verwende ResidualUNet für Produktion")
    elseif improvement > 0
        println("  ✓ Residual Learning ist leicht besser")
        println("  → Beide Ansätze sind vielversprechend")
    else
        println("  → Standard UNet performt in diesem Fall besser")
        println("  → Eventuell mehr Training oder Daten für Residual UNet nötig")
    end
    
    return (
        standard = (model_standard, train_std, val_std),
        residual = (model_residual, train_res, val_res, comp_res)
    )
end

# =============================================================================
# HILFSFUNKTIONEN
# =============================================================================

"""
Sicherer System-Test
"""
function quick_test_safe()
    try
        # Test mit 2 Kristallen
        x, z, phase, vx, vz, exx, ezz, v_stokes = LaMEM_Multi_crystal(
            resolution=(64, 64),
            n_crystals=2,
            radius_crystal=[0.05, 0.05],
            cen_2D=[(0.0, 0.3), (0.0, 0.7)]
        )
        
        phase_tensor, velocity_tensor = preprocess_lamem_sample(
            x, z, phase, vx, vz, v_stokes, target_resolution=64
        )
        
        # Test Residual UNet
        model = create_residual_unet(base_filters=16)
        test_input = randn(Float32, 64, 64, 1, 1)
        output = model(test_input)
        
        return size(output) == (64, 64, 2, 1)
        
    catch e
        println("System-Test Fehler: $e")
        return false
    end
end

# =============================================================================
# USAGE EXAMPLES & DOCUMENTATION
# =============================================================================

println("\n" * "="^80)
println("RESIDUAL LEARNING - BEREINIGT (OHNE STREAM FUNCTION)")
println("="^80)
println("\nAnsatz 3: Residual Learning")
println("  v_total = v_stokes(analytisch) + Δv(gelernt)")
println("\nVerfügbare Funktionen:")
println("\n1. Residual Training:")
println("   julia> run_residual_10_crystal_training()")
println("\n2. Vergleich mit Standard UNet:")
println("   julia> compare_standard_vs_residual()")
println("\n3. Individuelles Training:")
println("   julia> model = create_residual_unet()")
println("   julia> train_residual_unet(model, dataset, 256)")
println("\n4. Konfiguration anpassen:")
println("   julia> # Editiere SERVER_CONFIG oben im Script")
println("="^80)

# =============================================================================
# PROGRAMMSTART (Optional auskommentiert)
# =============================================================================

# Uncomment die folgenden Zeilen um das Training zu starten:
#
println("\nStarte Residual UNet Training...")
trained_model, train_losses, val_losses, components = run_residual_10_crystal_training()

if trained_model !== nothing
    println("\n✓ Training erfolgreich!")
    exit(0)
else
    println("\n✗ Training fehlgeschlagen!")
    exit(1)
end