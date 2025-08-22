# =============================================================================
# TRAINING MODULE - ZYGOTE-SICHER MIT PHYSICS-INFORMED LOSS
# =============================================================================
# Speichern als: training.jl

using Flux
using Flux: mse
using CUDA
using Optimisers
using BSON: @save
using Statistics
using Random

"""
Training-Konfiguration
"""
struct TrainingConfig
    learning_rate::Float32
    num_epochs::Int
    batch_size::Int
    checkpoint_dir::String
    save_every_n_epochs::Int
    use_gpu::Bool
    validation_split::Float32
    early_stopping_patience::Int
    lambda_physics_initial::Float32      
    lambda_physics_final::Float32        
    physics_warmup_epochs::Int           
end

"""
Erstellt Standard-Training-Konfiguration
"""
function create_training_config(;
    learning_rate = 0.001f0,
    num_epochs = 50,
    batch_size = 4,
    checkpoint_dir = "checkpoints",
    save_every_n_epochs = 10,
    use_gpu = false,
    validation_split = 0.1f0,
    early_stopping_patience = 10,
    lambda_physics_initial = 0.01f0,
    lambda_physics_final = 0.1f0,
    physics_warmup_epochs = 10
)
    return TrainingConfig(
        learning_rate, num_epochs, batch_size, checkpoint_dir,
        save_every_n_epochs, use_gpu, validation_split, early_stopping_patience,
        lambda_physics_initial, lambda_physics_final, physics_warmup_epochs
    )
end

"""
Berechnet die Divergenz des Geschwindigkeitsfeldes
Kontinuitätsgleichung: ∂vx/∂x + ∂vz/∂z = 0 für inkompressible Strömung
"""
function compute_divergence(velocity_pred)
    # velocity_pred hat Shape: (H, W, 2, B)
    # Kanal 1: vx, Kanal 2: vz
    
    vx = velocity_pred[:, :, 1, :]
    vz = velocity_pred[:, :, 2, :]
    
    # Zentrale Differenzen für Ableitungen
    # ∂vx/∂x
    dvx_dx = zeros(eltype(vx), size(vx))
    dvx_dx[2:end-1, :, :] = (vx[3:end, :, :] - vx[1:end-2, :, :]) / 2.0
    
    # ∂vz/∂z 
    dvz_dz = zeros(eltype(vz), size(vz))
    dvz_dz[:, 2:end-1, :] = (vz[:, 3:end, :] - vz[:, 1:end-2, :]) / 2.0
    
    # Divergenz = ∂vx/∂x + ∂vz/∂z
    divergence = dvx_dx + dvz_dz
    
    return divergence
end

"""
Physics-Informed Loss Function mit Kontinuitäts-Constraint
"""
function physics_informed_loss(prediction, velocity_batch; lambda_physics=0.1f0)
    # Standard MSE für Daten-Treue
    data_loss = mse(prediction, velocity_batch)
    
    # Physikalischer Constraint: Divergenz sollte null sein
    divergence = compute_divergence(prediction)
    physics_loss = mean(abs2, divergence)  # L2-Norm der Divergenz
    
    # Kombinierte Verlustfunktion
    total_loss = data_loss + lambda_physics * physics_loss
    
    return total_loss, data_loss, physics_loss
end

"""
Teilt Dataset in Training und Validation auf
"""
function split_dataset(dataset, validation_split=0.1)
    n_total = length(dataset)
    n_val = max(1, Int(round(n_total * validation_split)))
    n_train = n_total - n_val
    
    if n_train < 1
        return dataset, dataset[1:1]
    end
    
    shuffled_indices = randperm(n_total)
    train_indices = shuffled_indices[1:n_train]
    val_indices = shuffled_indices[n_train+1:end]
    
    train_dataset = dataset[train_indices]
    val_dataset = dataset[val_indices]
    
    return train_dataset, val_dataset
end

"""
Sichere Evaluierung auf CPU mit Physics Loss
"""
function evaluate_model_safe(model, val_dataset, target_resolution; check_physics=true)
    if length(val_dataset) == 0
        return Inf32
    end
    
    total_data_loss = 0.0f0
    total_physics_loss = 0.0f0
    n_batches = 0
    
    # Sichere Batch-Größe
    val_batch_size = min(2, length(val_dataset))
    max_batches = min(5, length(val_dataset))
    
    for i in 1:val_batch_size:max_batches
        end_idx = min(i + val_batch_size - 1, length(val_dataset))
        
        if i > length(val_dataset)
            break
        end
        
        batch_samples = val_dataset[i:min(end_idx, length(val_dataset))]
        
        try
            phase_batch, velocity_batch, successful = create_adaptive_batch(
                batch_samples, target_resolution, verbose=false
            )
            
            if successful == 0
                continue
            end
            
            # Evaluation immer auf CPU
            phase_batch = cpu(phase_batch)
            velocity_batch = cpu(velocity_batch)
            model = cpu(model)
            
            prediction = model(phase_batch)
            batch_data_loss = mse(prediction, velocity_batch)
            
            if check_physics
                divergence = compute_divergence(prediction)
                batch_physics_loss = mean(abs2, divergence)
            else
                batch_physics_loss = 0.0f0
            end
            
            if isfinite(batch_data_loss)
                total_data_loss += batch_data_loss
                total_physics_loss += batch_physics_loss
                n_batches += 1
            end
            
        catch e
            println("Validation Batch $i fehlgeschlagen: $e")
            continue
        end
    end
    
    # Berechne durchschnittliche Losses
    avg_data_loss = n_batches > 0 ? total_data_loss / n_batches : Inf32
    avg_physics_loss = n_batches > 0 ? total_physics_loss / n_batches : Inf32
    
    # Kombinierte Validation Loss (gleiche Gewichtung wie im Training)
    combined_loss = avg_data_loss + 0.1f0 * avg_physics_loss
    
    return combined_loss
end

"""
Haupttraining-Funktion mit Physics-Informed Loss
"""
function train_velocity_unet_safe(
    model, 
    dataset, 
    target_resolution;
    config = create_training_config()
)
    println("=== STARTE PHYSICS-INFORMED UNET TRAINING ===")
    println("Konfiguration:")
    println("  Epochen: $(config.num_epochs)")
    println("  Lernrate: $(config.learning_rate)")
    println("  Batch-Größe: $(config.batch_size)")
    println("  Dataset-Größe: $(length(dataset))")
    println("  Physics Lambda: $(config.lambda_physics_initial) → $(config.lambda_physics_final)")
    println("  Physics Warmup: $(config.physics_warmup_epochs) Epochen")
    
    # Validiere Dataset
    if length(dataset) == 0
        error("Dataset ist leer!")
    end
    
    # Checkpoint-Verzeichnis erstellen
    mkpath(config.checkpoint_dir)
    
    # Dataset aufteilen
    train_dataset, val_dataset = split_dataset(dataset, config.validation_split)
    println("  Training: $(length(train_dataset)) Samples")
    println("  Validation: $(length(val_dataset)) Samples")
    
    # Optimizer setup
    opt_state = Optimisers.setup(Optimisers.Adam(config.learning_rate), model)
    
    # Training-Geschichte
    train_losses = Float32[]
    val_losses = Float32[]
    physics_losses = Float32[]  # NEU: Physics Loss tracking
    best_val_loss = Inf32
    patience_counter = 0
    
    # Training-Loop
    for epoch in 1:config.num_epochs
        println("\n--- Epoche $epoch/$(config.num_epochs) ---")
        
        # Berechne lambda_physics für diese Epoche
        if epoch <= config.physics_warmup_epochs
            # Lineares Warm-up
            lambda_physics = config.lambda_physics_initial + 
                            (config.lambda_physics_final - config.lambda_physics_initial) * 
                            (epoch - 1) / config.physics_warmup_epochs
        else
            lambda_physics = config.lambda_physics_final
        end
        
        println("  Lambda Physics: $(round(lambda_physics, digits=4))")
        
        # Training
        epoch_loss = 0.0f0
        epoch_data_loss = 0.0f0
        epoch_physics_loss = 0.0f0
        n_batches = 0
        
        # Sichere Batch-Verarbeitung
        max_training_batches = min(20, length(train_dataset))
        
        for i in 1:config.batch_size:max_training_batches
            end_idx = min(i + config.batch_size - 1, length(train_dataset))
            
            if i > length(train_dataset)
                break
            end
            
            batch_samples = train_dataset[i:min(end_idx, length(train_dataset))]
            
            try
                phase_batch, velocity_batch, successful = create_adaptive_batch(
                    batch_samples, target_resolution, verbose=false
                )
                
                if successful == 0
                    continue
                end
                
                # Loss-Funktion mit Physics
                function loss_fn(m)
                    pred = m(phase_batch)
                    total_loss, _, _ = physics_informed_loss(
                        pred, velocity_batch, 
                        lambda_physics=lambda_physics
                    )
                    return total_loss
                end
                
                # Gradient-Berechnung
                loss_val, grads = Flux.withgradient(loss_fn, model)
                
                # Für Logging: Separate Berechnung der Loss-Komponenten
                pred_for_logging = model(phase_batch)
                _, data_loss_val, physics_loss_val = physics_informed_loss(
                    pred_for_logging, velocity_batch,
                    lambda_physics=lambda_physics
                )
                
                # Parameter-Update
                if grads[1] !== nothing
                    opt_state, model = Optimisers.update!(opt_state, model, grads[1])
                end
                
                # Prüfe auf valide Loss
                if isfinite(loss_val)
                    epoch_loss += loss_val
                    epoch_data_loss += data_loss_val
                    epoch_physics_loss += physics_loss_val
                    n_batches += 1
                end
                
                if i % (config.batch_size * 5) == 1
                    println("  Batch $i:")
                    println("    Total Loss = $(round(Float32(loss_val), digits=6))")
                    println("    Data Loss = $(round(Float32(data_loss_val), digits=6))")
                    println("    Physics Loss = $(round(Float32(physics_loss_val), digits=6))")
                end
                
            catch e
                println("  Fehler bei Batch $i: $e")
                continue
            end
        end
        
        # Durchschnittliche Training Losses
        avg_train_loss = n_batches > 0 ? Float32(epoch_loss / n_batches) : Inf32
        avg_data_loss = n_batches > 0 ? Float32(epoch_data_loss / n_batches) : Inf32
        avg_physics_loss = n_batches > 0 ? Float32(epoch_physics_loss / n_batches) : Inf32
        
        push!(train_losses, avg_train_loss)
        push!(physics_losses, avg_physics_loss)
        
        # Validation
        val_loss = evaluate_model_safe(model, val_dataset, target_resolution)
        push!(val_losses, val_loss)
        
        println("Epoche $epoch abgeschlossen:")
        println("  Training Loss: $(round(avg_train_loss, digits=6))")
        println("  Data Loss: $(round(avg_data_loss, digits=6))")
        println("  Physics Loss: $(round(avg_physics_loss, digits=6))")
        println("  Validation Loss: $(round(val_loss, digits=6))")
        
        # Early Stopping Check
        if val_loss < best_val_loss
            best_val_loss = val_loss
            patience_counter = 0
            
            # Speichere bestes Modell
            best_model_path = joinpath(config.checkpoint_dir, "best_model.bson")
            @save best_model_path model
            println("  Neues bestes Modell gespeichert!")
            
        else
            patience_counter += 1
            println("  Validation Loss nicht verbessert (Patience: $patience_counter/$(config.early_stopping_patience))")
            
            if patience_counter >= config.early_stopping_patience
                println("  Early Stopping aktiviert!")
                break
            end
        end
        
        # Checkpoint speichern
        if epoch % config.save_every_n_epochs == 0
            checkpoint_path = joinpath(config.checkpoint_dir, "checkpoint_epoch_$(epoch).bson")
            @save checkpoint_path model epoch train_losses val_losses physics_losses
            println("  Checkpoint gespeichert: $checkpoint_path")
        end
        
        # Memory cleanup
        GC.gc()
    end
    
    # Finales Modell speichern
    final_path = joinpath(config.checkpoint_dir, "final_model.bson")
    @save final_path model train_losses val_losses physics_losses
    
    println("\n" * "="^50)
    println("TRAINING ABGESCHLOSSEN")
    println("="^50)
    println("Beste Validation Loss: $(round(best_val_loss, digits=6))")
    if length(train_losses) > 0
        println("Finale Training Loss: $(round(train_losses[end], digits=6))")
        println("Finale Physics Loss: $(round(physics_losses[end], digits=6))")
    end
    println("Modelle gespeichert in: $(config.checkpoint_dir)")
    
    # In training.jl, am Ende von train_velocity_unet_safe:
    return model, train_losses, val_losses, physics_losses  # physics_losses hinzugefügt
end

"""
Lädt ein gespeichertes Modell
"""
function load_trained_model(model_path::String)
    println("Lade Modell: $model_path")
    
    if !isfile(model_path)
        error("Modelldatei nicht gefunden: $model_path")
    end
    
    model_dict = BSON.load(model_path)
    
    # Versuche verschiedene Schlüssel
    for key in [:model, :best_model, :final_model]
        if haskey(model_dict, key)
            model = model_dict[key]
            println("Modell unter Schlüssel '$key' gefunden")
            return model
        end
    end
    
    # Fallback: Nehme ersten Wert der ein Modell sein könnte
    for (key, value) in model_dict
        if isa(value, Flux.Chain) || hasproperty(value, :layers) || string(typeof(value)) |> x -> contains(x, "UNet")
            println("Modell unter Schlüssel '$key' gefunden")
            return value
        end
    end
    
    error("Kein Modell in der BSON-Datei gefunden")
end

println("Physics-Informed Training Module geladen!")
println("Verfügbare Funktionen:")
println("  - train_velocity_unet_safe(model, dataset, resolution)")
println("  - load_trained_model(path)")
println("  - compute_divergence(velocity_field)")