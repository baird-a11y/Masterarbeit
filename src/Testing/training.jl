# =============================================================================
# TRAINING MODULE - ZYGOTE-SICHER (BSON IMPORT KORRIGIERT)
# =============================================================================
# 

using Flux
using Flux: mse
using CUDA
using Optimisers
using BSON
using BSON: @load, @save
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
    use_gpu = false,  # Default auf CPU für Stabilität
    validation_split = 0.1f0,
    early_stopping_patience = 10
)
    return TrainingConfig(
        learning_rate, num_epochs, batch_size, checkpoint_dir,
        save_every_n_epochs, use_gpu, validation_split, early_stopping_patience
    )
end

"""
Teilt Dataset in Training und Validation auf
"""
function split_dataset(dataset, validation_split=0.1)
    n_total = length(dataset)
    n_val = max(1, Int(round(n_total * validation_split)))  # Mindestens 1 Validation-Sample
    n_train = n_total - n_val
    
    if n_train < 1
        # Falls zu wenig Daten: Verwende alles für Training
        return dataset, dataset[1:1]  # Dummy Validation
    end
    
    shuffled_indices = randperm(n_total)
    train_indices = shuffled_indices[1:n_train]
    val_indices = shuffled_indices[n_train+1:end]
    
    train_dataset = dataset[train_indices]
    val_dataset = dataset[val_indices]
    
    return train_dataset, val_dataset
end

"""
Sichere Evaluierung auf CPU
"""
function evaluate_model_safe(model, val_dataset, target_resolution)
    if length(val_dataset) == 0
        return Inf32
    end
    
    total_loss = 0.0f0
    n_batches = 0
    
    # Sichere Batch-Größe
    val_batch_size = min(2, length(val_dataset))
    max_batches = min(5, length(val_dataset))  # Maximal 5 Batches für Validation
    
    # Sichere Range-Erstellung
    for i in 1:val_batch_size:max_batches
        end_idx = min(i + val_batch_size - 1, length(val_dataset))
        
        # Überspringe wenn Start-Index größer als Dataset
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
            batch_loss = mse(prediction, velocity_batch)
            
            if isfinite(batch_loss)
                total_loss += batch_loss
                n_batches += 1
            end
            
        catch e
            println("Validation Batch $i fehlgeschlagen: $e")
            continue
        end
    end
    
    return n_batches > 0 ? total_loss / n_batches : Inf32
end

"""
Haupttraining-Funktion - CPU-fokussiert und Zygote-sicher
"""
function train_velocity_unet_safe(
    model, 
    dataset, 
    target_resolution;
    config = create_training_config()
)
    println("=== STARTE SICHERES UNET TRAINING ===")
    println("Konfiguration:")
    println("  Epochen: $(config.num_epochs)")
    println("  Lernrate: $(config.learning_rate)")
    println("  Batch-Größe: $(config.batch_size)")
    println("  Dataset-Größe: $(length(dataset))")
    
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
    best_val_loss = Inf32
    patience_counter = 0
    
    # Training-Loop
    for epoch in 1:config.num_epochs
        println("\n--- Epoche $epoch/$(config.num_epochs) ---")
        
        # Training
        epoch_loss = 0.0f0
        n_batches = 0
        
        # Sichere Batch-Verarbeitung
        max_training_batches = min(20, length(train_dataset))  # Limitiere für Stabilität
        
        for i in 1:config.batch_size:max_training_batches
            end_idx = min(i + config.batch_size - 1, length(train_dataset))
            
            # Überspringe wenn Start-Index zu groß
            if i > length(train_dataset)
                break
            end
            
            batch_samples = train_dataset[i:min(end_idx, length(train_dataset))]
            
            try
                phase_batch, velocity_batch, successful = create_adaptive_batch(
                    batch_samples, target_resolution, verbose=false
                )
                
                if successful == 0
                    continue  # Skip leere Batches
                end
                
                # ZYGOTE-SICHERE Verlustfunktion
                function loss_fn(m)
                    pred = m(phase_batch)
                    return mse(pred, velocity_batch)
                end
                
                # Gradient-Berechnung
                loss_val, grads = Flux.withgradient(loss_fn, model)
                
                # Parameter-Update
                if grads[1] !== nothing
                    opt_state, model = Optimisers.update!(opt_state, model, grads[1])
                end
                
                # Prüfe auf valide Loss
                if isfinite(loss_val)
                    epoch_loss += loss_val
                    n_batches += 1
                end
                
                if i % (config.batch_size * 5) == 1
                    println("  Batch $i: Loss = $(round(Float32(loss_val), digits=6))")
                end
                
            catch e
                println("  Fehler bei Batch $i: $e")
                continue
            end
        end
        
        # Durchschnittlicher Training-Loss
        avg_train_loss = if n_batches > 0
            Float32(epoch_loss / n_batches)
        else
            Inf32
        end
        push!(train_losses, avg_train_loss)
        
        # Validation
        val_loss = evaluate_model_safe(model, val_dataset, target_resolution)
        push!(val_losses, val_loss)
        
        println("Epoche $epoch abgeschlossen:")
        println("  Training Loss: $(round(avg_train_loss, digits=6))")
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
            @save checkpoint_path model epoch train_losses val_losses
            println("  Checkpoint gespeichert: $checkpoint_path")
        end
        
        # Memory cleanup
        GC.gc()
    end
    
    # Finales Modell speichern
    final_path = joinpath(config.checkpoint_dir, "final_model.bson")
    @save final_path model train_losses val_losses
    
    println("\n" * "="^50)
    println("TRAINING ABGESCHLOSSEN")
    println("="^50)
    println("Beste Validation Loss: $(round(best_val_loss, digits=6))")
    if length(train_losses) > 0
        println("Finale Training Loss: $(round(train_losses[end], digits=6))")
    end
    println("Modelle gespeichert in: $(config.checkpoint_dir)")
    
    return model, train_losses, val_losses
end

"""
Lädt ein gespeichertes Modell - KORRIGIERT
"""
function load_trained_model(model_path::String)
    println("Lade Modell: $model_path")
    
    if !isfile(model_path)
        error("Modelldatei nicht gefunden: $model_path")
    end
    
    # KORRIGIERT: Verwende BSON.load() statt model_dict
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

println("Sicheres Training Module geladen!")
println("Verfügbare Funktionen:")
println("  - train_velocity_unet_safe(model, dataset, resolution)")
println("  - load_trained_model(path)")