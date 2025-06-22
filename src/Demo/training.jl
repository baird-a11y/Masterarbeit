# =============================================================================
# TRAINING MODULE
# =============================================================================
# Speichern als: training.jl

using Flux
using Flux: mse
using CUDA
using CUDA: @allowscalar  # HINZUGEFÜGT: Import @allowscalar Makro
using Optimisers
using BSON: @save
using Statistics

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
    use_gpu = CUDA.functional(),
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
    n_val = Int(round(n_total * validation_split))
    n_train = n_total - n_val
    
    shuffled_indices = randperm(n_total)
    train_indices = shuffled_indices[1:n_train]
    val_indices = shuffled_indices[n_train+1:end]
    
    train_dataset = dataset[train_indices]
    val_dataset = dataset[val_indices]
    
    return train_dataset, val_dataset
end

"""
Evaluiert Modell auf Validation-Set
"""
function evaluate_model(model, val_dataset, target_resolution; use_gpu=false)
    total_loss = 0.0f0
    n_batches = 0
    
    # Kleine Batches für Validation
    val_batch_size = min(4, length(val_dataset))
    
    for i in 1:val_batch_size:length(val_dataset)
        end_idx = min(i + val_batch_size - 1, length(val_dataset))
        batch_samples = val_dataset[i:end_idx]
        
        try
            phase_batch, velocity_batch, _ = create_adaptive_batch(
                batch_samples, target_resolution, verbose=false
            )
            
            if use_gpu && CUDA.functional()
                phase_batch = gpu(phase_batch)
                velocity_batch = gpu(velocity_batch)
            end
            
            prediction = model(phase_batch)
            batch_loss = mse(prediction, velocity_batch)
            
            total_loss += batch_loss
            n_batches += 1
            
        catch e
            println("Validation Batch $i fehlgeschlagen: $e")
            continue
        end
    end
    
    return n_batches > 0 ? total_loss / n_batches : Inf32
end

"""
Haupttraining-Funktion - KORRIGIERT für GPU-Kompatibilität
"""
function train_velocity_unet(
    model, 
    dataset, 
    target_resolution;
    config = create_training_config()
)
    
    println("=== STARTE UNET TRAINING ===")
    println("Konfiguration:")
    println("  Epochen: $(config.num_epochs)")
    println("  Lernrate: $(config.learning_rate)")
    println("  Batch-Größe: $(config.batch_size)")
    println("  GPU: $(config.use_gpu)")
    println("  Dataset-Größe: $(length(dataset))")
    
    # Checkpoint-Verzeichnis erstellen
    mkpath(config.checkpoint_dir)
    
    # Dataset aufteilen
    train_dataset, val_dataset = split_dataset(dataset, config.validation_split)
    println("  Training: $(length(train_dataset)) Samples")
    println("  Validation: $(length(val_dataset)) Samples")
    
    # Optimizer setup
    opt_state = Optimisers.setup(Optimisers.Adam(config.learning_rate), model)
    
    # GPU setup
    if config.use_gpu && CUDA.functional()
        model = gpu(model)
        println("  Modell auf GPU verschoben")
        # ENTFERNT: Globales allowscalar - wird lokal verwendet
    end
    model_cpu = cpu(model)
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
        
        # Vereinfachtes Batch-Management für GPU-Kompatibilität
        for i in 1:config.batch_size:min(50, length(train_dataset))  # Limitiere für ersten Test
            end_idx = min(i + config.batch_size - 1, length(train_dataset))
            batch_samples = train_dataset[i:end_idx]
            
            try
                phase_batch, velocity_batch, successful = deepcopy(create_adaptive_batch(
                    batch_samples, target_resolution, verbose=false
                ))
                
                if successful == 0
                    continue  # Skip leere Batches
                end
                
                # GPU-Transfer
                if config.use_gpu && CUDA.functional()
                    phase_batch = gpu(phase_batch)
                    velocity_batch = gpu(velocity_batch)
                end
                
                # VOLLSTÄNDIG KORRIGIERTES CPU-TRAINING
                # Alle Daten auf CPU
                phase_cpu = cpu(phase_batch)
                velocity_cpu = cpu(velocity_batch)
                
                # Training auf CPU-Modell
                loss_fn(m) = mse(m(phase_cpu), velocity_cpu)
                ∇model_cpu = gradient(loss_fn, model_cpu)[1]
                batch_loss = loss_fn(model_cpu)
                
                # Update CPU-Modell
                opt_state, model_cpu = Optimisers.update!(opt_state, model_cpu, ∇model_cpu)
                
                # Prüfe auf valide Loss
                if isfinite(batch_loss)
                    epoch_loss += batch_loss
                    n_batches += 1
                end
                
                if i % (config.batch_size * 10) == 1
                    println("  Batch $i: Loss = $(round(Float32(batch_loss), digits=6))")
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
        train_losses = vcat(train_losses, avg_train_loss)
        
        # Validation (auf CPU-Modell)
        val_loss = evaluate_model_safe(
            model_cpu, val_dataset, target_resolution
        )
        val_losses = vcat(val_losses, val_loss)
        
        println("Epoche $epoch abgeschlossen:")
        println("  Training Loss: $(round(avg_train_loss, digits=6))")
        println("  Validation Loss: $(round(val_loss, digits=6))")
        
        # Early Stopping Check
        if val_loss < best_val_loss
            best_val_loss = val_loss
            patience_counter = 0
            
            # Speichere bestes Modell (CPU-Version)
            best_model_path = joinpath(config.checkpoint_dir, "best_model.bson")
            @save best_model_path model_cpu
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
            @save checkpoint_path model_cpu epoch train_losses val_losses
            println("  Checkpoint gespeichert: $checkpoint_path")
        end
        
        # Memory cleanup
        GC.gc()
        if CUDA.functional()
            CUDA.reclaim()
        end
    end
    
    # Finales Modell speichern
    final_path = joinpath(config.checkpoint_dir, "final_model.bson")
    @save final_path model_cpu train_losses val_losses
    
    println("\n" * "="^50)
    println("TRAINING ABGESCHLOSSEN")
    println("="^50)
    println("Beste Validation Loss: $(round(best_val_loss, digits=6))")
    println("Finale Training Loss: $(round(train_losses[end], digits=6))")
    println("Modelle gespeichert in: $(config.checkpoint_dir)")
    
    return model_cpu, train_losses, val_losses  # Returniere CPU-Modell
end

"""
Sichere Evaluierung auf CPU
"""
"""
Sichere Evaluierung auf CPU
"""
function evaluate_model_safe(model, val_dataset, target_resolution)
    total_loss = 0.0f0
    n_batches = 0
    
    # Kleine Batches für Validation
    val_batch_size = min(2, length(val_dataset))
    
    for i in 1:val_batch_size:length(val_dataset)  # Ensure valid range
        end_idx = min(i + val_batch_size - 1, length(val_dataset))
        
        # Überprüfen Sie, ob der Bereich gültig ist
        if i > end_idx
            println("Ungültiger Bereich: i=$i, end_idx=$end_idx")
            continue
        end
        
        batch_samples = val_dataset[i:end_idx]
        
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
Lädt ein gespeichertes Modell
"""
function load_trained_model(model_path::String)
    println("Lade Modell: $model_path")
    
    if !isfile(model_path)
        error("Modelldatei nicht gefunden: $model_path")
    end
    
    model_dict = BSON.load(model_path)
    
    # Versuche verschiedene Schlüssel
    for key in [:model_cpu, :model, :final_model_cpu, :best_model]
        if haskey(model_dict, key)
            model = model_dict[key]
            println("Modell unter Schlüssel '$key' gefunden")
            return model
        end
    end
    
    # Fallback: Nehme ersten Wert
    model = first(values(model_dict))
    println("Verwende ersten Wert aus BSON-Datei")
    return model
end

"""
Demo für komplettes Training
"""
function demo_training(;
    n_samples = 100,
    target_resolution = 128,
    num_epochs = 20
)
    println("=== TRAINING DEMO ===")
    
    # 1. Daten generieren
    println("\n1. Generiere Trainingsdaten...")
    dataset = generate_mixed_resolution_dataset(
        n_samples, 
        resolutions=[64, 128, 256],
        verbose=true
    )
    
    if length(dataset) < 10
        error("Zu wenige Samples generiert!")
    end
    
    # 2. UNet erstellen
    println("\n2. Erstelle UNet...")
    config = design_adaptive_unet(target_resolution)
    model = create_corrected_adaptive_unet(config)
    
    # 3. Training-Konfiguration
    train_config = create_training_config(
        num_epochs = num_epochs,
        learning_rate = 0.001f0,
        checkpoint_dir = "demo_training_checkpoints"
    )
    
    # 4. Training
    println("\n3. Starte Training...")
    trained_model, train_losses, val_losses = train_velocity_unet(
        model, dataset, target_resolution, config=train_config
    )
    
    # 5. Evaluierung
    println("\n4. Evaluiere trainiertes Modell...")
    if length(dataset) > 0
        test_sample = dataset[1]
        x, z, phase, vx, vz, exx, ezz, v_stokes = test_sample
        
        phase_tensor, velocity_tensor = preprocess_lamem_sample(
            x, z, phase, vx, vz, v_stokes,
            target_resolution=target_resolution
        )
        
        prediction = cpu(trained_model(phase_tensor))
        
        mse_vx = mean((prediction[:,:,1,1] .- velocity_tensor[:,:,1,1]).^2)
        mse_vz = mean((prediction[:,:,2,1] .- velocity_tensor[:,:,2,1]).^2)
        
        println("Test-Evaluierung:")
        println("  MSE Vx: $(round(mse_vx, digits=6))")
        println("  MSE Vz: $(round(mse_vz, digits=6))")
        println("  MSE Gesamt: $(round((mse_vx + mse_vz)/2, digits=6))")
    end
    
    println("\n✓ Training Demo abgeschlossen!")
    return trained_model, train_losses, val_losses
end

println("Training Module geladen!")
println("Verfügbare Funktionen:")
println("  - train_velocity_unet(model, dataset, resolution)")
println("  - demo_training()")
println("  - load_trained_model(path)")