# =============================================================================
# TRAINING MODULE - ZYGOTE-SICHER MIT PHYSICS-INFORMED LOSS
# =============================================================================


using Flux
using Flux: mse
using CUDA
using Optimisers
using BSON: @save
using Statistics
using Random

include("gpu_utils.jl")  # GPU Utilities laden

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
GPU-kompatible Divergenz-Berechnung (vereinfacht)
"""
function compute_divergence(velocity_pred)
    # velocity_pred hat Shape: (H, W, 2, B)
    vx = velocity_pred[:, :, 1, :]
    vz = velocity_pred[:, :, 2, :]
    
    H, W, B = size(vx)
    
    # Vereinfachte Differenzen ohne explizite GPU-Arrays
    # Verwende Padding um Größenprobleme zu vermeiden
    dvx_dx = (vx[2:end, :, :] .- vx[1:end-1, :, :])
    dvz_dz = (vz[:, 2:end, :] .- vz[:, 1:end-1, :])
    
    # Padding hinzufügen um gleiche Größe zu behalten
    dvx_dx_padded = vcat(zeros(eltype(vx), 1, W, B), dvx_dx)
    dvz_dz_padded = hcat(zeros(eltype(vz), H, 1, B), dvz_dz)
    
    # Divergenz
    divergence = dvx_dx_padded .+ dvz_dz_padded
    
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
    # GPU-Check
    use_gpu = config.use_gpu && check_gpu_availability()
    
    if use_gpu
        println("GPU-Training aktiviert")
        model = safe_gpu(model)
    else
        println("CPU-Training aktiviert")
        model = safe_cpu(model)
    end
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
                
                # Device-Transfer
                if use_gpu
                    phase_batch = safe_gpu(phase_batch)
                    velocity_batch = safe_gpu(velocity_batch)
                else
                    phase_batch = safe_cpu(phase_batch)
                    velocity_batch = safe_cpu(velocity_batch)
                end
                
                # GPU-sichere Loss-Funktion (NUR EINMAL DEFINIEREN)
                function loss_fn(m)
                    pred = m(phase_batch)
                    total_loss, _, _ = physics_informed_loss(
                        pred, velocity_batch, 
                        lambda_physics=lambda_physics
                    )
                    return total_loss
                end
                
                # Gradient-Berechnung mit GPU-Fehlerbehandlung (NUR EINMAL)
                loss_val = nothing
                grads = nothing
                
                try
                    loss_val, grads = Flux.withgradient(loss_fn, model)
                catch gpu_error
                    if use_gpu && occursin("CUDA", string(gpu_error))
                        println("  GPU-Fehler, wechsle zu CPU für diesen Batch")
                        model = safe_cpu(model)
                        phase_batch = safe_cpu(phase_batch)
                        velocity_batch = safe_cpu(velocity_batch)
                        opt_state = Optimisers.setup(Optimisers.Adam(config.learning_rate), model)  # Optimizer neu setup für CPU
                        loss_val, grads = Flux.withgradient(loss_fn, model)
                        use_gpu = false  # Bleibe bei CPU für Rest des Trainings
                    else
                        rethrow(gpu_error)
                    end
                end
                
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
            
            # Periodisches GPU-Memory Cleanup
            if use_gpu && i % 10 == 0
                gpu_memory_cleanup()
            end
        end  # Ende der Batch-Schleife

        # Nach der Batch-Schleife, aber noch in der Epochen-Schleife:
        # Memory cleanup am Ende jeder Epoche
        if use_gpu
            gpu_memory_cleanup()
        else
            GC.gc()
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

        # Memory cleanup am Ende jeder Epoche (INNERHALB der Epochen-Schleife)
        if use_gpu
            gpu_memory_cleanup()
        else
            GC.gc()
        end

        end  # Ende der Epochen-Schleife

        # Modell zurück auf CPU für Speicherung
        model = safe_cpu(model)

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

        return model, train_losses, val_losses, physics_losses
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


# =============================================================================
# RESIDUAL LEARNING TRAINING MODULE
# =============================================================================
# Erweiterte Loss-Funktionen für Residual Learning + Stream Function
# Füge diese Funktionen zu deiner existierenden training.jl hinzu

using Flux
using Flux: mse
using Statistics

# =============================================================================
# RESIDUAL LEARNING LOSS FUNCTIONS
# =============================================================================

"""
    loss_residual(model, phase, velocity_target; kwargs...)

Loss-Funktion für ResidualUNet (Residual Learning).

# Argumente
- `model`: ResidualUNet Modell
- `phase`: Phasenfeld [H, W, 1, B]
- `velocity_target`: Ground Truth Geschwindigkeiten [H, W, 2, B]

# Keyword-Argumente
- `lambda_residual`: Gewicht für Residuum-Regularisierung (default: 0.01)
- `lambda_sparsity`: Gewicht für Sparsity-Loss (default: 0.001)
- `lambda_physics`: Gewicht für Divergenz-Loss (default: 0.1, nur ohne Stream Function)

# Rückgabe
- `total_loss`: Gesamtverlust
- `components`: Dict mit Loss-Komponenten für Monitoring

# Funktionsweise
1. Modell gibt v_total, v_stokes, Δv zurück
2. Haupt-Loss: MSE zwischen v_total und Ground Truth
3. Regularisierung: Residuum klein halten (L2)
4. Sparsity: Residuum spärlich halten (L1)
5. Optional: Divergenz-Loss für nicht-Stream-Function Modelle
"""
function loss_residual(
    model, 
    phase, 
    velocity_target; 
    lambda_residual = 0.01f0, 
    lambda_sparsity = 0.001f0,
    lambda_physics = 0.1f0
)
    # Forward Pass mit allen Komponenten
    # model gibt zurück: (v_total, v_stokes, Δv)
    v_pred, v_stokes, Δv = forward_with_components(model, phase)
    
    # 1. Haupt-Loss: Gesamtgeschwindigkeit gegen Ground Truth
    velocity_loss = mse(v_pred, velocity_target)
    
    # 2. Regularisierung: Residuen klein halten
    # Bestraft große Abweichungen von der Stokes-Lösung
    residual_penalty = lambda_residual * mean(abs2, Δv)
    
    # 3. Sparsity Loss: Fördert spärliche Residuen
    # Viele Bereiche sollten nahe Null sein (nur wo nötig korrigieren)
    sparsity_loss = lambda_sparsity * mean(abs, Δv)
    
    # 4. Optional: Divergenz-Loss (nur wenn keine Stream Function)
    if !model.use_stream_function
        divergence = compute_divergence(v_pred)
        physics_loss = mean(abs2, divergence)
        
        total_loss = velocity_loss + residual_penalty + sparsity_loss + 
                     lambda_physics * physics_loss
        
        # Komponenten für Monitoring
        components = Dict(
            "velocity_loss" => velocity_loss,
            "residual_penalty" => residual_penalty,
            "sparsity_loss" => sparsity_loss,
            "physics_loss" => physics_loss,
            "total_loss" => total_loss
        )
        
        return total_loss, components
    end
    
    # Stream Function Fall: keine Divergenz-Regularisierung nötig
    total_loss = velocity_loss + residual_penalty + sparsity_loss
    
    components = Dict(
        "velocity_loss" => velocity_loss,
        "residual_penalty" => residual_penalty,
        "sparsity_loss" => sparsity_loss,
        "total_loss" => total_loss
    )
    
    return total_loss, components
end

"""
    loss_residual_adaptive(model, phase, velocity_target, epoch; kwargs...)

Adaptive Loss-Funktion mit zeitabhängigen Gewichten.

Die Gewichte werden während des Trainings angepasst:
- Frühe Epochen: Fokus auf Velocity-Treue
- Mittlere Epochen: Balance zwischen Treue und Regularisierung
- Späte Epochen: Stärkere Regularisierung für Generalisierung

# Zusätzliche Argumente
- `epoch`: Aktuelle Epoche
- `total_epochs`: Gesamtanzahl Epochen
- `residual_warmup`: Epochen bis volle Residual-Regularisierung (default: 10)
- `sparsity_warmup`: Epochen bis volle Sparsity-Regularisierung (default: 15)
"""
function loss_residual_adaptive(
    model,
    phase,
    velocity_target,
    epoch;
    lambda_residual_initial = 0.001f0,
    lambda_residual_final = 0.01f0,
    lambda_sparsity_initial = 0.0001f0,
    lambda_sparsity_final = 0.001f0,
    lambda_physics = 0.1f0,
    residual_warmup = 10,
    sparsity_warmup = 15
)
    # Berechne adaptive Gewichte
    # Residual-Gewicht: Lineares Warm-up
    if epoch <= residual_warmup
        lambda_residual = lambda_residual_initial + 
                         (lambda_residual_final - lambda_residual_initial) * 
                         (epoch / residual_warmup)
    else
        lambda_residual = lambda_residual_final
    end
    
    # Sparsity-Gewicht: Verzögertes Warm-up
    if epoch <= sparsity_warmup
        lambda_sparsity = lambda_sparsity_initial + 
                         (lambda_sparsity_final - lambda_sparsity_initial) * 
                         (epoch / sparsity_warmup)
    else
        lambda_sparsity = lambda_sparsity_final
    end
    
    # Rufe normale Loss-Funktion mit adaptiven Gewichten auf
    total_loss, components = loss_residual(
        model, phase, velocity_target;
        lambda_residual = lambda_residual,
        lambda_sparsity = lambda_sparsity,
        lambda_physics = lambda_physics
    )
    
    # Füge aktuelle Gewichte zu Komponenten hinzu für Logging
    components["lambda_residual"] = lambda_residual
    components["lambda_sparsity"] = lambda_sparsity
    
    return total_loss, components
end

"""
    loss_residual_with_stokes_supervision(model, phase, velocity_target, v_stokes_target; kwargs...)

Erweiterte Loss mit expliziter Stokes-Supervision.

Nutzt vorberechnete Stokes-Lösung als zusätzliches Supervisionssignal.
Nützlich wenn analytische Stokes-Lösung verfügbar ist.

# Zusätzliche Argumente
- `v_stokes_target`: Vorberechnete Stokes-Lösung [H, W, 2, B]
- `lambda_stokes`: Gewicht für Stokes-Supervision (default: 0.05)
"""
function loss_residual_with_stokes_supervision(
    model,
    phase,
    velocity_target,
    v_stokes_target;
    lambda_residual = 0.01f0,
    lambda_sparsity = 0.001f0,
    lambda_physics = 0.1f0,
    lambda_stokes = 0.05f0
)
    # Standard Residual Loss
    total_loss, components = loss_residual(
        model, phase, velocity_target;
        lambda_residual = lambda_residual,
        lambda_sparsity = lambda_sparsity,
        lambda_physics = lambda_physics
    )
    
    # Zusätzliche Stokes-Supervision
    # Vergleiche interne Stokes-Berechnung mit vorberechneter
    v_pred, v_stokes_pred, Δv = forward_with_components(model, phase)
    stokes_supervision = mse(v_stokes_pred, v_stokes_target)
    
    # Addiere zu Total Loss
    total_loss = total_loss + lambda_stokes * stokes_supervision
    components["stokes_supervision"] = stokes_supervision
    components["total_loss"] = total_loss
    
    return total_loss, components
end

# =============================================================================
# UTILITY FUNCTIONS FÜR RESIDUAL ANALYSIS
# =============================================================================

"""
    compute_residual_statistics(model, dataset)

Berechnet Statistiken über gelernte Residuen für ein Dataset.

Nützlich für Analyse und Debugging:
- Durchschnittliche Residuum-Magnitude
- Sparsity (Anteil kleiner Werte)
- Räumliche Verteilung

# Rückgabe
Dict mit Statistiken:
- `mean_residual_magnitude`: Durchschnittliche Größe
- `sparsity_ratio`: Anteil Werte < Schwellwert
- `residual_to_total_ratio`: Verhältnis Residuum zu Gesamtgeschwindigkeit
"""
function compute_residual_statistics(model, dataset; threshold=1e-3)
    n_samples = length(dataset)
    
    total_residual_mag = 0.0f0
    total_velocity_mag = 0.0f0
    sparse_count = 0
    total_count = 0
    
    for sample in dataset
        # Extrahiere Phase und Velocity aus Sample
        # Annahme: sample ist Tuple (phase, ..., velocity, ...)
        phase = sample[1]
        velocity_target = sample[end-1]  # Passe Index an deine Datenstruktur an
        
        # Berechne Residuum
        v_pred, v_stokes, Δv = forward_with_components(model, phase)
        
        # Statistiken
        total_residual_mag += mean(abs, Δv)
        total_velocity_mag += mean(abs, v_pred)
        
        # Sparsity: Wie viele Residuen sind klein?
        sparse_count += sum(abs.(Δv) .< threshold)
        total_count += length(Δv)
    end
    
    return Dict(
        "mean_residual_magnitude" => total_residual_mag / n_samples,
        "mean_velocity_magnitude" => total_velocity_mag / n_samples,
        "sparsity_ratio" => sparse_count / total_count,
        "residual_to_total_ratio" => (total_residual_mag / n_samples) / 
                                     (total_velocity_mag / n_samples)
    )
end

"""
    monitor_residual_training(components_history)

Analysiert Trainings-Verlauf für Residual Learning.

Gibt Empfehlungen basierend auf Loss-Komponenten:
- Ist Residuum zu groß/klein?
- Sind Gewichte gut balanciert?
- Konvergiert das Training?
"""
function monitor_residual_training(components_history)
    println("\n=== RESIDUAL LEARNING MONITORING ===")
    
    # Extrahiere letzte Werte
    latest = components_history[end]
    
    velocity_loss = latest["velocity_loss"]
    residual_penalty = latest["residual_penalty"]
    sparsity_loss = latest["sparsity_loss"]
    
    # Analyse
    println("Loss-Komponenten:")
    println("  Velocity Loss: $(round(velocity_loss, digits=6))")
    println("  Residual Penalty: $(round(residual_penalty, digits=6))")
    println("  Sparsity Loss: $(round(sparsity_loss, digits=6))")
    
    # Balance-Check
    total = velocity_loss + residual_penalty + sparsity_loss
    vel_ratio = velocity_loss / total
    res_ratio = residual_penalty / total
    spar_ratio = sparsity_loss / total
    
    println("\nRelative Anteile:")
    println("  Velocity: $(round(vel_ratio*100, digits=1))%")
    println("  Residual: $(round(res_ratio*100, digits=1))%")
    println("  Sparsity: $(round(spar_ratio*100, digits=1))%")
    
    # Empfehlungen
    println("\nEmpfehlungen:")
    if vel_ratio > 0.95
        println("  ⚠️  Velocity Loss dominiert → Erhöhe lambda_residual/sparsity")
    elseif vel_ratio < 0.7
        println("  ⚠️  Regularisierung zu stark → Reduziere lambda_residual/sparsity")
    else
        println("  ✓ Gute Balance zwischen Losses")
    end
    
    if haskey(latest, "physics_loss")
        physics_loss = latest["physics_loss"]
        println("  Physics Loss: $(round(physics_loss, digits=6))")
        if physics_loss > 0.01
            println("  ⚠️  Hohe Divergenz → Überprüfe Modell oder erhöhe lambda_physics")
        end
    end
    
    println("="^40)
end

# =============================================================================
# TRAINING WRAPPER FÜR RESIDUAL LEARNING
# =============================================================================

"""
    train_residual_unet(model, dataset, target_resolution; config, use_adaptive=true)

Haupt-Trainings-Funktion speziell für ResidualUNet.

Erweitert die Standard-Training-Funktion um:
- Residual-spezifische Loss
- Adaptive Gewichte
- Residual-Monitoring
- Zusätzliche Metriken

# Keyword-Argumente
- `config`: TrainingConfig Objekt
- `use_adaptive`: Verwende adaptive Gewichte (empfohlen)
- `monitor_residuals`: Zeige detaillierte Residual-Statistiken (default: true)
"""
function train_residual_unet(
    model,
    dataset,
    target_resolution;
    config = create_training_config(),
    use_adaptive = true,
    monitor_residuals = true
)
    println("=== RESIDUAL UNET TRAINING ===")
    println("Modus: $(model.use_stream_function ? "Stream Function" : "Direct Residual")")
    
    # GPU Setup
    use_gpu = config.use_gpu && check_gpu_availability()
    if use_gpu
        println("GPU-Training aktiviert")
        model = safe_gpu(model)
    else
        println("CPU-Training aktiviert")
        model = safe_cpu(model)
    end
    
    # Dataset Split
    train_dataset, val_dataset = split_dataset(dataset, config.validation_split)
    println("Training: $(length(train_dataset)) Samples")
    println("Validation: $(length(val_dataset)) Samples")
    
    # Optimizer
    opt_state = Optimisers.setup(Optimisers.Adam(config.learning_rate), model)
    
    # Tracking
    train_losses = Float32[]
    val_losses = Float32[]
    components_history = []
    
    best_val_loss = Inf32
    patience_counter = 0
    
    # Training Loop
    for epoch in 1:config.num_epochs
        println("\n--- Epoche $epoch/$(config.num_epochs) ---")
        
        epoch_loss = 0.0f0
        epoch_components = Dict{String, Float32}()
        n_batches = 0
        
        # Training
        for batch_idx in 1:config.batch_size:length(train_dataset)
            batch_end = min(batch_idx + config.batch_size - 1, length(train_dataset))
            batch_samples = train_dataset[batch_idx:batch_end]
            
            # Erstelle Batch (anpassen an deine Datenstruktur)
            phase_batch, velocity_batch = create_batch(batch_samples, target_resolution)
            
            if use_gpu
                phase_batch = safe_gpu(phase_batch)
                velocity_batch = safe_gpu(velocity_batch)
            end
            
            # Gradient Step
            loss_value, grads = Flux.withgradient(model) do m
                if use_adaptive
                    loss, comps = loss_residual_adaptive(
                        m, phase_batch, velocity_batch, epoch;
                        total_epochs = config.num_epochs
                    )
                else
                    loss, comps = loss_residual(m, phase_batch, velocity_batch)
                end
                
                # Akkumuliere Komponenten
                for (key, val) in comps
                    if !haskey(epoch_components, key)
                        epoch_components[key] = 0.0f0
                    end
                    epoch_components[key] += Float32(val)
                end
                
                return loss
            end
            
            # Update
            if !isnothing(grads[1])
                Optimisers.update!(opt_state, model, grads[1])
            end
            
            epoch_loss += loss_value
            n_batches += 1
        end
        
        # Durchschnittliche Loss für Epoche
        avg_loss = epoch_loss / n_batches
        push!(train_losses, avg_loss)
        
        # Durchschnittliche Komponenten
        for key in keys(epoch_components)
            epoch_components[key] /= n_batches
        end
        push!(components_history, epoch_components)
        
        # Validation
        val_loss = evaluate_model_safe(model, val_dataset, target_resolution, check_physics=false)
        push!(val_losses, val_loss)
        
        println("Train Loss: $(round(avg_loss, digits=6))")
        println("Val Loss: $(round(val_loss, digits=6))")
        
        # Optional: Detailliertes Monitoring
        if monitor_residuals && epoch % 5 == 0
            monitor_residual_training(components_history)
        end
        
        # Early Stopping
        if val_loss < best_val_loss
            best_val_loss = val_loss
            patience_counter = 0
            # Speichere bestes Modell
        else
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience
                println("\nEarly Stopping nach $epoch Epochen")
                break
            end
        end
        
        # Checkpoint
        if epoch % config.save_every_n_epochs == 0
            checkpoint_path = joinpath(config.checkpoint_dir, "residual_epoch_$epoch.bson")
            save_model(model, checkpoint_path)
        end
    end
    
    # Finale Analyse
    if monitor_residuals
        println("\n=== FINALE RESIDUAL ANALYSE ===")
        stats = compute_residual_statistics(model, train_dataset[1:min(50, length(train_dataset))])
        for (key, val) in stats
            println("$key: $(round(val, digits=6))")
        end
    end
    
    return model, train_losses, val_losses, components_history
end

println("Residual Learning Training-Funktionen geladen!")
println("Verfügbare Loss-Funktionen:")
println("  - loss_residual() - Standard Residual Loss")
println("  - loss_residual_adaptive() - Mit adaptiven Gewichten")
println("  - loss_residual_with_stokes_supervision() - Mit Stokes-Supervision")
println("  - train_residual_unet() - Komplette Training-Pipeline")