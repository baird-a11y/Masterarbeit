##################################
# Training.jl - Optimiert für Standard Precision
##################################
module Training

include("Model.jl")
include("Data.jl")
using Flux
using Flux: onehotbatch, logitcrossentropy, gpu, cpu
using Statistics
using LinearAlgebra
using Optimisers
using CUDA
using .Model
using .Data
using BSON: @save, @load
using ProgressMeter

# Optimierte One-Hot-Kodierung für Batches
# Korrigierte Version
# Füge diese korrigierte Funktion in deine Training.jl Datei ein

function batch_one_hot(batch, num_classes)
    # Extrahiere die Labels als Integer
    batch_int = Int.(selectdim(batch, 3, 1))
    
    # Prüfe auf ungültige Labelwerte
    if any(batch_int .< 0) || any(batch_int .>= num_classes)
        println("WARNING: Label values outside valid range found!")
        println("Range: ", minimum(batch_int), " to ", maximum(batch_int))
        # Begrenze die Werte auf den gültigen Bereich
        batch_int = clamp.(batch_int, 0, num_classes-1)
    end
    
    # Manuelle One-Hot-Kodierung erstellen
    batch_size = size(batch, 4)
    height, width = size(batch, 1), size(batch, 2)
    one_hot = zeros(Float32, height, width, num_classes, batch_size)
    
    for b in 1:batch_size
        for j in 1:width
            for i in 1:height
                label = batch_int[i, j, b]
                if 0 <= label < num_classes
                    one_hot[i, j, label+1, b] = 1.0f0
                end
            end
        end
    end
    
    return gpu(one_hot)
end

# Einfache Loss-Funktion für semantische Segmentierung
function loss_fn(model, x, y)
    pred = model(x)
    return logitcrossentropy(pred, y)
end

# Checkpoint-Speicherfunktion
function save_checkpoint(model, epoch, loss, filename)
    model_cpu = cpu(model)
    @save filename model=model_cpu epoch=epoch loss=loss
    println("Saved checkpoint to $filename")
end

# ===================================================
# Standard Precision Training Funktion mit Batch-Verifizierung
# ===================================================
function train_unet(model, train_data, num_epochs, learning_rate, output_channels; 
                    checkpoint_dir="checkpoints", checkpoint_freq=5, verify_batches=true)

    mkpath(checkpoint_dir)
    opt = Optimisers.Adam(learning_rate)
    opt_state = Optimisers.setup(opt, model)
    
    losses = Float32[]
    
    # Batch-Verifizierung vor dem Training
    if verify_batches
        println("\n======= BATCH VERIFICATION =======")
        println("Checking $(min(3, length(train_data))) batches for correct order and consistency:")
        
        for batch_idx in 1:min(3, length(train_data))
            input_batch, mask_batch = train_data[batch_idx]
            batch_size = size(input_batch, 4)
            
            # Bestimme Anzahl der Eingabekanäle direkt aus dem Batch
            channels = size(input_batch, 3)
            
            println("Batch $batch_idx: $(size(input_batch)) inputs, $(size(mask_batch)) masks")
            
            # Überprüfe, ob die Dimensionen korrekt sind
            # Verwende feste Werte statt Konstanten aus dem Data-Modul
            expected_input_dims = (368, 1232, channels, batch_size)
            expected_mask_dims = (368, 1232, 1, batch_size)
            
            if size(input_batch) != expected_input_dims
                println("  WARNING: Input batch has unexpected dimensions!")
                println("    Expected: $expected_input_dims, Got: $(size(input_batch))")
            end
            
            if size(mask_batch) != expected_mask_dims
                println("  WARNING: Mask batch has unexpected dimensions!")
                println("    Expected: $expected_mask_dims, Got: $(size(mask_batch))")
            end
            
            # Überprüfe Ranges und statistische Werte
            input_stats = (min=minimum(input_batch), max=maximum(input_batch), mean=mean(input_batch))
            mask_stats = (min=minimum(mask_batch), max=maximum(mask_batch))
            
            println("  Input stats: min=$(input_stats.min), max=$(input_stats.max), mean=$(input_stats.mean)")
            println("  Mask stats: min=$(mask_stats.min), max=$(mask_stats.max) (should be 0-34)")
            println("")
        end
        println("================================\n")
    end
    
    for epoch in 1:num_epochs
        println("====== Epoch $epoch / $num_epochs ======")
        
        total_loss = 0f0
        batch_count = 0
        
        p = Progress(length(train_data), 1, "Training epoch $epoch...")
        
        for (batch_idx, (input_batch, mask_batch)) in enumerate(train_data)
            # Vergewissere dich, dass Dimensionen korrekt sind
            if verify_batches && batch_idx == 1
                channels = size(input_batch, 3)
                println("First batch dimensions: Input $(size(input_batch)), Mask $(size(mask_batch))")
                println("  Input channels: $channels")
            end
            
            # One-Hot-Kodierung der Labels
            mask_batch_oh = batch_one_hot(mask_batch, output_channels)
            
            # Daten auf die GPU verschieben
            input_batch = gpu(input_batch)
            
            # Gradienten und Loss berechnen
            gradients = gradient(model -> loss_fn(model, input_batch, mask_batch_oh), model)[1]
            batch_loss = loss_fn(model, input_batch, mask_batch_oh)
            
            # Modellparameter aktualisieren
            opt_state, model = Optimisers.update!(opt_state, model, gradients)
            
            total_loss += batch_loss
            batch_count += 1
            
            next!(p; showvalues = [(:batch, batch_idx), (:loss, batch_loss)])
            
            # GPU-Speicher freigeben
            CUDA.reclaim()
        end
        
        avg_loss = total_loss / batch_count
        push!(losses, avg_loss)
        println("Epoch $epoch finished. Average Loss: $avg_loss")
        
        # Checkpoint speichern
        if epoch % checkpoint_freq == 0 || epoch == num_epochs
            save_checkpoint(model, epoch, avg_loss, 
                           joinpath(checkpoint_dir, "checkpoint_epoch$(epoch).bson"))
        end
        
        println("--------------------------------------------------")
    end
    
    # Finales Modell speichern
    save_checkpoint(model, num_epochs, losses[end], 
                   joinpath(checkpoint_dir, "final_model.bson"))
    
    return model, losses
end

end # module Training