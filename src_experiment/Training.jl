##################################
# Training.jl - Optimiert für Standard Precision
##################################
module Training

include("Model.jl")
using Flux
using Flux: onehotbatch, logitcrossentropy, gpu, cpu
using Statistics
using LinearAlgebra
using Optimisers
using CUDA
using .Model
using BSON: @save, @load
using ProgressMeter

# Optimierte One-Hot-Kodierung für Batches
function batch_one_hot(batch, num_classes)
    batch_int = Int.(selectdim(batch, 3, 1))
    return gpu(Float32.(permutedims(onehotbatch(batch_int, 0:(num_classes-1)), (2,3,1,4))))
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
# Standard Precision Training Funktion
# ===================================================
function train_unet(model, train_data, num_epochs, learning_rate, output_channels; 
                    checkpoint_dir="checkpoints", checkpoint_freq=5)

    mkpath(checkpoint_dir)
    opt = Optimisers.Adam(learning_rate)
    opt_state = Optimisers.setup(opt, model)
    
    losses = Float32[]
    
    for epoch in 1:num_epochs
        println("====== Epoch $epoch / $num_epochs ======")
        
        total_loss = 0f0
        batch_count = 0
        
        p = Progress(length(train_data), 1, "Training epoch $epoch...")
        
        for (batch_idx, (input_batch, mask_batch)) in enumerate(train_data)
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