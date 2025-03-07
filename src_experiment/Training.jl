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

function debug_gradients(model, x, y, y_oh)
    # Berechne Gradienten
    grads = gradient(model -> loss_fn(model, x, y_oh), model)[1]
    
    # Überprüfe Gradienten für einige Schichten
    total_grad_norm = 0.0
    for layer_name in [:encoder1, :bottleneck, :decoder1_1]
        layer = getproperty(model, layer_name)
        if isa(layer, Chain) && !isempty(layer)
            first_layer = layer[1]
            if hasproperty(first_layer, :weight)
                param = first_layer.weight
                param_grads = grads.params[layer][1].weight
                
                param_cpu = cpu(param_grads)
                grad_norm = norm(param_cpu)
                total_grad_norm += grad_norm
                
                println("Gradients for $layer_name - norm: $grad_norm")
                println("  mean: $(mean(param_cpu)), std: $(std(param_cpu))")
                println("  min: $(minimum(param_cpu)), max: $(maximum(param_cpu))")
                
                # Prüfe auf Vanishing/Exploding Gradients
                if maximum(abs.(param_cpu)) < 1e-6
                    println("  WARNING: Possible vanishing gradients")
                elseif maximum(abs.(param_cpu)) > 1.0
                    println("  WARNING: Possible exploding gradients")
                end
            end
        end
    end
    println("Total gradient norm: $total_grad_norm")
    return grads
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
function train_simple(model, train_data, num_epochs, learning_rate, output_channels;
        checkpoint_dir="checkpoints", checkpoint_freq=1)
    mkpath(checkpoint_dir)

    # Neues Optimierer-Setup mit Flux.setup statt Optimisers.setup
    opt = Flux.Adam(learning_rate)
    opt_state = Flux.setup(opt, model)

    losses = Float32[]

    # Berechne initialen Loss
    x, y = first(train_data)
    x_device = to_device(x)
    y_oh = manual_one_hot(y, output_channels)
    initial_loss = simple_loss_fn(model, x_device, y_oh)
    println("Initial loss: $initial_loss")

    for epoch in 1:num_epochs
        println("====== Epoch $epoch / $num_epochs ======")

        total_loss = 0f0
        batch_count = 0

        for (batch_idx, (input_batch, mask_batch)) in enumerate(train_data)
        # Auf GPU/CPU verschieben
        input_device = to_device(input_batch)

        # One-Hot Encoding
        mask_oh = manual_one_hot(mask_batch, output_channels)

        # Trainingsprozess in try-catch Block
            try
                # Gradienten und Loss berechnen mit der neuen API
                loss, grads = Flux.withgradient(model) do m
                    simple_loss_fn(m, input_device, mask_oh)
                end

                # Modell aktualisieren mit der neuen API
                Flux.update!(opt_state, model, grads[1])

                total_loss += loss
                batch_count += 1

                # Ausgabe
                if batch_idx % 10 == 0 || batch_idx == 1
                    println("  Batch $batch_idx/$(length(train_data)) - Loss: $loss")
                end
            catch e
                println("Error in batch $batch_idx: $e")
                println(stacktrace())
            end

            # Speicher freigeben
            if batch_idx % 5 == 0
                clear_memory()
            end
        end

        # Berechne durchschnittlichen Loss
        avg_loss = total_loss / batch_count
        push!(losses, avg_loss)
        println("Epoch $epoch completed. Average Loss: $avg_loss")

        # Speichere Checkpoint
        if epoch % checkpoint_freq == 0 || epoch == num_epochs
            checkpoint_path = joinpath(checkpoint_dir, "checkpoint_epoch$(epoch).bson")
            model_cpu = cpu(model)
            @save checkpoint_path model=model_cpu epoch=epoch loss=avg_loss
            println("Saved checkpoint to $checkpoint_path")
        end
    end

    return model, losses
end

end # module Training