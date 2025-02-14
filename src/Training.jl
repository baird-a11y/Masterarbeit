module Training

using Flux
using Flux: onehotbatch, logitcrossentropy
using Statistics

# Trainingsfunktion für das UNet-Modell
function train_unet(model, train_data, num_epochs, learning_rate, output_channels)
    opt = Adam(learning_rate)
    opt_state = Flux.setup(opt, model)
    trainable = Flux.trainable(model)

    function loss_fn(x, y)
        pred = model(x)
        println("DEBUG: Prediction shape: ", size(pred))
        println("DEBUG: Prediction Min/Max: ", minimum(pred), " / ", maximum(pred))
        return logitcrossentropy(pred, y)
    end

    for epoch in 1:num_epochs
        println("====== Epoch $epoch ======")
        total_loss = 0f0
        for (batch_idx, (input_batch, mask_batch)) in enumerate(train_data)
            println("\n--- Batch $batch_idx ---")
            # One-Hot-Kodierung der Labels
            mask_batch_int = Int.(mask_batch[:, :, 1, :])
            mask_batch_oh = permutedims(onehotbatch(mask_batch_int, 0:(output_channels-1)), (2, 3, 1, 4))
            mask_batch_oh = Float32.(mask_batch_oh)
            println("DEBUG: Mask Batch nach One-Hot, shape: ", size(mask_batch_oh))
            println("DEBUG: Mask Batch Unique Values: ", unique(mask_batch_int))

            input_batch, mask_batch_oh = Flux.gpu(input_batch), Flux.gpu(mask_batch_oh)
            println("DEBUG: Input Batch shape: ", size(input_batch))

            gs = gradient(Flux.trainable(model)) do
                loss = logitcrossentropy(model(input_batch), mask_batch_oh)
                println("DEBUG: Loss im Gradienten-Block: ", loss)
                return loss
            end

            # Debug: Ausgabe der Gradienten-Normen
            for (p, g) in zip(Flux.trainable(model), gs)
                println("DEBUG: Gradient norm für Parameter (Größe ", size(p), "): ", norm(g))
            end

            println("DEBUG: Beispiel-Gewicht vor Update: ", Flux.params(model)[1])
            Flux.update!(opt_state, trainable, gs)
            println("DEBUG: Beispiel-Gewicht nach Update: ", Flux.params(model)[1])

            batch_loss = loss_fn(input_batch, mask_batch_oh)
            println("DEBUG: Batch Loss: ", batch_loss)
            total_loss += mean(batch_loss)
        end
        println("Epoch $epoch abgeschlossen. Durchschnittlicher Loss: ", total_loss / length(train_data))
    end
end

end # module Training
