##############################
# Training.jl
##############################

module Training
include("Model.jl")

using Flux
using Flux: onehotbatch, logitcrossentropy
using Statistics
using LinearAlgebra  # Fügt norm hinzu
using .Model

# Loss-Funktion mit zusätzlichen Debug-Ausgaben
function loss_fn(model, x, y)
    pred = model(x)
    println("DEBUG: Prediction shape: ", size(pred))
    println("DEBUG: Prediction Mean/Std: ", mean(pred), " / ", std(pred))
    println("DEBUG: Prediction Min/Max: ", minimum(pred), " / ", maximum(pred))
    loss = logitcrossentropy(pred, y)
    println("DEBUG: Berechneter Loss: ", loss)
    return loss
end

# Trainingsfunktion mit schrittweiser Debug-Ausgabe
function train_unet(model, train_data, num_epochs, learning_rate, output_channels)
    opt = Adam(learning_rate)
    opt_state = Flux.setup(opt, model)
    
    for epoch in 1:num_epochs
        println("====== Epoch $epoch ======")
        total_loss = 0f0
        for (batch_idx, (input_batch, mask_batch)) in enumerate(train_data)
            println("\n--- Batch $batch_idx ---")
            
            # One-Hot-Kodierung der Labels
            mask_batch_int = Int.(mask_batch[:, :, 1, :])
            mask_batch_oh = permutedims(onehotbatch(mask_batch_int, 0:(output_channels-1)), (2,3,1,4))
            mask_batch_oh = Float32.(mask_batch_oh)
            println("DEBUG: Mask Batch nach One-Hot, shape: ", size(mask_batch_oh))
            println("DEBUG: Mask Batch Unique Values: ", unique(mask_batch_int))
            
            # Sicherstellen, dass die Eingaben auf der GPU liegen (sofern verfügbar)
            input_batch, mask_batch_oh = Flux.gpu(input_batch), Flux.gpu(mask_batch_oh)
            println("DEBUG: Input Batch shape: ", size(input_batch))
            
            # Gradientenberechnung mit expliziter Übergabe des Modells an das Lambda:
            gs = gradient(m -> logitcrossentropy(m(input_batch), mask_batch_oh), model)
            println("DEBUG: Loss im Gradienten-Block: ", logitcrossentropy(model(input_batch), mask_batch_oh))


            print("Hier bin ich")
            Flux.update!(opt_state, model, gs)
            print("Hier bin nicht ich")
            
            # Berechnung des Batch-Loss und aufsummieren
            batch_loss = loss_fn(model, input_batch, mask_batch_oh)
            println("DEBUG: Batch Loss: ", batch_loss)
            total_loss += batch_loss
        end
        println("Epoch $epoch abgeschlossen. Durchschnittlicher Loss: ", total_loss / length(train_data))
    end
end

end # module Training
