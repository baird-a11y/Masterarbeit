module Training

include("Model.jl")  # Damit wir auf UNet und Co. zugreifen können

using Flux
using Flux: onehotbatch, logitcrossentropy, binarycrossentropy
using Statistics
using LinearAlgebra
using Optimisers
using .Model

##############################################################################
# 1) Mehrklassen-Training -> logitcrossentropy
##############################################################################

function train_unet(model, train_data, num_epochs, learning_rate, output_channels)
    # Bereite Optimizer-State vor
    opt_state = Optimisers.setup(Optimisers.Adam(learning_rate), model)
    loss_over_time = Float32[]

    println("Total number of batches: ", length(train_data))

    for epoch in 1:num_epochs
        println("====== Epoch $epoch ======")
        total_loss = 0f0

        for (batch_idx, (input_batch, mask_batch)) in enumerate(train_data)
            #println("\n--- Batch $batch_idx ---")

            # Falls die Maske z. B. (H, W, 1, N) mit Int-Werten hat
            mask_batch_int = Int.(mask_batch[:, :, 1, :])

            # One-Hot (Klassen = 0..output_channels-1)
            mask_batch_oh = permutedims(onehotbatch(mask_batch_int, 0:(output_channels-1)), (2,3,1,4))
            mask_batch_oh = Float32.(mask_batch_oh)

            # Gradient über das Modell
            #   logitcrossentropy erwartet ungeclipte Logits und One-Hot-Targets
            #   "[1]" => wir holen die Gradienten w.r.t. 'model' aus dem Rückgabewert
            ∇model = gradient(m -> logitcrossentropy(m(input_batch), mask_batch_oh), model)[1]

            # Modell-Update
            opt_state, model = Optimisers.update!(opt_state, model, ∇model)

            # Loss zum Monitoring
            batch_loss = logitcrossentropy(model(input_batch), mask_batch_oh)
            total_loss += batch_loss
            #println("DEBUG: Batch Loss: ", batch_loss)
        end

        avg_loss = total_loss / length(train_data)
        push!(loss_over_time, avg_loss)
        println("Epoch $epoch finished. Average Loss: ", avg_loss)
        println("--------------------------------------------------")
    end

    return loss_over_time
end

##############################################################################
# 2) Binär-Training -> binarycrossentropy (z. B. 1-Kanal-Ausgabe)
##############################################################################

function train_unet_synthetic(model, train_data, num_epochs, learning_rate)
    opt_state = Optimisers.setup(Optimisers.Adam(learning_rate), model)
    losses = Float32[]

    for epoch in 1:num_epochs
        println("====== Epoch $epoch ======")
        epoch_loss = 0f0

        for (batch_idx, (x_batch, y_batch)) in enumerate(train_data)
            println("\n--- Batch $batch_idx ---")

            # Gradientenberechnung (Binary Cross Entropy)
            ∇model = gradient(m -> binarycrossentropy(m(x_batch), y_batch), model)[1]

            # Parameter-Update
            opt_state, model = Optimisers.update!(opt_state, model, ∇model)

            # Monitoring-Loss
            batch_loss = binarycrossentropy(model(x_batch), y_batch)
            epoch_loss += batch_loss
            println("DEBUG: Batch Loss: ", batch_loss)
        end

        avg_loss = epoch_loss / length(train_data)
        push!(losses, avg_loss)
        println("Epoch $epoch finished. Average Loss: $avg_loss")
        println("--------------------------------------------------")
    end
    return losses
end

end # module
