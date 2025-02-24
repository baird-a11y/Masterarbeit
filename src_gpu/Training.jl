##################################
# Training.jl
##################################
module Training

include("Model.jl")
using Flux
using Flux: onehotbatch, logitcrossentropy, gpu
using Statistics
using LinearAlgebra
using Optimisers
using .Model

function loss_fn(model, x, y)
    pred = model(x)
    return logitcrossentropy(pred, y)
end

function train_unet(model, train_data, num_epochs, learning_rate, output_channels)
    # Optimierer anlegen (Adam)
    opt_state = Optimisers.setup(Optimisers.Adam(learning_rate), model)
    loss_over_time = Float32[]

    for epoch in 1:num_epochs
        println("====== Epoch $epoch ======")
        total_loss = 0f0

        for (batch_idx, (input_batch, mask_batch)) in enumerate(train_data)
            println("\n--- Batch $batch_idx ---")

            # One-Hot-Encoding
            mask_batch_int = Int.(mask_batch[:, :, 1, :])
            mask_batch_oh = permutedims(onehotbatch(mask_batch_int, 0:(output_channels-1)), (2,3,1,4))
            mask_batch_oh = Float32.(mask_batch_oh)

            # Auf GPU verschieben
            input_batch   = gpu(input_batch)
            mask_batch_oh = gpu(mask_batch_oh)

            # Compute gradients with respect to the model using an explicit lambda
            ∇model = gradient(m -> logitcrossentropy(m(input_batch), mask_batch_oh), model)[1]
            println("DEBUG: Loss in gradient block: ", logitcrossentropy(model(input_batch), mask_batch_oh))

            # Update the optimizer state and model parameters explicitly
            opt_state, model = Optimisers.update!(opt_state, model, ∇model)

            # Compute the batch loss for monitoring
            batch_loss = loss_fn(model, input_batch, mask_batch_oh)
            println("DEBUG: Batch Loss: ", batch_loss)
            total_loss += batch_loss
        end

        push!(loss_over_time, total_loss / length(train_data))
        println("Epoch $epoch finished. Average Loss: ", total_loss / length(train_data))
        println("--------------------------------------------------")
    end

    return loss_over_time
end

end # module Training
