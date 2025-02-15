##############################
# Training.jl
##############################
module Training
include("Model.jl")

using Flux
using Flux: onehotbatch, logitcrossentropy
using Statistics
using LinearAlgebra
using Optimisers
using CUDA          # GPU-Unterstützung
using .Model

# Loss function with additional debug outputs.
function loss_fn(model, x, y)
    pred = model(x)
    println("DEBUG: Prediction shape: ", size(pred))
    println("DEBUG: Prediction Mean/Std: ", mean(pred), " / ", std(pred))
    println("DEBUG: Prediction Min/Max: ", minimum(pred), " / ", maximum(pred))
    loss = logitcrossentropy(pred, y)
    println("DEBUG: Calculated Loss: ", loss)
    return loss
end

# Train the UNet model using the explicit update with Optimisers.jl.
# train_data is assumed to be an array of (input_batch, mask_batch) tuples.
function train_unet(model, train_data, num_epochs, learning_rate, output_channels)
    # Set up the optimizer state using Optimisers.jl (using Adam)
    opt_state = Optimisers.setup(Optimisers.Adam(learning_rate), model)
    loss_over_time = Float32[]
    println("Total number of batches: ", length(train_data))
    
    for epoch in 1:num_epochs
        println("====== Epoch $epoch ======")
        total_loss = 0f0
        
        for (batch_idx, (input_batch, mask_batch)) in enumerate(train_data)
            println("\n--- Batch $batch_idx ---")
            
            # One-hot encode the labels.
            mask_batch_int = Int.(mask_batch[:, :, 1, :])
            mask_batch_oh = permutedims(onehotbatch(mask_batch_int, 0:(output_channels-1)), (2,3,1,4))
            mask_batch_oh = Float32.(mask_batch_oh)
            println("DEBUG: Mask Batch after One-Hot, shape: ", size(mask_batch_oh))
            println("DEBUG: Mask Batch Unique Values: ", unique(mask_batch_int))
            
            # Ensure data is on the GPU.
            input_batch, mask_batch_oh = Flux.gpu(input_batch), Flux.gpu(mask_batch_oh)
            println("DEBUG: Input Batch shape: ", size(input_batch))
            
            # Compute gradients with respect to the model.
            ∇model = gradient(m -> logitcrossentropy(m(input_batch), mask_batch_oh), model)[1]
            println("DEBUG: Loss in gradient block: ", logitcrossentropy(model(input_batch), mask_batch_oh))
            
            # Update the optimizer state and model parameters explicitly.
            opt_state, model = Optimisers.update!(opt_state, model, ∇model)
            
            # After each batch, force a GPU garbage collection to free memory.
            CUDA.gc()
            
            # Compute the batch loss for monitoring.
            batch_loss = loss_fn(model, input_batch, mask_batch_oh)
            println("DEBUG: Batch Loss: ", batch_loss)
            total_loss += batch_loss
        end
        
        avg_loss = total_loss / length(train_data)
        push!(loss_over_time, avg_loss)
        println("Epoch $epoch finished. Average Loss: ", avg_loss)
        println("--------------------------------------------------")
    end
    return loss_over_time
end

end # module Training
