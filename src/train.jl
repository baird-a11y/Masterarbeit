module Train

export train_model

include("dataloader.jl")
include("model.jl")
using Flux
using Flux.Optimise: update!
using .DataLoader
using .Model



# Training function
function train_model(model, train_data, val_data; epochs, learning_rate)
    """
    Train the U-Net model using the specified hyperparameters.

    Args:
        model: The U-Net model instance to be trained.
        train_data: Training dataset (batches).
        val_data: Validation dataset (batches).
        epochs: Number of training epochs.
        learning_rate: Learning rate for the optimizer.

    Returns:
        The trained model.
    """
    # Set up the optimizer
    opt_state = Flux.setup(Flux.ADAM(learning_rate), model)

    # Loss function
    loss(m, x, y) = Flux.logitcrossentropy(m(x), y)

    for epoch in 1:epochs
        println("Epoch $epoch/$epochs")
        epoch_loss = 0.0

        # Training loop
        for (x_batch, y_batch) in train_data
            # Normalize masks
            y_batch = Array{Float32}(y_batch)
            y_batch .= y_batch ./ maximum(y_batch)

            # Debugging statements
            println("Typ von m(x): ", eltype(model(x_batch)))
            println("Typ von y: ", eltype(y_batch))
            println("Wertebereich von y: ", minimum(y_batch), " - ", maximum(y_batch))

            # Compute gradients and update model parameters
            grads = Flux.gradient(m -> loss(m, x_batch, y_batch), model)
            Flux.update!(opt_state, model, grads)

            epoch_loss += loss(model, x_batch, y_batch)
        end

        println("Training Loss: ", epoch_loss / length(train_data))

        # Validation loop
        val_loss = 0.0
        for (x_val, y_val) in val_data
            y_val = Array{Float32}(y_val)
            y_val .= y_val ./ maximum(y_val)

            ŷ_val = model(x_val)
            val_loss += loss(model, x_val, y_val)
        end

        println("Validation Loss: ", val_loss / length(val_data))
    end

    # Save the trained model
    println("Training complete. Saving model...")
    Flux.save("unet_model.bson", model)

    return model
end


end # module
