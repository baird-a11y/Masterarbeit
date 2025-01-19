module Train

export train_model

using Flux
using Flux.Optimise: update!
using DataLoader
using Model

# Training function
function train_model(; epochs=10, batch_size=4, learning_rate=0.001, target_size=(128, 128))
    """
    Train the U-Net model using the specified hyperparameters.

    Args:
        epochs: Number of training epochs.
        batch_size: Size of each batch.
        learning_rate: Learning rate for the optimizer.
        target_size: Target size for resizing input images.

    Returns:
        The trained model.
    """
    # Load and preprocess data
    images, labels = load_data("data_semantics/training/image_2", "data_semantics/training/semantic")
    train_data = create_batches(images, labels, batch_size, target_size)

    # Initialize model
    model = create_unet(1, 1)  # Assuming single-channel input and output

    # Define optimizer
    optimizer = Flux.ADAM(learning_rate)

    # Training loop
    for epoch in 1:epochs
        println("Epoch $epoch/$epochs")
        epoch_loss = 0.0

        for (x_batch, y_batch) in train_data
            # Forward and backward pass
            loss, grads = Flux.withgradient(Flux.params(model)) do
                ŷ = model(x_batch)
                Flux.logitcrossentropy(ŷ, y_batch)
            end

            # Update model parameters
            update!(optimizer, Flux.params(model), grads)

            epoch_loss += loss
        end

        println("Epoch $epoch completed with Loss: ", epoch_loss / length(train_data))
    end

    # Save the trained model
    println("Training complete. Saving model...")
    Flux.save("unet_model.bson", model)

    return model
end

end # module
