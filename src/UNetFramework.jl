module UNetFramework

export train_model, evaluate_model, preprocess_data, visualize_results

using Flux
using Flux.Optimise: update!
using Statistics
using ImageCore
using FileIO
using ImageShow

# Include all necessary submodules
include("dataloader.jl")
include("model.jl")  # Bindet Model.jl ein
using .Model         # Nutzt das lokale Modul Model
include("utils.jl")


# Function to train the U-Net model
function train_model(model, train_data, val_data; epochs=10, learning_rate=0.001)
    """
    Train the U-Net model.

    Args:
        model: The U-Net model instance.
        train_data: Training dataset (batches).
        val_data: Validation dataset (batches).
        epochs: Number of training epochs.
        learning_rate: Learning rate for the optimizer.

    Returns:
        Trained model.
    """
    optimizer = Flux.ADAM(learning_rate)

    for epoch in 1:epochs
        println("Epoch $epoch/$epochs")

        # Training Loop
        for (x_batch, y_batch) in train_data
            loss, grads = Flux.withgradient(Flux.params(model)) do
                ŷ = model(x_batch)
                Flux.logitcrossentropy(ŷ, y_batch)
            end
            update!(optimizer, Flux.params(model), grads)
        end

        # Validation Step
        val_loss = 0.0
        for (x_val, y_val) in val_data
            ŷ_val = model(x_val)
            val_loss += Flux.logitcrossentropy(ŷ_val, y_val)
        end

        println("Validation Loss: ", val_loss / length(val_data))
    end

    return model
end

# Function to evaluate the U-Net model
function evaluate_model(model, test_data)
    """
    Evaluate the U-Net model.

    Args:
        model: The trained U-Net model.
        test_data: Testing dataset.

    Returns:
        Accuracy of the model.
    """
    total_accuracy = 0.0

    for (x_test, y_test) in test_data
        ŷ_test = model(x_test)
        accuracy = mean(argmax(ŷ_test, dims=3) .== argmax(y_test, dims=3))
        total_accuracy += accuracy
    end

    println("Test Accuracy: ", total_accuracy / length(test_data))
    return total_accuracy / length(test_data)
end

# Function to preprocess data
function preprocess_data(data)
    """
    Preprocess the input data.

    Args:
        data: Raw input data.

    Returns:
        Preprocessed data suitable for training or evaluation.
    """
    return Float32.(data) / 255.0
end

# Function to visualize results
function visualize_results(image, prediction)
    """
    Visualize predictions.

    Args:
        image: Original input image.
        prediction: Predicted segmentation mask.
    """
    

    println("Visualizing results...")
    display(image)
    display(prediction)
end

end # module
