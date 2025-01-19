module UNetFramework

export train_model, evaluate_model, preprocess_data, visualize_results

using Flux
using Flux.Optimise: update!
using Statistics
using ImageCore
using FileIO
using ImageShow
using Images
using Plots

# Include all necessary submodules
include("dataloader.jl")
include("model.jl")  # Bindet Model.jl ein
using .Model         # Nutzt das lokale Modul Model

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
        @info "Epoch $epoch/$epochs"

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
        num_batches = length(val_data)
        for (x_val, y_val) in val_data
            ŷ_val = model(x_val)
            val_loss += Flux.logitcrossentropy(ŷ_val, y_val)
        end

        @info "Validation Loss: $(val_loss / num_batches)"
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
    num_batches = length(test_data)

    for (x_test, y_test) in test_data
        ŷ_test = model(x_test)
        accuracy = mean(argmax(ŷ_test, dims=3) .== argmax(y_test, dims=3))
        total_accuracy += accuracy
    end

    accuracy = total_accuracy / num_batches
    @info "Test Accuracy: $accuracy"
    return accuracy
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
function visualize_results(model, input_image, ground_truth)
    """
    Visualize the input image, ground truth, and model prediction.

    Args:
        model: Trained U-Net model for prediction.
        input_image: Input image tensor (H, W, C, Batch).
        ground_truth: Ground-truth mask tensor (H, W, C, Batch).
    """
    # Führe Vorhersage durch
    prediction = model(input_image)

    # Übertrage Arrays zurück auf die CPU für die Visualisierung
    input_image_cpu = Array(input_image[:, :, 1, 1])  # Extrahiere die ersten Dimensionen
    ground_truth_cpu = Array(ground_truth[:, :, 1, 1])  # Gleiche Extraktion für Ground Truth
    prediction_cpu = Array(prediction[:, :, 1, 1])  # Vorhersage ebenfalls auf CPU übertragen

    # Flip-Operation für die Darstellung
    input_image_flipped = reverse(input_image_cpu, dims=1)
    ground_truth_flipped = reverse(ground_truth_cpu, dims=1)
    prediction_flipped = reverse(prediction_cpu, dims=1)

    # Visualisierung mit vertikalem Layout
    plot(
        heatmap(input_image_flipped, title="Input Image", color=:viridis),
        heatmap(ground_truth_flipped, title="Ground Truth Mask", color=:viridis),
        heatmap(prediction_flipped, title="Predicted Mask", color=:viridis),
        layout=(3, 1),  # Vertikales Layout
        size=(600, 900)  # Größere Plotgröße für bessere Darstellung
    )
end


end # module
