##################################
# Visualization.jl - Optimized
##################################
module Visualization

using Plots
using Flux: cpu
using Images
using ColorSchemes
using Statistics
using BSON: @load

# Helper function to convert prediction to class labels
function prediction_to_class_labels(prediction)
    # Get the maximum class along dimension 3
    _, indices = findmax(prediction, dims=3)
    
    # Extract the indices and subtract 1 to get 0-based class labels
    return dropdims(getindex.(indices, 3), dims=3) .- 1
end

# Improved visualization with better color mapping
function visualize_results(model, input_image, ground_truth, losses;
                        save_path="results", show_plots=true, 
                        color_scheme=:viridis)
    # Create directory if it doesn't exist
    mkpath(save_path)
    
    # Ensure model is on CPU for visualization
    model_cpu = cpu(model)

    # Add batch dimension if not present
    if ndims(input_image) == 3
        input_image = reshape(input_image, size(input_image)..., 1)
    end

    # Move to CPU
    input_cpu = cpu(input_image)
    ground_truth_cpu = cpu(ground_truth)
    
    # Run prediction
    prediction_raw = model_cpu(input_cpu)
    
    # Convert to class labels
    prediction_classes = prediction_to_class_labels(prediction_raw)
    
    # Reshape for plotting
    img_for_plot = dropdims(selectdim(selectdim(input_cpu, 3, 1), 4, 1), dims=3)
    gt_for_plot = dropdims(ground_truth_cpu, dims=3)
    pred_for_plot = dropdims(prediction_classes, dims=3)
    # gt_for_plot = dropdims(ground_truth_cpu[:,:,1,1], dims=3)
    # pred_for_plot = dropdims(prediction_classes[:,:,1], dims=3)
    
    # Custom colormap for easier visualization
    num_classes = maximum(gt_for_plot) + 1
    color_map = get(ColorSchemes.viridis, range(0, 1, length=num_classes))
    
    # Create figures with improved titles and layout
    p1 = heatmap(img_for_plot, title="Input Image", c=color_scheme, aspect_ratio=:equal)
    p2 = heatmap(gt_for_plot, title="Ground Truth", c=color_scheme, aspect_ratio=:equal)
    p3 = heatmap(pred_for_plot, title="Prediction", c=color_scheme, aspect_ratio=:equal)
    
    # Calculate metrics
    accuracy = mean(pred_for_plot .== gt_for_plot)
    
    # Create combined plot
    p_combined = plot(p1, p2, p3, layout=(1, 3), size=(900, 300),
                      title="Semantic Segmentation (Accuracy: $(round(accuracy*100, digits=2))%)")
    
    # Save the plot
    savefig(p_combined, joinpath(save_path, "prediction_result.png"))
    
    # Plot for loss over time
    if !isempty(losses)
        p_loss = plot(1:length(losses), losses, xlabel="Epoch", ylabel="Loss",
                     title="Loss Over Time", marker=:circle, linewidth=2,
                     legend=false, c=:blue)
        savefig(p_loss, joinpath(save_path, "training_loss.png"))
        
        if show_plots
            display(p_loss)
        end
    end
    
    if show_plots
        display(p_combined)
    end
    
    return p_combined
end

# Function to visualize multiple examples
function visualize_batch(model, data_batch, indices=1:4; save_path="results/batch")
    mkpath(save_path)
    
    model_cpu = cpu(model)
    
    for i in indices
        if i <= size(data_batch[1], 4)
            input_img = data_batch[1][:,:,:,i:i]
            true_mask = data_batch[2][:,:,:,i:i]
            
            # Create visualization
            p = visualize_results(model_cpu, input_img, true_mask, Float32[]; 
                                save_path=save_path, show_plots=false)
                                
            # Save with index
            savefig(p, joinpath(save_path, "sample_$(i).png"))
        end
    end
    
    println("Saved batch visualizations to $(save_path)")
end

# Function to visualize class distribution
function visualize_class_distribution(dataset, num_classes; save_path="results")
    mkpath(save_path)
    
    # Count class occurrences
    class_counts = zeros(Int, num_classes)
    
    for (_, label) in dataset
        # Flatten label
        flat_label = reshape(label, :)
        
        # Count occurrences
        for c in 0:(num_classes-1)
            class_counts[c+1] += count(==(c), flat_label)
        end
    end
    
    # Create bar plot
    p = bar(0:(num_classes-1), class_counts, 
            xlabel="Class", ylabel="Frequency",
            title="Class Distribution in Dataset",
            legend=false)
            
    # Save plot
    savefig(p, joinpath(save_path, "class_distribution.png"))
    
    return p, class_counts
end

# Function to visualize confusion matrix
function visualize_confusion_matrix(model, validation_data, num_classes; save_path="results")
    mkpath(save_path)
    
    # Initialize confusion matrix
    conf_matrix = zeros(Int, num_classes, num_classes)
    
    # Ensure model is on CPU
    model_cpu = cpu(model)
    
    for (input_batch, true_labels) in validation_data
        # Get predictions
        pred_raw = model_cpu(cpu(input_batch))
        pred_classes = prediction_to_class_labels(pred_raw)
        
        # Get true classes
        true_classes = dropdims(cpu(true_labels), dims=3)
        
        # Update confusion matrix
        for i in 1:length(pred_classes)
            pred_class = pred_classes[i]
            true_class = true_classes[i]
            if 0 <= pred_class < num_classes && 0 <= true_class < num_classes
                conf_matrix[true_class+1, pred_class+1] += 1
            end
        end
    end
    
    # Create heatmap
    p = heatmap(0:(num_classes-1), 0:(num_classes-1), conf_matrix,
                xlabel="Predicted Class", ylabel="True Class",
                title="Confusion Matrix",
                c=:viridis)
                
    # Save plot
    savefig(p, joinpath(save_path, "confusion_matrix.png"))
    
    return p, conf_matrix
end

# Function to visualize model architecture
function visualize_model_architecture(model; save_path="results")
    mkpath(save_path)
    
    # Convert model to string representation
    model_str = string(model)
    
    # Create text file with model architecture
    open(joinpath(save_path, "model_architecture.txt"), "w") do io
        println(io, "Model Architecture:")
        println(io, "==================")
        println(io, model_str)
    end
    
    println("Saved model architecture to $(joinpath(save_path, "model_architecture.txt"))")
end

# Function to visualize training metrics over time
function visualize_training_metrics(train_losses, val_losses=nothing; save_path="results")
    mkpath(save_path)
    
    epochs = 1:length(train_losses)
    
    if isnothing(val_losses)
        # Plot only training loss
        p = plot(epochs, train_losses,
                xlabel="Epoch", ylabel="Loss",
                label="Training Loss",
                marker=:circle, linewidth=2)
    else
        # Plot both training and validation loss
        p = plot(epochs, [train_losses val_losses],
                xlabel="Epoch", ylabel="Loss",
                label=["Training Loss" "Validation Loss"],
                marker=:circle, linewidth=2)
    end
    
    title!(p, "Training Metrics")
    
    # Save plot
    savefig(p, joinpath(save_path, "training_metrics.png"))
    
    return p
end

end # module Visualization