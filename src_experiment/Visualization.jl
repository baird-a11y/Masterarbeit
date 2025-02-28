##################################
# Visualization.jl - Enhanced
##################################
module Visualization

using Plots
using Flux: cpu
using Images
using ColorSchemes
using Statistics
using BSON: @load
using LinearAlgebra
using Measures

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
    
    # Calculate metrics
    accuracy = mean(pred_for_plot .== gt_for_plot)
    
    # Calculate IoU for each class
    unique_classes = sort(unique(vcat(unique(gt_for_plot), unique(pred_for_plot))))
    iou_values = zeros(Float32, length(unique_classes))
    
    for (i, class_id) in enumerate(unique_classes)
        gt_mask = gt_for_plot .== class_id
        pred_mask = pred_for_plot .== class_id
        
        intersection = sum(gt_mask .& pred_mask)
        union = sum(gt_mask .| pred_mask)
        
        if union > 0
            iou_values[i] = intersection / union
        end
    end
    
    mean_iou = mean(filter(!isnan, iou_values))
    
    # Custom colormap for easier visualization
    num_classes = maximum([maximum(gt_for_plot), maximum(pred_for_plot)]) + 1
    color_map = get(ColorSchemes.viridis, range(0, 1, length=num_classes))
    
    # Create figures with improved titles and layout
    p1 = heatmap(img_for_plot, title="Input Image", c=color_scheme, aspect_ratio=:equal)
    p2 = heatmap(gt_for_plot, title="Ground Truth", c=color_scheme, aspect_ratio=:equal)
    p3 = heatmap(pred_for_plot, title="Prediction", c=color_scheme, aspect_ratio=:equal)
    
    # Create difference map
    diff_map = pred_for_plot .== gt_for_plot
    p4 = heatmap(diff_map, title="Correct Predictions", c=:RdYlGn, aspect_ratio=:equal)
    
    # Create combined plot
    p_combined = plot(p1, p2, p3, p4, layout=(2, 2), size=(900, 900),
                      title="Segmentation Results\nAccuracy: $(round(accuracy*100, digits=2))%, IoU: $(round(mean_iou*100, digits=2))%",
                      margin=5mm)
    
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
    
    # Generate HTML report with metrics
    generate_html_report(save_path, accuracy, mean_iou, iou_values, unique_classes, losses)
    
    return p_combined, Dict("accuracy" => accuracy, "mean_iou" => mean_iou, "class_iou" => Dict(zip(unique_classes, iou_values)))
end

# Generate HTML report with detailed metrics
function generate_html_report(save_path, accuracy, mean_iou, iou_values, unique_classes, losses)
    html_path = joinpath(save_path, "segmentation_report.html")
    
    open(html_path, "w") do io
        write(io, """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Semantic Segmentation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2 { color: #333; }
                .metrics { margin: 20px 0; }
                table { border-collapse: collapse; width: 100%; max-width: 800px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .good { color: green; }
                .medium { color: orange; }
                .poor { color: red; }
                .summary { background-color: #eef; padding: 15px; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>Semantic Segmentation Results</h1>
            
            <div class="summary">
                <h2>Summary Metrics</h2>
                <p><strong>Pixel Accuracy:</strong> $(round(accuracy*100, digits=2))%</p>
                <p><strong>Mean IoU:</strong> $(round(mean_iou*100, digits=2))%</p>
            </div>
            
            <div class="metrics">
                <h2>Per-Class IoU</h2>
                <table>
                    <tr>
                        <th>Class ID</th>
                        <th>IoU (%)</th>
                        <th>Performance</th>
                    </tr>
        """)
        
        # Add rows for each class
        for (i, class_id) in enumerate(unique_classes)
            iou = iou_values[i]
            iou_percent = round(iou * 100, digits=2)
            
            # Determine performance level
            performance = if iou >= 0.7
                "<span class=\"good\">Good</span>"
            elseif iou >= 0.5
                "<span class=\"medium\">Medium</span>"
            else
                "<span class=\"poor\">Poor</span>"
            end
            
            write(io, """
                    <tr>
                        <td>$(class_id)</td>
                        <td>$(iou_percent)%</td>
                        <td>$(performance)</td>
                    </tr>
            """)
        end
        
        # Close table and add loss graph if available
        write(io, """
                </table>
            </div>
        """)
        
        # Add loss chart if available
        if !isempty(losses)
            write(io, """
            <div class="metrics">
                <h2>Training Loss</h2>
                <img src="training_loss.png" alt="Training Loss Chart" width="600">
            </div>
            """)
        end
        
        # Add visualization images
        write(io, """
            <div class="metrics">
                <h2>Segmentation Visualization</h2>
                <img src="prediction_result.png" alt="Segmentation Results" width="900">
            </div>
            
            <footer>
                <p>Report generated on $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM"))</p>
            </footer>
        </body>
        </html>
        """)
    end
    
    println("HTML report generated at: $html_path")
end

# Function to visualize multiple examples
function visualize_batch(model, data_batch, indices=1:4; save_path="results/batch")
    mkpath(save_path)
    
    model_cpu = cpu(model)
    
    # Initialize metrics collection
    batch_metrics = Dict{Int, Dict{String, Any}}()
    
    for i in indices
        if i <= size(data_batch[1], 4)
            input_img = data_batch[1][:,:,:,i:i]
            true_mask = data_batch[2][:,:,:,i:i]
            
            # Create visualization and get metrics
            p, metrics = visualize_results(model_cpu, input_img, true_mask, Float32[]; 
                              save_path=joinpath(save_path, "sample_$(i)"), show_plots=false)
                              
            # Save with index
            savefig(p, joinpath(save_path, "sample_$(i).png"))
            
            # Store metrics
            batch_metrics[i] = metrics
        end
    end
    
    # Generate batch summary report
    generate_batch_summary(save_path, batch_metrics)
    
    println("Saved batch visualizations to $(save_path)")
    
    return batch_metrics
end

# Generate a summary report for the batch
function generate_batch_summary(save_path, batch_metrics)
    # Calculate average metrics
    accuracies = [metrics["accuracy"] for (_, metrics) in batch_metrics]
    mean_ious = [metrics["mean_iou"] for (_, metrics) in batch_metrics]
    
    avg_accuracy = mean(accuracies)
    avg_iou = mean(mean_ious)
    
    # Save summary to file
    open(joinpath(save_path, "batch_summary.txt"), "w") do io
        println(io, "Batch Summary Report")
        println(io, "====================")
        println(io, "Number of samples: $(length(batch_metrics))")
        println(io, "Average Accuracy: $(round(avg_accuracy*100, digits=2))%")
        println(io, "Average Mean IoU: $(round(avg_iou*100, digits=2))%")
        println(io, "")
        println(io, "Individual Sample Metrics:")
        
        for (idx, metrics) in sort(collect(batch_metrics))
            println(io, "  Sample $idx:")
            println(io, "    Accuracy: $(round(metrics["accuracy"]*100, digits=2))%")
            println(io, "    Mean IoU: $(round(metrics["mean_iou"]*100, digits=2))%")
        end
    end
    
    # Create a bar chart of metrics
    sample_indices = sort(collect(keys(batch_metrics)))
    
    p = plot(
        sample_indices, 
        [accuracies mean_ious],
        labels=["Accuracy" "Mean IoU"],
        title="Batch Metrics by Sample",
        xlabel="Sample Index", 
        ylabel="Metric Value",
        xticks=sample_indices,
        marker=:circle,
        linewidth=2,
        legend=:bottomright,
        ylims=(0, 1)
    )
    
    # Add average lines
    hline!([avg_accuracy], line=:dash, color=:blue, label="Avg Accuracy")
    hline!([avg_iou], line=:dash, color=:red, label="Avg IoU")
    
    savefig(p, joinpath(save_path, "batch_metrics.png"))
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
    
    # Calculate class frequencies
    total_pixels = sum(class_counts)
    class_frequencies = class_counts ./ total_pixels
    
    # Create bar plot
    p = bar(0:(num_classes-1), class_counts, 
            xlabel="Class", 
            ylabel="Frequency",
            title="Class Distribution in Dataset",
            legend=false,
            color=:viridis,
            size=(900, 500),
            xticks=0:2:num_classes-1)  # Show every other class for clarity
            
    # Save plot
    savefig(p, joinpath(save_path, "class_distribution.png"))
    
    # Create frequency plot
    p2 = bar(0:(num_classes-1), class_frequencies .* 100, 
            xlabel="Class", 
            ylabel="Percentage (%)",
            title="Class Frequency Distribution",
            legend=false,
            color=:viridis,
            size=(900, 500),
            xticks=0:2:num_classes-1)
            
    # Save plot
    savefig(p2, joinpath(save_path, "class_frequency.png"))
    
    # Create cumulative distribution
    sorted_freqs = sort(class_frequencies, rev=true)
    cumulative = cumsum(sorted_freqs)
    
    p3 = plot(1:num_classes, cumulative .* 100,
              xlabel="Number of Classes", 
              ylabel="Cumulative Percentage",
              title="Cumulative Class Distribution",
              legend=false,
              marker=:circle,
              linewidth=2,
              size=(900, 500))
              
    # Add lines for 80% and 95%
    hline!([80], line=:dash, color=:red, label="80%")
    hline!([95], line=:dash, color=:blue, label="95%")
    
    # Find how many classes make up 80% and 95% of data
    classes_80 = findfirst(x -> x >= 0.8, cumulative)
    classes_95 = findfirst(x -> x >= 0.95, cumulative)
    
    # Add annotations
    if !isnothing(classes_80)
        annotate!([(classes_80, 82, text("$(classes_80) classes ≈ 80%", 10, :red))])
    end
    if !isnothing(classes_95)
        annotate!([(classes_95, 97, text("$(classes_95) classes ≈ 95%", 10, :blue))])
    end
    
    savefig(p3, joinpath(save_path, "cumulative_distribution.png"))
    
    # Save detailed class distribution to CSV
    open(joinpath(save_path, "class_distribution.csv"), "w") do io
        println(io, "Class,Count,Percentage")
        for c in 0:(num_classes-1)
            println(io, "$c,$(class_counts[c+1]),$(round(class_frequencies[c+1]*100, digits=4))")
        end
    end
    
    return p, class_counts, class_frequencies
end

# Function to visualize confusion matrix
function visualize_confusion_matrix(model, validation_data, num_classes; save_path="results", normalized=true)
    mkpath(save_path)
    
    # Initialize confusion matrix
    conf_matrix = zeros(Int, num_classes, num_classes)
    
    # Ensure model is on CPU
    model_cpu = cpu(model)
    
    # Set model to evaluation mode
    model_mode = Flux.trainmode(model_cpu, false)
    
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
            
            # Only consider valid classes
            if 0 <= pred_class < num_classes && 0 <= true_class < num_classes
                conf_matrix[true_class+1, pred_class+1] += 1
            end
        end
    end
    
    # Set model back to training mode
    Flux.trainmode(model_cpu, true)
    
    # Normalize by row (true class) if requested
    if normalized
        norm_conf_matrix = zeros(Float32, size(conf_matrix))
        
        for i in 1:num_classes
            row_sum = sum(conf_matrix[i, :])
            if row_sum > 0
                norm_conf_matrix[i, :] = conf_matrix[i, :] ./ row_sum
            end
        end
        
        # Create heatmap with normalized values
        p = heatmap(0:(num_classes-1), 0:(num_classes-1), norm_conf_matrix,
                    xlabel="Predicted Class", ylabel="True Class",
                    title="Normalized Confusion Matrix",
                    c=:viridis,
                    aspect_ratio=:equal,
                    size=(900, 800),
                    xticks=0:5:num_classes-1,  # Show every 5th class
                    yticks=0:5:num_classes-1,
                    clims=(0, 1))
                    
        # Save plot
        savefig(p, joinpath(save_path, "normalized_confusion_matrix.png"))
    end
    
    # Always create a raw count heatmap as well
    p_raw = heatmap(0:(num_classes-1), 0:(num_classes-1), log1p.(conf_matrix),
                xlabel="Predicted Class", ylabel="True Class",
                title="Confusion Matrix (log scale)",
                c=:viridis,
                aspect_ratio=:equal,
                size=(900, 800),
                xticks=0:5:num_classes-1,
                yticks=0:5:num_classes-1)
                
    # Save plot
    savefig(p_raw, joinpath(save_path, "raw_confusion_matrix.png"))
    
    # Calculate metrics from confusion matrix
    diagonal = diag(conf_matrix)
    row_sums = sum(conf_matrix, dims=2)
    col_sums = sum(conf_matrix, dims=1)
    
    # Per-class precision
    precision = zeros(Float32, num_classes)
    for i in 1:num_classes
        if col_sums[i] > 0
            precision[i] = conf_matrix[i, i] / col_sums[i]
        end
    end
    
    # Per-class recall
    recall = zeros(Float32, num_classes)
    for i in 1:num_classes
        if row_sums[i] > 0
            recall[i] = conf_matrix[i, i] / row_sums[i]
        end
    end
    
    # Per-class F1 score
    f1_score = zeros(Float32, num_classes)
    for i in 1:num_classes
        if precision[i] + recall[i] > 0
            f1_score[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
        end
    end
    
    # Save detailed metrics to CSV
    open(joinpath(save_path, "confusion_matrix_metrics.csv"), "w") do io
        println(io, "Class,Precision,Recall,F1_Score")
        for c in 0:(num_classes-1)
            println(io, "$c,$(round(precision[c+1], digits=4)),$(round(recall[c+1], digits=4)),$(round(f1_score[c+1], digits=4))")
        end
    end
    
    # Plot metrics
    p_metrics = plot(0:(num_classes-1), 
                     [precision recall f1_score],
                     labels=["Precision" "Recall" "F1 Score"],
                     title="Per-class Metrics",
                     xlabel="Class",
                     ylabel="Score",
                     linewidth=2,
                     marker=:circle,
                     legend=:bottomright,
                     size=(900, 500),
                     xticks=0:5:num_classes-1)
                     
    savefig(p_metrics, joinpath(save_path, "class_metrics.png"))
    
    return normalized ? (p, conf_matrix, norm_conf_matrix, Dict("precision" => precision, "recall" => recall, "f1" => f1_score)) : 
                       (p_raw, conf_matrix, Dict("precision" => precision, "recall" => recall, "f1" => f1_score))
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
    
    # Create an HTML visualization of the architecture
    open(joinpath(save_path, "model_architecture.html"), "w") do io
        write(io, """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Architecture Visualization</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                .architecture { margin: 20px 0; }
                .layer { margin: 5px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
                .encoder { background-color: #e1f5fe; }
                .bottleneck { background-color: #ffecb3; }
                .decoder { background-color: #e8f5e9; }
                .header { font-weight: bold; }
                pre { margin: 0; white-space: pre-wrap; }
            </style>
        </head>
        <body>
            <h1>UNet Architecture Visualization</h1>
            
            <div class="architecture">
        """)
        
        # Simple parsing to highlight different parts of the architecture
        sections = split(model_str, ")")
        for section in sections
            if contains(section, "encoder")
                write(io, """
                <div class="layer encoder">
                    <div class="header">Encoder:</div>
                    <pre>$(section))</pre>
                </div>
                """)
            elseif contains(section, "bottleneck")
                write(io, """
                <div class="layer bottleneck">
                    <div class="header">Bottleneck:</div>
                    <pre>$(section))</pre>
                </div>
                """)
            elseif contains(section, "decoder")
                write(io, """
                <div class="layer decoder">
                    <div class="header">Decoder:</div>
                    <pre>$(section))</pre>
                </div>
                """)
            elseif length(strip(section)) > 0
                write(io, """
                <div class="layer">
                    <pre>$(section))</pre>
                </div>
                """)
            end
        end
        
        write(io, """
            </div>
        </body>
        </html>
        """)
    end
    
    println("Saved model architecture to $(joinpath(save_path, "model_architecture.txt"))")
    println("Saved model visualization to $(joinpath(save_path, "model_architecture.html"))")
end

# Function to visualize training metrics over time
function visualize_training_metrics(train_losses, val_losses=nothing; save_path="results", metrics=nothing)
    mkpath(save_path)
    
    epochs = 1:length(train_losses)
    
    if isnothing(val_losses)
        # Plot only training loss
        p = plot(epochs, train_losses,
                xlabel="Epoch", ylabel="Loss",
                label="Training Loss",
                marker=:circle, linewidth=2,
                title="Training Loss Over Time",
                size=(900, 500))
    else
        # Plot both training and validation loss
        p = plot(epochs, [train_losses val_losses],
                xlabel="Epoch", ylabel="Loss",
                label=["Training Loss" "Validation Loss"],
                marker=:circle, linewidth=2,
                title="Training and Validation Loss",
                size=(900, 500))
    end
    
    # Save plot
    savefig(p, joinpath(save_path, "training_metrics.png"))
    
    # If additional metrics are provided (e.g., accuracy, IoU)
    if !isnothing(metrics) && !isempty(metrics)
        metric_keys = sort(collect(keys(metrics)))
        
        for key in metric_keys
            metric_values = metrics[key]
            
            if length(metric_values) == length(epochs)
                p_metric = plot(epochs, metric_values,
                               xlabel="Epoch", ylabel=capitalize(key),
                               label=capitalize(key),
                               marker=:circle, linewidth=2,
                               title="$(capitalize(key)) Over Time",
                               size=(900, 500))
                               
                savefig(p_metric, joinpath(save_path, "$(lowercase(key))_over_time.png"))
            end
        end
        
        # Create a summary report
        open(joinpath(save_path, "training_summary.txt"), "w") do io
            println(io, "Training Summary")
            println(io, "===============")
            println(io, "Total Epochs: $(length(epochs))")
            println(io, "")
            println(io, "Final Metrics:")
            
            # Print final loss
            println(io, "  Training Loss: $(round(train_losses[end], digits=6))")
            if !isnothing(val_losses)
                println(io, "  Validation Loss: $(round(val_losses[end], digits=6))")
            end
            
            # Print other final metrics
            for key in metric_keys
                if length(metrics[key]) == length(epochs)
                    println(io, "  $(capitalize(key)): $(round(metrics[key][end], digits=6))")
                end
            end
            
            # Print best metrics
            println(io, "")
            println(io, "Best Metrics:")
            
            if !isnothing(val_losses)
                best_val_idx = argmin(val_losses)
                println(io, "  Best Validation Loss: $(round(val_losses[best_val_idx], digits=6)) (Epoch $best_val_idx)")
            end
            
            for key in metric_keys
                if length(metrics[key]) == length(epochs)
                    best_idx = argmax(metrics[key])  # Assuming higher is better for all other metrics
                    println(io, "  Best $(capitalize(key)): $(round(metrics[key][best_idx], digits=6)) (Epoch $best_idx)")
                end
            end
        end
    end
    
    return p
end

# Helper function to capitalize strings
function capitalize(s::AbstractString)
    if isempty(s)
        return s
    end
    return uppercase(s[1]) * lowercase(s[2:end])
end

end # module Visualization