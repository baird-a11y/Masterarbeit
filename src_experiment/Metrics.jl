##################################
# Metrics.jl - Advanced Evaluation Metrics
##################################
module Metrics

using Statistics
using LinearAlgebra
using Flux: cpu

# Helper function to convert prediction to class labels
function prediction_to_class_labels(prediction)
    # Get the maximum class along dimension 3
    _, indices = findmax(prediction, dims=3)
    
    # Extract the indices and subtract 1 to get 0-based class labels
    return dropdims(getindex.(indices, 3), dims=3) .- 1
end

"""
    pixel_accuracy(y_pred, y_true)

Calculate pixel-wise accuracy between predicted and true segmentation masks.
"""
function pixel_accuracy(y_pred, y_true)
    # Convert prediction to class labels if needed
    if ndims(y_pred) > 2 && size(y_pred, 3) > 1
        y_pred = prediction_to_class_labels(y_pred)
    end
    
    # Flatten if necessary
    if ndims(y_true) > 2
        y_true = dropdims(y_true, dims=tuple(findall(size(y_true) .== 1)...))
    end
    
    # Calculate accuracy
    return mean(y_pred .== y_true)
end

"""
    class_accuracy(y_pred, y_true, num_classes)

Calculate per-class accuracy between predicted and true segmentation masks.
"""
function class_accuracy(y_pred, y_true, num_classes)
    # Convert prediction to class labels if needed
    if ndims(y_pred) > 2 && size(y_pred, 3) > 1
        y_pred = prediction_to_class_labels(y_pred)
    end
    
    # Flatten if necessary
    if ndims(y_true) > 2
        y_true = dropdims(y_true, dims=tuple(findall(size(y_true) .== 1)...))
    end
    
    # Initialize per-class accuracies
    class_accs = zeros(Float32, num_classes)
    
    # Calculate accuracy for each class
    for c in 0:(num_classes-1)
        # Create mask for this class
        mask = y_true .== c
        
        # If class exists in ground truth
        if sum(mask) > 0
            # Calculate accuracy for this class
            correct = sum((y_pred .== c) .& mask)
            class_accs[c+1] = correct / sum(mask)
        end
    end
    
    return class_accs
end

"""
    mean_class_accuracy(y_pred, y_true, num_classes)

Calculate mean class accuracy (mean of per-class accuracies).
"""
function mean_class_accuracy(y_pred, y_true, num_classes)
    class_accs = class_accuracy(y_pred, y_true, num_classes)
    
    # Only consider classes that appear in the ground truth
    valid_classes = findall(x -> x > 0, class_accs)
    
    if isempty(valid_classes)
        return 0.0f0
    else
        return mean(class_accs[valid_classes])
    end
end

"""
    iou(y_pred, y_true, class_id)

Calculate Intersection over Union (IoU) for a specific class.
"""
function iou(y_pred, y_true, class_id)
    # Convert prediction to class labels if needed
    if ndims(y_pred) > 2 && size(y_pred, 3) > 1
        y_pred = prediction_to_class_labels(y_pred)
    end
    
    # Flatten if necessary
    if ndims(y_true) > 2
        y_true = dropdims(y_true, dims=tuple(findall(size(y_true) .== 1)...))
    end
    
    # Create masks for this class
    pred_mask = y_pred .== class_id
    true_mask = y_true .== class_id
    
    # Calculate intersection and union
    intersection = sum(pred_mask .& true_mask)
    union = sum(pred_mask .| true_mask)
    
    # Return IoU (handle case where class doesn't exist)
    return union > 0 ? intersection / union : 0.0f0
end

"""
    mean_iou(y_pred, y_true, num_classes)

Calculate mean IoU across all classes.
"""
function mean_iou(y_pred, y_true, num_classes)
    # Calculate IoU for each class
    iou_values = [iou(y_pred, y_true, c) for c in 0:(num_classes-1)]
    
    # Only consider classes that appear in the ground truth
    valid_classes = findall(x -> x > 0, iou_values)
    
    if isempty(valid_classes)
        return 0.0f0
    else
        return mean(iou_values[valid_classes])
    end
end

"""
    frequency_weighted_iou(y_pred, y_true, num_classes)

Calculate frequency weighted IoU where each class IoU is weighted by its frequency.
"""
function frequency_weighted_iou(y_pred, y_true, num_classes)
    # Convert prediction to class labels if needed
    if ndims(y_pred) > 2 && size(y_pred, 3) > 1
        y_pred = prediction_to_class_labels(y_pred)
    end
    
    # Flatten if necessary
    if ndims(y_true) > 2
        y_true = dropdims(y_true, dims=tuple(findall(size(y_true) .== 1)...))
    end
    
    # Calculate IoU for each class
    iou_values = zeros(Float32, num_classes)
    
    # Count occurrences of each class
    class_counts = zeros(Int, num_classes)
    for c in 0:(num_classes-1)
        # Create masks for this class
        pred_mask = y_pred .== c
        true_mask = y_true .== c
        
        # Calculate intersection and union
        intersection = sum(pred_mask .& true_mask)
        union = sum(pred_mask .| true_mask)
        
        # Calculate IoU
        iou_values[c+1] = union > 0 ? intersection / union : 0.0f0
        
        # Count occurrences
        class_counts[c+1] = sum(true_mask)
    end
    
    # Calculate weights based on frequency
    total_pixels = sum(class_counts)
    weights = class_counts ./ total_pixels
    
    # Calculate weighted average
    return sum(weights .* iou_values)
end

"""
    dice_coefficient(y_pred, y_true, class_id; smooth=1.0f0)

Calculate Dice coefficient for a specific class.
"""
function dice_coefficient(y_pred, y_true, class_id; smooth=1.0f0)
    # Convert prediction to class labels if needed
    if ndims(y_pred) > 2 && size(y_pred, 3) > 1
        y_pred = prediction_to_class_labels(y_pred)
    end
    
    # Flatten if necessary
    if ndims(y_true) > 2
        y_true = dropdims(y_true, dims=tuple(findall(size(y_true) .== 1)...))
    end
    
    # Create masks for this class
    pred_mask = y_pred .== class_id
    true_mask = y_true .== class_id
    
    # Calculate Dice coefficient
    intersection = sum(pred_mask .& true_mask)
    return (2.0f0 * intersection + smooth) / (sum(pred_mask) + sum(true_mask) + smooth)
end

"""
    mean_dice_coefficient(y_pred, y_true, num_classes; smooth=1.0f0)

Calculate mean Dice coefficient across all classes.
"""
function mean_dice_coefficient(y_pred, y_true, num_classes; smooth=1.0f0)
    # Calculate Dice for each class
    dice_values = [dice_coefficient(y_pred, y_true, c, smooth=smooth) for c in 0:(num_classes-1)]
    
    # Only consider classes that appear in the ground truth
    valid_classes = findall(x -> x > 0, dice_values)
    
    if isempty(valid_classes)
        return 0.0f0
    else
        return mean(dice_values[valid_classes])
    end
end

"""
    precision(y_pred, y_true, class_id)

Calculate precision for a specific class.
"""
function precision(y_pred, y_true, class_id)
    # Convert prediction to class labels if needed
    if ndims(y_pred) > 2 && size(y_pred, 3) > 1
        y_pred = prediction_to_class_labels(y_pred)
    end
    
    # Flatten if necessary
    if ndims(y_true) > 2
        y_true = dropdims(y_true, dims=tuple(findall(size(y_true) .== 1)...))
    end
    
    # Create masks
    pred_mask = y_pred .== class_id
    true_mask = y_true .== class_id
    
    # Calculate true positives and false positives
    tp = sum(pred_mask .& true_mask)
    fp = sum(pred_mask .& .!true_mask)
    
    # Return precision
    return (tp + fp) > 0 ? tp / (tp + fp) : 0.0f0
end

"""
    recall(y_pred, y_true, class_id)

Calculate recall for a specific class.
"""
function recall(y_pred, y_true, class_id)
    # Convert prediction to class labels if needed
    if ndims(y_pred) > 2 && size(y_pred, 3) > 1
        y_pred = prediction_to_class_labels(y_pred)
    end
    
    # Flatten if necessary
    if ndims(y_true) > 2
        y_true = dropdims(y_true, dims=tuple(findall(size(y_true) .== 1)...))
    end
    
    # Create masks
    pred_mask = y_pred .== class_id
    true_mask = y_true .== class_id
    
    # Calculate true positives and false negatives
    tp = sum(pred_mask .& true_mask)
    fn = sum(.!pred_mask .& true_mask)
    
    # Return recall
    return (tp + fn) > 0 ? tp / (tp + fn) : 0.0f0
end

"""
    f1_score(y_pred, y_true, class_id)

Calculate F1 score for a specific class.
"""
function f1_score(y_pred, y_true, class_id)
    # Calculate precision and recall
    prec = precision(y_pred, y_true, class_id)
    rec = recall(y_pred, y_true, class_id)
    
    # Calculate F1 score
    return (prec + rec) > 0 ? 2.0f0 * prec * rec / (prec + rec) : 0.0f0
end

"""
    mean_f1_score(y_pred, y_true, num_classes)

Calculate mean F1 score across all classes.
"""
function mean_f1_score(y_pred, y_true, num_classes)
    # Calculate F1 for each class
    f1_values = [f1_score(y_pred, y_true, c) for c in 0:(num_classes-1)]
    
    # Only consider classes that appear in the ground truth
    valid_classes = findall(x -> x > 0, f1_values)
    
    if isempty(valid_classes)
        return 0.0f0
    else
        return mean(f1_values[valid_classes])
    end
end

"""
    evaluate_batch(model, batch, num_classes)

Evaluate a model on a batch with multiple metrics.
"""
function evaluate_batch(model, batch, num_classes)
    # Get model predictions
    input_batch, true_labels = batch
    
    # Move to CPU
    input_cpu = cpu(input_batch)
    true_cpu = cpu(true_labels)
    
    # Get model prediction
    model_cpu = cpu(model)
    pred_raw = model_cpu(input_cpu)
    
    # Convert to class labels
    pred_classes = prediction_to_class_labels(pred_raw)
    
    # Reshape true labels
    true_classes = dropdims(true_cpu, dims=3)
    
    # Calculate metrics
    metrics = Dict{String, Any}()
    
    # Overall metrics
    metrics["pixel_accuracy"] = pixel_accuracy(pred_classes, true_classes)
    metrics["mean_iou"] = mean_iou(pred_classes, true_classes, num_classes)
    metrics["freq_weighted_iou"] = frequency_weighted_iou(pred_classes, true_classes, num_classes)
    metrics["mean_dice"] = mean_dice_coefficient(pred_classes, true_classes, num_classes)
    metrics["mean_f1"] = mean_f1_score(pred_classes, true_classes, num_classes)
    
    # Per-class metrics
    metrics["class_accuracy"] = class_accuracy(pred_classes, true_classes, num_classes)
    metrics["class_iou"] = [iou(pred_classes, true_classes, c) for c in 0:(num_classes-1)]
    metrics["class_dice"] = [dice_coefficient(pred_classes, true_classes, c) for c in 0:(num_classes-1)]
    metrics["class_precision"] = [precision(pred_classes, true_classes, c) for c in 0:(num_classes-1)]
    metrics["class_recall"] = [recall(pred_classes, true_classes, c) for c in 0:(num_classes-1)]
    metrics["class_f1"] = [f1_score(pred_classes, true_classes, c) for c in 0:(num_classes-1)]
    
    return metrics
end

"""
    evaluate_dataset(model, dataset, num_classes)

Evaluate a model on the entire dataset.
"""
function evaluate_dataset(model, dataset, num_classes)
    # Initialize overall metrics
    overall_metrics = Dict{String, Any}(
        "pixel_accuracy" => 0.0f0,
        "mean_iou" => 0.0f0,
        "freq_weighted_iou" => 0.0f0,
        "mean_dice" => 0.0f0,
        "mean_f1" => 0.0f0,
        "class_accuracy" => zeros(Float32, num_classes),
        "class_iou" => zeros(Float32, num_classes),
        "class_dice" => zeros(Float32, num_classes),
        "class_precision" => zeros(Float32, num_classes),
        "class_recall" => zeros(Float32, num_classes),
        "class_f1" => zeros(Float32, num_classes)
    )
    
    # Evaluate each batch
    for batch in dataset
        batch_metrics = evaluate_batch(model, batch, num_classes)
        
        # Accumulate batch metrics
        for (key, value) in batch_metrics
            if key in ["pixel_accuracy", "mean_iou", "freq_weighted_iou", "mean_dice", "mean_f1"]
                overall_metrics[key] += value / length(dataset)
            else
                overall_metrics[key] .+= value ./ length(dataset)
            end
        end
    end
    
    return overall_metrics
end

"""
    generate_metrics_report(metrics, save_path)

Generate a detailed metrics report.
"""
function generate_metrics_report(metrics, save_path)
    # Create directory if it doesn't exist
    mkpath(save_path)
    
    # Save metrics to a text file
    open(joinpath(save_path, "metrics_report.txt"), "w") do io
        println(io, "Semantic Segmentation Metrics Report")
        println(io, "===================================")
        
        println(io, "\nOverall Metrics:")
        println(io, "  Pixel Accuracy: $(round(metrics["pixel_accuracy"]*100, digits=2))%")
        println(io, "  Mean IoU: $(round(metrics["mean_iou"]*100, digits=2))%")
        println(io, "  Frequency Weighted IoU: $(round(metrics["freq_weighted_iou"]*100, digits=2))%")
        println(io, "  Mean Dice Coefficient: $(round(metrics["mean_dice"]*100, digits=2))%")
        println(io, "  Mean F1 Score: $(round(metrics["mean_f1"]*100, digits=2))%")
        
        println(io, "\nPer-Class Metrics:")
        println(io, "  Class\tAccuracy\tIoU\tDice\tPrecision\tRecall\tF1")
        
        for c in 0:(length(metrics["class_accuracy"])-1)
            acc = metrics["class_accuracy"][c+1]
            iou_val = metrics["class_iou"][c+1]
            dice = metrics["class_dice"][c+1]
            prec = metrics["class_precision"][c+1]
            rec = metrics["class_recall"][c+1]
            f1 = metrics["class_f1"][c+1]
            
            println(io, "  $c\t$(round(acc*100, digits=2))%\t$(round(iou_val*100, digits=2))%\t$(round(dice*100, digits=2))%\t$(round(prec*100, digits=2))%\t$(round(rec*100, digits=2))%\t$(round(f1*100, digits=2))%")
        end
        
        println(io, "\nTop 5 Best Classes by IoU:")
        top_indices = sortperm(metrics["class_iou"], rev=true)[1:min(5, length(metrics["class_iou"]))]
        for i in top_indices
            println(io, "  Class $(i-1): IoU=$(round(metrics["class_iou"][i]*100, digits=2))%, Dice=$(round(metrics["class_dice"][i]*100, digits=2))%")
        end
        
        println(io, "\nTop 5 Worst Classes by IoU:")
        bottom_indices = sortperm(metrics["class_iou"])[1:min(5, length(metrics["class_iou"]))]
        for i in bottom_indices
            println(io, "  Class $(i-1): IoU=$(round(metrics["class_iou"][i]*100, digits=2))%, Dice=$(round(metrics["class_dice"][i]*100, digits=2))%")
        end
    end
    
    # Save metrics to a CSV file for further analysis
    open(joinpath(save_path, "metrics_by_class.csv"), "w") do io
        println(io, "Class,Accuracy,IoU,Dice,Precision,Recall,F1")
        
        for c in 0:(length(metrics["class_accuracy"])-1)
            acc = metrics["class_accuracy"][c+1]
            iou_val = metrics["class_iou"][c+1]
            dice = metrics["class_dice"][c+1]
            prec = metrics["class_precision"][c+1]
            rec = metrics["class_recall"][c+1]
            f1 = metrics["class_f1"][c+1]
            
            println(io, "$c,$(acc),$(iou_val),$(dice),$(prec),$(rec),$(f1)")
        end
    end
    
    println("Metrics report generated at: $(save_path)")
end

end # module Metrics