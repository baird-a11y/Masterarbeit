# ======================================================
# RESULTS VISUALIZATION FOR SEMANTIC SEGMENTATION MODEL
# ======================================================

# Load necessary packages
include("Model.jl")
include("Data.jl")
include("Visualization.jl")
using Plots
using Flux: gpu, cpu
using CUDA
using BSON: @load
using Images
using Statistics
using ColorSchemes
using .Model
using .Data
using .Visualization

# === CONFIGURATION ===
# Global variables declaration
# Set this to false if you want to run without masks
use_masks = false
# if use_masks
#     test_img_dir = "S:/Masterarbeit/Datensatz/Testing/image_2"  # Test images
# else
#     test_img_dir = "S:/Masterarbeit/Datensatz/Training/image_2"  # Test images
#     test_mask_dir = "S:/Masterarbeit/Datensatz/Training/semantic"  # Test masks
# end
# checkpoint_path = "S:/Masterarbeit/checkpoints/final_model.bson"  # Path to your saved model

if use_masks
    test_img_dir = "/local/home/baselt/Datensatz/Testing/image_2"  # Test images
else
    test_img_dir = "/local/home/baselt/Datensatz/Training/image_2"  # Test images
    test_mask_dir = "/local/home/baselt/Datensatz/Training/semantic"  # Test masks
end
checkpoint_path = "/local/home/baselt/checkpoints/final_model.bson"  # Path to your saved model

um_classes = 35  # Number of classes in your segmentation model
results_dir = "results_presentation"  # Output directory
num_examples = 5  # Number of examples to visualize


# Clear GPU memory
function clear_gpu_memory()
    GC.gc()
    CUDA.reclaim()
end

# === HELPER FUNCTIONS ===
# Downsample array to make visualization more manageable
function downsample_array(arr, factor=4)
    h, w = size(arr)
    new_h, new_w = div(h, factor), div(w, factor)
    
    # Ensure dimensions are at least 1
    new_h = max(1, new_h)
    new_w = max(1, new_w)
    
    # Create downsampled array
    result = similar(arr, new_h, new_w)
    
    for i in 1:new_h
        for j in 1:new_w
            # Get region in original array
            i_start = (i-1)*factor + 1
            j_start = (j-1)*factor + 1
            i_end = min(i_start + factor - 1, h)
            j_end = min(j_start + factor - 1, w)
            
            # For class labels (Int), use mode (most common value)
            if eltype(arr) <: Integer
                # Extract block and find most common value
                block = arr[i_start:i_end, j_start:j_end]
                # Count occurrences
                counts = Dict{eltype(arr), Int}()
                for val in block
                    counts[val] = get(counts, val, 0) + 1
                end
                # Find most common
                if !isempty(counts)
                    max_count = 0
                    max_val = first(keys(counts))
                    for (val, count) in counts
                        if count > max_count
                            max_count = count
                            max_val = val
                        end
                    end
                    result[i, j] = max_val
                else
                    result[i, j] = 0  # Default if empty
                end
            else
                # For floating point (image), use mean
                result[i, j] = mean(arr[i_start:i_end, j_start:j_end])
            end
        end
    end
    
    return result
end

# Visualize example with flexibility for mask presence
function visualize_example(img, mask, pred, index, dir, has_mask=true)
    # Get prediction classes
    _, pred_indices = findmax(pred, dims=3)
    pred_classes = dropdims(getindex.(pred_indices, 3), dims=3) .- 1
    
    # Get ground truth if available
    if has_mask
        gt_classes = dropdims(cpu(mask), dims=(3,4))
        # Calculate accuracy before downsampling
        accuracy = mean(pred_classes .== gt_classes)
        # Create difference map
        diff_map = (pred_classes .== gt_classes)
    else
        # No ground truth available
        accuracy = NaN
        gt_classes = similar(pred_classes)
        fill!(gt_classes, 0)  # Empty ground truth
        diff_map = similar(pred_classes, Bool)
        fill!(diff_map, false)  # Empty diff map
    end
    
    # Convert image for display (take first channel if needed)
    if size(img, 3) > 1
        img_display = Float32.(cpu(img[:,:,1,1]))
    else
        img_display = Float32.(cpu(img[:,:,1,1]))
    end
    
    # Downsample for visualization to avoid stack overflow
    ds_factor = 8  # Increased to ensure smaller arrays
    img_small = downsample_array(img_display, ds_factor)
    pred_small = downsample_array(pred_classes, ds_factor)
    
    if has_mask
        gt_small = downsample_array(gt_classes, ds_factor)
        diff_small = downsample_array(diff_map, ds_factor)
    else
        # Create empty placeholders with same dimensions
        gt_small = similar(pred_small)
        fill!(gt_small, 0)
        diff_small = similar(pred_small, Bool)
        fill!(diff_small, false)
    end
    
    # Flip the arrays to correct the orientation
    img_small = reverse(img_small, dims=1)
    pred_small = reverse(pred_small, dims=1)
    gt_small = reverse(gt_small, dims=1)
    diff_small = reverse(diff_small, dims=1)
    
    # Create visualizations with better color scheme
    p1 = heatmap(img_small, title="Input Image", c=:grays, aspect_ratio=:equal)
    p3 = heatmap(pred_small, title="Prediction", c=:viridis, aspect_ratio=:equal)
    
    if has_mask
        # Include ground truth and difference map if masks are available
        p2 = heatmap(gt_small, title="Ground Truth", c=:viridis, aspect_ratio=:equal)
        p4 = heatmap(diff_small, title="Correct Predictions", c=:RdYlGn, aspect_ratio=:equal)
        title_text = "Example $index (Accuracy: $(round(accuracy*100, digits=2))%)"
    else
        # For no masks, show prediction with different angles/views
        p2 = heatmap(pred_small, title="Prediction (alt. colormap)", c=:turbo, aspect_ratio=:equal)
        p4 = plot(title="No ground truth available", grid=false, showaxis=false, ticks=false)
        title_text = "Example $index (No ground truth)"
    end
    
    # Create combined plot
    p_combined = plot(p1, p2, p3, p4, layout=(2, 2), size=(800, 600), title=title_text)
    
    # Save the plot
    savefig(p_combined, joinpath(dir, "example_$(index).png"))
    
    # Save full-resolution visualization as image
    save_summary_image(img_display, gt_classes, pred_classes, diff_map, index, dir, has_mask)
end

# Save full resolution visualization as image
function save_summary_image(img, ground_truth, prediction, diff_map, index, dir, has_mask=true)
    # Normalize image to 0-1 range
    img_norm = (img .- minimum(img)) ./ max(maximum(img) - minimum(img), 1e-6)
    
    # Create colormap for ground truth and prediction
    # Use ColorSchemes to create a consistent color mapping
    cmap = get(ColorSchemes.viridis, range(0, 1, length=num_classes))
    
    # Function to get color for a class ID
    function class_to_color(class_id, max_class=num_classes-1)
        # Map class_id to a color from the colormap
        # Normalize class_id to 0-1 range
        normalized_id = clamp(class_id / max_class, 0, 1)
        return get(ColorSchemes.viridis, normalized_id)
    end
    
    # Convert image to RGB
    img_rgb = Gray.(img_norm)
    
    # Convert prediction map to RGB
    h, w = size(prediction)
    pred_rgb = Array{RGB{Float32}}(undef, h, w)
    for i in 1:h
        for j in 1:w
            pred_rgb[i,j] = class_to_color(prediction[i,j])
        end
    end
    
    # Handle ground truth and difference map based on availability
    if has_mask
        # Convert class maps to RGB images
        gt_rgb = Array{RGB{Float32}}(undef, h, w)
        for i in 1:h
            for j in 1:w
                gt_rgb[i,j] = class_to_color(ground_truth[i,j])
            end
        end
        
        # Convert difference map to RGB
        diff_rgb = Array{RGB{Float32}}(undef, h, w)
        for i in 1:h
            for j in 1:w
                diff_rgb[i,j] = diff_map[i,j] ? RGB(0.0, 0.8, 0.0) : RGB(0.8, 0.0, 0.0)
            end
        end
    else
        # Create empty placeholders
        gt_rgb = Array{RGB{Float32}}(undef, h, w)
        fill!(gt_rgb, RGB(0.0, 0.0, 0.0))
        
        # For the case without ground truth, use a duplicate of prediction with alternate colormap
        diff_rgb = Array{RGB{Float32}}(undef, h, w)
        for i in 1:h
            for j in 1:w
                normalized_id = clamp(prediction[i,j] / (num_classes-1), 0, 1)
                diff_rgb[i,j] = get(ColorSchemes.turbo, normalized_id)
            end
        end
    end
    
    try
        # Determine output dimensions
        # For very large images, downsample to a reasonable size for disk storage
        max_dim = 1024  # Maximum dimension for output image
        
        orig_h, orig_w = h, w
        scale_factor = min(1.0, max_dim / max(h, w))
        
        if scale_factor < 1.0
            # Need to resize for output
            new_h = round(Int, h * scale_factor)
            new_w = round(Int, w * scale_factor)
            
            # Resize the components
            img_rgb_resized = imresize(img_rgb, (new_h, new_w))
            pred_rgb_resized = imresize(pred_rgb, (new_h, new_w))
            
            if has_mask
                gt_rgb_resized = imresize(gt_rgb, (new_h, new_w))
                diff_rgb_resized = imresize(diff_rgb, (new_h, new_w))
            else
                # For no masks, use alternate visualization
                gt_rgb_resized = imresize(diff_rgb, (new_h, new_w))  # Use alternate colormap viz
                diff_rgb_resized = Array{RGB{Float32}}(undef, new_h, new_w)
                fill!(diff_rgb_resized, RGB(0.0, 0.0, 0.0))
            end
            
            # Update dimensions
            h, w = new_h, new_w
        else
            # No resizing needed
            img_rgb_resized = img_rgb
            pred_rgb_resized = pred_rgb
            
            if has_mask
                gt_rgb_resized = gt_rgb
                diff_rgb_resized = diff_rgb
            else
                gt_rgb_resized = diff_rgb  # Use alternate colormap viz
                diff_rgb_resized = Array{RGB{Float32}}(undef, h, w)
                fill!(diff_rgb_resized, RGB(0.0, 0.0, 0.0))
            end
        end
        
        # Create a 2x2 grid
        composite = Array{RGB{Float32}}(undef, h*2, w*2)
        
        # Fill the panels
        composite[1:h, 1:w] .= img_rgb_resized
        composite[1:h, (w+1):(2*w)] .= has_mask ? gt_rgb_resized : pred_rgb_resized
        composite[(h+1):(2*h), 1:w] .= pred_rgb_resized
        
        if has_mask
            composite[(h+1):(2*h), (w+1):(2*w)] .= diff_rgb_resized
        else
            # Use alternative visualization when no masks - prediction with alternative colormap
            composite[(h+1):(2*h), (w+1):(2*w)] .= gt_rgb_resized
        end
        
        # Save the image
        save(joinpath(dir, "fullres_example_$(index).png"), composite)
        
        println("Saved full-resolution visualization for example $index")
        
    catch e
        println("Error saving full-resolution image: $e")
        println("Falling back to individual image saving...")
        
        # Save individual components
        try
            save(joinpath(dir, "img_$(index).png"), img_rgb)
            save(joinpath(dir, "pred_$(index).png"), pred_rgb)
            
            if has_mask
                save(joinpath(dir, "gt_$(index).png"), gt_rgb)
                save(joinpath(dir, "diff_$(index).png"), diff_rgb)
            end
        catch e2
            println("Error saving individual images: $e2")
        end
    end
end

# Load image dataset without masks
function load_image_dataset(image_files, num_examples)
    println("Loading $(length(image_files)) images without masks")
    
    dataset = []
    
    for i in 1:min(num_examples, length(image_files))
        img_path = image_files[i]
        println("Loading image $i: $(basename(img_path))")
        
        # Load and preprocess image - using Data module's function
        img_data = Data.load_and_preprocess_image(img_path)
        
        # Create a dummy mask (all zeros) with the correct dimensions
        h, w, c, n = size(img_data)
        dummy_mask = zeros(Int, h, w, 1, 1)
        
        push!(dataset, (img_data, dummy_mask))
    end
    
    return dataset
end

# Main function to run the evaluation
function main(use_masks=true)
    # Create results directory
    mkpath(results_dir)

    # === LOAD MODEL ===
    println("Loading model from $checkpoint_path...")
    @load checkpoint_path model epoch loss
    model = gpu(model)  # Move to GPU

    # === LOAD TEST DATA ===
    println("Loading test data...")
    test_files = sort(readdir(test_img_dir, join=true))[1:min(num_examples, length(readdir(test_img_dir)))]
    
    if use_masks
        mask_files = sort(readdir(test_mask_dir, join=true))[1:min(num_examples, length(readdir(test_mask_dir)))]
        test_dataset = Data.load_dataset(test_files, mask_files)
        println("Using images with corresponding masks for evaluation")
    else
        test_dataset = load_image_dataset(test_files, num_examples)
        println("Using images without masks for visualization only")
    end

    # === PERFORMANCE METRICS ===
    if use_masks
        println("Calculating performance metrics...")
        
        # Initialize metrics
        total_accuracy = 0.0
        class_accuracies = zeros(Float32, num_classes)
        class_counts = zeros(Int, num_classes)
    end

    # Process each test image
    for (i, (img, mask)) in enumerate(test_dataset)
        println("Processing test example $i...")
        
        # Add batch dimension if needed
        if ndims(img) == 3
            img = reshape(img, size(img)..., 1)
        end
        
        # Move to GPU for inference
        img_gpu = gpu(img)
        
        # Run prediction
        pred_raw = model(img_gpu)
        pred_raw_cpu = cpu(pred_raw)
        
        # Get class predictions
        _, pred_indices = findmax(pred_raw_cpu, dims=3)
        pred_classes = dropdims(getindex.(pred_indices, 3), dims=3) .- 1
        
        # Calculate metrics if masks are available
        if use_masks
            # Get ground truth classes
            gt_classes = dropdims(cpu(mask), dims=(3,4))
            
            # Calculate accuracy
            accuracy = mean(pred_classes .== gt_classes)
            total_accuracy += accuracy
            
            # Calculate per-class accuracy
            for c in 0:(num_classes-1)
                class_mask = gt_classes .== c
                class_count = sum(class_mask)
                if class_count > 0
                    class_acc = sum((pred_classes .== c) .& class_mask) / class_count
                    class_accuracies[c+1] += class_acc
                    class_counts[c+1] += 1
                end
            end
        end
        
        # Visualize this example
        visualize_example(img, mask, pred_raw_cpu, i, results_dir, use_masks)
        
        # Clean up GPU memory
        clear_gpu_memory()
    end

    # Generate metrics report if masks were used
    if use_masks
        # Calculate final metrics
        mean_accuracy = total_accuracy / length(test_dataset)
        mean_class_accuracies = zeros(Float32, num_classes)
        for c in 1:num_classes
            if class_counts[c] > 0
                mean_class_accuracies[c] = class_accuracies[c] / class_counts[c]
            end
        end
        
        # Filter out NaN values before calculating mean
        valid_accuracies = filter(!isnan, mean_class_accuracies)
        mean_iou = mean(valid_accuracies)

        # === GENERATE SUMMARY REPORT ===
        println("Generating summary report...")

        # Create summary plot - use only classes that appeared in the test set
        valid_classes = findall(class_counts .> 0) .- 1
        valid_accuracies_plot = mean_class_accuracies[class_counts .> 0]
        
        p1 = bar(valid_classes, valid_accuracies_plot, 
            xlabel="Class ID", ylabel="Accuracy", 
            title="Per-class Accuracy", 
            legend=false,
            ylims=(0, 1))

        # Save metrics to file
        open(joinpath(results_dir, "metrics_summary.txt"), "w") do io
            println(io, "===== SEMANTIC SEGMENTATION MODEL RESULTS =====")
            println(io, "Model evaluated on $(length(test_dataset)) test images")
            println(io, "")
            println(io, "OVERALL METRICS:")
            println(io, "Mean Pixel Accuracy: $(round(mean_accuracy*100, digits=2))%")
            println(io, "Mean IoU: $(round(mean_iou*100, digits=2))%")
            println(io, "")
            println(io, "TOP 5 BEST PERFORMING CLASSES:")
            top_classes = sortperm(mean_class_accuracies, rev=true)[1:min(5, num_classes)]
            for i in 1:length(top_classes)
                c = top_classes[i]
                if class_counts[c] > 0
                    println(io, "  Class $(c-1): $(round(mean_class_accuracies[c]*100, digits=2))%")
                end
            end
            println(io, "")
            println(io, "BOTTOM 5 WORST PERFORMING CLASSES:")
            bottom_classes = sortperm(mean_class_accuracies)[1:min(5, num_classes)]
            for i in 1:length(bottom_classes)
                c = bottom_classes[i]
                if class_counts[c] > 0  # Only include classes that appear in the test set
                    println(io, "  Class $(c-1): $(round(mean_class_accuracies[c]*100, digits=2))%")
                end
            end
        end

        # Save class accuracy plot
        savefig(p1, joinpath(results_dir, "class_accuracies.png"))

        println("Results saved to $results_dir")
        println("Overall Mean Accuracy: $(round(mean_accuracy*100, digits=2))%")
        println("Mean IoU: $(round(mean_iou*100, digits=2))%")
    else
        println("Visualization complete. No metrics calculated (masks not provided).")
    end
end

# Run the main function with the use_masks configuration
main(use_masks)