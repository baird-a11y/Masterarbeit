##############################
# Visualization.jl
##############################
module Visualization

using Plots
using CUDA

function visualize_results(model, input_image, ground_truth, losses)
    # Ensure input data is on CPU for plotting.
    input_image = cpu(input_image)
    ground_truth = cpu(ground_truth)
    
    # Run prediction on GPU model and then transfer to CPU
    prediction = cpu(model(Flux.gpu(input_image)))
    
    # Remove batch dimension if present
    input_image = input_image[:, :, 1, 1]
    ground_truth = ground_truth[:, :, 1, 1]
    prediction = prediction[:, :, 1, 1]
    
    # Optionally, reverse if necessary (depends on your data)
    input_image = reverse(input_image, dims=1)
    ground_truth = reverse(ground_truth, dims=1)
    prediction = reverse(prediction, dims=1)
    
    result = plot(
        heatmap(input_image, title="Input Image", color=:viridis),
        heatmap(ground_truth, title="Ground Truth Mask", color=:viridis),
        heatmap(prediction, title="Predicted Mask", color=:viridis),
        layout=(1, 3),
        size=(900, 300)
    )
    savefig(result, "S:/Masterarbeit/results/result.png")
    
    p = scatter(1:length(losses), losses, xlabel="Epoch", ylabel="Loss", title="Loss Over Time", marker=:o)
    savefig(p, "S:/Masterarbeit/results/losses.png")
end

end # module Visualization
