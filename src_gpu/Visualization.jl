##################################
# Visualization.jl
##################################
module Visualization

using Plots
using Flux: cpu

function visualize_results(model, input_image, ground_truth, losses)
    # Für Visualisierung (CPU-Seite)
    model_cpu = cpu(model)

    # Falls keine Batch-Dimension vorliegt
    if ndims(input_image) == 3
        input_image = reshape(input_image, size(input_image)..., 1)
    end

    # Prediction auf CPU
    input_cpu       = cpu(input_image)
    ground_truth_cpu = cpu(ground_truth)
    prediction_cpu   = model_cpu(input_cpu)

    # Dimensionen reduzieren
    img_for_plot   = dropdims(input_cpu[:,:,1,1], dims=3)      
    gt_for_plot    = dropdims(ground_truth_cpu[:,:,1,1], dims=3)
    pred_for_plot  = dropdims(prediction_cpu[:,:,1,1], dims=3)

    # Für Demo ggf. flip/reverse
    img_for_plot   = reverse(img_for_plot, dims=1)
    gt_for_plot    = reverse(gt_for_plot, dims=1)
    pred_for_plot  = reverse(pred_for_plot, dims=1)

    # Heatmaps
    result = plot(
        heatmap(img_for_plot, title="Input Image", color=:viridis),
        heatmap(gt_for_plot,  title="Ground Truth Mask", color=:viridis),
        heatmap(pred_for_plot, title="Predicted Mask", color=:viridis),
        layout=(1, 3),
        size=(900, 300)
    )
    savefig(result, "S:/Masterarbeit/results/result.png")

    # Loss-Verlauf
    p = plot(1:length(losses), losses, xlabel="Epoch", ylabel="Loss",
             title="Loss Over Time", marker=:o)
    savefig(p, "S:/Masterarbeit/results/losses.png")
end

end # module Visualization
