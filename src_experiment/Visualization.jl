module Visualization

using Plots
using CUDA

##############################################################################
# 1) Bestehende Visualisierung (Mehrklassen-Fall)
#    -> Zeigt Input, Ground Truth, Prediction als Heatmaps
##############################################################################
function visualize_results(model, input_image, ground_truth, losses)
    # Ensure input data is on CPU for plotting.
    input_image = cpu(input_image)
    ground_truth = cpu(ground_truth)
    
    # Run prediction on GPU model and then transfer to CPU
    prediction = cpu(model(Flux.gpu(input_image)))
    
    # Remove batch dimension if present
    # Annahme: (H, W, 1, 1) oder (H, W, C, 1)
    # Im ursprünglichen Code nimmst Du den 1. Kanal, die 1. Batch
    # für eine 2D-Heatmap:
    input_image = input_image[:, :, 1, 1]
    ground_truth = ground_truth[:, :, 1, 1]
    prediction   = prediction[:, :, 1, 1]
    
    # Evtl. Flip/Reihenfolge anpassen (abh. von Deinen Daten)
    input_image = reverse(input_image, dims=1)
    ground_truth = reverse(ground_truth, dims=1)
    prediction = reverse(prediction, dims=1)
    
    # Plot
    result = plot(
        heatmap(input_image, title="Input Image", color=:viridis),
        heatmap(ground_truth, title="Ground Truth Mask", color=:viridis),
        heatmap(prediction, title="Predicted Mask", color=:viridis),
        layout=(1, 3),
        size=(900, 300)
    )
    savefig(result, "S:/Masterarbeit/results/result.png")
    
    p = scatter(1:length(losses), losses,
        xlabel="Epoch", ylabel="Loss", title="Loss Over Time", marker=:o)
    savefig(p, "S:/Masterarbeit/results/losses.png")
end

##############################################################################
# 2) Neue Visualisierung (Ein-Kanal-Fall, z. B. Proof-of-Concept)
##############################################################################

"""
    visualize_synthetic_results(model, sample;
        threshold::Float32 = 0.5,
        reverse_y::Bool = false)

Zeigt ein Beispiel aus dem synthetischen Datensatz (bewegter Punkt) an:
- `sample` = (input_image, target_image).
- Model-Prediction via GPU.
- Optionales Thresholding (0..1).
- Optionales Reverse im Y-Dim (abhängig von Deiner Daten-Orientierung).
"""
function visualize_synthetic_results(model, sample; threshold=0.5, reverse_y=false)
    input_image, target_image = sample

    # Bringe die Daten auf CPU, 4D-Format => (H, W, C, 1)
    xb = reshape(input_image, size(input_image,1), size(input_image,2), size(input_image,3), 1)
    yb = reshape(target_image, size(target_image,1), size(target_image,2), size(target_image,3), 1)

    # GPU vor Vorhersage
    pred_gpu = model(Flux.gpu(xb))
    pred = cpu(pred_gpu)  # (H, W, 1, 1)

    # Reduziere auf 2D
    # Input hier 3-Kanal: [Frame, vx, vy], du könntest z. B. NUR den 1. Kanal visualisieren, 
    # oder eine Summe etc.:
    frame_channel = xb[:, :, 1, 1]             # "Frame_t"
    target_2d = yb[:, :, 1, 1]                # Ground Truth
    pred_2d   = pred[:, :, 1, 1]              # Prediction

    # Optional threshold
    pred_bin = map(x -> x > threshold ? 1f0 : 0f0, pred_2d)

    # Evtl. reverse in y-Dim
    if reverse_y
        frame_channel = reverse(frame_channel, dims=1)
        target_2d = reverse(target_2d, dims=1)
        pred_2d   = reverse(pred_2d, dims=1)
        pred_bin  = reverse(pred_bin, dims=1)
    end

    # Plot
    plt1 = heatmap(frame_channel, title="Input Frame", color=:viridis)
    plt2 = heatmap(target_2d, title="Target Next Frame", color=:viridis)
    plt3 = heatmap(pred_2d, title="Predicted Next Frame (Continuous)", color=:viridis)
    plt4 = heatmap(pred_bin, title="Predicted Next Frame (Threshold)", color=:viridis)

    result = plot(plt1, plt2, plt3, plt4, layout=(1,4), size=(1200,300))
    savefig(result, "S:/Masterarbeit/results/synthetic_result.png")
end

end # module Visualization
