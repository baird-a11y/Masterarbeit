module Visualization

using Plots
using Flux

##############################################################################
# 1) Bestehende Visualisierung (Mehrklassen-Fall)
#    -> Zeigt Input, Ground Truth, Prediction als Heatmaps
##############################################################################
function visualize_results(model, input_image, ground_truth, losses)
    # Wir gehen davon aus, dass die Daten bereits auf der CPU liegen.
    # Wenn du 100% sicherstellen willst, dass es CPU-Daten sind, 
    # könntest du input_image = cpu(input_image) machen.
    # Hier entfernen wir sämtliche GPU-Aufrufe.

    # Modellvorhersage auf CPU
    prediction = model(input_image)

    # Remove batch dimension if present
    # Angenommen: (H, W, 1, 1) oder (H, W, C, 1)
    # Wir nehmen den 1. Kanal und 1. Batch
    input_2d      = input_image[:, :, 1, 1]
    ground_truth2d = ground_truth[:, :, 1, 1]
    prediction2d  = prediction[:, :, 1, 1]

    # Optionales Reverse, je nach Daten-Orientierung
    input_2d       = reverse(input_2d, dims=1)
    ground_truth2d = reverse(ground_truth2d, dims=1)
    prediction2d   = reverse(prediction2d, dims=1)

    # Plot
    result = plot(
        heatmap(input_2d, title="Input Image", color=:viridis),
        heatmap(ground_truth2d, title="Ground Truth Mask", color=:viridis),
        heatmap(prediction2d, title="Predicted Mask", color=:viridis),
        layout=(1, 3),
        size=(900, 300)
    )
    savefig(result, "S:/Masterarbeit/results/result.png")
    
    p = scatter(
        1:length(losses), losses,
        xlabel="Epoch", ylabel="Loss", 
        title="Loss Over Time", marker=:o
    )
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
- Model-Prediction auf CPU.
- Optionales Thresholding (0..1).
- Optionales Reverse im Y-Dim (abhängig von Deiner Daten-Orientierung).
"""

function visualize_synthetic_results(model, sample; threshold=0.2, reverse_y=false)
    input_image, target_image = sample

    # (H, W, C, 1)
    xb = reshape(input_image, size(input_image,1), size(input_image,2), size(input_image,3), 1)
    yb = reshape(target_image, size(target_image,1), size(target_image,2), size(target_image,3), 1)

    # Modellvorhersage (CPU-Variante)
    pred = model(xb)  # -> (H, W, 1, 1)

    # 2D-Ableitung
    pred_2d   = pred[:, :, 1, 1]
    frame_2d  = xb[:, :, 1, 1]  # Erster Kanal vom Input
    target_2d = yb[:, :, 1, 1]

    # -- Neu: Ausgabe min/max --
    println("Continuous prediction: min=", minimum(pred_2d), "  max=", maximum(pred_2d))

    # Threshold
    pred_bin = map(x -> x > threshold ? 1f0 : 0f0, pred_2d)

    if reverse_y
        frame_2d  = reverse(frame_2d, dims=1)
        target_2d = reverse(target_2d, dims=1)
        pred_2d   = reverse(pred_2d, dims=1)
        pred_bin  = reverse(pred_bin, dims=1)
    end

    plt1 = heatmap(frame_2d,   title="Input Frame",                 color=:viridis)
    plt2 = heatmap(target_2d,  title="Target Next Frame",           color=:viridis)
    plt3 = heatmap(pred_2d,    title="Predicted Next Frame (Cont.)",color=:viridis)
    plt4 = heatmap(pred_bin,   title="Predicted Next Frame (Thresh)", color=:viridis)

    result = plot(plt1, plt2, plt3, plt4, layout=(1,4), size=(1200,300))
    savefig(result, "S:/Masterarbeit/results/synthetic_result.png")
end

end # module Visualization
