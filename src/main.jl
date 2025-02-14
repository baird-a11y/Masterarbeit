##############################
# Main.jl
##############################
include("Data.jl")
include("Model.jl")
include("Training.jl")
include("Visualization.jl")

using Test
using Flux
using Flux: onehotbatch, logitcrossentropy
using .Model
using .Data
using .Training
using .Visualization
using Statistics
using FileIO
using Images
using LinearAlgebra
using Optimisers

# Beispiel-Daten laden
img_path = "G:/Meine Ablage/Geowissenschaften/Masterarbeit/Masterarbeit/Datensatz/Training/Bilder_1/000101_10.png"
mask_path = "G:/Meine Ablage/Geowissenschaften/Masterarbeit/Masterarbeit/Datensatz/Training/Masken_1/000101_10.png"

# Bild und Maske laden und vorverarbeiten
input_image = Data.load_and_preprocess_image(img_path)
ground_truth = Data.load_and_preprocess_label(mask_path)
ground_truth_int = Int.(ground_truth[:, :, 1, :])
ground_truth_oh = permutedims(onehotbatch(ground_truth_int, 0:34), (2,3,1,4))
ground_truth_oh = Float32.(ground_truth_oh)

# Modell initialisieren (3 Eingangskanäle, 35 Ausgangskanäle)
model = Model.UNet(3, 35)

# Mit Optimisers.jl: Initialisiere den Optimierer-Zustand (hier Adam mit Lernrate 0.001)
opt_state = Optimisers.setup(Optimisers.Adam(0.001), model)

# Schritt 1a: Berechne den Loss für den aktuellen Batch
loss_value = logitcrossentropy(model(input_image), ground_truth_oh)
println("Initialer Loss: ", loss_value)

# Schritt 1b: Berechne die Gradienten mit expliziter Übergabe des Modells
∇model = gradient(m -> logitcrossentropy(m(input_image), ground_truth_oh), model)[1]
println("Gradienten berechnet.")

# Debug: Ausgabe der Gradienten-Normen für jeden Parameter
for p in Flux.params(model)
    println("Gradient norm (Parameter mit Größe ", size(p), "): ", norm(cpu(∇model[p])))
end

println("Beispiel-Gewicht vor Update: ", cpu(first(Flux.params(model))))

# Schritt 1c: Update: Aktualisiere Optimizer-Zustand und Modell explizit
opt_state, model = Optimisers.update!(opt_state, model, ∇model)
println("Beispiel-Gewicht nach Update: ", cpu(first(Flux.params(model))))

# Optional: Visualisierung der Ergebnisse
Visualization.visualize_results(model, input_image, ground_truth)
