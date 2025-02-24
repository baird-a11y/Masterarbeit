##################################
# Main.jl
##################################

# Einbinden der anderen Dateien
include("Data.jl")
include("Model.jl")
include("Training.jl")
include("Visualization.jl")

using Test
using Flux
using Flux: onehotbatch, logitcrossentropy, gpu
using CUDA  # <-- GPU-Support
using .Model
using .Data
using .Training
using .Visualization
using Statistics
using FileIO
using Images
using LinearAlgebra
using Optimisers
using Plots
using Dates

# Hyperparameter
num_epochs = 2       # Anzahl Trainings-Epochen
learning_rate = 0.001
input_channels = 3   # z.B. RGB
batch_size = 4

# Verzeichnisse
img_dir = "S:/Masterarbeit/Datensatz/Training/image_2"
mask_dir = "S:/Masterarbeit/Datensatz/Training/semantic"

# Datensatz laden
dataset = Data.load_dataset(img_dir, mask_dir)
println("Number of samples in dataset: ", length(dataset))

# Batches erstellen
train_data = Data.create_batches(dataset, batch_size)
println("Number of batches: ", length(train_data))

# Anzahl Output-Kanäle (Klassenanzahl). In dem Beispiel 35.
output_channels = 35
println("Overall output channels (classes): ", output_channels)

# UNet-Modell erstellen
model = Model.UNet(input_channels, output_channels)

# --------------------------------------------------
# Modell auf die GPU verschieben:
# --------------------------------------------------
model = gpu(model)

start_time = now()

# Training
losses = Training.train_unet(model, train_data, num_epochs, learning_rate, output_channels)

end_time = now()
println("Training took ", end_time - start_time, " seconds.")

# Beispielhafte Visualisierung mit dem 1. Sample
input_image = dataset[1][1]
ground_truth = dataset[1][2]
Visualization.visualize_results(model, input_image, ground_truth, losses)

println("Done!")
