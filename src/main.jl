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
using Plots

num_epochs = 10 # Number of epochs to train
learning_rate = 0.001 # Learning rate
input_channels = 3 # RGB input channels
batch_size = 4 # Desired batch size


# Define the directories for images and masks
img_dir = "G:/Meine Ablage/Geowissenschaften/Masterarbeit/Masterarbeit/Datensatz/Training/Bilder_alle"
mask_dir = "G:/Meine Ablage/Geowissenschaften/Masterarbeit/Masterarbeit/Datensatz/Training/Masken_alle"

# Load the dataset as an array of (input_image, ground_truth) tuples.
dataset = Data.load_dataset(img_dir, mask_dir)

# Create batches from the dataset.
train_data = Data.create_batches(dataset, batch_size)


ground_truth, num_classes = Data.load_and_preprocess_label(mask_path)
# num_classes gibt die Anzahl der Klassen an, die im Label vorhanden sind. Da man von 0 bis max geht muss man +1 rechnen
output_channels = num_classes+1 # Number of classes in the output mask
    
model = Model.UNet(input_channels, output_channels)
# Create a dummy training set containing one batch (the same image/mask pair)
train_data = [(input_image, ground_truth)]

# # Run the training loop for a specified number of epochs
# Training.train_unet(model, train_data, num_epochs, learning_rate, output_channels)

losses = Training.train_unet(model, train_data, num_epochs, learning_rate, output_channels)


p = scatter(1:num_epochs, losses, xlabel="Epoch", ylabel="Loss", title="Loss Over Time", marker=:o)

# Dient nur zur Visualisierung der Losses falls es ausreißer gibt
# scatter!(p, 5:7, losses[7:8], marker=:o)
# # scatter!(p, 9:10, losses[9:10], marker=:o)
display(p)

# # Optionally visualize the updated predictions
# Visualization.visualize_results(model, input_image, ground_truth)
