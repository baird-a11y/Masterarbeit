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
img_dir = "G:/Meine Ablage/Geowissenschaften/Masterarbeit/Masterarbeit/Datensatz/Training/Bilder_5"
mask_dir = "G:/Meine Ablage/Geowissenschaften/Masterarbeit/Masterarbeit/Datensatz/Training/Masken_5"

# Load the dataset as an array of (input_image, ground_truth) tuples.
dataset = Data.load_dataset(img_dir, mask_dir)
println("Loaded $(length(dataset)) samples.")

# Create batches from the dataset.
train_data = Data.create_batches(dataset, batch_size)
println("Created $(length(train_data)) batches from the dataset.")

ground_truth, num_classes = Data.load_and_preprocess_label(mask_path)
output_channels = num_classes # Number of classes in the output mask
    
model = Model.UNet(input_channels, output_channels)
# Create a dummy training set containing one batch (the same image/mask pair)
train_data = [(input_image, ground_truth)]

# # Run the training loop for a specified number of epochs
# Training.train_unet(model, train_data, num_epochs, learning_rate, output_channels)

losses = Training.train_unet(model, train_data, num_epochs, learning_rate, output_channels)


# p = scatter(1:3, losses[2:4], xlabel="Epoch", ylabel="Loss", title="Loss Over Time", marker=:o)
# scatter!(p, 5:7, losses[7:8], marker=:o)
# # scatter!(p, 9:10, losses[9:10], marker=:o)
# display(p)

# # Optionally visualize the updated predictions
# Visualization.visualize_results(model, input_image, ground_truth)
