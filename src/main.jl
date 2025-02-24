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
using Dates


num_epochs = 10 # Number of epochs to train
learning_rate = 0.001 # Learning rate
input_channels = 3 # RGB input channels
batch_size = 4 # Desired batch size


# Define the directories for images and masks
img_dir = "S:/Masterarbeit/Datensatz/Training/image_2"
mask_dir = "S:/Masterarbeit/Datensatz/Training/semantic"
mask_files = sort(readdir(mask_dir, join=true))

# Load the dataset as an array of (input_image, ground_truth) tuples.
dataset = Data.load_dataset(img_dir, mask_dir)
println("Number of samples in dataset: ", length(dataset))
# Create batches from the dataset.
train_data = Data.create_batches(dataset, batch_size)
println("Number of batches: ", length(train_data))

global_max_class = 0
# for mask_path in mask_files
#     _, max_class = Data.load_and_preprocess_label(mask_path)
#     global_max_class = max(global_max_class, max_class)
#     println("Loaded mask from ", mask_path, " with largest class: ", max_class)
#     println("Global max class: ", global_max_class)
# end

# Da die Klassen von 0 bis max_class gehen, entspricht die Anzahl der Klassen global_max_class + 1.
output_channels = 35
println("Overall output channels (classes): ", output_channels)
    
model = Model.UNet(input_channels, output_channels)

start_time = now() 
losses = Training.train_unet(model, train_data, num_epochs, learning_rate, output_channels)

# Use the first sample from the dataset for visualization.
input_image = dataset[1][1]
ground_truth = dataset[1][2]


# Optionally visualize the updated predictions
Visualization.visualize_results(model, input_image, ground_truth,losses)

end_time = now() 
println("Training took ", end_time - start_time, " seconds.") 