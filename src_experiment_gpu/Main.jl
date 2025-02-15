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
using CUDA  # <-- load CUDA to use GPU

num_epochs = 1         # Number of epochs to train
learning_rate = 0.001  # Learning rate
input_channels = 3     # RGB input channels
batch_size = 4         # Desired batch size

# Define the directories for images and masks
img_dir = "G:/Meine Ablage/Geowissenschaften/Masterarbeit/Masterarbeit/Datensatz/Training/Bilder_alle"
mask_dir = "G:/Meine Ablage/Geowissenschaften/Masterarbeit/Masterarbeit/Datensatz/Training/Masken_alle"
mask_files = sort(readdir(mask_dir, join=true))

# Load the dataset as an array of (input_image, ground_truth) tuples.
dataset = Data.load_dataset(img_dir, mask_dir)
println("Number of samples in dataset: ", length(dataset))
# Create batches from the dataset.
train_data = Data.create_batches(dataset, batch_size)
println("Number of batches: ", length(train_data))

# Determine the overall maximum class from all masks.
global_max_class = 0
for mask_path in mask_files
    _, max_class = Data.load_and_preprocess_label(mask_path)
    global global_max_class  # explizit angeben, dass wir die globale Variable ändern
    global_max_class = max(global_max_class, max_class)
    println("Loaded mask from ", mask_path, " with largest class: ", max_class)
end

# Since classes range from 0 to global_max_class, the number of output channels is:
output_channels = global_max_class + 1
println("Overall output channels (classes): ", output_channels)

    
# Initialize the model and move it to the GPU.
model = Model.UNet(input_channels, output_channels)
model = Flux.gpu(model)

# (Optional) If you want to use a specific sample for visualization later,
# you can extract it from your dataset:
input_image = dataset[1][1]  # first sample's image
ground_truth = dataset[1][2] # first sample's label

# Run the training loop (the training function will also move batch data to GPU)
losses = Training.train_unet(model, train_data, num_epochs, learning_rate, output_channels)

# For visualization, convert the data back to the CPU for plotting.
Visualization.visualize_results(model, cpu(input_image), cpu(ground_truth), losses)
println("Done!")