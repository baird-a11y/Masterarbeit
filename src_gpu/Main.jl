##################################
# Main.jl - Memory-Optimized Version
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
import Base.GC

# Hyperparameter
num_epochs = 2       # Anzahl Trainings-Epochen
learning_rate = 0.001
input_channels = 3   # z.B. RGB
batch_size = 2       # Reduced batch size from 4 to 2
use_mixed_precision = true  # Enable mixed precision to reduce memory usage
checkpoint_dir = "checkpoints"  # Directory to save model checkpoints
checkpoint_freq = 1  # Save checkpoint every N epochs

# GPU memory management
function clear_gpu_memory()
    GC.gc()
    CUDA.reclaim()
    println("GPU memory cleared")
end

# Verzeichnisse
img_dir = "S:/Masterarbeit/Datensatz/Training/image_2"
mask_dir = "S:/Masterarbeit/Datensatz/Training/semantic"

println("Loading dataset...")
# Datensatz laden - option 1: load all data at once
# dataset = Data.load_dataset(img_dir, mask_dir)

# Option 2: Load a subset of data for testing
image_files = sort(readdir(img_dir, join=true))
label_files = sort(readdir(mask_dir, join=true))
subset_size = min(50, length(image_files))  # Limit to 50 images or fewer

println("Loading subset of $subset_size images (out of $(length(image_files)) total)")
subset_img_files = image_files[1:subset_size]
subset_label_files = label_files[1:subset_size]

dataset = Data.load_dataset(subset_img_files, subset_label_files)
println("Number of samples in dataset: ", length(dataset))

clear_gpu_memory()

# Batches erstellen
train_data = Data.create_batches(dataset, batch_size)
println("Number of batches: ", length(train_data))

# Anzahl Output-Kanäle (Klassenanzahl). In dem Beispiel 35.
output_channels = 35
println("Overall output channels (classes): ", output_channels)

# UNet-Modell erstellen - using smaller feature maps to reduce memory
# Standard UNet with full feature maps
# model = Model.UNet(input_channels, output_channels, memory_efficient=false)

# Memory-efficient UNet with reduced feature maps
model = Model.UNet(input_channels, output_channels, memory_efficient=true)
println("UNet model created")

# --------------------------------------------------
# Modell auf die GPU verschieben:
# --------------------------------------------------
model = gpu(model)
println("Model moved to GPU")

clear_gpu_memory()

# Create checkpoint directory
mkpath(checkpoint_dir)

start_time = now()
println("Starting training at ", start_time)

# Training - use mixed precision to reduce memory usage
if use_mixed_precision
    println("Using mixed precision training")
    model, losses, val_losses = Training.train_unet_mixed_precision(
        model, train_data, num_epochs, learning_rate, output_channels;
        checkpoint_dir=checkpoint_dir, checkpoint_freq=checkpoint_freq
    )
else
    println("Using standard precision training")
    model, losses, val_losses = Training.train_unet(
        model, train_data, num_epochs, learning_rate, output_channels;
        checkpoint_dir=checkpoint_dir, checkpoint_freq=checkpoint_freq
    )
end

end_time = now()
println("Training took ", end_time - start_time)

clear_gpu_memory()

# Beispielhafte Visualisierung mit dem 1. Sample
input_image = dataset[1][1]
ground_truth = dataset[1][2]
Visualization.visualize_results(model, input_image, ground_truth, losses; 
                              save_path="results")

# Visualize model architecture
Visualization.visualize_model_architecture(model)

# Visualize training metrics
Visualization.visualize_training_metrics(losses)

println("Done!")