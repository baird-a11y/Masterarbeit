##################################
# Main.jl - Updated for optimized modules
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
use_mixed_precision = false  # Set to true for faster training on compatible GPUs
checkpoint_dir = "checkpoints"  # Directory to save model checkpoints
checkpoint_freq = 1  # Save checkpoint every N epochs

# Verzeichnisse
img_dir = "S:/Masterarbeit/Datensatz/Training/image_2"
mask_dir = "S:/Masterarbeit/Datensatz/Training/semantic"

println("Loading dataset...")
# Datensatz laden
dataset = Data.load_dataset(img_dir, mask_dir)
println("Number of samples in dataset: ", length(dataset))

# Optional: Create augmented dataset
# Uncomment to use data augmentation
# augmented_dataset = Data.create_augmented_dataset(dataset, 2)
# println("Number of samples after augmentation: ", length(augmented_dataset))
# dataset = augmented_dataset

# Batches erstellen
train_data = Data.create_batches(dataset, batch_size)
println("Number of batches: ", length(train_data))

# Anzahl Output-Kanäle (Klassenanzahl). In dem Beispiel 35.
output_channels = 35
println("Overall output channels (classes): ", output_channels)

# Calculate class weights for imbalanced dataset (optional)
# Uncomment to use class weighting
# class_weights = Training.calculate_class_weights(dataset, output_channels)
# println("Calculated class weights for imbalanced dataset")

# UNet-Modell erstellen
model = Model.UNet(input_channels, output_channels)
println("UNet model created")

# --------------------------------------------------
# Modell auf die GPU verschieben:
# --------------------------------------------------
model = gpu(model)
println("Model moved to GPU")

# Create checkpoint directory
mkpath(checkpoint_dir)

start_time = now()
println("Starting training at ", start_time)

# Training - choose standard or mixed precision
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

# Beispielhafte Visualisierung mit dem 1. Sample
input_image = dataset[1][1]
ground_truth = dataset[1][2]
Visualization.visualize_results(model, input_image, ground_truth, losses; 
                              save_path="results")

# Visualize class distribution (optional)
# Visualization.visualize_class_distribution(dataset, output_channels)

# Visualize model architecture
Visualization.visualize_model_architecture(model)

# Visualize training metrics
Visualization.visualize_training_metrics(losses)

println("Done!")