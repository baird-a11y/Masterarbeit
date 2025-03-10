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
num_epochs = 1       # Anzahl Trainings-Epochen
learning_rate = 0.001
input_channels = 3   # z.B. RGB
batch_size = 2       # Reduced batch size from 4 to 2
use_mixed_precision = false  # Enable mixed precision to reduce memory usage
# checkpoint_dir = "checkpoints"  # Directory to save model checkpoints
checkpoint_dir = "/local/home/baselt/checkpoints"  # Directory to save model checkpoints
checkpoint_freq = 1  # Save checkpoint every N epochs
sub_size = 50  # Number of samples to load for trainin
sub_set = false  # Load entire dataset or a subset
# GPU memory management
function clear_gpu_memory()
    GC.gc()
    CUDA.reclaim()
    println("GPU memory cleared")
end
#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

# labels = [
#     #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
#     Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
#     Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
#     Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
#     Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
#     Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
#     Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
#     Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
#     Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
#     Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
#     Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
#     Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
#     Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
#     Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
#     Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
#     Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
#     Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
#     Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
#     Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
#     Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
#     Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
#     Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
#     Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
#     Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
#     Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
#     Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
#     Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
#     Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
#     Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
#     Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
#     Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
# ]
# # Verzeichnisse
# img_dir = "S:/Masterarbeit/Datensatz/Training/image_2"
# mask_dir = "S:/Masterarbeit/Datensatz/Training/semantic"

# Verzeichnisse
img_dir = "/local/home/baselt/Datensatz/Training/image_2"
mask_dir = "/local/home/baselt/Datensatz/Training/semantic_rgb"
checkpoint_dir = "/local/home/baselt/checkpoints"

# Ich will wissen wie viele Bilder in img_dir sind
println("Number of images in img_dir: ", length(readdir(img_dir)))
println("Number of images in mask_dir: ", length(readdir(mask_dir)))



println("Loading dataset...")

if sub_set
    image_files = sort(readdir(img_dir, join=true))
    label_files = sort(readdir(mask_dir, join=true))
    subset_size = min(sub_size, length(image_files))  # Limit to 50 images or fewer
    println("Loading subset of $subset_size images (out of $(length(image_files)) total)")
    subset_img_files = image_files[1:subset_size]
    subset_label_files = label_files[1:subset_size]
    dataset = Data.load_dataset(subset_img_files, subset_label_files)
else
    dataset = Data.load_dataset(img_dir, mask_dir)
end

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

#include("Results.jl")
# run(Results.jl)
println("Done!")
