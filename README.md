# U-Net for Image Segmentation

This repository contains an implementation of the U-Net architecture for image segmentation tasks. The project is built using the Julia programming language with `Flux.jl` as the deep learning framework.

---

## Features

- **Customizable U-Net Architecture:**  
  Flexible definitions for encoder and decoder blocks, with skip connections and a bottleneck layer.
  
- **Preprocessing Pipeline:**  
  - Images are loaded and normalized (converted to `Float32`, scaled to [0,1]).
  - Labels are loaded, normalized, and scaled to discrete integer classes (by multiplying by 34), then one-hot encoded.
  - The pipeline can also determine the number of classes by computing the unique values in the processed mask.
  
- **Training with Flux.jl and Optimisers.jl:**  
  The training loop uses an explicit parameter update mechanism with Optimisers.jl (e.g., Adam), making it easy to monitor gradients, losses, and parameter updates.
  
- **Batching:**  
  Utility functions group individual image/mask pairs into batches along the batch dimension.
  
- **GPU Support:**  
  GPU-accelerated version available in the `src_gpu` directory, while the standard implementation in `src` operates on CPU.
  
- **Visualization:**  
  Model predictions can be visualized alongside input images and ground-truth masks using Plots.jl.

---

## Directory Structure

```
checkpoints/
    Storage for model checkpoints during training.
Datensatz/
    Contains the training and validation datasets (images and labels).
experiments/
    Scripts for testing and evaluating different ideas.
results/
    Directory for saving visualization results and generated images.
src/
    Standard CPU implementation of the U-Net architecture, data loaders, training pipeline, and visualization.
src_experiment/
    Experimental versions of the code for testing new approaches.
src_gpu/
    GPU-accelerated implementation of the U-Net framework (requires CUDA).
.gitignore
    Git ignore file specifying which files and directories to exclude from version control.
README.md
    Documentation for the repository.
unet_model.pth
    Pretrained model weights.
```

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. **Install Julia packages:**  
   Open Julia in the repository directory and run:
   ```julia
   using Pkg
   Pkg.activate(".")
   Pkg.instantiate()
   ```

---

## Usage

### 1. Load and Preprocess Data

Organize your images and labels in the `Datensatz` directory:

```
Datensatz/
└── Training/
    ├── Bilder_alle/
    └── Masken_alle/
```

In your main script, load the dataset and create batches:
```julia
using Data

img_dir = "Datensatz/Training/Bilder_alle"
mask_dir = "Datensatz/Training/Masken_alle"

# Load dataset as an array of (input_image, ground_truth) tuples.
dataset = Data.load_dataset(img_dir, mask_dir)
println("Number of samples in dataset: ", length(dataset))

# Create batches from the dataset (e.g., batch_size = 4)
batch_size = 4
train_data = Data.create_batches(dataset, batch_size)
println("Number of batches: ", length(train_data))
```

The label preprocessing function also returns the largest class value, from which you can determine the number of output channels (by adding one).

### 2. Train the Model

The training process is defined in the `train_unet` function in `Training.jl`, which uses explicit gradient computation and parameter updates via `Optimisers.jl`. For example, in your main script:

```julia
using Model, Training

# Determine output channels from a sample mask:
_, max_class = Data.load_and_preprocess_label("Datensatz/Training/Masken_alle/sample_mask.png")
output_channels = max_class + 1  # Classes from 0 to max_class

# Initialize the model (e.g., 3 input channels, determined output channels)
input_channels = 3
model = Model.UNet(input_channels, output_channels)

# Train the model for a set number of epochs
num_epochs = 10
losses = Training.train_unet(model, train_data, num_epochs, 0.001, output_channels)
```

The training loop prints debug information (shapes, means, losses, etc.) for each batch and epoch. The loss for each epoch is stored in `losses`, which you can later use for plotting.

### 3. Visualize Results

After training, visualize the model's predictions alongside the input image and ground truth mask:

```julia
using Visualization

Visualization.visualize_results(model, input_image, ground_truth)
```

The visualization results will be saved in the `results` directory.

### 4. Plot Loss Over Time

If you saved the epoch loss values in an array (returned by `train_unet`), you can plot the loss progression:

```julia
using Plots

scatter(1:num_epochs, losses, xlabel="Epoch", ylabel="Loss", title="Loss Over Time", marker=:o)
```

### 5. GPU Acceleration

To use the GPU-accelerated implementation, import from the `src_gpu` directory instead of `src`:

```julia
include("src_gpu/Model.jl")
include("src_gpu/Training.jl")
# ... and so on
```

Make sure you have CUDA installed and configured properly for GPU support.

---

## Requirements

- Julia 1.8+
- Packages:
  - Flux.jl
  - Optimisers.jl
  - CUDA.jl (optional, for GPU support)
  - Images.jl
  - ImageTransformations.jl
  - Plots.jl
  - FileIO.jl

---

## Contribution

Contributions are welcome! Please feel free to submit pull requests or open issues if you have any suggestions or encounter any problems.