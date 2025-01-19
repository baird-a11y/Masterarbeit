# U-Net for Image Segmentation

This repository contains an implementation of the U-Net architecture for image segmentation tasks. The project is built using the Julia programming language with `Flux.jl` as the deep learning framework.

---

## Features

- **Customizable U-Net architecture**: Flexible encoder and decoder block definitions.
- **Simplified U-Net architecture**: Currently, the implementation does not use bottleneck layers or skip-connections for simplicity.
- **Preprocessing pipeline**: Converts images and labels to grayscale, normalizes them, resizes them to the nearest higher power of 2 dimensions for compatibility, and batches them for training.
- **Training with Flux.jl**: Supports training with `Adam` optimizer and cross-entropy loss.
- **GPU Support**: Fully compatible with CUDA for GPU acceleration.
- **Visualization**: Displays predictions alongside input images and ground-truth masks.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. Install Julia packages:
   Open Julia in the repository directory and run:
   ```julia
   using Pkg
   Pkg.activate(".")
   Pkg.instantiate()
   ```

3. Install CUDA support (if using GPU):
   ```julia
   Pkg.add("CUDA")
   ```

---

## Directory Structure

- `src/`
  - Contains the implementation of the U-Net architecture, data loaders, and training pipeline.
- `Datensatz/`
  - Placeholder for training and validation datasets.
- `experiments/`
  - Scripts for testing and evaluating some ideas.
- `results/`
  - Old results from the experiment codes.
- `README.md`
  - Documentation for the repository.

---

## Usage

### **1. Load and Preprocess Data**
Images and labels are automatically resized to the nearest higher power of 2 dimensions (e.g., `512x512`, `1024x1024`) to simplify operations during training and inference. Ensure your images and labels are organized in the following directory structure:
```
data/
├── Training/
│   ├── Bilder/
│   └── Masken/
```
Run the `main.jl` script to load and preprocess the data:
```julia
julia src/main.jl
```

### **2. Train the Model**
The training process is defined in the `train_model` function. Adjust hyperparameters such as `epochs`, `batch_size`, and `learning_rate` in `main.jl`.

### **3. Visualize Results**
After training, visualize the model's predictions alongside the ground-truth masks:
```julia
UNetFramework.visualize_results(trained_model, test_image, test_label)
```

---

## Requirements

- Julia 1.8+
- Packages:
  - Flux.jl
  - CUDA.jl (optional, for GPU support)
  - Images.jl
  - ImageTransformations.jl
  - Plots.jl

---

## Contribution

Feel free to submit pull requests or report issues. Contributions to improve the code or add new features are welcome.

---

