##################################
# Main.jl - Enhanced with Better Training
##################################

# Import modules
include("Data.jl")
include("Model.jl")
include("Training.jl")
include("Visualization.jl")
include("Metrics.jl")  # Optional: Will need to create this file

using Test
using Flux
using Flux: onehotbatch, logitcrossentropy, gpu
using CUDA
using .Model
using .Data
using .Training
using .Visualization
using .Metrics
using Statistics
using FileIO
using Images
using LinearAlgebra
using Optimisers
using Plots
using Dates
using Random
using ArgParse  # Add for command-line arguments support
import Base.GC

# Function to parse command-line arguments
function parse_commandline()
    s = ArgParseSettings(description="Semantic Segmentation Training Script")
    
    @add_arg_table s begin
        "--epochs", "-e"
            help = "Number of training epochs"
            arg_type = Int
            default = 20
        "--batch-size", "-b"
            help = "Batch size"
            arg_type = Int
            default = 2
        "--learning-rate", "-l"
            help = "Learning rate"
            arg_type = Float32
            default = 0.001f0
        "--subset"
            help = "Use a subset of the dataset for testing"
            action = :store_true
        "--subset-size"
            help = "Size of subset if --subset is used"
            arg_type = Int
            default = 50
        "--checkpoint-dir"
            help = "Directory to save model checkpoints"
            default = "checkpoints"
        "--checkpoint-freq"
            help = "Save checkpoint every N epochs"
            arg_type = Int
            default = 1
        "--data-dir"
            help = "Base directory for the dataset"
            default = "S:/Masterarbeit/Datensatz"
        "--output-dir"
            help = "Directory for output"
            default = "results"
        "--model-type"
            help = "UNet model type (standard or efficient)"
            default = "efficient"
        "--loss-type"
            help = "Loss function type (ce, dice, combo, focal, lovasz)"
            default = "combo"
        "--augmentation"
            help = "Augmentation factor (1 = no augmentation)"
            arg_type = Int
            default = 2
        "--use-class-balancing"
            help = "Use class balancing for dataset"
            action = :store_true
        "--mixed-precision"
            help = "Use mixed precision training"
            action = :store_true
        "--lr-scheduler"
            help = "Learning rate scheduler (constant, cosine, onecycle)"
            default = "onecycle"
        "--early-stopping"
            help = "Enable early stopping"
            action = :store_true
        "--patience"
            help = "Patience for early stopping"
            arg_type = Int
            default = 10
        "--validation-split"
            help = "Fraction of data to use for validation"
            arg_type = Float32
            default = 0.1f0
        "--use-residual"
            help = "Use residual connections in model"
            action = :store_true
        "--use-attention"
            help = "Use attention mechanism in model"
            action = :store_true
        "--dropout-rate"
            help = "Dropout rate for regularization"
            arg_type = Float32
            default = 0.3f0
        "--seed"
            help = "Random seed for reproducibility"
            arg_type = Int
            default = 42
    end
    
    return parse_args(s)
end

# GPU memory management
function clear_gpu_memory()
    GC.gc()
    CUDA.reclaim()
    println("GPU memory cleared")
end

# Main function
function main()
    # Parse command-line arguments
    args = parse_commandline()
    
    # Set random seed for reproducibility
    Random.seed!(args["seed"])
    
    # Extract parameters
    num_epochs = args["epochs"]
    learning_rate = args["learning-rate"]
    batch_size = args["batch-size"]
    sub_set = args["subset"]
    sub_size = args["subset-size"]
    checkpoint_dir = args["checkpoint-dir"]
    checkpoint_freq = args["checkpoint-freq"]
    use_mixed_precision = args["mixed-precision"]
    loss_type = args["loss-type"]
    model_type = args["model-type"]
    lr_scheduler = args["lr-scheduler"]
    early_stopping = args["early-stopping"]
    patience = args["patience"]
    validation_split = args["validation-split"]
    use_residual = args["use-residual"]
    use_attention = args["use-attention"]
    dropout_rate = args["dropout-rate"]
    augmentation_factor = args["augmentation"]
    use_class_balancing = args["use-class-balancing"]
    
    # Set up directories
    data_dir = args["data-dir"]
    img_dir = joinpath(data_dir, "Training/image_2")
    mask_dir = joinpath(data_dir, "Training/semantic")
    output_dir = args["output-dir"]
    
    # Create timestamp for this run
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM")
    run_dir = joinpath(output_dir, "run_$(timestamp)")
    mkpath(run_dir)
    
    # Save all parameters to a log file
    open(joinpath(run_dir, "parameters.txt"), "w") do io
        println(io, "Parameters for run at $timestamp")
        println(io, "==================================")
        for (key, value) in args
            println(io, "$key: $value")
        end
    end
    
    println("Loading dataset...")
    
    # Load dataset
    if sub_set
        image_files = sort(readdir(img_dir, join=true))
        label_files = sort(readdir(mask_dir, join=true))
        subset_size = min(sub_size, length(image_files))
        println("Loading subset of $subset_size images (out of $(length(image_files)) total)")
        subset_img_files = image_files[1:subset_size]
        subset_label_files = label_files[1:subset_size]
        dataset = Data.load_dataset(subset_img_files, subset_label_files)
    else
        dataset = Data.load_dataset(img_dir, mask_dir)
    end
    
    println("Number of samples in dataset: ", length(dataset))
    
    # Split into training and validation sets
    dataset_size = length(dataset)
    val_size = round(Int, dataset_size * validation_split)
    indices = randperm(dataset_size)
    
    train_indices = indices[1:dataset_size-val_size]
    val_indices = indices[dataset_size-val_size+1:end]
    
    train_dataset = dataset[train_indices]
    val_dataset = dataset[val_indices]
    
    println("Training set size: ", length(train_dataset))
    println("Validation set size: ", length(val_dataset))
    
    # Apply data augmentation
    if augmentation_factor > 1
        if use_class_balancing
            train_dataset = Data.create_balanced_augmented_dataset(train_dataset, 35, augmentation_factor)
        else
            train_dataset = Data.create_augmented_dataset(train_dataset, augmentation_factor)
        end
    end
    
    clear_gpu_memory()
    
    # Visualize class distribution
    Visualization.visualize_class_distribution(train_dataset, 35, save_path=joinpath(run_dir, "class_distribution"))
    
    # Calculate class weights based on the training set
    class_weights = Training.calculate_class_weights(train_dataset, 35, method="inverse")
    
    # Create batches
    train_data = Data.create_batches(train_dataset, batch_size)
    val_data = Data.create_batches(val_dataset, batch_size)
    
    println("Number of training batches: ", length(train_data))
    println("Number of validation batches: ", length(val_data))
    
    # Anzahl Output-Kanäle (Klassenanzahl). In dem Beispiel 35.
    input_channels = 3  # RGB
    output_channels = 35
    println("Input channels: ", input_channels)
    println("Output channels (classes): ", output_channels)
    
    # Model creation
    memory_efficient = model_type == "efficient"
    println("Creating UNet model with type: $(model_type)")
    println("Using residual connections: $(use_residual)")
    println("Using attention mechanism: $(use_attention)")
    
    model = Model.UNet(input_channels, output_channels, 
                      memory_efficient=memory_efficient,
                      dropout_rate=dropout_rate,
                      use_attention=use_attention)
    
    println("UNet model created")
    
    # Move model to GPU
    model = gpu(model)
    println("Model moved to GPU")
    
    clear_gpu_memory()
    
    # Create checkpoint directory
    cp_dir = joinpath(checkpoint_dir, timestamp)
    mkpath(cp_dir)
    
    # Save model architecture visualization
    Visualization.visualize_model_architecture(model, save_path=joinpath(run_dir, "architecture"))
    
    # Start training
    start_time = now()
    println("Starting training at ", start_time)
    
    # Set up training parameters
    training_args = Dict(
        :validation_data => val_data,
        :checkpoint_dir => cp_dir,
        :checkpoint_freq => checkpoint_freq,
        :class_weights => class_weights,
        :loss_type => loss_type,
        :early_stopping_patience => patience,
        :lr_scheduler => lr_scheduler
    )
    
    # Run training with selected method
    if use_mixed_precision
        println("Using mixed precision training")
        model, losses, val_losses = Training.train_unet_mixed_precision(
            model, train_data, num_epochs, learning_rate, output_channels;
            training_args...
        )
    else
        println("Using standard precision training")
        model, losses, val_losses = Training.train_unet(
            model, train_data, num_epochs, learning_rate, output_channels;
            training_args...
        )
    end
    
    end_time = now()
    training_duration = end_time - start_time
    println("Training completed in ", training_duration)
    
    clear_gpu_memory()
    
    # Visualize training metrics
    Visualization.visualize_training_metrics(
        losses, val_losses, 
        save_path=joinpath(run_dir, "training_metrics")
    )
    
    # Evaluate on validation set
    println("Evaluating model on validation set...")
    
    # Create confusion matrix
    Visualization.visualize_confusion_matrix(
        model, val_data, output_channels,
        save_path=joinpath(run_dir, "confusion_matrix")
    )
    
    # Generate example visualizations
    println("Generating example visualizations...")
    
    for i in 1:min(5, length(val_data))
        Visualization.visualize_results(
            model, val_data[i][1][:,:,:,1:1], val_data[i][2][:,:,:,1:1], losses,
            save_path=joinpath(run_dir, "examples", "example_$(i)"),
            show_plots=false
        )
    end
    
    # Create a summary report
    open(joinpath(run_dir, "training_summary.txt"), "w") do io
        println(io, "Semantic Segmentation Training Summary")
        println(io, "=====================================")
        println(io, "Run timestamp: ", timestamp)
        println(io, "")
        
        println(io, "Dataset:")
        println(io, "  Total samples: ", dataset_size)
        println(io, "  Training samples: ", length(train_dataset))
        println(io, "  Validation samples: ", length(val_dataset))
        if augmentation_factor > 1
            println(io, "  Training samples after augmentation: ", length(train_dataset))
        end
        println(io, "")
        
        println(io, "Model Configuration:")
        println(io, "  Type: ", model_type)
        println(io, "  Input channels: ", input_channels)
        println(io, "  Output channels: ", output_channels)
        println(io, "  Using residual connections: ", use_residual)
        println(io, "  Using attention mechanism: ", use_attention)
        println(io, "  Dropout rate: ", dropout_rate)
        println(io, "")
        
        println(io, "Training Configuration:")
        println(io, "  Batch size: ", batch_size)
        println(io, "  Learning rate: ", learning_rate)
        println(io, "  Number of epochs: ", num_epochs)
        println(io, "  Loss function: ", loss_type)
        println(io, "  LR scheduler: ", lr_scheduler)
        println(io, "  Mixed precision: ", use_mixed_precision)
        println(io, "  Early stopping: ", early_stopping)
        if early_stopping
            println(io, "  Patience: ", patience)
        end
        println(io, "")
        
        println(io, "Training Results:")
        println(io, "  Training duration: ", training_duration)
        println(io, "  Final training loss: ", losses[end])
        println(io, "  Final validation loss: ", val_losses[end])
        println(io, "  Best validation loss: ", minimum(val_losses))
        println(io, "  Best epoch: ", argmin(val_losses))
        println(io, "")
        
        println(io, "Checkpoints saved to: ", cp_dir)
        println(io, "Results saved to: ", run_dir)
    end
    
    println("Training summary saved to $(joinpath(run_dir, "training_summary.txt"))")
    println("All results saved to $run_dir")
    println("Done!")
end

# Execute main function if run as script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end