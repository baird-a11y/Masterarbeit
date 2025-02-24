##################################
# simplified_debug.jl - Step-by-step debugging
##################################

# Include only what's necessary
include("Data.jl")

using Flux
using CUDA
using Statistics
using Optimisers

println("===== SCRIPT STARTED =====")

# Extremely simple model - just one convolutional layer
function create_minimal_model(input_channels, output_channels)
    return Chain(
        Conv((3, 3), input_channels => output_channels, relu, pad=SamePad())
    )
end

# # Print basic information about a tensor
# function print_tensor_info(name, x)
#     println("$name: shape=$(size(x)), min=$(minimum(x)), max=$(maximum(x)), mean=$(mean(x))")
#     if any(isnan, x) || any(isinf, x)
#         println("  WARNING: Tensor contains NaN or Inf values!")
#     end
# end

# Convert target to one-hot encoding (simplified)
function simple_onehot(target, num_classes)
    # Move to CPU for scalar operations
    target_cpu = cpu(target)
    target_indices = Int.(target_cpu[:,:,1,1])
    
    # Get dimensions
    h, w = size(target_indices)
    batch_size = 1
    
    # Create one-hot tensor
    result = zeros(Float32, h, w, num_classes, batch_size)
    for i in 1:h
        for j in 1:w
            class_idx = target_indices[i, j] + 1  # +1 for 1-based indexing
            if 1 <= class_idx <= num_classes
                result[i, j, class_idx, 1] = 1.0
            end
        end
    end
    
    # Move back to GPU
    return gpu(result)
end

# Loss function with additional debug outputs.
function loss_fn(model, x, y)
    pred = model(x)
    # println("DEBUG: Prediction shape: ", size(pred))
    # println("DEBUG: Prediction Mean/Std: ", mean(pred), " / ", std(pred))
    # println("DEBUG: Prediction Min/Max: ", minimum(pred), " / ", maximum(pred))
    loss = Flux.logitcrossentropy(pred, y)
    println("DEBUG: Calculated Loss: ", loss)
    return loss
end

# Main debugging function with incremental steps
function debug_steps()
    println("\n===== STEP 1: LOAD DATA =====")
    # Use Data module to load a single image
    img_dir = "S:/Masterarbeit/Datensatz/Training/image_2"
    mask_dir = "S:/Masterarbeit/Datensatz/Training/semantic"
    
    # Get only the first image and mask
    img_files = sort(readdir(img_dir, join=true))
    mask_files = sort(readdir(mask_dir, join=true))
    
    println("Loading first image: $(basename(img_files[1]))")
    # Load just one sample
    sample_dataset = Data.load_dataset([img_files[1]], [mask_files[1]])
    
    # Extract the data
    input = sample_dataset[1][1]
    target = sample_dataset[1][2]
    
    println("Input: $(size(input)), Target: $(size(target))")
    
    println("\n===== STEP 2: CREATE MINIMAL MODEL =====")
    # Number of classes (0-34)
    num_classes = 35
    
    # Create the simplest possible model
    model = create_minimal_model(3, num_classes)
    println("Model structure: $model")
    
    println("\n===== STEP 3: MOVE DATA TO GPU =====")
    # Enable scalar indexing for debugging
    CUDA.allowscalar(true)
    
    # Move data to GPU
    input_gpu = gpu(input)
    target_gpu = gpu(target)
    model_gpu = gpu(model)
    
    println("Data moved to GPU successfully")
    
    println("\n===== STEP 4: ONE-HOT ENCODE TARGET =====")
    # Convert target to one-hot
    target_onehot = simple_onehot(target_gpu, num_classes)
    println("One-hot target shape: $(size(target_onehot))")
    
    println("\n===== STEP 5: FORWARD PASS =====")
    # Run model forward
    pred = model_gpu(input_gpu)
    # print_tensor_info("Prediction", pred)
    
    println("\n===== STEP 6: COMPUTE LOSS =====")
    # Calculate loss
    loss = loss_fn(model_gpu, input_gpu, target_onehot)
    println("Loss value: $loss")
    
    println("\n===== STEP 7: COMPUTE GRADIENT AND UPDATE MODEL =====")
    # Set up the optimizer state using Optimisers.jl (here using Adam)
    opt_state = Optimisers.setup(Optimisers.Adam(0.001), model_gpu)
    
    # Compute gradients with respect to the model using an explicit lambda.
    ∇model = gradient(m -> Flux.logitcrossentropy(m(input_gpu), target_onehot), model_gpu)[1]
    # println("DEBUG: Loss in gradient block: ", Flux.logitcrossentropy(model_gpu(input_gpu), target_onehot))
    
    # Update the optimizer state and model parameters explicitly.
    opt_state, model_gpu = Optimisers.update!(opt_state, model_gpu, ∇model)
    
    println("\n===== STEP 8: TRY FLUX TRAINING FUNCTION =====")
    # Try using Flux's built-in training function
    try
        opt = Flux.setup(Flux.Adam(0.001), model_gpu)
        println("Running Flux.train!...")

        # Define a simple training loss
        function training_loss(model, x, y)
            pred = model(x)
            return Flux.logitcrossentropy(pred, y)
        end
        
        try
            # Run a single training step
            Flux.train!(training_loss, model_gpu, [(input_gpu, target_onehot)], opt)
            println("Flux.train! completed successfully")
            
            # Verify model works after training
            new_loss = loss_fn(model_gpu, input_gpu, target_onehot)
            println("Loss after training step: $new_loss")
        catch e
            if isa(e, InterruptException)
                println("Training interrupted by the user.")
                # Perform any necessary cleanup or save model state
            else
                println("Error during Flux.train!: ", e)
                println("Stacktrace: ", stacktrace(catch_backtrace()))
            end
        end
    catch e
        if isa(e, InterruptException)
            println("Setup interrupted by the user.")
            # Perform any necessary cleanup
        else
            println("Error setting up Flux.train!: ", e)
            println("Stacktrace: ", stacktrace(catch_backtrace()))
        end
    end
    
    println("\n===== DEBUGGING COMPLETE =====")
end
# Run the step-by-step debugging
debug_steps()