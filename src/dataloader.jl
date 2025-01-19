module DataLoader

export load_data, create_batches, preprocess_image

using FileIO
using ImageCore
using ImageTransformations: imresize

# Function to load images and labels
function load_data(image_path::String, label_path::String)
    """
    Load images and corresponding labels from the specified directories.

    Args:
        image_path: Path to the directory containing images.
        label_path: Path to the directory containing labels.

    Returns:
        A tuple of arrays (images, labels).
    """
    images = []
    labels = []

    # Load all images and labels
    for image_file in readdir(image_path)
        push!(images, load(joinpath(image_path, image_file)))
    end

    for label_file in readdir(label_path)
        push!(labels, load(joinpath(label_path, label_file)))
    end

    println("Loaded ", length(images), " images and ", length(labels), " labels.")
    return images, labels
end

function preprocess_and_batch(images::Vector, labels::Vector, batch_size::Int, target_size::Tuple{Int, Int})
    """
    Preprocess images and labels, and create batches for training.

    Args:
        images: Array of input images.
        labels: Array of corresponding labels.
        batch_size: Number of samples per batch.
        target_size: Target size for resizing images.

    Returns:
        An array of batches (x_batch, y_batch).
    """
    @assert length(images) == length(labels) "Number of images and labels must match."

    function preprocess_image(img, target_size)
        # Convert the image to grayscale and normalize pixel values to [0, 1]
        gray_img = Float32.(Gray.(img)) ./ 255.0

        # Resize the image to the target size (height, width)
        resized_img = imresize(gray_img, target_size)

        # Reshape the resized image for U-Net-compatible dimensions: (H, W, C, BatchSize)
        return reshape(resized_img, size(resized_img, 1), size(resized_img, 2), 1, 1)
    end

    batches = []
    for i in 1:batch_size:length(images)
        x_batch = images[i:min(i+batch_size-1, end)]
        y_batch = labels[i:min(i+batch_size-1, end)]

        # Preprocess images and labels
        x_batch = [preprocess_image(x, target_size) for x in x_batch]
        y_batch = [preprocess_image(y, target_size) for y in y_batch]

        # Combine preprocessed images and labels into batches
        push!(batches, (cat(x_batch..., dims=4), cat(y_batch..., dims=4)))
    end

    println("Created ", length(batches), " batches.")
    return batches
end


end # module
