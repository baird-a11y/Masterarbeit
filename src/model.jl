module Model

export create_unet

using Flux

# Define a single convolutional block
function conv_block(in_channels::Int, out_channels::Int)
    """
    Create a convolutional block with Conv, BatchNorm, and ReLU layers.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.

    Returns:
        A Chain representing the convolutional block.
    """
    return Chain(
        (x -> begin println("Input to conv_block: ", size(x)); x end),
        Conv((3, 3), in_channels => out_channels, relu, pad=1),
        (x -> begin println("After Conv: ", size(x)); x end),
        BatchNorm(out_channels),
        (x -> begin println("After BatchNorm: ", size(x)); x end)
    )
end

# Define the encoder block
function encoder_block(in_channels::Int, out_channels::Int)
    """
    Create an encoder block with two convolutional blocks and max pooling.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.

    Returns:
        A tuple (encoder, pool) with the encoder layers and max pooling layer.
    """
    encoder = Chain(
        (x -> begin println("Encoder input size: ", size(x)); x end),
        conv_block(in_channels, out_channels),
        conv_block(out_channels, out_channels)
    )
    pool = Chain(
        MaxPool((2, 2), stride=(2, 2)),
        (x -> begin println("After max pooling: ", size(x)); x end)
    )
    return encoder, pool
end

# Define the decoder block
function decoder_block(in_channels::Int, out_channels::Int)
    """
    Create a decoder block with a transposed convolution and a convolutional block.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.

    Returns:
        A Chain representing the decoder block.
    """
    return Chain(
        (x -> begin println("Decoder input size: ", size(x)); x end),
        ConvTranspose((2, 2), in_channels => out_channels, stride=(2, 2)),
        (x -> begin println("After transposed convolution: ", size(x)); x end),
        conv_block(out_channels, out_channels),
        (x -> begin println("After conv block in decoder: ", size(x)); x end)
    )
end

# Define the U-Net architecture without bottleneck or skip connections
function create_unet(input_channels::Int, output_channels::Int)
    """
    Create a U-Net model without bottleneck or skip connections.

    Args:
        input_channels: Number of input channels.
        output_channels: Number of output channels.

    Returns:
        The U-Net model.
    """
    # Encoder layers
    enc1, pool1 = encoder_block(input_channels, 64)
    enc2, pool2 = encoder_block(64, 128)
    enc3, pool3 = encoder_block(128, 256)
    enc4, pool4 = encoder_block(256, 512)

    # Decoder layers
    dec4 = decoder_block(512, 256)
    dec3 = decoder_block(256, 128)
    dec2 = decoder_block(128, 64)
    dec1 = decoder_block(64, output_channels)

    # Combine all layers into a Chain without skip connections
    return Chain(
        (x -> begin
            println("Input size: ", size(x))
            x1 = enc1(x); p1 = pool1(x1)
            x2 = enc2(p1); p2 = pool2(x2)
            x3 = enc3(p2); p3 = pool3(x3)
            x4 = enc4(p3); p4 = pool4(x4)
            
            # Decoder without skip connections
            d4 = dec4(p4)
            println("After decoding layer 4: ", size(d4))
            d3 = dec3(d4)
            println("After decoding layer 3: ", size(d3))
            d2 = dec2(d3)
            println("After decoding layer 2: ", size(d2))
            d1 = dec1(d2)
            println("After decoding layer 1: ", size(d1))
            d1
        end)
    )
end

end # module
