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
        Conv((3, 3), in_channels => out_channels, relu, pad=1),
        BatchNorm(out_channels)
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
        conv_block(in_channels, out_channels),
        conv_block(out_channels, out_channels)
    )
    pool = MaxPool((2, 2), stride=(2, 2))
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
        ConvTranspose((2, 2), in_channels => out_channels, stride=(2, 2)),
        conv_block(out_channels * 2, out_channels) # Skip connections double the channels
    )
end

# Define the U-Net architecture
function create_unet(input_channels::Int, output_channels::Int)
    """
    Create a U-Net model.

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

    # Bottleneck
    bottleneck = Chain(
        conv_block(512, 1024),
        conv_block(1024, 1024)
    )

    # Decoder layers
    dec4 = decoder_block(1024, 512)
    dec3 = decoder_block(512, 256)
    dec2 = decoder_block(256, 128)
    dec1 = decoder_block(128, 64)

    # Final output layer
    final_layer = Conv((1, 1), 64 => output_channels, relu)

    # Combine all layers into a Chain with skip connections
    return Chain(
        (x -> begin
            x1 = enc1(x); p1 = pool1(x1)
            x2 = enc2(p1); p2 = pool2(x2)
            x3 = enc3(p2); p3 = pool3(x3)
            x4 = enc4(p3); p4 = pool4(x4)
            bottleneck_out = bottleneck(p4)
            d4 = dec4(cat(bottleneck_out, x4, dims=3))
            d3 = dec3(cat(d4, x3, dims=3))
            d2 = dec2(cat(d3, x2, dims=3))
            d1 = dec1(cat(d2, x1, dims=3))
            final_layer(d1)
        end)
    )
end

end # module
