module Model

export create_unet

using Flux
using ImageTransformations: imresize

# Define a single convolutional block
function conv_block(in_channels::Int, out_channels::Int)
    return Chain(
        Conv((3, 3), in_channels => out_channels, relu, pad=1),
        BatchNorm(out_channels)
    )
end

# Define the encoder block
function encoder_block(in_channels::Int, out_channels::Int)
    encoder = Chain(
        conv_block(in_channels, out_channels),
        conv_block(out_channels, out_channels)
    )
    pool = MaxPool((2, 2), stride=(2, 2))
    return encoder, pool
end

# Define the bottleneck block
function bottleneck(in_channels::Int, out_channels::Int)
    return Chain(
        conv_block(in_channels, out_channels),
        conv_block(out_channels, out_channels)
    )
end

# Define the decoder block
function decoder_block(in_channels::Int, out_channels::Int)
    upsample = ConvTranspose((2, 2), in_channels => out_channels, stride=(2, 2))
    return Chain(
        upsample,
        conv_block(out_channels, out_channels)
    )
end

# Resize function to match spatial dimensions
function resize_to_match(input, target)
    size_target = size(target)
    imresize(input, (size_target[1], size_target[2]))
end

# Define the U-Net architecture with controlled upsampling
function create_unet(input_channels::Int, output_channels::Int)
    # Encoder
    enc1, pool1 = encoder_block(input_channels, 64)
    enc2, pool2 = encoder_block(64, 128)
    enc3, pool3 = encoder_block(128, 256)
    enc4, pool4 = encoder_block(256, 512)

    # Bottleneck
    bottleneck_layer = bottleneck(512, 1024)

    # Decoder
    dec4 = decoder_block(1024 + 512, 512)
    dec3 = decoder_block(512 + 256, 256)
    dec2 = decoder_block(256 + 128, 128)
    dec1 = decoder_block(128 + 64, 64)

    # Final Convolution for output
    final_conv = Conv((1, 1), 64 => output_channels, pad=0)

    return Chain(x -> begin
        # Encoder Pass
        x1 = enc1(x); p1 = pool1(x1)
        x2 = enc2(p1); p2 = pool2(x2)
        x3 = enc3(p2); p3 = pool3(x3)
        x4 = enc4(p3); p4 = pool4(x4)
        println("x1 Größe: ", size(x1))
        println("p4 Größe: ", size(p4))
        
        # Bottleneck
        b = bottleneck_layer(p4)
        println("Bottleneck Größe: ", size(b))
        b_resized = resize_to_match(b, x4)
        println("Bottleneck Größe nach resize: ", size(b_resized))
        println("x4 Größe: ", size(x4))
        
        # Decoder Pass with controlled resizing
        d4 = resize_to_match(dec4(cat(b_resized, x4; dims=3)), x3)
        d3 = resize_to_match(dec3(cat(d4, x3; dims=3)), x2)
        d2 = resize_to_match(dec2(cat(d3, x2; dims=3)), x1)
        d1 = resize_to_match(dec1(cat(d2, x1; dims=3)), x1)
        println("d1 Größe: ", size(d1))

        # Output with final resizing to input dimensions
        output = resize_to_match(final_conv(d1), x)
        println("Finale Ausgabegröße: ", size(output))
        output
    end)
end

end # module

# Testing the U-Net model
using Test
using .Model

@testset "U-Net Model" begin
    input = rand(Float32, 512, 2048, 1, 4)  # Input: 512x2048, 1 channel, batch size 4
    model = create_unet(1, 1)  # Input channels: 1, Output channels: 1
    output = model(input)
    println("Test-Ausgabegröße: ", size(output))
    @test size(output) == (512, 2048, 1, 4)  # Expected output dimensions
end
