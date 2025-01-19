module Model

export create_unet

using Flux

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

# Define the decoder block
function decoder_block(in_channels::Int, out_channels::Int)
    return Chain(
        ConvTranspose((2, 2), in_channels => out_channels, stride=(2, 2)),
        conv_block(out_channels, out_channels)
    )
end

# Define the U-Net architecture without bottleneck or skip connections
function create_unet(input_channels::Int, output_channels::Int)
    enc1, pool1 = encoder_block(input_channels, 64)
    enc2, pool2 = encoder_block(64, 128)
    enc3, pool3 = encoder_block(128, 256)
    enc4, pool4 = encoder_block(256, 512)

    dec4 = decoder_block(512, 256)
    dec3 = decoder_block(256, 128)
    dec2 = decoder_block(128, 64)
    dec1 = decoder_block(64, output_channels)

    return Chain(x -> begin
        x1 = enc1(x); p1 = pool1(x1)
        x2 = enc2(p1); p2 = pool2(x2)
        x3 = enc3(p2); p3 = pool3(x3)
        x4 = enc4(p3); p4 = pool4(x4)

        d4 = dec4(p4)
        d3 = dec3(d4)
        d2 = dec2(d3)
        d1 = dec1(d2)
        d1
    end)
end

end # module
