##################################
# Model.jl - Optimized
##################################
module Model

using Flux
using Functors
using CUDA
using Statistics

# UNet als strukturiertes Modell mit verbesserten skip connections
struct UNet
    encoder1
    encoder2
    encoder3
    encoder4
    bottleneck
    decoder4
    decoder4_1
    decoder3
    decoder3_1
    decoder2
    decoder2_1
    decoder1
    decoder1_1
    dropout  # Add dropout for regularization
end
Functors.@functor UNet

function UNet(input_channels::Int, output_channels::Int; dropout_rate=0.2)
    # Encoder - Use BatchNorm for better training stability
    encoder1 = Chain(
        Conv((3, 3), input_channels => 64, relu, pad=SamePad()),
        BatchNorm(64),
        Conv((3, 3), 64 => 64, relu, pad=SamePad()),
        BatchNorm(64)
    )
    encoder2 = Chain(
        MaxPool((2,2)),
        Conv((3, 3), 64 => 128, relu, pad=SamePad()),
        BatchNorm(128),
        Conv((3, 3), 128 => 128, relu, pad=SamePad()),
        BatchNorm(128)
    )
    encoder3 = Chain(
        MaxPool((2,2)),
        Conv((3, 3), 128 => 256, relu, pad=SamePad()),
        BatchNorm(256),
        Conv((3, 3), 256 => 256, relu, pad=SamePad()),
        BatchNorm(256)
    )
    encoder4 = Chain(
        MaxPool((2,2)),
        Conv((3, 3), 256 => 512, relu, pad=SamePad()),
        BatchNorm(512),
        Conv((3, 3), 512 => 512, relu, pad=SamePad()),
        BatchNorm(512)
    )
    bottleneck = Chain(
        MaxPool((2,2)),
        Conv((3,3), 512 => 1024, relu, pad=SamePad()),
        BatchNorm(1024),
        Dropout(dropout_rate),
        Conv((3,3), 1024 => 1024, relu, pad=SamePad()),
        BatchNorm(1024)
    )
    
    # Decoder - Use SamePad for more consistent dimensions
    decoder4 = ConvTranspose((2, 2), 1024 => 512, stride=2)
    
    decoder4_1 = Chain(
        Conv((3,3), 1024 => 512, relu, pad=SamePad()),
        BatchNorm(512),
        Conv((3,3), 512 => 512, relu, pad=SamePad()),
        BatchNorm(512)
    )
    
    decoder3 = ConvTranspose((2,2), 512 => 256, stride=2)
    
    decoder3_1 = Chain(
        Conv((3,3), 512 => 256, relu, pad=SamePad()),
        BatchNorm(256),
        Conv((3,3), 256 => 256, relu, pad=SamePad()),
        BatchNorm(256)
    )
    
    decoder2 = ConvTranspose((2,2), 256 => 128, stride=2)
    
    decoder2_1 = Chain(
        Conv((3,3), 256 => 128, relu, pad=SamePad()),
        BatchNorm(128),
        Conv((3,3), 128 => 128, relu, pad=SamePad()),
        BatchNorm(128)
    )
    
    decoder1 = ConvTranspose((2,2), 128 => 64, stride=2)
    
    decoder1_1 = Chain(
        Conv((3,3), 128 => 64, relu, pad=SamePad()),
        BatchNorm(64),
        Conv((3,3), 64 => 64, relu, pad=SamePad()),
        BatchNorm(64),
        Dropout(dropout_rate/2),  # Less dropout in final layers
        Conv((1,1), 64 => output_channels)
    )
    
    dropout = Dropout(dropout_rate)
    
    return UNet(encoder1, encoder2, encoder3, encoder4, bottleneck,
                decoder4, decoder4_1, decoder3, decoder3_1,
                decoder2, decoder2_1, decoder1, decoder1_1, dropout)
end

# Helper function to handle center cropping for skip connections
function crop_and_concat(x, skip, dims=3)
    # Get dimensions
    x_size = size(x)
    skip_size = size(skip)
    
    # Calculate padding/cropping
    height_diff = skip_size[1] - x_size[1]
    width_diff = skip_size[2] - x_size[2]
    
    if height_diff < 0 || width_diff < 0
        # Skip connection is smaller, pad it
        padded_skip = zeros(eltype(skip), max(x_size[1], skip_size[1]), 
                            max(x_size[2], skip_size[2]), 
                            skip_size[3], skip_size[4])
        
        h_start = abs(min(0, height_diff)) ÷ 2 + 1
        w_start = abs(min(0, width_diff)) ÷ 2 + 1
        
        padded_skip[h_start:h_start+skip_size[1]-1, 
                   w_start:w_start+skip_size[2]-1, :, :] .= skip
                   
        return cat(x, padded_skip, dims=dims)
    else
        # Skip connection is larger, crop it
        h_start = height_diff ÷ 2 + 1
        w_start = width_diff ÷ 2 + 1
        
        cropped_skip = skip[h_start:h_start+x_size[1]-1, 
                           w_start:w_start+x_size[2]-1, :, :]
                           
        return cat(x, cropped_skip, dims=dims)
    end
end

function (model::UNet)(x)
    # Apply dropout at training time only
    e1 = model.encoder1(x)
    e2 = model.encoder2(e1)
    e3 = model.encoder3(e2)
    e4 = model.encoder4(e3)
    b  = model.bottleneck(e4)

    # More robust skip connections with crop_and_concat
    d4 = model.decoder4(b)
    d4 = model.decoder4_1(crop_and_concat(d4, e4))

    d3 = model.decoder3(d4)
    d3 = model.decoder3_1(crop_and_concat(d3, e3))

    d2 = model.decoder2(d3)
    d2 = model.decoder2_1(crop_and_concat(d2, e2))

    d1 = model.decoder1(d2)
    d1 = model.decoder1_1(crop_and_concat(d1, e1))

    return d1
end

# Utility to convert model to half precision
function f16(model)
    return fmap(x -> isa(x, AbstractArray) ? Float16.(x) : x, model)
end

# Utility to convert model back to single precision
function f32(model)
    return fmap(x -> isa(x, AbstractArray) ? Float32.(x) : x, model)
end

end # module Model