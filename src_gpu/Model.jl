##################################
# Model.jl - With Dimension Debugging and Memory Efficiency
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

# --------------------------------------------------------------------------------
# Debug-Funktion: Schichtweiser Forward-Pass mit Min/Max/NaN/Inf-Check
# --------------------------------------------------------------------------------
"""
    debug_forward_unet(model::UNet, x; verbose=true)

Führt einen Forward-Pass auf dem UNet schrittweise aus und gibt nach jedem 
Teilschritt (encoder1..4, bottleneck, decoder4..1) min, max, anyNaN, anyInf aus.
So können Sie feststellen, an welcher Stelle Werte NaN oder Inf werden.

Setzen Sie `verbose=false`, um die Ausgaben zu unterdrücken.
"""
function debug_forward_unet(model::UNet, x; verbose=true)
    # Hilfsfunktion für Debug-Ausgabe
    show_stats(name, t) = verbose && println("  $name: size=", size(t),
        ", min=", minimum(t), ", max=", maximum(t),
        ", anyNaN=", any(isnan, t), ", anyInf=", any(isinf, t))

    # 1) encoder1
    e1 = model.encoder1(x)
    show_stats("encoder1", e1)
    if any(isinf, e1) || any(isnan, e1)
        return e1
    end

    # 2) encoder2
    e2 = model.encoder2(e1)
    show_stats("encoder2", e2)
    if any(isinf, e2) || any(isnan, e2)
        return e2
    end

    # 3) encoder3
    e3 = model.encoder3(e2)
    show_stats("encoder3", e3)
    if any(isinf, e3) || any(isnan, e3)
        return e3
    end

    # 4) encoder4
    e4 = model.encoder4(e3)
    show_stats("encoder4", e4)
    if any(isinf, e4) || any(isnan, e4)
        return e4
    end

    # 5) bottleneck
    b  = model.bottleneck(e4)
    show_stats("bottleneck", b)
    if any(isinf, b) || any(isnan, b)
        return b
    end

    # 6) decoder4
    d4 = model.decoder4(b)
    show_stats("decoder4 (before concat)", d4)
    # skip-connection + decoder4_1
    d4 = model.decoder4_1(crop_and_concat(d4, e4))
    show_stats("decoder4_1", d4)
    if any(isinf, d4) || any(isnan, d4)
        return d4
    end

    # 7) decoder3
    d3 = model.decoder3(d4)
    show_stats("decoder3 (before concat)", d3)
    d3 = model.decoder3_1(crop_and_concat(d3, e3))
    show_stats("decoder3_1", d3)
    if any(isinf, d3) || any(isnan, d3)
        return d3
    end

    # 8) decoder2
    d2 = model.decoder2(d3)
    show_stats("decoder2 (before concat)", d2)
    d2 = model.decoder2_1(crop_and_concat(d2, e2))
    show_stats("decoder2_1", d2)
    if any(isinf, d2) || any(isnan, d2)
        return d2
    end

    # 9) decoder1
    d1 = model.decoder1(d2)
    show_stats("decoder1 (before concat)", d1)
    d1 = model.decoder1_1(crop_and_concat(d1, e1))
    show_stats("decoder1_1 (final)", d1)

    return d1
end

# --------------------------------------------------------------------------------
# Original UNet function to be called when memory_efficient=false
# --------------------------------------------------------------------------------
function UNet_full(input_channels::Int, output_channels::Int; dropout_rate=0.2)
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
        Dropout(dropout_rate/2),
        Conv((1,1), 64 => output_channels)
    )
    
    dropout = Dropout(dropout_rate)
    
    return UNet(encoder1, encoder2, encoder3, encoder4, bottleneck,
                decoder4, decoder4_1, decoder3, decoder3_1,
                decoder2, decoder2_1, decoder1, decoder1_1, dropout)
end

# --------------------------------------------------------------------------------
# Memory-efficient UNet with smaller feature maps
# --------------------------------------------------------------------------------
function UNet(input_channels::Int, output_channels::Int; dropout_rate=0.2, memory_efficient=false)
    if !memory_efficient
        return UNet_full(input_channels, output_channels, dropout_rate=dropout_rate)
    end
    
    # Reduced size for feature maps
    encoder1 = Chain(
        Conv((3, 3), input_channels => 48, relu, pad=SamePad()),
        BatchNorm(48),
        Conv((3, 3), 48 => 48, relu, pad=SamePad()),
        BatchNorm(48)
    )
    encoder2 = Chain(
        MaxPool((2,2)),
        Conv((3, 3), 48 => 96, relu, pad=SamePad()),
        BatchNorm(96),
        Conv((3, 3), 96 => 96, relu, pad=SamePad()),
        BatchNorm(96)
    )
    encoder3 = Chain(
        MaxPool((2,2)),
        Conv((3, 3), 96 => 192, relu, pad=SamePad()),
        BatchNorm(192),
        Conv((3, 3), 192 => 192, relu, pad=SamePad()),
        BatchNorm(192)
    )
    encoder4 = Chain(
        MaxPool((2,2)),
        Conv((3, 3), 192 => 384, relu, pad=SamePad()),
        BatchNorm(384),
        Conv((3, 3), 384 => 384, relu, pad=SamePad()),
        BatchNorm(384)
    )
    bottleneck = Chain(
        MaxPool((2,2)),
        Conv((3,3), 384 => 768, relu, pad=SamePad()),
        BatchNorm(768),
        Dropout(dropout_rate),
        Conv((3,3), 768 => 768, relu, pad=SamePad()),
        BatchNorm(768)
    )
    
    decoder4 = ConvTranspose((2, 2), 768 => 384, stride=2)
    decoder4_1 = Chain(
        Conv((3,3), 768 => 384, relu, pad=SamePad()),
        BatchNorm(384),
        Conv((3,3), 384 => 384, relu, pad=SamePad()),
        BatchNorm(384)
    )
    decoder3 = ConvTranspose((2,2), 384 => 192, stride=2)
    decoder3_1 = Chain(
        Conv((3,3), 384 => 192, relu, pad=SamePad()),
        BatchNorm(192),
        Conv((3,3), 192 => 192, relu, pad=SamePad()),
        BatchNorm(192)
    )
    decoder2 = ConvTranspose((2,2), 192 => 96, stride=2)
    decoder2_1 = Chain(
        Conv((3,3), 192 => 96, relu, pad=SamePad()),
        BatchNorm(96),
        Conv((3,3), 96 => 96, relu, pad=SamePad()),
        BatchNorm(96)
    )
    decoder1 = ConvTranspose((2,2), 96 => 48, stride=2)
    decoder1_1 = Chain(
        Conv((3,3), 96 => 48, relu, pad=SamePad()),
        BatchNorm(48),
        Conv((3,3), 48 => 48, relu, pad=SamePad()),
        BatchNorm(48),
        Dropout(dropout_rate/2),
        Conv((1,1), 48 => output_channels)
    )
    
    dropout = Dropout(dropout_rate)
    
    return UNet(encoder1, encoder2, encoder3, encoder4, bottleneck,
                decoder4, decoder4_1, decoder3, decoder3_1,
                decoder2, decoder2_1, decoder1, decoder1_1, dropout)
end


# --------------------------------------------------------------------------------
# Helper function to handle center cropping for skip connections
# --------------------------------------------------------------------------------
function crop_and_concat(x, skip, dims=3)
    x_size = size(x)
    skip_size = size(skip)
    
    height_diff = skip_size[1] - x_size[1]
    width_diff  = skip_size[2] - x_size[2]
    
    if height_diff < 0 || width_diff < 0
        # Skip connection is smaller, pad it
        padded_skip = zeros(eltype(skip),
                            max(x_size[1], skip_size[1]),
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
        w_start = width_diff  ÷ 2 + 1
        
        cropped_skip = skip[h_start:h_start+x_size[1]-1,
                            w_start:w_start+x_size[2]-1, :, :]
        
        return cat(x, cropped_skip, dims=dims)
    end
end

# --------------------------------------------------------------------------------
# Standard Forward Pass (no debug)
# --------------------------------------------------------------------------------
function (model::UNet)(x)
    e1 = model.encoder1(x)
    e2 = model.encoder2(e1)
    e3 = model.encoder3(e2)
    e4 = model.encoder4(e3)
    b  = model.bottleneck(e4)

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

# --------------------------------------------------------------------------------
# Utility to convert model to half precision
# --------------------------------------------------------------------------------
function f16(model)
    return fmap(x -> isa(x, AbstractArray) ? Float16.(x) : x, model)
end

# --------------------------------------------------------------------------------
# Utility to convert model back to single precision
# --------------------------------------------------------------------------------
function f32(model)
    return fmap(x -> isa(x, AbstractArray) ? Float32.(x) : x, model)
end

end # module Model
