##################################
# Model.jl - Optimiert für UNet mit Memory Efficiency
##################################
module Model

using Flux
using Functors
using CUDA
using Statistics

# UNet als strukturiertes Modell mit skip connections
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
    dropout
end
Functors.@functor UNet

# Optimierte Implementierung von UNet
function UNet(input_channels::Int, output_channels::Int; dropout_rate=0.2, memory_efficient=true)
    # Feature Map Größen basierend auf memory_efficient Parameter
    if memory_efficient
        # Reduzierte Feature Map Größen für weniger Speicherverbrauch
        init_filters = 48
    else
        # Standard Feature Map Größen
        init_filters = 64
    end
    
    # Skalierungsfaktoren für die Feature Map Größen in den verschiedenen Stufen
    f1 = init_filters
    f2 = 2 * f1
    f3 = 2 * f2
    f4 = 2 * f3
    f5 = 2 * f4
    
    # Encoder Blöcke
    encoder1 = Chain(
        Conv((3, 3), input_channels => f1, relu, pad=SamePad()),
        BatchNorm(f1),
        Conv((3, 3), f1 => f1, relu, pad=SamePad()),
        BatchNorm(f1)
    )
    encoder2 = Chain(
        MaxPool((2,2)),
        Conv((3, 3), f1 => f2, relu, pad=SamePad()),
        BatchNorm(f2),
        Conv((3, 3), f2 => f2, relu, pad=SamePad()),
        BatchNorm(f2)
    )
    encoder3 = Chain(
        MaxPool((2,2)),
        Conv((3, 3), f2 => f3, relu, pad=SamePad()),
        BatchNorm(f3),
        Conv((3, 3), f3 => f3, relu, pad=SamePad()),
        BatchNorm(f3)
    )
    encoder4 = Chain(
        MaxPool((2,2)),
        Conv((3, 3), f3 => f4, relu, pad=SamePad()),
        BatchNorm(f4),
        Conv((3, 3), f4 => f4, relu, pad=SamePad()),
        BatchNorm(f4)
    )
    bottleneck = Chain(
        MaxPool((2,2)),
        Conv((3,3), f4 => f5, relu, pad=SamePad()),
        BatchNorm(f5),
        Dropout(dropout_rate),
        Conv((3,3), f5 => f5, relu, pad=SamePad()),
        BatchNorm(f5)
    )
    
    # Decoder Blöcke
    decoder4 = ConvTranspose((2, 2), f5 => f4, stride=2)
    decoder4_1 = Chain(
        Conv((3,3), 2*f4 => f4, relu, pad=SamePad()),
        BatchNorm(f4),
        Conv((3,3), f4 => f4, relu, pad=SamePad()),
        BatchNorm(f4)
    )
    decoder3 = ConvTranspose((2,2), f4 => f3, stride=2)
    decoder3_1 = Chain(
        Conv((3,3), 2*f3 => f3, relu, pad=SamePad()),
        BatchNorm(f3),
        Conv((3,3), f3 => f3, relu, pad=SamePad()),
        BatchNorm(f3)
    )
    decoder2 = ConvTranspose((2,2), f3 => f2, stride=2)
    decoder2_1 = Chain(
        Conv((3,3), 2*f2 => f2, relu, pad=SamePad()),
        BatchNorm(f2),
        Conv((3,3), f2 => f2, relu, pad=SamePad()),
        BatchNorm(f2)
    )
    decoder1 = ConvTranspose((2,2), f2 => f1, stride=2)
    decoder1_1 = Chain(
        Conv((3,3), 2*f1 => f1, relu, pad=SamePad()),
        BatchNorm(f1),
        Conv((3,3), f1 => f1, relu, pad=SamePad()),
        BatchNorm(f1),
        Dropout(dropout_rate/2),
        Conv((1,1), f1 => output_channels)
    )
    
    dropout = Dropout(dropout_rate)
    
    return UNet(encoder1, encoder2, encoder3, encoder4, bottleneck,
                decoder4, decoder4_1, decoder3, decoder3_1,
                decoder2, decoder2_1, decoder1, decoder1_1, dropout)
end

# Hilfsfunktion zum Zuschneiden und Konkatenieren für Skip-Connections
function crop_and_concat(x, skip, dims=3)
    x_size = size(x)
    skip_size = size(skip)
    
    # Handle ungleiche Dimensionen
    if x_size[1:2] == skip_size[1:2]
        # Einfacher Fall: Dimensionen passen bereits
        return cat(x, skip, dims=dims)
    else
        # Komplexerer Fall: Dimensionen passen nicht, Anpassung notwendig
        height_diff = skip_size[1] - x_size[1]
        width_diff  = skip_size[2] - x_size[2]
        
        if height_diff < 0 || width_diff < 0
            # Skip connection ist kleiner, auffüllen
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
            # Skip connection ist größer, zuschneiden
            h_start = height_diff ÷ 2 + 1
            w_start = width_diff  ÷ 2 + 1
            
            cropped_skip = skip[h_start:h_start+x_size[1]-1,
                                w_start:w_start+x_size[2]-1, :, :]
            
            return cat(x, cropped_skip, dims=dims)
        end
    end
end

# Standard Forward Pass
function (model::UNet)(x)
    # Encoder-Pfad
    e1 = model.encoder1(x)
    e2 = model.encoder2(e1)
    e3 = model.encoder3(e2)
    e4 = model.encoder4(e3)
    b  = model.bottleneck(e4)

    # Decoder-Pfad mit Skip-Connections
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

end # module Model