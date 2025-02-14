module Model

using Flux
using Functors

# UNet-Struktur mit Functors, damit Flux.trainable() funktioniert
struct UNet
    layers
end
Functors.@functor UNet

function UNet(input_channels::Int, output_channels::Int)
    # Encoder
    encoder1 = Chain(Conv((3, 3), input_channels => 64, relu, pad=1),
                     Conv((3, 3), 64 => 64, relu, pad=1))
    encoder2 = Chain(MaxPool((2, 2)),
                     Conv((2, 2), 64 => 128, relu, pad=1),
                     Conv((3, 3), 128 => 128, relu, pad=1))
    encoder3 = Chain(MaxPool((2, 2)),
                     Conv((2, 2), 128 => 256, relu, pad=1),
                     Conv((3, 3), 256 => 256, relu, pad=1))
    encoder4 = Chain(MaxPool((2, 2)),
                     Conv((2, 2), 256 => 512, relu, pad=1),
                     Conv((3, 3), 512 => 512, relu, pad=1))
    bottleneck = Chain(MaxPool((2, 2)),
                       Conv((2, 2), 512 => 1024, relu, pad=1),
                       Conv((3, 3), 1024 => 1024, relu, pad=1))
    # Decoder (ohne echte Skip-Verbindungen – das muss später ergänzt werden)
    decoder4 = Chain(ConvTranspose((2, 3), 1024 => 512, stride=2, pad=1))
    decoder4_1 = Chain(Conv((3, 3), 1024 => 512, relu, pad=1),
                       Conv((3, 3), 512 => 512, relu, pad=1))
    decoder3 = Chain(ConvTranspose((3, 2), 512 => 256, stride=2, pad=1))
    decoder3_1 = Chain(Conv((3, 3), 512 => 256, relu, pad=1),
                       Conv((3, 3), 256 => 256, relu, pad=1))
    decoder2 = Chain(ConvTranspose((2, 2), 256 => 128, stride=2, pad=1))
    decoder2_1 = Chain(Conv((3, 3), 256 => 128, relu, pad=1),
                       Conv((3, 3), 128 => 128, relu, pad=1))
    decoder1 = Chain(ConvTranspose((3, 2), 128 => 64, stride=2, pad=1))
    decoder1_1 = Chain(Conv((3, 3), 128 => 64, relu, pad=1),
                       Conv((3, 3), 64 => 64, relu, pad=1),
                       Conv((1, 1), 64 => output_channels))
    layers = Chain(
        encoder1, encoder2, encoder3, encoder4, bottleneck, 
        decoder4, decoder4_1, decoder3, decoder3_1, decoder2, decoder2_1, 
        decoder1, decoder1_1
    )
    return UNet(layers)
end

# Definiert den Aufruf, sodass model(x) die Daten durch alle Layer leitet
function (model::UNet)(x)
    return model.layers(x)
end

end # module Model
