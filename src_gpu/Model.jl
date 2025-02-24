##################################
# Model.jl
##################################
module Model

using Flux
using Functors
using Statistics

# UNet als strukturiertes Modell
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
end
Functors.@functor UNet

function UNet(input_channels::Int, output_channels::Int)
    # Encoder
    encoder1 = Chain(
        Conv((3, 3), input_channels => 64, relu, pad=1),
        Conv((3, 3), 64 => 64, relu, pad=1)
    )
    encoder2 = Chain(
        MaxPool((2,2)),
        Conv((2, 2), 64 => 128, relu, pad=1),
        Conv((3, 3), 128 => 128, relu, pad=1)
    )
    encoder3 = Chain(
        MaxPool((2,2)),
        Conv((3, 3), 128 => 256, relu, pad=1),
        Conv((3, 3), 256 => 256, relu, pad=1)
    )
    encoder4 = Chain(
        MaxPool((2,2)),
        Conv((2, 2), 256 => 512, relu, pad=1),
        Conv((3, 3), 512 => 512, relu, pad=1)
    )
    bottleneck = Chain(
        MaxPool((2,2)),
        Conv((2,2), 512 => 1024, relu, pad=1),
        Conv((2,2), 1024 => 1024, relu, pad=1)
    )
    # Decoder
    decoder4 = Chain(
        ConvTranspose((2, 2), 1024 => 512, stride=2, pad=2)
    )
    decoder4_1 = Chain(
        Conv((2,2), 1024 => 512, relu, pad=1),
        Conv((2,2), 512 => 512, relu, pad=1)
    )
    decoder3 = Chain(
        ConvTranspose((2,2), 512 => 256, stride=2, pad=3)
    )
    decoder3_1 = Chain(
        Conv((3,3), 512 => 256, relu, pad=1),
        Conv((3,3), 256 => 256, relu, pad=1)
    )
    decoder2 = Chain(
        ConvTranspose((2,2), 256 => 128, stride=2, pad=0)
    )
    decoder2_1 = Chain(
        Conv((3,3), 256 => 128, relu, pad=1),
        Conv((3,3), 128 => 128, relu, pad=1)
    )
    decoder1 = Chain(
        ConvTranspose((2,2), 128 => 64, stride=2, pad=1)
    )
    decoder1_1 = Chain(
        Conv((3,3), 128 => 64, relu, pad=1),
        Conv((3,3), 64 => 64, relu, pad=1),
        Conv((1,1), 64 => output_channels)
    )
    return UNet(encoder1, encoder2, encoder3, encoder4, bottleneck,
                decoder4, decoder4_1, decoder3, decoder3_1,
                decoder2, decoder2_1, decoder1, decoder1_1)
end

function (model::UNet)(x)
    e1 = model.encoder1(x)
    e2 = model.encoder2(e1)
    e3 = model.encoder3(e2)
    e4 = model.encoder4(e3)
    b  = model.bottleneck(e4)

    d4 = model.decoder4(b)
    d4 = model.decoder4_1(cat(d4, e4, dims=3))

    d3 = model.decoder3(d4)
    d3 = model.decoder3_1(cat(d3, e3, dims=3))

    d2 = model.decoder2(d3)
    d2 = model.decoder2_1(cat(d2, e2, dims=3))

    d1 = model.decoder1(d2)
    d1 = model.decoder1_1(cat(d1, e1, dims=3))

    return d1
end

end # module Model
