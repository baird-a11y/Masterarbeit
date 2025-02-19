module Model

using Flux
using Functors
using Statistics

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
    output_channels::Int
end

Functors.@functor UNet

function UNet(input_channels::Int, output_channels::Int)
    # Encoder-Stufe 1
    encoder1 = Chain(
        Conv((3,3), input_channels => 64, relu, pad=1),
        Conv((3,3), 64 => 64, relu, pad=1)
    )
    # Encoder-Stufe 2
    encoder2 = Chain(
        MaxPool((2,2)),
        Conv((3,3), 64 => 128, relu, pad=1),
        Conv((3,3), 128 => 128, relu, pad=1)
    )
    # Encoder-Stufe 3
    encoder3 = Chain(
        MaxPool((2,2)),
        Conv((3,3), 128 => 256, relu, pad=1),
        Conv((3,3), 256 => 256, relu, pad=1)
    )
    # Encoder-Stufe 4
    encoder4 = Chain(
        MaxPool((2,2)),
        Conv((3,3), 256 => 512, relu, pad=1),
        Conv((3,3), 512 => 512, relu, pad=1)
    )
    # Bottleneck
    bottleneck = Chain(
        MaxPool((2,2)),
        Conv((3,3), 512 => 1024, relu, pad=1),
        Conv((3,3), 1024 => 1024, relu, pad=1)
    )

    # Decoder-Stufe 4
    decoder4 = Chain(
        ConvTranspose((2,2), 1024 => 512, stride=2)
    )
    decoder4_1 = Chain(
        Conv((3,3), 1024 => 512, relu, pad=1),
        Conv((3,3), 512 => 512, relu, pad=1)
    )
    # Decoder-Stufe 3
    decoder3 = Chain(
        ConvTranspose((2,2), 512 => 256, stride=2)
    )
    decoder3_1 = Chain(
        Conv((3,3), 512 => 256, relu, pad=1),
        Conv((3,3), 256 => 256, relu, pad=1)
    )
    # Decoder-Stufe 2
    decoder2 = Chain(
        ConvTranspose((2,2), 256 => 128, stride=2)
    )
    decoder2_1 = Chain(
        Conv((3,3), 256 => 128, relu, pad=1),
        Conv((3,3), 128 => 128, relu, pad=1)
    )
    # Decoder-Stufe 1
    decoder1 = Chain(
        ConvTranspose((2,2), 128 => 64, stride=2)
    )
    decoder1_1 = Chain(
        Conv((3,3), 128 => 64, relu, pad=1),
        Conv((3,3), 64 => 64, relu, pad=1),
        Conv((1,1), 64 => output_channels)  # letztes Conv
    )
    
    return UNet(encoder1, encoder2, encoder3, encoder4, bottleneck,
                decoder4, decoder4_1, decoder3, decoder3_1,
                decoder2, decoder2_1, decoder1, decoder1_1, output_channels)
end

function (model::UNet)(x)
    # ================== Encoder ==================
    println("DEBUG: Input shape: ", size(x))
    e1 = model.encoder1(x)
    #println("DEBUG: e1 shape: ", size(e1), " Mean: ", mean(e1), " Std: ", std(e1))

    e2 = model.encoder2(e1)
    #println("DEBUG: e2 shape: ", size(e2), " Mean: ", mean(e2), " Std: ", std(e2))

    e3 = model.encoder3(e2)
    #println("DEBUG: e3 shape: ", size(e3), " Mean: ", mean(e3), " Std: ", std(e3))

    e4 = model.encoder4(e3)
    #println("DEBUG: e4 shape: ", size(e4), " Mean: ", mean(e4), " Std: ", std(e4))
    
    b = model.bottleneck(e4)
    #println("DEBUG: Bottleneck shape: ", size(b), " Mean: ", mean(b), " Std: ", std(b))
    
    # ================== Decoder ==================
    # Decoder 4 (skip mit e4)
    d4 = model.decoder4(b)
    #println("DEBUG: d4 shape (before skip): ", size(d4), " Mean: ", mean(d4), " Std: ", std(d4))
    d4 = model.decoder4_1(cat(d4, e4, dims=3))
    #println("DEBUG: d4 shape (after skip): ", size(d4), " Mean: ", mean(d4), " Std: ", std(d4))
    
    # Decoder 3 (skip mit e3)
    d3 = model.decoder3(d4)
    #println("DEBUG: d3 shape (before skip): ", size(d3), " Mean: ", mean(d3), " Std: ", std(d3))
    d3 = model.decoder3_1(cat(d3, e3, dims=3))
    #println("DEBUG: d3 shape (after skip): ", size(d3), " Mean: ", mean(d3), " Std: ", std(d3))
    
    # Decoder 2 (skip mit e2)
    d2 = model.decoder2(d3)
    #println("DEBUG: d2 shape (before skip): ", size(d2), " Mean: ", mean(d2), " Std: ", std(d2))
    d2 = model.decoder2_1(cat(d2, e2, dims=3))
    #println("DEBUG: d2 shape (after skip): ", size(d2), " Mean: ", mean(d2), " Std: ", std(d2))
    
    # Decoder 1 (skip mit e1)
    d1 = model.decoder1(d2)
    #println("DEBUG: d1 shape (before skip): ", size(d1), " Mean: ", mean(d1), " Std: ", std(d1))
    d1 = model.decoder1_1(cat(d1, e1, dims=3))
    #println("DEBUG: d1 shape (final): ", size(d1), " Mean: ", mean(d1), " Std: ", std(d1))
    
    # ================== Output-Aktivierung ==================
    if model.output_channels == 1
        # z.B. binäre Segmentierung
        return sigmoid.(d1)
    else
        # Mehrklassen: Softmax über Kanal-Dimension
        return softmax(d1, dims=3)
    end
end

end # module Model
