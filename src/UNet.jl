
function unet(input_channels::Int, output_channels::Int)
    # Encoder
    encoder = Chain(
        (x -> begin
            println("Eingabegröße Encoder: ", size(x))
            x
        end),
        Conv((2, 3), input_channels => 64, relu, pad=1),  # Erste Convolution
        (x -> begin
            println("Nach Conv: ", size(x))
            x
        end),
        MaxPool((2, 2), stride=(2, 2)),                   # Einmaliges Downsampling
        (x -> begin
            println("Nach MaxPool: ", size(x))
            x
        end),
        Conv((3, 3), 64 => 128, relu, pad=1),
        (x -> begin
            println("Nach Conv_2: ", size(x))
            x
        end),             # Zweite Convolution
        MaxPool((2, 2), stride=(2, 3)),                   # Zweites Downsampling
        (x -> begin
            println("Nach zweiter MaxPool: ", size(x))
            x
        end)
        
    )
    
    # Decoder
    decoder = Chain(
        (x -> begin
            println("Eingabegröße Decoder: ", size(x))
            x
        end),
        
        ConvTranspose((5, 3), 128 => 64, stride=(2,3),pad=(1,0)),   # Zweites Upsampling
        (x -> begin
            println("Nach erstem Upsampling: ", size(x))
            x
        end),
        ConvTranspose((1, 4), 64 => output_channels, stride=(2,2),pad=(1, 1)),  # Upsampling
        (x -> begin
            println("Nach finalem ConvTranspose: ", size(x))
            x
        end)
        
        
    )
    
    return Chain(encoder, decoder)
end