# =============================================================================
# ZYGOTE-SICHERE UNET ARCHITECTURE
# =============================================================================
# Speichern als: unet_architecture_zygote_safe.jl

using Flux
"""
GPU-kompatible Skip-Dimensionen-Anpassung
"""
function adapt_skip_dimensions_safe(skip_features, decoder_features)
    skip_size = size(skip_features)
    decoder_size = size(decoder_features)
    
    if skip_size[1:2] == decoder_size[1:2]
        return skip_features
    end
    
    target_h, target_w = decoder_size[1:2]
    current_h, current_w = skip_size[1:2]
    
    # Center-Crop für GPU-Kompatibilität
    if current_h >= target_h && current_w >= target_w
        h_start = (current_h - target_h) ÷ 2 + 1
        w_start = (current_w - target_w) ÷ 2 + 1
        
        # GPU-sicheres Slicing mit @views
        @views cropped = skip_features[h_start:(h_start+target_h-1), 
                                      w_start:(w_start+target_w-1), :, :]
        return cropped
    else
        # GPU-sichere Zero-Padding
        T = eltype(skip_features)
        
        if isa(skip_features, CuArray)
            padded = CUDA.zeros(T, target_h, target_w, skip_size[3], skip_size[4])
        else
            padded = zeros(T, target_h, target_w, skip_size[3], skip_size[4])
        end
        
        h_copy = min(current_h, target_h)
        w_copy = min(current_w, target_w)
        
        @views padded[1:h_copy, 1:w_copy, :, :] .= skip_features[1:h_copy, 1:w_copy, :, :]
        
        return padded
    end
end

"""
Vereinfachte UNet-Struktur mit Batch Normalization
"""
struct SimplifiedUNetBN
    # Encoder
    enc1_conv1::Conv
    enc1_bn1::BatchNorm
    enc1_conv2::Conv
    enc1_bn2::BatchNorm
    enc1_pool::MaxPool
    
    enc2_conv1::Conv
    enc2_bn1::BatchNorm
    enc2_conv2::Conv
    enc2_bn2::BatchNorm
    enc2_pool::MaxPool
    
    enc3_conv1::Conv
    enc3_bn1::BatchNorm
    enc3_conv2::Conv
    enc3_bn2::BatchNorm
    enc3_pool::MaxPool
    
    # Bottleneck
    bottleneck_conv1::Conv
    bottleneck_bn1::BatchNorm
    bottleneck_conv2::Conv
    bottleneck_bn2::BatchNorm
    
    # Decoder
    dec3_up::ConvTranspose
    dec3_conv1::Conv
    dec3_bn1::BatchNorm
    dec3_conv2::Conv
    dec3_bn2::BatchNorm
    
    dec2_up::ConvTranspose
    dec2_conv1::Conv
    dec2_bn1::BatchNorm
    dec2_conv2::Conv
    dec2_bn2::BatchNorm
    
    dec1_up::ConvTranspose
    dec1_conv1::Conv
    dec1_bn1::BatchNorm
    dec1_conv2::Conv
    dec1_bn2::BatchNorm
    
    # Output mit Tanh für begrenzte Ausgabe
    output_conv::Conv
end

Flux.@layer SimplifiedUNetBN

"""
Erstellt UNet mit Batch Normalization und Output-Skalierung
"""
function create_simplified_unet_bn(input_channels=1, output_channels=2, base_filters=32)
    f = base_filters
    
    SimplifiedUNetBN(
        # Encoder 1
        Conv((3, 3), input_channels => f, pad=SamePad()),
        BatchNorm(f, relu),
        Conv((3, 3), f => f, pad=SamePad()),
        BatchNorm(f, relu),
        MaxPool((2, 2)),
        
        # Encoder 2
        Conv((3, 3), f => 2f, pad=SamePad()),
        BatchNorm(2f, relu),
        Conv((3, 3), 2f => 2f, pad=SamePad()),
        BatchNorm(2f, relu),
        MaxPool((2, 2)),
        
        # Encoder 3
        Conv((3, 3), 2f => 4f, pad=SamePad()),
        BatchNorm(4f, relu),
        Conv((3, 3), 4f => 4f, pad=SamePad()),
        BatchNorm(4f, relu),
        MaxPool((2, 2)),
        
        # Bottleneck
        Conv((3, 3), 4f => 8f, pad=SamePad()),
        BatchNorm(8f, relu),
        Conv((3, 3), 8f => 8f, pad=SamePad()),
        BatchNorm(8f, relu),
        
        # Decoder 3
        ConvTranspose((2, 2), 8f => 4f, stride=2),
        Conv((3, 3), 8f => 4f, pad=SamePad()),
        BatchNorm(4f, relu),
        Conv((3, 3), 4f => 4f, pad=SamePad()),
        BatchNorm(4f, relu),
        
        # Decoder 2
        ConvTranspose((2, 2), 4f => 2f, stride=2),
        Conv((3, 3), 4f => 2f, pad=SamePad()),
        BatchNorm(2f, relu),
        Conv((3, 3), 2f => 2f, pad=SamePad()),
        BatchNorm(2f, relu),
        
        # Decoder 1
        ConvTranspose((2, 2), 2f => f, stride=2),
        Conv((3, 3), 2f => f, pad=SamePad()),
        BatchNorm(f, relu),
        Conv((3, 3), f => f, pad=SamePad()),
        BatchNorm(f, relu),
        
        # Output - KEIN Tanh, damit Modell freien Wertebereich lernen kann
        Conv((1, 1), f => output_channels)
    )
end

"""
Forward-Pass mit Batch Normalization
"""
function (model::SimplifiedUNetBN)(x)
    # Encoder 1
    enc1 = model.enc1_bn1(model.enc1_conv1(x))
    enc1 = model.enc1_bn2(model.enc1_conv2(enc1))
    enc1_pooled = model.enc1_pool(enc1)
    
    # Encoder 2
    enc2 = model.enc2_bn1(model.enc2_conv1(enc1_pooled))
    enc2 = model.enc2_bn2(model.enc2_conv2(enc2))
    enc2_pooled = model.enc2_pool(enc2)
    
    # Encoder 3
    enc3 = model.enc3_bn1(model.enc3_conv1(enc2_pooled))
    enc3 = model.enc3_bn2(model.enc3_conv2(enc3))
    enc3_pooled = model.enc3_pool(enc3)
    
    # Bottleneck
    bottleneck = model.bottleneck_bn1(model.bottleneck_conv1(enc3_pooled))
    bottleneck = model.bottleneck_bn2(model.bottleneck_conv2(bottleneck))
    
    # Decoder 3
    dec3_up = model.dec3_up(bottleneck)
    enc3_adapted = adapt_skip_dimensions_safe(enc3, dec3_up)
    dec3_concat = cat(dec3_up, enc3_adapted, dims=3)
    dec3 = model.dec3_bn1(model.dec3_conv1(dec3_concat))
    dec3 = model.dec3_bn2(model.dec3_conv2(dec3))
    
    # Decoder 2
    dec2_up = model.dec2_up(dec3)
    enc2_adapted = adapt_skip_dimensions_safe(enc2, dec2_up)
    dec2_concat = cat(dec2_up, enc2_adapted, dims=3)
    dec2 = model.dec2_bn1(model.dec2_conv1(dec2_concat))
    dec2 = model.dec2_bn2(model.dec2_conv2(dec2))
    
    # Decoder 1
    dec1_up = model.dec1_up(dec2)
    enc1_adapted = adapt_skip_dimensions_safe(enc1, dec1_up)
    dec1_concat = cat(dec1_up, enc1_adapted, dims=3)
    dec1 = model.dec1_bn1(model.dec1_conv1(dec1_concat))
    dec1 = model.dec1_bn2(model.dec1_conv2(dec1))
    
    # Output
    output = model.output_conv(dec1)
    
    return output
end
"""
Test-Funktion für vereinfachtes UNet
"""
function test_simplified_unet(resolution=256; batch_size=2)
    model = create_simplified_unet()
    test_input = randn(Float32, resolution, resolution, 1, batch_size)
    
    try
        output = model(test_input)
        expected_shape = (resolution, resolution, 2, batch_size)
        success = size(output) == expected_shape
        
        println("Test UNet:")
        println("  Input: $(size(test_input))")
        println("  Output: $(size(output))")
        println("  Erwartet: $expected_shape")
        println("  Erfolg: $success")
        
        return success, model
    catch e
        println("UNet Test fehlgeschlagen: $e")
        return false, nothing
    end
end

println("Zygote-sichere UNet Architecture geladen!")
println("Verwende: create_simplified_unet()")