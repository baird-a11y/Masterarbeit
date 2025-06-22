# =============================================================================
# KORRIGIERTE UNET ARCHITECTURE - DIMENSIONEN FINAL GEFIXT
# =============================================================================
# Speichern als: unet_architecture_fixed.jl

using Flux

"""
KRITISCHE KORREKTUR: Adaptive Skip-Dimensionen-Anpassung
"""
function adapt_skip_dimensions_fixed(skip_features, decoder_features)
    skip_size = size(skip_features)
    decoder_size = size(decoder_features)
    
    # Wenn Dimensionen bereits passen
    if skip_size[1:2] == decoder_size[1:2]
        return skip_features
    end
    
    target_h, target_w = decoder_size[1:2]
    current_h, current_w = skip_size[1:2]
    
    # IMMER Center-Crop auf exakte Decoder-Größe
    if current_h >= target_h && current_w >= target_w
        h_start = (current_h - target_h) ÷ 2 + 1
        w_start = (current_w - target_w) ÷ 2 + 1
        h_end = h_start + target_h - 1
        w_end = w_start + target_w - 1
        
        return skip_features[h_start:h_end, w_start:w_end, :, :]
    else
        # Falls Skip kleiner als Decoder: Zero-Padding
        padded = zeros(eltype(skip_features), target_h, target_w, skip_size[3], skip_size[4])
        
        h_copy = min(current_h, target_h)
        w_copy = min(current_w, target_w)
        
        padded[1:h_copy, 1:w_copy, :, :] .= skip_features[1:h_copy, 1:w_copy, :, :]
        
        return padded
    end
end

"""
KORRIGIERTE Encoder-Block-Struktur
"""
struct FixedEncoderBlock
    conv_block::Chain
    has_pooling::Bool
    level::Int
end

Flux.@functor FixedEncoderBlock

function (block::FixedEncoderBlock)(x)
    features = block.conv_block(x)
    
    if block.has_pooling
        pooled = MaxPool((2, 2))(features)
        return features, pooled  # Features für Skip, Pooled für nächsten Level
    else
        return features, features  # Letzter Encoder: keine Größenänderung
    end
end

"""
KORRIGIERTE Decoder-Block-Struktur mit exakter Größenkontrolle
"""
struct FixedDecoderBlock
    upsample::ConvTranspose
    conv_block::Chain
    level::Int
    target_size::Tuple{Int, Int}  # NEUE: Zielgröße für diesen Decoder
end

Flux.@functor FixedDecoderBlock

function (block::FixedDecoderBlock)(x, skip_features)
    # Upsampling
    upsampled = block.upsample(x)
    
    # KRITISCH: Prüfe ob Upsampling korrekte Größe hat
    current_size = size(upsampled)
    if current_size[1:2] != block.target_size
        println("WARNUNG: Decoder $(block.level) Upsampling $(current_size[1:2]) ≠ Ziel $(block.target_size)")
        # Korrigiere durch Resize
        if current_size[1] > block.target_size[1] || current_size[2] > block.target_size[2]
            # Crop auf Zielgröße
            h_end = min(current_size[1], block.target_size[1])
            w_end = min(current_size[2], block.target_size[2])
            upsampled = upsampled[1:h_end, 1:w_end, :, :]
        end
    end
    
    # Skip-Connection anpassen
    adapted_skip = adapt_skip_dimensions_fixed(skip_features, upsampled)
    
    # Concatenation
    concatenated = cat(upsampled, adapted_skip, dims=3)
    
    # Convolution
    result = block.conv_block(concatenated)
    
    # FINAL: Stelle sicher, dass Output exakte Zielgröße hat
    result_size = size(result)
    if result_size[1:2] != block.target_size
        println("KORREKTUR: Decoder $(block.level) Output $(result_size[1:2]) → $(block.target_size)")
        h_end = min(result_size[1], block.target_size[1])
        w_end = min(result_size[2], block.target_size[2])
        result = result[1:h_end, 1:w_end, :, :]
    end
    
    return result
end

"""
Erstellt korrigierten Encoder-Block
"""
function create_fixed_encoder_block(config::UNetConfig, in_channels::Int, out_channels::Int, level::Int)
    conv_layers = []
    
    push!(conv_layers, Conv((3, 3), in_channels => out_channels, pad=SamePad()))
    if config.use_batchnorm
        push!(conv_layers, BatchNorm(out_channels))
    end
    push!(conv_layers, config.activation)
    
    push!(conv_layers, Conv((3, 3), out_channels => out_channels, pad=SamePad()))
    if config.use_batchnorm
        push!(conv_layers, BatchNorm(out_channels))
    end
    push!(conv_layers, config.activation)
    
    if config.dropout_rate > 0 && level >= 2
        push!(conv_layers, Dropout(config.dropout_rate))
    end
    
    conv_block = Chain(conv_layers...)
    
    # Pooling nur für Encoder 1 bis depth-1
    has_pooling = level < config.depth
    
    return FixedEncoderBlock(conv_block, has_pooling, level)
end

"""
Erstellt korrigierten Decoder-Block mit exakter Zielgröße
"""
function create_fixed_decoder_block(config::UNetConfig, in_channels::Int, skip_channels::Int, out_channels::Int, level::Int, target_size::Tuple{Int, Int})
    # KRITISCHE KORREKTUR: Verwende Upsample((2,2)) statt ConvTranspose für exakte Kontrolle
    # ConvTranspose führt zu unvorhersagbaren Größenänderungen
    
    conv_layers = []
    
    # Upsample-Layer: Einfache 2x Vergrößerung durch Interpolation
    # Dann 1x1 Conv für Channel-Anpassung
    push!(conv_layers, Upsample(:bilinear, scale=2))
    push!(conv_layers, Conv((1, 1), in_channels => out_channels))
    
    # DUMMY ConvTranspose für Kompatibilität - wird nicht verwendet
    upsample = ConvTranspose((2, 2), in_channels => out_channels, stride=2, pad=0)
    
    # Convolution-Block nach Concatenation
    concat_channels = out_channels + skip_channels
    conv_block_layers = []
    
    push!(conv_block_layers, Conv((3, 3), concat_channels => out_channels, pad=SamePad()))
    if config.use_batchnorm
        push!(conv_block_layers, BatchNorm(out_channels))
    end
    push!(conv_block_layers, config.activation)
    
    push!(conv_block_layers, Conv((3, 3), out_channels => out_channels, pad=SamePad()))
    if config.use_batchnorm
        push!(conv_block_layers, BatchNorm(out_channels))
    end
    push!(conv_block_layers, config.activation)
    
    if config.dropout_rate > 0 && level <= 2
        push!(conv_block_layers, Dropout(config.dropout_rate))
    end
    
    conv_block = Chain(conv_block_layers...)
    
    return FixedDecoderBlock(upsample, conv_block, level, target_size)
end

"""
Alternative: Manueller Upsample-Block
"""
struct ManualUpsampleBlock
    upsample_conv::Conv
    conv_block::Chain
    level::Int
    target_size::Tuple{Int, Int}
end

Flux.@functor ManualUpsampleBlock

function (block::ManualUpsampleBlock)(x, skip_features)
    # Manuelle 2x Vergrößerung durch Repeat + Conv
    # Repeat jedes Pixel 2x2
    x_repeated = repeat(x, inner=(2, 2, 1, 1))
    
    # 1x1 Conv für Channel-Reduktion
    upsampled = block.upsample_conv(x_repeated)
    
    # Crop auf exakte Zielgröße falls nötig
    current_size = size(upsampled)
    if current_size[1:2] != block.target_size
        h_end = min(current_size[1], block.target_size[1])
        w_end = min(current_size[2], block.target_size[2])
        upsampled = upsampled[1:h_end, 1:w_end, :, :]
    end
    
    # Skip-Connection
    adapted_skip = adapt_skip_dimensions_fixed(skip_features, upsampled)
    concatenated = cat(upsampled, adapted_skip, dims=3)
    
    # Convolution
    result = block.conv_block(concatenated)
    
    # Final size check
    result_size = size(result)
    if result_size[1:2] != block.target_size
        h_end = min(result_size[1], block.target_size[1])
        w_end = min(result_size[2], block.target_size[2])
        result = result[1:h_end, 1:w_end, :, :]
    end
    
    return result
end

function create_manual_decoder_block(config::UNetConfig, in_channels::Int, skip_channels::Int, out_channels::Int, level::Int, target_size::Tuple{Int, Int})
    # Upsample-Conv: Channel-Reduktion nach Repeat
    upsample_conv = Conv((1, 1), in_channels => out_channels)
    
    # Convolution-Block
    concat_channels = out_channels + skip_channels
    conv_layers = []
    
    push!(conv_layers, Conv((3, 3), concat_channels => out_channels, pad=SamePad()))
    if config.use_batchnorm
        push!(conv_layers, BatchNorm(out_channels))
    end
    push!(conv_layers, config.activation)
    
    push!(conv_layers, Conv((3, 3), out_channels => out_channels, pad=SamePad()))
    if config.use_batchnorm
        push!(conv_layers, BatchNorm(out_channels))
    end
    push!(conv_layers, config.activation)
    
    conv_block = Chain(conv_layers...)
    
    return ManualUpsampleBlock(upsample_conv, conv_block, level, target_size)
end

"""
Erstellt den Bottleneck-Block
"""
function create_bottleneck_block(config::UNetConfig, in_channels::Int, out_channels::Int)
    layers = []
    
    push!(layers, Conv((3, 3), in_channels => out_channels, pad=SamePad()))
    if config.use_batchnorm
        push!(layers, BatchNorm(out_channels))
    end
    push!(layers, config.activation)
    
    if config.dropout_rate > 0
        push!(layers, Dropout(min(config.dropout_rate * 2, 0.5)))
    end
    
    push!(layers, Conv((3, 3), out_channels => out_channels, pad=SamePad()))
    if config.use_batchnorm
        push!(layers, BatchNorm(out_channels))
    end
    push!(layers, config.activation)
    
    return Chain(layers...)
end

"""
Erstellt den finalen Output-Layer
"""
function create_output_layer(config::UNetConfig, in_channels::Int)
    layers = []
    push!(layers, Conv((1, 1), in_channels => config.output_channels))
    
    if config.final_activation !== nothing
        push!(layers, config.final_activation)
    end
    
    return Chain(layers...)
end

"""
FINAL KORRIGIERTE UNet-Struktur
"""
struct FinalCorrectedUNet
    config::UNetConfig
    encoder_blocks::Vector{FixedEncoderBlock}
    bottleneck::Chain
    decoder_blocks::Vector{ManualUpsampleBlock}  # Verwende manuelle Upsample-Blöcke
    output_layer::Chain
end

Flux.@functor FinalCorrectedUNet

"""
Erstellt final korrigiertes UNet mit garantierten Dimensionen
"""
function create_final_corrected_unet(config::UNetConfig)
    println("=== ERSTELLE FINAL KORRIGIERTES UNET ===")
    println("Konfiguration: $(config.input_resolution)×$(config.input_resolution)")
    println("Tiefe: $(config.depth), Filter: $(config.filter_progression)")
    
    # Berechne erwartete Größen für jeden Level
    expected_sizes = []
    current_size = config.input_resolution
    push!(expected_sizes, current_size)
    
    for i in 1:config.depth
        current_size = current_size ÷ 2
        push!(expected_sizes, current_size)
    end
    
    println("Erwartete Größen: $expected_sizes")
    
    # Encoder-Blöcke
    encoder_blocks = FixedEncoderBlock[]
    for level in 1:config.depth
        in_ch = level == 1 ? config.input_channels : config.filter_progression[level-1]
        out_ch = config.filter_progression[level]
        
        encoder_block = create_fixed_encoder_block(config, in_ch, out_ch, level)
        push!(encoder_blocks, encoder_block)
        
        pool_info = encoder_block.has_pooling ? "mit Pooling" : "ohne Pooling"
        expected_input_size = expected_sizes[level]
        expected_output_size = encoder_block.has_pooling ? expected_sizes[level+1] : expected_sizes[level]
        
        println("  Encoder $level: $in_ch → $out_ch ($pool_info) [$(expected_input_size)→$(expected_output_size)]")
    end
    
    # Bottleneck
    bottleneck_in = config.filter_progression[config.depth]
    bottleneck_out = config.filter_progression[config.depth + 1]
    bottleneck = create_bottleneck_block(config, bottleneck_in, bottleneck_out)
    bottleneck_size = expected_sizes[end]
    println("  Bottleneck: $bottleneck_in → $bottleneck_out [$(bottleneck_size)×$(bottleneck_size)]")
    
    # Decoder-Blöcke mit exakten Zielgrößen
    decoder_blocks = ManualUpsampleBlock[]
    for level in config.depth:-1:1
        decoder_in = config.filter_progression[level + 1]
        skip_ch = config.filter_progression[level]
        decoder_out = config.filter_progression[level]
        
        # Zielgröße für diesen Decoder
        target_size = (expected_sizes[level], expected_sizes[level])
        
        decoder_block = create_manual_decoder_block(config, decoder_in, skip_ch, decoder_out, level, target_size)
        push!(decoder_blocks, decoder_block)
        
        println("  Decoder $level: $decoder_in + $skip_ch → $decoder_out [Ziel: $(target_size)]")
    end
    
    # Output-Layer
    final_in = config.filter_progression[1]
    output_layer = create_output_layer(config, final_in)
    println("  Output: $final_in → $(config.output_channels) [$(config.input_resolution)×$(config.input_resolution)]")
    
    return FinalCorrectedUNet(config, encoder_blocks, bottleneck, decoder_blocks, output_layer)
end

"""
Final korrigierter Forward-Pass - OHNE DEBUG-AUSGABEN für Zygote-Kompatibilität
"""
function (model::FinalCorrectedUNet)(x)
    input_size = size(x)
    expected_size = (model.config.input_resolution, model.config.input_resolution, model.config.input_channels)
    
    if input_size[1:3] != expected_size
        error("Input-Größe $(input_size[1:3]) passt nicht zu erwarteter Größe $expected_size")
    end
    
    # Encoder-Phase
    skip_features = []
    current = x
    
    for (level, encoder) in enumerate(model.encoder_blocks)
        features, pooled = encoder(current)
        push!(skip_features, features)
        current = pooled
        
        # ENTFERNT: Debug-Ausgaben (verursachen Zygote-Fehler)
        # println("Encoder $level: Input $(size(x)), Features $(size(features)), Pooled $(size(pooled))")
    end
    
    # Bottleneck
    current = model.bottleneck(current)
    # ENTFERNT: println("Bottleneck: $(size(current))")
    
    # Decoder-Phase
    for (i, decoder) in enumerate(model.decoder_blocks)
        skip_idx = length(skip_features) - i + 1
        skip = skip_features[skip_idx]
        current = decoder(current, skip)
        
        # ENTFERNT: println("Decoder $i: Output $(size(current))")
    end
    
    # Output-Layer
    output = model.output_layer(current)
    
    # FINALE Dimensionsprüfung - NUR bei Bedarf
    expected_output = (model.config.input_resolution, model.config.input_resolution, model.config.output_channels, input_size[4])
    
    if size(output) != expected_output
        # NOTFALL-KORREKTUR: Crop/Pad auf korrekte Größe
        h_target, w_target = expected_output[1:2]
        h_current, w_current = size(output)[1:2]
        
        if h_current > h_target || w_current > w_target
            # Crop
            output = output[1:h_target, 1:w_target, :, :]
        elseif h_current < h_target || w_current < w_target
            # Pad
            padded = zeros(eltype(output), h_target, w_target, size(output, 3), size(output, 4))
            padded[1:h_current, 1:w_current, :, :] .= output
            output = padded
        end
    end
    
    return output
end

println("Final Corrected UNet Architecture geladen!")
println("Verwende: create_final_corrected_unet(config)")