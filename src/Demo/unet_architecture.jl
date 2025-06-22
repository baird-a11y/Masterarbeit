# =============================================================================
# UNET ARCHITECTURE MODULE
# =============================================================================
# Speichern als: unet_architecture.jl

using Flux

"""
Korrigierte Skip-Dimensionen-Anpassung mit exakter Kontrolle
"""
function adapt_skip_dimensions_corrected(skip_features, decoder_features)
    skip_size = size(skip_features)
    decoder_size = size(decoder_features)
    
    if skip_size[1:2] == decoder_size[1:2]
        return skip_features
    end
    
    target_h, target_w = decoder_size[1:2]
    current_h, current_w = skip_size[1:2]
    
    if current_h >= target_h && current_w >= target_w
        # Center-Crop
        h_start = (current_h - target_h) ÷ 2 + 1
        w_start = (current_w - target_w) ÷ 2 + 1
        h_end = h_start + target_h - 1
        w_end = w_start + target_w - 1
        
        return skip_features[h_start:h_end, w_start:w_end, :, :]
    else
        # Zero-Padding
        padded = zeros(eltype(skip_features), target_h, target_w, skip_size[3], skip_size[4])
        
        h_copy = min(current_h, target_h)
        w_copy = min(current_w, target_w)
        
        padded[1:h_copy, 1:w_copy, :, :] .= skip_features[1:h_copy, 1:w_copy, :, :]
        
        return padded
    end
end

"""
Korrigierte Encoder-Block-Struktur mit separater Pooling-Kontrolle
"""
struct CorrectedEncoderBlock
    conv_block::Chain
    pool_layer::Union{MaxPool, Nothing}
    level::Int
end

Flux.@functor CorrectedEncoderBlock

function (block::CorrectedEncoderBlock)(x)
    features = block.conv_block(x)
    
    if block.pool_layer !== nothing
        pooled = block.pool_layer(features)
        return features, pooled  # Beide für Skip-Connections
    else
        return features, features  # Letzter Encoder ohne Pooling
    end
end

"""
Korrigierte Decoder-Block-Struktur mit exakter Dimensionskontrolle
"""
struct CorrectedDecoderBlock
    upsample::ConvTranspose
    conv_block::Chain
    level::Int
end

Flux.@functor CorrectedDecoderBlock

function (block::CorrectedDecoderBlock)(x, skip_features)
    upsampled = block.upsample(x)
    adapted_skip = adapt_skip_dimensions_corrected(skip_features, upsampled)
    concatenated = cat(upsampled, adapted_skip, dims=3)
    
    return block.conv_block(concatenated)
end

"""
Erstellt korrigierten Encoder-Block
"""
function create_corrected_encoder_block(config::UNetConfig, in_channels::Int, out_channels::Int, level::Int)
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
    pool_layer = level < config.depth ? MaxPool((2, 2)) : nothing
    
    return CorrectedEncoderBlock(conv_block, pool_layer, level)
end

"""
Erstellt korrigierten Decoder-Block
"""
function create_corrected_decoder_block(config::UNetConfig, in_channels::Int, skip_channels::Int, out_channels::Int, level::Int)
    # ConvTranspose mit korrektem Padding für exakt 2x Vergrößerung
    upsample = ConvTranspose((2, 2), in_channels => out_channels, stride=2, pad=0)
    
    # Convolution-Block nach Concatenation
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
    
    if config.dropout_rate > 0 && level <= 2
        push!(conv_layers, Dropout(config.dropout_rate))
    end
    
    conv_block = Chain(conv_layers...)
    
    return CorrectedDecoderBlock(upsample, conv_block, level)
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
Korrigierte Hauptstruktur des adaptiven UNet
"""
struct CorrectedAdaptiveUNet
    config::UNetConfig
    encoder_blocks::Vector{CorrectedEncoderBlock}
    bottleneck::Chain
    decoder_blocks::Vector{CorrectedDecoderBlock}
    output_layer::Chain
end

Flux.@functor CorrectedAdaptiveUNet

"""
Erstellt korrigiertes adaptives UNet mit exakter Dimensionskontrolle
"""
function create_corrected_adaptive_unet(config::UNetConfig)
    println("=== ERSTELLE KORRIGIERTES ADAPTIVE UNET ===")
    println("Konfiguration: $(config.input_resolution)×$(config.input_resolution)")
    println("Tiefe: $(config.depth), Filter: $(config.filter_progression)")
    
    # Encoder-Blöcke
    encoder_blocks = CorrectedEncoderBlock[]
    for level in 1:config.depth
        in_ch = level == 1 ? config.input_channels : config.filter_progression[level-1]
        out_ch = config.filter_progression[level]
        
        encoder_block = create_corrected_encoder_block(config, in_ch, out_ch, level)
        push!(encoder_blocks, encoder_block)
        
        pool_info = encoder_block.pool_layer !== nothing ? "mit Pooling" : "ohne Pooling"
        println("  Encoder $level: $in_ch → $out_ch ($pool_info)")
    end
    
    # Bottleneck
    bottleneck_in = config.filter_progression[config.depth]
    bottleneck_out = config.filter_progression[config.depth + 1]
    bottleneck = create_bottleneck_block(config, bottleneck_in, bottleneck_out)
    println("  Bottleneck: $bottleneck_in → $bottleneck_out")
    
    # Decoder-Blöcke
    decoder_blocks = CorrectedDecoderBlock[]
    for level in config.depth:-1:1
        decoder_in = config.filter_progression[level + 1]
        skip_ch = config.filter_progression[level]
        decoder_out = config.filter_progression[level]
        
        decoder_block = create_corrected_decoder_block(config, decoder_in, skip_ch, decoder_out, level)
        push!(decoder_blocks, decoder_block)
        
        println("  Decoder $level: $decoder_in + $skip_ch → $decoder_out")
    end
    
    # Output-Layer
    final_in = config.filter_progression[1]
    output_layer = create_output_layer(config, final_in)
    println("  Output: $final_in → $(config.output_channels)")
    
    return CorrectedAdaptiveUNet(config, encoder_blocks, bottleneck, decoder_blocks, output_layer)
end

"""
Korrigierter Forward-Pass mit Dimensionskontrolle
"""
function (model::CorrectedAdaptiveUNet)(x)
    # Input-Validierung
    input_size = size(x)
    expected_size = (model.config.input_resolution, model.config.input_resolution, model.config.input_channels)
    
    if input_size[1:3] != expected_size
        error("Input-Größe $(input_size[1:3]) passt nicht zu erwarteter Größe $expected_size")
    end
    
    # Encoder-Phase: Sammle Skip-Features
    skip_features = []
    current = x
    
    for (level, encoder) in enumerate(model.encoder_blocks)
        features, pooled = encoder(current)
        push!(skip_features, features)  # Für Skip-Connections
        current = pooled  # Für nächsten Level
    end
    
    # Bottleneck
    current = model.bottleneck(current)
    
    # Decoder-Phase: Verwende Skip-Features in umgekehrter Reihenfolge
    for (i, decoder) in enumerate(model.decoder_blocks)
        skip_idx = length(skip_features) - i + 1
        skip = skip_features[skip_idx]
        current = decoder(current, skip)
    end
    
    # Output-Layer
    output = model.output_layer(current)
    
    # Finale Dimensionsprüfung
    expected_output = (model.config.input_resolution, model.config.input_resolution, model.config.output_channels, input_size[4])
    if size(output) != expected_output
        println("WARNUNG: Output-Größe $(size(output)) ≠ erwartet $expected_output")
    end
    
    return output
end

"""
Testet das korrigierte UNet
"""
function test_corrected_unet(resolution::Int=256; batch_size::Int=2, verbose::Bool=true)
    if verbose
        println("=== TESTE KORRIGIERTES UNET ===")
        println("Auflösung: $(resolution)×$(resolution)")
        println("Batch-Größe: $batch_size")
    end
    
    # Erstelle Konfiguration
    config = design_adaptive_unet(resolution)
    
    # Erstelle korrigiertes UNet
    model = create_corrected_adaptive_unet(config)
    
    # Test-Input
    test_input = randn(Float32, resolution, resolution, 1, batch_size)
    
    if verbose
        println("Input-Shape: $(size(test_input))")
    end
    
    # Forward-Pass
    try
        output = model(test_input)
        expected_shape = (resolution, resolution, 2, batch_size)
        
        if verbose
            println("Output-Shape: $(size(output))")
            println("Erwartet: $expected_shape")
        end
        
        success = size(output) == expected_shape
        
        if verbose
            status = success ? "✓ ERFOLGREICH" : "✗ FEHLGESCHLAGEN"
            println("Ergebnis: $status")
        end
        
        return success, size(output)
        
    catch e
        if verbose
            println("✗ FEHLER: $e")
        end
        return false, nothing
    end
end

"""
Teste mehrere Auflösungen
"""
function test_all_resolutions()
    println("=== TESTE ALLE AUFLÖSUNGEN ===")
    
    resolutions = [64, 128, 256, 512]
    results = Dict{Int, Bool}()
    
    for res in resolutions
        println("\n" * "="^50)
        success, output_shape = test_corrected_unet(res, verbose=true)
        results[res] = success
    end
    
    # Zusammenfassung
    println("\n" * "="^50)
    println("ZUSAMMENFASSUNG:")
    successful = sum(values(results))
    total = length(results)
    println("Erfolgreich: $successful/$total")
    
    for (res, success) in sort(collect(results))
        status = success ? "✓" : "✗"
        println("  $(res)×$(res): $status")
    end
    
    return results
end

println("UNet Architecture Module geladen!")
println("Verfügbare Funktionen:")
println("  - create_corrected_adaptive_unet(config)")
println("  - test_corrected_unet(resolution)")
println("  - test_all_resolutions()")