##################################
# Model.jl - Enhanced Architecture
##################################
module Model

using Flux
using Functors
using CUDA
using Statistics

# Residual block for improved feature extraction
struct ResidualBlock
    path
    shortcut
end
Functors.@functor ResidualBlock

function (rb::ResidualBlock)(x)
    return rb.path(x) + rb.shortcut(x)
end

# Attention block for focusing on relevant features
struct AttentionBlock
    query_conv
    key_conv
    value_conv
    gamma
end
Functors.@functor AttentionBlock

function (attn::AttentionBlock)(x)
    batch_size, h, w, C, N = size(x)

    # Reshape for attention operations
    proj_query = reshape(attn.query_conv(x), :, C, N)
    proj_key = permutedims(reshape(attn.key_conv(x), :, C, N), (2, 1, 3))
    energy = batched_mul(proj_query, proj_key)

    # Softmax along spatial dimension
    attention = softmax(energy, dims=2)
    
    # Apply attention weights
    proj_value = reshape(attn.value_conv(x), :, C, N)
    out = batched_mul(attention, proj_value)
    out = reshape(out, h, w, C, N)
    
    # Apply learnable scaling factor and residual connection
    return attn.gamma * out + x
end

# Enhanced UNet with residual connections and attention mechanism
struct UNet
    encoder1
    encoder2
    encoder3
    encoder4
    bottleneck
    attention
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

# Helper function for creating a residual block
function create_residual_block(channels::Int, kernel_size=(3,3))
    return ResidualBlock(
        Chain(
            Conv(kernel_size, channels => channels, relu, pad=SamePad()),
            BatchNorm(channels),
            Conv(kernel_size, channels => channels, pad=SamePad()),
            BatchNorm(channels)
        ),
        identity
    )
end

# Helper function for creating an attention block
function create_attention_block(channels::Int)
    return AttentionBlock(
        Conv((1,1), channels => channels ÷ 8),
        Conv((1,1), channels => channels ÷ 8),
        Conv((1,1), channels => channels),
        Float32(0.1)  # Initial scaling parameter
    )
end

# Full UNet with enhancements
function UNet_full(input_channels::Int, output_channels::Int; 
                  dropout_rate=0.3, use_attention=true)
    # Encoder with residual connections
    encoder1 = Chain(
        Conv((3, 3), input_channels => 64, relu, pad=SamePad()),
        BatchNorm(64),
        Conv((3, 3), 64 => 64, relu, pad=SamePad()),
        BatchNorm(64),
        create_residual_block(64)  # Add residual connection
    )
    
    encoder2 = Chain(
        MaxPool((2,2)),
        Conv((3, 3), 64 => 128, relu, pad=SamePad()),
        BatchNorm(128),
        Conv((3, 3), 128 => 128, relu, pad=SamePad()),
        BatchNorm(128),
        create_residual_block(128)  # Add residual connection
    )
    
    encoder3 = Chain(
        MaxPool((2,2)),
        Conv((3, 3), 128 => 256, relu, pad=SamePad()),
        BatchNorm(256),
        Conv((3, 3), 256 => 256, relu, pad=SamePad()),
        BatchNorm(256),
        create_residual_block(256)  # Add residual connection
    )
    
    encoder4 = Chain(
        MaxPool((2,2)),
        Conv((3, 3), 256 => 512, relu, pad=SamePad()),
        BatchNorm(512),
        Conv((3, 3), 512 => 512, relu, pad=SamePad()),
        BatchNorm(512),
        create_residual_block(512)  # Add residual connection
    )
    
    # Bottleneck with attention
    bottleneck = Chain(
        MaxPool((2,2)),
        Conv((3,3), 512 => 1024, relu, pad=SamePad()),
        BatchNorm(1024),
        Dropout(dropout_rate),
        Conv((3,3), 1024 => 1024, relu, pad=SamePad()),
        BatchNorm(1024),
        create_residual_block(1024)  # Add residual connection
    )
    
    # Attention module at bottleneck
    attention = use_attention ? create_attention_block(1024) : identity
    
    # Decoders with better feature fusion
    decoder4 = ConvTranspose((2, 2), 1024 => 512, stride=2)
    decoder4_1 = Chain(
        Conv((3,3), 1024 => 512, relu, pad=SamePad()),
        BatchNorm(512),
        Conv((3,3), 512 => 512, relu, pad=SamePad()),
        BatchNorm(512),
        create_residual_block(512)  # Add residual connection
    )
    
    decoder3 = ConvTranspose((2,2), 512 => 256, stride=2)
    decoder3_1 = Chain(
        Conv((3,3), 512 => 256, relu, pad=SamePad()),
        BatchNorm(256),
        Conv((3,3), 256 => 256, relu, pad=SamePad()),
        BatchNorm(256),
        create_residual_block(256)  # Add residual connection
    )
    
    decoder2 = ConvTranspose((2,2), 256 => 128, stride=2)
    decoder2_1 = Chain(
        Conv((3,3), 256 => 128, relu, pad=SamePad()),
        BatchNorm(128),
        Conv((3,3), 128 => 128, relu, pad=SamePad()),
        BatchNorm(128),
        create_residual_block(128)  # Add residual connection
    )
    
    decoder1 = ConvTranspose((2,2), 128 => 64, stride=2)
    decoder1_1 = Chain(
        Conv((3,3), 128 => 64, relu, pad=SamePad()),
        BatchNorm(64),
        Conv((3,3), 64 => 64, relu, pad=SamePad()),
        BatchNorm(64),
        create_residual_block(64),  # Add residual connection
        Dropout(dropout_rate/2),
        Conv((1,1), 64 => output_channels)
    )
    
    dropout = Dropout(dropout_rate)
    
    return UNet(encoder1, encoder2, encoder3, encoder4, bottleneck, attention,
                decoder4, decoder4_1, decoder3, decoder3_1,
                decoder2, decoder2_1, decoder1, decoder1_1, dropout)
end

# Memory-efficient UNet with enhancements
function UNet(input_channels::Int, output_channels::Int; 
             dropout_rate=0.3, memory_efficient=false, use_attention=true)
    if !memory_efficient
        return UNet_full(input_channels, output_channels, 
                         dropout_rate=dropout_rate, 
                         use_attention=use_attention)
    end
    
    # Reduced size for feature maps but with architectural improvements
    encoder1 = Chain(
        Conv((3, 3), input_channels => 48, relu, pad=SamePad()),
        BatchNorm(48),
        Conv((3, 3), 48 => 48, relu, pad=SamePad()),
        BatchNorm(48),
        create_residual_block(48)  # Add residual connection
    )
    
    encoder2 = Chain(
        MaxPool((2,2)),
        Conv((3, 3), 48 => 96, relu, pad=SamePad()),
        BatchNorm(96),
        Conv((3, 3), 96 => 96, relu, pad=SamePad()),
        BatchNorm(96),
        create_residual_block(96)  # Add residual connection
    )
    
    encoder3 = Chain(
        MaxPool((2,2)),
        Conv((3, 3), 96 => 192, relu, pad=SamePad()),
        BatchNorm(192),
        Conv((3, 3), 192 => 192, relu, pad=SamePad()),
        BatchNorm(192),
        create_residual_block(192)  # Add residual connection
    )
    
    encoder4 = Chain(
        MaxPool((2,2)),
        Conv((3, 3), 192 => 384, relu, pad=SamePad()),
        BatchNorm(384),
        Conv((3, 3), 384 => 384, relu, pad=SamePad()),
        BatchNorm(384),
        create_residual_block(384)  # Add residual connection
    )
    
    bottleneck = Chain(
        MaxPool((2,2)),
        Conv((3,3), 384 => 768, relu, pad=SamePad()),
        BatchNorm(768),
        Dropout(dropout_rate),
        Conv((3,3), 768 => 768, relu, pad=SamePad()),
        BatchNorm(768),
        create_residual_block(768)  # Add residual connection
    )
    
    # Attention module at bottleneck (simplified for memory efficiency)
    attention = use_attention ? create_attention_block(768) : identity
    
    decoder4 = ConvTranspose((2, 2), 768 => 384, stride=2)
    decoder4_1 = Chain(
        Conv((3,3), 768 => 384, relu, pad=SamePad()),
        BatchNorm(384),
        Conv((3,3), 384 => 384, relu, pad=SamePad()),
        BatchNorm(384),
        create_residual_block(384)  # Add residual connection
    )
    
    decoder3 = ConvTranspose((2,2), 384 => 192, stride=2)
    decoder3_1 = Chain(
        Conv((3,3), 384 => 192, relu, pad=SamePad()),
        BatchNorm(192),
        Conv((3,3), 192 => 192, relu, pad=SamePad()),
        BatchNorm(192),
        create_residual_block(192)  # Add residual connection
    )
    
    decoder2 = ConvTranspose((2,2), 192 => 96, stride=2)
    decoder2_1 = Chain(
        Conv((3,3), 192 => 96, relu, pad=SamePad()),
        BatchNorm(96),
        Conv((3,3), 96 => 96, relu, pad=SamePad()),
        BatchNorm(96),
        create_residual_block(96)  # Add residual connection
    )
    
    decoder1 = ConvTranspose((2,2), 96 => 48, stride=2)
    decoder1_1 = Chain(
        Conv((3,3), 96 => 48, relu, pad=SamePad()),
        BatchNorm(48),
        Conv((3,3), 48 => 48, relu, pad=SamePad()),
        BatchNorm(48),
        create_residual_block(48),  # Add residual connection
        Dropout(dropout_rate/2),
        Conv((1,1), 48 => output_channels)
    )
    
    dropout = Dropout(dropout_rate)
    
    return UNet(encoder1, encoder2, encoder3, encoder4, bottleneck, attention,
                decoder4, decoder4_1, decoder3, decoder3_1,
                decoder2, decoder2_1, decoder1, decoder1_1, dropout)
end

# Debug function for forward pass
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
    b = model.bottleneck(e4)
    show_stats("bottleneck", b)
    if any(isinf, b) || any(isnan, b)
        return b
    end
    
    # Apply attention if not identity
    if model.attention !== identity
        b = model.attention(b)
        show_stats("attention", b)
        if any(isinf, b) || any(isnan, b)
            return b
        end
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

# Helper function to handle center cropping for skip connections
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

# Enhanced Forward Pass with attention mechanism
function (model::UNet)(x)
    e1 = model.encoder1(x)
    e2 = model.encoder2(e1)
    e3 = model.encoder3(e2)
    e4 = model.encoder4(e3)
    b  = model.bottleneck(e4)
    
    # Apply attention if not identity
    if model.attention !== identity
        b = model.attention(b)
    end

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