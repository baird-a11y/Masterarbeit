module UNetPsi

using Flux
using Flux: Conv, ConvTranspose, BatchNorm, relu

# ============================================================
#  Conv-Block: zwei 3×3-Convs mit BatchNorm + ReLU
# ============================================================
function conv_block(cin::Int, cout::Int)
    return Chain(
        Conv((3, 3), cin => cout, pad=1),
        BatchNorm(cout),
        relu,
        Conv((3, 3), cout => cout, pad=1),
        BatchNorm(cout),
        relu,
    )
end

# ============================================================
# Downsampling-Block: STRIDED CONV (statt MaxPool)
# → verhindert Grid-Artefakte
# ============================================================
function down_block(cin::Int, cout::Int)
    return Chain(
        Conv((3,3), cin => cout, stride=2, pad=1),   # Downsampling
        BatchNorm(cout),
        relu,
        Conv((3,3), cout => cout, pad=1),
        BatchNorm(cout),
        relu
    )
end

# ============================================================
# Upsampling-Block: ConvTranspose + zwei Convs
# ============================================================
function up_block(cin::Int, cout::Int)
    return Chain(
        ConvTranspose((2,2), cin => cout, stride=2),
        Conv((3,3), cout => cout, pad=1),
        BatchNorm(cout),
        relu,
        Conv((3,3), cout => cout, pad=1),
        BatchNorm(cout),
        relu
    )
end

# ============================================================
# UNet Struct
# ============================================================
struct UNet2D
    enc1
    enc2
    enc3
    enc4
    bottom

    up4
    dec4
    up3
    dec3
    up2
    dec2
    up1
    dec1

    final
end

Flux.@functor UNet2D

# ============================================================
# Forward Pass
# ============================================================
function (m::UNet2D)(x)
    # Encoder
    x1 = m.enc1(x)
    x2 = m.enc2(x1)
    x3 = m.enc3(x2)
    x4 = m.enc4(x3)
    xb = m.bottom(x4)

    # Decoder
    u4 = m.up4(xb)
    d4 = m.dec4(cat(u4, x4; dims=3))

    u3 = m.up3(d4)
    d3 = m.dec3(cat(u3, x3; dims=3))

    u2 = m.up2(d3)
    d2 = m.dec2(cat(u2, x2; dims=3))

    u1 = m.up1(d2)
    d1 = m.dec1(cat(u1, x1; dims=3))

    return m.final(d1)
end

# ============================================================
# Build Function
# ============================================================
function build_unet(in_channels::Int, out_channels::Int; base_channels::Int=32)

    # Encoder (mit strided Convs)
    enc1 = conv_block(in_channels, base_channels)
    enc2 = down_block(base_channels, base_channels*2)
    enc3 = down_block(base_channels*2, base_channels*4)
    enc4 = down_block(base_channels*4, base_channels*8)

    # Bottom
    bottom = down_block(base_channels*8, base_channels*16)

    # Decoder
    up4  = up_block(base_channels*16, base_channels*8)
    dec4 = conv_block(base_channels*16, base_channels*8)

    up3  = up_block(base_channels*8, base_channels*4)
    dec3 = conv_block(base_channels*8, base_channels*4)

    up2  = up_block(base_channels*4, base_channels*2)
    dec2 = conv_block(base_channels*4, base_channels*2)

    up1  = up_block(base_channels*2, base_channels)
    dec1 = conv_block(base_channels*2, base_channels)

    final = Conv((1, 1), base_channels => out_channels, pad=0)

    return UNet2D(
        enc1, enc2, enc3, enc4, bottom,
        up4, dec4,
        up3, dec3,
        up2, dec2,
        up1, dec1,
        final
    )
end

end # module
