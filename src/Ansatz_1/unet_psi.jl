module UNetPsi

using Flux

# Für Klarheit: explizit importieren
using Flux: Conv, ConvTranspose, MaxPool, BatchNorm, relu

"""
    conv_block(cin, cout)

Zwei 3x3-Convs mit BatchNorm + ReLU.
"""
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

"""
    UNet2D – Struct für das U-Net
"""
struct UNet2D
    enc1
    enc2
    enc3
    enc4
    bottom

    pool

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

# Vorwärtsdurchlauf
function (m::UNet2D)(x)
    # Encoder
    x1 = m.enc1(x)
    x2 = m.enc2(m.pool(x1))
    x3 = m.enc3(m.pool(x2))
    x4 = m.enc4(m.pool(x3))
    xb = m.bottom(m.pool(x4))

    # Decoder
    u4 = m.up4(xb)
    d4 = m.dec4(cat(u4, x4; dims=3))

    u3 = m.up3(d4)
    d3 = m.dec3(cat(u3, x3; dims=3))

    u2 = m.up2(d3)
    d2 = m.dec2(cat(u2, x2; dims=3))

    u1 = m.up1(d2)
    d1 = m.dec1(cat(u1, x1; dims=3))

    y = m.final(d1)
    return y
end

"""
    build_unet(in_channels, out_channels; base_channels=32)

Baut ein U-Net für Eingaben (H, W, in_channels, B).
Standard: 256×256×1 → 256×256×1
"""
function build_unet(in_channels::Int, out_channels::Int; base_channels::Int=32)
    pool = MaxPool((2, 2); stride=2)

    enc1 = conv_block(in_channels, base_channels)
    enc2 = conv_block(base_channels, base_channels * 2)
    enc3 = conv_block(base_channels * 2, base_channels * 4)
    enc4 = conv_block(base_channels * 4, base_channels * 8)

    bottom = conv_block(base_channels * 8, base_channels * 16)

    up4  = ConvTranspose((2, 2), base_channels * 16 => base_channels * 8, stride=2)
    dec4 = conv_block(base_channels * 16, base_channels * 8)

    up3  = ConvTranspose((2, 2), base_channels * 8 => base_channels * 4, stride=2)
    dec3 = conv_block(base_channels * 8, base_channels * 4)

    up2  = ConvTranspose((2, 2), base_channels * 4 => base_channels * 2, stride=2)
    dec2 = conv_block(base_channels * 4, base_channels * 2)

    up1  = ConvTranspose((2, 2), base_channels * 2 => base_channels, stride=2)
    dec1 = conv_block(base_channels * 2, base_channels)

    final = Conv((1, 1), base_channels => out_channels, pad=0)

    return UNet2D(
        enc1, enc2, enc3, enc4, bottom,
        pool,
        up4, dec4,
        up3, dec3,
        up2, dec2,
        up1, dec1,
        final
    )
end

end # module
