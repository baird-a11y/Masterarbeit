# =============================================================================
# 3D UNET ARCHITEKTUR
# =============================================================================


using Flux

# =============================================================================
# 3D UNET STRUKTUR
# =============================================================================

"""
3D SimplifiedUNet - Direkte Erweiterung deiner 2D Version
"""
struct SimplifiedUNet3D
    # Encoder 1 (64³ → 32³)
    enc1_conv1::Conv
    enc1_conv2::Conv  
    enc1_pool::MaxPool
    
    # Encoder 2 (32³ → 16³)
    enc2_conv1::Conv
    enc2_conv2::Conv
    enc2_pool::MaxPool
    
    # Encoder 3 (16³ → 8³)
    enc3_conv1::Conv
    enc3_conv2::Conv
    enc3_pool::MaxPool
    
    # Bottleneck (8³)
    bottleneck_conv1::Conv
    bottleneck_conv2::Conv
    
    # Decoder 3 (8³ → 16³)
    dec3_up::ConvTranspose
    dec3_conv1::Conv
    dec3_conv2::Conv
    
    # Decoder 2 (16³ → 32³)
    dec2_up::ConvTranspose
    dec2_conv1::Conv
    dec2_conv2::Conv
    
    # Decoder 1 (32³ → 64³)
    dec1_up::ConvTranspose
    dec1_conv1::Conv
    dec1_conv2::Conv
    
    # Output Layer
    output_conv::Conv
end

# Flux integration
Flux.@functor SimplifiedUNet3D

# =============================================================================
# KONSTRUKTOR
# =============================================================================

"""
Erstellt 3D UNet - gleiche Parameter wie deine 2D Version
"""
function create_simplified_unet_3d(input_channels=1, output_channels=3, base_filters=32)
    f = base_filters
    
    SimplifiedUNet3D(
        # Encoder 1: 64³ → 32³
        Conv((3, 3, 3), input_channels => f, relu; pad=1),
        Conv((3, 3, 3), f => f, relu; pad=1),
        MaxPool((2, 2, 2)),
        
        # Encoder 2: 32³ → 16³ 
        Conv((3, 3, 3), f => 2f, relu; pad=1),
        Conv((3, 3, 3), 2f => 2f, relu; pad=1),
        MaxPool((2, 2, 2)),
        
        # Encoder 3: 16³ → 8³
        Conv((3, 3, 3), 2f => 4f, relu; pad=1),
        Conv((3, 3, 3), 4f => 4f, relu; pad=1),
        MaxPool((2, 2, 2)),
        
        # Bottleneck: 8³
        Conv((3, 3, 3), 4f => 8f, relu; pad=1),
        Conv((3, 3, 3), 8f => 8f, relu; pad=1),
        
        # Decoder 3: 8³ → 16³
        ConvTranspose((2, 2, 2), 8f => 4f; stride=2),
        Conv((3, 3, 3), 8f => 4f, relu; pad=1),  # 8f wegen Skip-Connection
        Conv((3, 3, 3), 4f => 4f, relu; pad=1),
        
        # Decoder 2: 16³ → 32³
        ConvTranspose((2, 2, 2), 4f => 2f; stride=2),
        Conv((3, 3, 3), 4f => 2f, relu; pad=1),  # 4f wegen Skip-Connection
        Conv((3, 3, 3), 2f => 2f, relu; pad=1),
        
        # Decoder 1: 32³ → 64³
        ConvTranspose((2, 2, 2), 2f => f; stride=2),
        Conv((3, 3, 3), 2f => f, relu; pad=1),   # 2f wegen Skip-Connection
        Conv((3, 3, 3), f => f, relu; pad=1),
        
        # Output: 64³ → 64³ mit 3 Kanälen (vx, vy, vz)
        Conv((1, 1, 1), f => output_channels)    # Kein ReLU - Geschwindigkeiten können negativ sein
    )
end

# =============================================================================
# FORWARD PASS
# =============================================================================

"""
3D Forward Pass mit Skip-Connections (identische Logik wie deine 2D Version)
"""
function (model::SimplifiedUNet3D)(x)
    # Encoder 1
    enc1 = model.enc1_conv1(x)
    enc1 = model.enc1_conv2(enc1)
    enc1_pooled = model.enc1_pool(enc1)
    
    # Encoder 2
    enc2 = model.enc2_conv1(enc1_pooled)
    enc2 = model.enc2_conv2(enc2)
    enc2_pooled = model.enc2_pool(enc2)
    
    # Encoder 3
    enc3 = model.enc3_conv1(enc2_pooled)
    enc3 = model.enc3_conv2(enc3)
    enc3_pooled = model.enc3_pool(enc3)
    
    # Bottleneck
    bottleneck = model.bottleneck_conv1(enc3_pooled)
    bottleneck = model.bottleneck_conv2(bottleneck)
    
    # Decoder 3 mit Skip-Connection
    dec3_up = model.dec3_up(bottleneck)
    
    # Skip-Connection von Encoder 3
    enc3_adapted = adapt_skip_dimensions_3d_safe(enc3, dec3_up)
    dec3_concat = cat(dec3_up, enc3_adapted, dims=4)  # Channel-Dimension ist 4 in 3D
    
    dec3 = model.dec3_conv1(dec3_concat)
    dec3 = model.dec3_conv2(dec3)
    
    # Decoder 2 mit Skip-Connection
    dec2_up = model.dec2_up(dec3)
    
    # Skip-Connection von Encoder 2
    enc2_adapted = adapt_skip_dimensions_3d_safe(enc2, dec2_up)
    dec2_concat = cat(dec2_up, enc2_adapted, dims=4)
    
    dec2 = model.dec2_conv1(dec2_concat)
    dec2 = model.dec2_conv2(dec2)
    
    # Decoder 1 mit Skip-Connection
    dec1_up = model.dec1_up(dec2)
    
    # Skip-Connection von Encoder 1
    enc1_adapted = adapt_skip_dimensions_3d_safe(enc1, dec1_up)
    dec1_concat = cat(dec1_up, enc1_adapted, dims=4)
    
    dec1 = model.dec1_conv1(dec1_concat)
    dec1 = model.dec1_conv2(dec1)
    
    # Output Layer
    output = model.output_conv(dec1)
    
    return output
end

# =============================================================================
# 3D UTILITY FUNCTIONS
# =============================================================================

"""
3D Version deiner adapt_skip_dimensions_safe Funktion
"""
function adapt_skip_dimensions_3d_safe(skip_feature, target_feature)
    skip_size = size(skip_feature)
    target_size = size(target_feature)
    
    # Falls Größen bereits identisch sind
    if skip_size[1:3] == target_size[1:3]
        return skip_feature
    end
    
    # Minimale Größe bestimmen
    min_w = min(skip_size[1], target_size[1])
    min_h = min(skip_size[2], target_size[2])
    min_d = min(skip_size[3], target_size[3])
    
    # Crop auf minimale Größe
    return skip_feature[1:min_w, 1:min_h, 1:min_d, :, :]
end

# =============================================================================
# TEST FUNCTIONS
# =============================================================================

"""
Test der 3D UNet Architektur
"""
function test_3d_unet_architecture()
    println("=== 3D UNET ARCHITEKTUR TEST ===")
    
    try
        # Modell erstellen
        model = create_simplified_unet_3d(1, 3, 32)  # 1 input (Phase), 3 outputs (vx,vy,vz)
        println("✓ 3D UNet erstellt")
        
        # Test Input: 64×64×64×1×1 (W×H×D×Channels×Batch)
        test_input = randn(Float32, 64, 64, 64, 1, 1)
        println("✓ Test Input erstellt: $(size(test_input))")
        
        # Forward Pass
        output = model(test_input)
        println("✓ Forward Pass erfolgreich")
        println("✓ Output Größe: $(size(output))")
        
        # Erwartete Größe prüfen
        expected_size = (64, 64, 64, 3, 1)
        if size(output) == expected_size
            println("✓ Output Größe korrekt: $(size(output))")
        else
            println("✗ Falsche Output Größe: $(size(output)), erwartet: $expected_size")
            return false
        end
        
        # Parameter zählen
        total_params = sum(length, Flux.params(model))
        println("✓ Gesamte Parameter: $(total_params)")
        
        # Memory Schätzung
        memory_mb = total_params * 4 / (1024^2)  # 4 bytes pro Float32
        println("✓ Geschätzter Speicherbedarf: $(round(memory_mb, digits=1)) MB")
        
        return true
        
    catch e
        println("✗ 3D UNet Test fehlgeschlagen: $e")
        return false
    end
end

# =============================================================================
# MIGRATION HELPERS - FÜR EINFACHEN ÜBERGANG
# =============================================================================

"""
Konvertiert 2D Konfiguration zu 3D Konfiguration
"""
function migrate_config_to_3d(config_2d)
    # Deine bestehende CONFIG erweitern
    config_3d = merge(config_2d, (
        # 3D spezifische Parameter
        image_size = (64, 64, 64),        # Kleinere 3D Auflösung
        target_resolution = (64, 64, 64),
        batch_size = 2,                   # Kleinere Batches wegen Speicher
        
        # LaMEM 3D Parameter
        nel = (63, 63, 63),
        coord_y = [-1.0, 1.0],           # Neue Y-Dimension
        
        # Training Anpassungen
        learning_rate = 0.0005,          # Langsameres Lernen für 3D
        num_epochs = 20,                 # Weniger Epochen für Tests
        
        # Speicher-Management
        use_mixed_precision = true,
        gradient_clip_norm = 1.0
    ))
    
    return config_3d
end

"""
Vergleicht 2D und 3D Modell-Größen
"""
function compare_2d_vs_3d_models()
    println("=== MODELL-VERGLEICH 2D vs 3D ===")
    
    # 2D Modell (deine bestehende Architektur)
    try
        model_2d = create_simplified_unet(1, 2, 32)  
        params_2d = sum(length, Flux.params(model_2d))
        println("2D UNet Parameter: $(params_2d)")
    catch
        println("2D UNet nicht verfügbar")
        params_2d = 0
    end
    
    # 3D Modell
    model_3d = create_simplified_unet_3d(1, 3, 32)
    params_3d = sum(length, Flux.params(model_3d))
    println("3D UNet Parameter: $(params_3d)")
    
    if params_2d > 0
        ratio = params_3d / params_2d
        println("3D/2D Verhältnis: $(round(ratio, digits=1))x")
    end
    
    # Memory Schätzung
    memory_3d_gb = params_3d * 4 / (1024^3)
    println("3D Speicherbedarf: $(round(memory_3d_gb, digits=2)) GB")
end
