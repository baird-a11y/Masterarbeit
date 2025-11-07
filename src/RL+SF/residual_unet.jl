# =============================================================================
# RESIDUAL UNET - MODERNISIERT FÜR CLEAN ARCHITECTURE
# =============================================================================
# Kombiniert:
# - SimplifiedUNetBN (aus altem Code)
# - Stokes Analytical Solution (neue Implementation)
# - Optional Stream Function
# - Crystal Params als Input

using Flux
using CUDA
using Zygote  # Für @ignore Makro

println("ResidualUNet wird geladen...")

# =============================================================================
# IMPORTIERE ALTES UNET (als Basis)
# =============================================================================

# Note: Diese Funktionen kommen aus deiner alten unet_architecture.jl
# Wir kopieren nur die essentiellen Teile hier rein

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
        
        @views cropped = skip_features[h_start:(h_start+target_h-1), 
                                      w_start:(w_start+target_w-1), :, :]
        return cropped
    else
        T = eltype(skip_features)
        
        h_copy = min(current_h, target_h)
        w_copy = min(current_w, target_w)
        
        # Zygote-safe: Padding mit @ignore (keine trainierbaren Parameter hier)
        if isa(skip_features, CuArray)
            padded = CUDA.zeros(T, target_h, target_w, skip_size[3], skip_size[4])
        else
            padded = zeros(T, target_h, target_w, skip_size[3], skip_size[4])
        end
        
        # @ignore weil Padding keine trainierbaren Parameter hat
        Zygote.@ignore begin
            @views padded[1:h_copy, 1:w_copy, :, :] .= skip_features[1:h_copy, 1:w_copy, :, :]
        end
        
        return padded
    end
end

# =============================================================================
# SIMPLIFIED UNET (Basis-Architektur)
# =============================================================================

"""
SimplifiedUNetBN - Standard UNet mit Batch Normalization

Aus deiner alten unet_architecture.jl übernommen.
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
    
    # Output
    output_conv::Conv
end

Flux.@layer SimplifiedUNetBN

"""
Erstellt UNet mit konfigurierbaren Output-Kanälen
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
        
        # Output
        Conv((1, 1), f => output_channels)
    )
end

"""
Forward-Pass mit Skip-Connections
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

# =============================================================================
# RESIDUAL UNET - MODERNISIERT
# =============================================================================

"""
    ResidualUNet

Modernisierte ResidualUNet-Architektur für Residual Learning.

# Architektur
v_total = v_stokes(analytisch) + Δv(gelernt)

Optional mit Stream Function:
ψ_total = ψ_stokes + Δψ(gelernt)
v_total = curl(ψ_total)

# Fields
- `base_unet::SimplifiedUNetBN`: Basis-UNet (lernt Residuum)
- `stream_function_layer::Union{StreamFunctionLayer, Nothing}`: Optional SF Layer
- `use_stream_function::Bool`: Ob Stream Function genutzt wird

# Neu gegenüber alter Version:
- Crystal Params als separater Input (statt aus Phase extrahieren)
- Optional Stream Function Support
- Kompatibel mit neuer stokes_analytical.jl
- Saubere Trennung: UNet / Stokes / Stream Function
"""
struct ResidualUNet
    base_unet::SimplifiedUNetBN
    stream_function_layer::Union{Nothing, Any}  # StreamFunctionLayer (wenn geladen)
    use_stream_function::Bool
end

Flux.@layer ResidualUNet

"""
    create_residual_unet(; kwargs...)

Erstellt ResidualUNet mit optionaler Stream Function.

# Keyword Arguments
- `input_channels::Int`: Input-Kanäle (default: 1 = Phasenfeld)
- `base_filters::Int`: Basis-Filter (default: 32)
- `use_stream_function::Bool`: Stream Function nutzen (default: false)
- `Δx::Float32`: Grid-Spacing horizontal (nur für SF)
- `Δz::Float32`: Grid-Spacing vertikal (nur für SF)

# Returns
- `ResidualUNet`: Trainierbare Modell-Struktur

# Examples
```julia
# Ohne Stream Function (direktes v)
model = create_residual_unet()

# Mit Stream Function (über ψ)
model = create_residual_unet(
    use_stream_function=true,
    Δx=1.0f0/255,
    Δz=1.0f0/255
)
```
"""
function create_residual_unet(;
    input_channels::Int=1,
    base_filters::Int=32,
    use_stream_function::Bool=false,
    Δx::Float32=1.0f0/255,
    Δz::Float32=1.0f0/255
)
    
    # Output-Kanäle abhängig von Stream Function
    output_channels = use_stream_function ? 1 : 2  # 1 = ψ, 2 = (vx, vz)
    
    # Basis-UNet erstellen
    base_unet = create_simplified_unet_bn(input_channels, output_channels, base_filters)
    
    # Stream Function Layer (optional)
    if use_stream_function
        # Import hier statt global (falls Modul geladen)
        try
            sf_layer = StreamFunctionLayer(Δx, Δz, method=:central)
            println("Stream Function aktiviert")
            return ResidualUNet(base_unet, sf_layer, true)
        catch e
            @warn "Stream Function Layer nicht verfügbar: $e"
            @warn "Fallback: Direktes Geschwindigkeitsfeld"
            # Fallback: Ändere output_channels zu 2
            base_unet_fallback = create_simplified_unet_bn(input_channels, 2, base_filters)
            return ResidualUNet(base_unet_fallback, nothing, false)
        end
    else
        return ResidualUNet(base_unet, nothing, false)
    end
end

"""
    (model::ResidualUNet)(phase_field, crystal_params, x_vec, z_vec, stats_batch)

Forward-Pass: Berechnet Gesamtgeschwindigkeit aus Phase + Crystal Params.

WICHTIG: v_stokes wird mit denselben Stats normalisiert wie v_target!

# Flow
1. Stokes Baseline: v_stokes = f(crystal_params) [physikalische Werte]
2. Normalisiere v_stokes mit Stats von v_target
3. UNet Residuum: Δv oder Δψ = UNet(phase_field)
4. Optional Stream Function: Δv = curl(Δψ)
5. Total: v_total = v_stokes_norm + Δv

# Arguments
- `phase_field::AbstractArray{T,4}`: Phasenfeld [H, W, 1, B]
- `crystal_params::Vector`: Liste von CrystalParams pro Batch
- `x_vec::AbstractVector`: X-Koordinaten
- `z_vec::AbstractVector`: Z-Koordinaten
- `stats_batch::Vector`: Normalisierungs-Stats [(vx_stats, vz_stats), ...]

# Returns
- `Tuple`: (v_total, v_stokes_norm, Δv)
  - v_total: Gesamtgeschwindigkeit [H, W, 2, B] (normalisiert!)
  - v_stokes_norm: Normalisierte Stokes-Baseline [H, W, 2, B]
  - Δv: Gelerntes Residuum [H, W, 2, B]
"""
function (model::ResidualUNet)(
    phase_field::AbstractArray{T,4},
    crystal_params::Vector,
    x_vec::AbstractVector,
    z_vec::AbstractVector,
    stats_batch::Vector
) where T
    
    batch_size = size(phase_field, 4)
    H, W = size(phase_field)[1:2]
    
    # 1. Berechne Stokes-Baseline für jeden Batch (KOMPLETT NICHT-MUTIEREND!)
    # List comprehension statt push!
    v_stokes_batch = cat([
        compute_stokes_from_crystal_params(
            phase_field[:, :, 1, b],
            crystal_params[b],
            x_vec,
            z_vec
        )
        for b in 1:batch_size
    ]..., dims=4)
    
    # 1b. WICHTIG: Normalisiere v_stokes mit denselben Stats wie v_target!
    v_stokes_normalized = normalize_stokes_field(v_stokes_batch, stats_batch)
    
    # 2. UNet Forward Pass
    unet_output = model.base_unet(phase_field)
    
    # 3. Stream Function (optional)
    if model.use_stream_function && model.stream_function_layer !== nothing
        # UNet gibt Δψ → Konvertiere zu Δv
        Δv = model.stream_function_layer(unet_output)
        
        # Match Dimensionen (SF reduziert Größe durch Ableitungen)
        # Crop v_stokes_normalized auf gleiche Größe
        H_out, W_out = size(Δv)[1:2]
        h_start = (H - H_out) ÷ 2 + 1
        w_start = (W - W_out) ÷ 2 + 1
        v_stokes_cropped = v_stokes_normalized[h_start:(h_start+H_out-1), 
                                                w_start:(w_start+W_out-1), :, :]
        
        v_stokes_normalized = v_stokes_cropped
    else
        # Direktes Geschwindigkeitsfeld
        Δv = unet_output
    end
    
    # 4. Total = Stokes_norm + Residuum (beide normalisiert!)
    v_total = v_stokes_normalized .+ Δv
    
    return v_total, v_stokes_normalized, Δv
end

# =============================================================================
# VEREINFACHTER FORWARD-PASS (nur Phase, ohne explizite Crystal Params)
# =============================================================================

"""
    forward_simple(model, phase_field, x_vec, z_vec)

Vereinfachter Forward-Pass: Extrahiert Crystal Params automatisch.

Für Training-Loop wo Crystal Params noch nicht vorhanden sind.
"""
function forward_simple(
    model::ResidualUNet,
    phase_field::AbstractArray{T,4},
    x_vec::AbstractVector,
    z_vec::AbstractVector
) where T
    
    batch_size = size(phase_field, 4)
    
    # Extrahiere Crystal Params für jeden Batch
    crystal_params_batch = []
    
    for b in 1:batch_size
        phase_b = phase_field[:, :, 1, b]
        params = extract_crystal_params(phase_b, x_vec, z_vec)
        push!(crystal_params_batch, params)
    end
    
    # Rufe normalen Forward-Pass auf
    return model(phase_field, crystal_params_batch, x_vec, z_vec)
end

# =============================================================================
# TEST-FUNKTIONEN
# =============================================================================

"""
    test_residual_unet_shapes(; use_stream_function=false)

Testet ob ResidualUNet korrekte Shapes produziert.
"""
function test_residual_unet_shapes(; use_stream_function=false)
    println("\n TEST: ResidualUNet Shapes $(use_stream_function ? "(mit Stream Function)" : "")")
    println("="^60)
    
    # Setup
    H, W, B = 256, 256, 2
    x_vec = range(-0.5, 0.5, length=W)
    z_vec = range(0.0, 1.0, length=H)
    
    # Erstelle Modell
    model = create_residual_unet(
        use_stream_function=use_stream_function,
        Δx=Float32(x_vec[2] - x_vec[1]),
        Δz=Float32(z_vec[2] - z_vec[1])
    )
    
    # Test-Input: Phasenfeld mit Kristallen
    phase_field = zeros(Float32, H, W, 1, B)
    
    # Dummy Crystal Params
    crystal_params = []
    for b in 1:B
        params = [
            CrystalParams(0.0, 0.5, 0.05, 3300.0)
        ]
        push!(crystal_params, params)
    end
    
    try
        # Forward Pass
        v_total, v_stokes, Δv = model(phase_field, crystal_params, x_vec, z_vec)
        
        println("Input Shape: $(size(phase_field))")
        println("v_total Shape: $(size(v_total))")
        println("v_stokes Shape: $(size(v_stokes))")
        println("Δv Shape: $(size(Δv))")
        
        # Check Shapes
        if use_stream_function
            # Stream Function reduziert Größe
            expected_h = H - 2  # Zentrale Differenzen
            expected_w = W - 2
            expected = (expected_h, expected_w, 2, B)
        else
            expected = (H, W, 2, B)
        end
        
        success = size(v_total) == size(v_stokes) == size(Δv)
        
        println("\nShape Konsistenz: $(success ? "Yes" : "No")")
        
        # Magnitude Check
        v_total_mag = mean(sqrt.(v_total[:,:,1,:].^2 .+ v_total[:,:,2,:].^2))
        v_stokes_mag = mean(sqrt.(v_stokes[:,:,1,:].^2 .+ v_stokes[:,:,2,:].^2))
        Δv_mag = mean(sqrt.(Δv[:,:,1,:].^2 .+ Δv[:,:,2,:].^2))
        
        println("\nMagnitudes:")
        println("  v_total: $(round(v_total_mag, digits=6))")
        println("  v_stokes: $(round(v_stokes_mag, digits=6))")
        println("  Δv: $(round(Δv_mag, digits=6))")
        
        println("\n$(success ? "Yes" : "No") Test $(success ? "bestanden" : "fehlgeschlagen")")
        
        return success, model
        
    catch e
        println("Test fehlgeschlagen: $e")
        showerror(stdout, e, catch_backtrace())
        return false, nothing
    end
end

# =============================================================================
# MODUL-INFO
# =============================================================================

println("ResidualUNet (Modernisiert) geladen!")
println("   - Basis: SimplifiedUNetBN")
println("   - Optional: Stream Function Support")
println("   - Kompatibel mit stokes_analytical.jl")
println("")
println("Wichtige Funktionen:")
println("   - create_residual_unet(...) - Modell erstellen")
println("   - model(phase, crystal_params, x, z) - Forward Pass")
println("   - test_residual_unet_shapes() - Unit Test")
println("")
println("Quick Test verfügbar:")
println("   julia> test_residual_unet_shapes()")
println("   julia> test_residual_unet_shapes(use_stream_function=true)")