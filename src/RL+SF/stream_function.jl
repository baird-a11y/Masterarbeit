# =============================================================================
# STREAM FUNCTION LAYER
# =============================================================================
# Konvertiert Stream Function ψ zu Geschwindigkeitsfeld (vx, vz)
# Garantiert Divergenzfreiheit durch Konstruktion: ∇·v = 0

using Flux
using Statistics

println("Stream Function Layer wird geladen...")

# =============================================================================
# STREAM FUNCTION LAYER
# =============================================================================

"""
    StreamFunctionLayer

Flux-kompatibler Layer zur Konvertierung von Stream Function zu Velocity.

# Mathematik
Aus Stream Function ψ:
- vx = ∂ψ/∂z
- vz = -∂ψ/∂x

# Divergenzfreiheit (automatisch garantiert)
∇·v = ∂vx/∂x + ∂vz/∂z 
    = ∂²ψ/(∂x∂z) - ∂²ψ/(∂z∂x) 
    = 0  (Schwarz's Theorem)

# Fields
- `Δx::Float32`: Grid-Spacing in x-Richtung
- `Δz::Float32`: Grid-Spacing in z-Richtung
- `method::Symbol`: Differenzierungs-Methode (:central, :forward, :backward)
"""
struct StreamFunctionLayer{T<:Real}
    Δx::T
    Δz::T
    method::Symbol
    
    # Haupt-Konstruktor (mit Keyword)
    function StreamFunctionLayer(Δx::T, Δz::T; method::Symbol=:central) where T<:Real
        if method ∉ [:central, :forward, :backward]
            error("method muss :central, :forward oder :backward sein")
        end
        new{T}(Δx, Δz, method)
    end
    
    # Zusätzlicher Konstruktor für Functors (mit positionellem Argument)
    function StreamFunctionLayer(Δx::T, Δz::T, method::Symbol) where T<:Real
        if method ∉ [:central, :forward, :backward]
            error("method muss :central, :forward oder :backward sein")
        end
        new{T}(Δx, Δz, method)
    end
end

# Flux Trainable Parameters (keine - Layer ist rein deterministisch)
Flux.@functor StreamFunctionLayer
Flux.trainable(::StreamFunctionLayer) = ()

"""
    (layer::StreamFunctionLayer)(ψ)

Forward Pass: Stream Function → Velocity Field

# Arguments
- `ψ::AbstractArray{T,4}`: Stream Function [H, W, 1, B]

# Returns
- `AbstractArray{T,4}`: Velocity Field [H, W, 2, B] mit (vx, vz)
"""
function (layer::StreamFunctionLayer)(ψ::AbstractArray{T,4}) where T
    H, W, C, B = size(ψ)
    
    if C != 1
        error("Stream Function muss 1 Kanal haben, hat: $C")
    end
    
    # Extrahiere ψ (remove channel dim temporär)
    ψ_2d = ψ[:, :, 1, :]  # [H, W, B]
    
    # Berechne Ableitungen
    vx = ∂z(ψ_2d, layer.Δz, layer.method)   # ∂ψ/∂z
    vz = .-∂x(ψ_2d, layer.Δx, layer.method)  # -∂ψ/∂x
    
    # Dimensionen angleichen
    if layer.method == :central
        vx = vx[:, 2:end-1, :]  # [H-2, W-2, B]
        vz = vz[2:end-1, :, :]  # [H-2, W-2, B]
    elseif layer.method == :forward
        vx = vx[:, 1:end-1, :]  # [H-1, W-1, B]
        vz = vz[1:end-1, :, :]  # [H-1, W-1, B]
    elseif layer.method == :backward
        vx = vx[:, 2:end, :]
        vz = vz[2:end, :, :]
    end
    
    # === FIX: Füge Channel-Dimension hinzu ===
    vx = reshape(vx, size(vx, 1), size(vx, 2), 1, size(vx, 3))  # [H', W', 1, B]
    vz = reshape(vz, size(vz, 1), size(vz, 2), 1, size(vz, 3))  # [H', W', 1, B]
    
    # Jetzt cat entlang Kanal-Dimension
    velocity = cat(vx, vz, dims=3)  # [H', W', 2, B] ✓
    
    return velocity
end

# =============================================================================
# FINITE DIFFERENZEN
# =============================================================================

"""
    ∂x(field, Δx, method)

Ableitung in x-Richtung (horizontal).

# Methods
- `:central`: (f[i+1] - f[i-1]) / (2Δx)  - Höchste Genauigkeit, reduziert Größe um 2
- `:forward`: (f[i+1] - f[i]) / Δx       - Einfach, reduziert Größe um 1
- `:backward`: (f[i] - f[i-1]) / Δx      - Einfach, reduziert Größe um 1
"""
function ∂x(field::AbstractArray{T,3}, Δx::Real, method::Symbol) where T
    H, W, B = size(field)
    
    if method == :central
        # Zentrale Differenzen (höchste Genauigkeit)
        # Nutze i-1, i, i+1 → Output: [H, W-2, B]
        if W < 3
            error("Zentrale Differenzen brauchen mindestens 3 Punkte, W=$W")
        end
        
        df = (field[:, 3:end, :] .- field[:, 1:end-2, :]) ./ (2 * Δx)
        return df
        
    elseif method == :forward
        # Vorwärts-Differenzen
        # Nutze i, i+1 → Output: [H, W-1, B]
        df = (field[:, 2:end, :] .- field[:, 1:end-1, :]) ./ Δx
        return df
        
    elseif method == :backward
        # Rückwärts-Differenzen
        # Nutze i-1, i → Output: [H, W-1, B]
        df = (field[:, 2:end, :] .- field[:, 1:end-1, :]) ./ Δx
        return df
    end
end

"""
    ∂z(field, Δz, method)

Ableitung in z-Richtung (vertikal).

Analog zu ∂x, aber für die erste Dimension.
"""
function ∂z(field::AbstractArray{T,3}, Δz::Real, method::Symbol) where T
    H, W, B = size(field)
    
    if method == :central
        # Zentrale Differenzen
        if H < 3
            error("Zentrale Differenzen brauchen mindestens 3 Punkte, H=$H")
        end
        
        df = (field[3:end, :, :] .- field[1:end-2, :, :]) ./ (2 * Δz)
        return df
        
    elseif method == :forward
        # Vorwärts-Differenzen
        df = (field[2:end, :, :] .- field[1:end-1, :, :]) ./ Δz
        return df
        
    elseif method == :backward
        # Rückwärts-Differenzen
        df = (field[2:end, :, :] .- field[1:end-1, :, :]) ./ Δz
        return df
    end
end

# =============================================================================
# HILFSFUNKTIONEN
# =============================================================================

"""
    compute_stream_function_from_velocity(vx, vz, Δx, Δz)

INVERSE Operation: Berechnet Stream Function aus Geschwindigkeitsfeld.

Nur für Validierung/Testing! Nicht für Training.

# Methode
Integration: ψ = ∫ vx dz (entlang z-Achse)

# Returns
- `AbstractArray`: Stream Function ψ
"""
function compute_stream_function_from_velocity(
    vx::AbstractArray{T,3},
    vz::AbstractArray{T,3},
    Δz::Real
) where T
    H, W, B = size(vx)
    
    # Integriere vx entlang z-Achse
    # ψ[i,j] = ∫ vx[i,j] dz = Σ vx[i,j] * Δz
    ψ = zeros(T, H, W, B)
    
    for i in 2:H
        ψ[i, :, :] = ψ[i-1, :, :] .+ vx[i-1, :, :] .* Δz
    end
    
    return ψ
end

"""
    validate_divergence_free(velocity::AbstractArray{T,4})

Validiert ob Geschwindigkeitsfeld divergenzfrei ist.

# Returns
- `Dict`: Diagnostics mit max/mean Divergenz
"""
function validate_divergence_free(velocity::AbstractArray{T,4}; threshold=1e-5) where T
    H, W, C, B = size(velocity)
    
    if C != 2
        error("Velocity muss 2 Kanäle haben (vx, vz), hat: $C")
    end
    
    vx = velocity[:, :, 1:1, :]
    vz = velocity[:, :, 2:2, :]
    
    # Finite Differenzen
    ∂vx_∂x = vx[:, 2:end, :, :] .- vx[:, 1:end-1, :, :]
    ∂vz_∂z = vz[2:end, :, :, :] .- vz[1:end-1, :, :, :]
    
    # Match Dimensionen
    min_h = min(size(∂vx_∂x, 1), size(∂vz_∂z, 1))
    min_w = min(size(∂vx_∂x, 2), size(∂vz_∂z, 2))
    
    ∂vx_∂x = ∂vx_∂x[1:min_h, 1:min_w, :, :]
    ∂vz_∂z = ∂vz_∂z[1:min_h, 1:min_w, :, :]
    
    # Divergenz
    div = ∂vx_∂x .+ ∂vz_∂z
    
    max_div = maximum(abs, div)
    mean_div = mean(abs, div)
    
    is_divergence_free = max_div < threshold
    
    diagnostics = Dict(
        "max_divergence" => max_div,
        "mean_divergence" => mean_div,
        "divergence_free" => is_divergence_free,
        "threshold" => threshold
    )
    
    return diagnostics
end

# =============================================================================
# TESTING & VALIDATION
# =============================================================================

"""
    test_stream_function_layer()

Unit Test für Stream Function Layer.

Testet:
1. Forward Pass (Dimensionen korrekt)
2. Divergenzfreiheit
3. Gradient Flow (für Backpropagation)
"""
function test_stream_function_layer()
    println("\n TEST: Stream Function Layer")
    println("="^60)
    
    # Setup
    H, W, B = 256, 256, 2
    Δx = 1.0f0 / (W - 1)
    Δz = 1.0f0 / (H - 1)
    
    layer = StreamFunctionLayer(Δx, Δz, method=:central)
    
    # Test 1: Random Stream Function
    println("\n Test: Random Stream Function")
    ψ_random = randn(Float32, H, W, 1, B)
    
    velocity = layer(ψ_random)
    println("   Input Shape: $(size(ψ_random))")
    println("   Output Shape: $(size(velocity))")
    
    # Check Dimensionen (zentrale Differenzen → H-2, W-2)
    expected_h = H - 2
    expected_w = W - 2
    if size(velocity) == (expected_h, expected_w, 2, B)
        println("Dimensionen korrekt")
    else
        println("Dimensionen falsch: $(size(velocity)) vs. ($expected_h, $expected_w, 2, $B)")
    end
    
    # Test 2: Divergenzfreiheit
    println("\n Test: Divergenzfreiheit")
    diagnostics = validate_divergence_free(velocity)
    
    println("   Max Divergenz: $(diagnostics["max_divergence"])")
    println("   Mean Divergenz: $(diagnostics["mean_divergence"])")
    
    if diagnostics["divergence_free"]
        println("Divergenzfrei (< $(diagnostics["threshold"]))")
    else
        println("Divergenz über Threshold")
    end
    
    # Test 3: Gradient Flow
    println("\n Test: Gradient Flow (Flux Compatibility)")
    try
        # Simple Loss
        loss_fn = () -> sum(abs2, layer(ψ_random))
        
        # Gradient berechnen
        grads = Flux.gradient(loss_fn, Flux.params([ψ_random]))
        
        if haskey(grads, ψ_random) && !all(isnan, grads[ψ_random])
            println("Gradienten fließen korrekt")
        else
            println("Gradienten-Problem")
        end
    catch e
        println("Gradient-Test fehlgeschlagen: $e")
    end
    
    # Test 4: Bekanntes Feld (Linear)
    println("\n Test: Lineares Stream Function Field")
    # ψ(x,z) = x*z → vx = x, vz = -z
    x_grid = repeat(range(0, 1, length=W)', H, 1, 1, B)
    z_grid = repeat(range(0, 1, length=H), 1, W, 1, B)
    
    ψ_linear = Float32.(x_grid .* z_grid)
    velocity_linear = layer(ψ_linear)
    
    vx_linear = velocity_linear[:, :, 1, 1]
    vz_linear = velocity_linear[:, :, 2, 1]
    
    # Analytisch: vx ≈ x, vz ≈ -z (im Zentrum)
    center_i, center_j = div(H, 2), div(W, 2)
    vx_center = vx_linear[center_i, center_j]
    vz_center = vz_linear[center_i, center_j]
    
    expected_vx = 0.5f0  # x = 0.5 im Zentrum
    expected_vz = -0.5f0 # -z = -0.5 im Zentrum
    
    error_vx = abs(vx_center - expected_vx)
    error_vz = abs(vz_center - expected_vz)
    
    println("   vx (Zentrum): $vx_center (erwartet: ~$expected_vx, Fehler: $error_vx)")
    println("   vz (Zentrum): $vz_center (erwartet: ~$expected_vz, Fehler: $error_vz)")
    
    if error_vx < 0.1 && error_vz < 0.1
        println("   Linearer Test bestanden")
    else
        println("   Größere Abweichungen")
    end
    
    println("\n" * "="^60)
    println("Stream Function Layer Tests abgeschlossen!")
    
    return diagnostics
end

# =============================================================================
# ALTERNATIVE: FORWARD DIFFERENZEN (für größere Output-Größe)
# =============================================================================

"""
    StreamFunctionLayerForward

Alternative mit Forward-Differenzen → Output-Größe nur um 1 reduziert.

Vorteil: Größere Output-Maps (H-1, W-1 statt H-2, W-2)
Nachteil: Etwas weniger genau als zentrale Differenzen
"""
const StreamFunctionLayerForward = StreamFunctionLayer

# Helper Constructor
stream_function_layer_forward(Δx, Δz) = StreamFunctionLayer(Δx, Δz, method=:forward)
stream_function_layer_central(Δx, Δz) = StreamFunctionLayer(Δx, Δz, method=:central)

# =============================================================================
# MODUL-INFO
# =============================================================================

println("Stream Function Layer geladen!")
println("   - Divergenzfrei durch Konstruktion")
println("   - Flux-kompatibel (trainierbar)")
println("   - Mehrere Differenzierungs-Methoden")
println("")
println("Wichtige Funktionen:")
println("   - StreamFunctionLayer(Δx, Δz) - Layer erstellen")
println("   - test_stream_function_layer() - Unit Tests")
println("   - validate_divergence_free(velocity) - Validierung")
println("")
println("Quick Test verfügbar:")
println("   julia> test_stream_function_layer()")