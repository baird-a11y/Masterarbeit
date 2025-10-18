# =============================================================================
# STREAM FUNCTION MODULE
# =============================================================================
# Dieses Modul implementiert die Berechnung von Geschwindigkeitsfeldern
# aus einer Stream Function ψ
#
# Physikalischer Hintergrund:
# - vx = ∂ψ/∂z  (vertikale Ableitung)
# - vz = -∂ψ/∂x (negative horizontale Ableitung)
# - Garantiert automatisch: ∇·v = 0 (Massenerhaltung)
# =============================================================================

"""
    compute_velocities_from_stream(ψ::AbstractArray{T,4}) where T

Berechnet Geschwindigkeitsfeld aus Stream Function.

# Argumente
- `ψ`: Stream Function mit Shape [H, W, 1, B]
  - H: Höhe
  - W: Breite
  - 1: Ein Kanal (Stream Function)
  - B: Batch-Größe

# Rückgabe
- Geschwindigkeitsfeld mit Shape [H, W, 2, B]
  - Kanal 1: vx (horizontale Geschwindigkeit)
  - Kanal 2: vz (vertikale Geschwindigkeit)

# Physik
Die Geschwindigkeiten werden aus der Stream Function berechnet:
- vx = ∂ψ/∂z  (Ableitung in z-Richtung, vertikal)
- vz = -∂ψ/∂x (negative Ableitung in x-Richtung, horizontal)

Dies garantiert automatisch die Inkompressibilität: ∇·v = 0
"""
function compute_velocities_from_stream(ψ::AbstractArray{T,4}) where T
    # ψ hat Shape: (H, W, 1, B)
    H, W, _, B = size(ψ)
    
    # Berechne Geschwindigkeiten mittels zentraler Differenzen
    vx = ∂z(ψ)    # ∂ψ/∂z (vertikale Ableitung)
    vz = -∂x(ψ)   # -∂ψ/∂x (negative horizontale Ableitung)
    
    # Kombiniere zu (H, W, 2, B)
    velocities = cat(vx, vz, dims=3)
    
    return velocities
end

"""
    ∂z(field::AbstractArray{T,4}) where T

Berechnet vertikale Ableitung (∂/∂z) mittels zentraler Differenzen.

# Argumente
- `field`: Eingabefeld mit Shape [H, W, C, B]

# Rückgabe
- Vertikale Ableitung mit gleicher Shape [H, W, C, B]

# Methode
Zentrale Differenzen mit periodischem Padding:
∂f/∂z ≈ (f[i+1] - f[i-1]) / 2Δz

Für Randpunkte wird periodisches Padding verwendet:
- Oben: f[H] wird verwendet
- Unten: f[1] wird verwendet
"""
function ∂z(field::AbstractArray{T,4}) where T
    # field Shape: (H, W, C, B)
    H, W, C, B = size(field)
    
    # Periodisches Padding: füge letzte Zeile oben und erste Zeile unten hinzu
    # Resultierende Shape: (H+2, W, C, B)
    padded = cat(
        field[end:end, :, :, :],  # Letzte Zeile (oben)
        field,                      # Original
        field[1:1, :, :, :],        # Erste Zeile (unten)
        dims=1
    )
    
    # Zentrale Differenzen: (f[i+1] - f[i-1]) / 2
    # padded[3:end] entspricht f[i+1]
    # padded[1:end-2] entspricht f[i-1]
    derivative = (padded[3:end, :, :, :] .- padded[1:end-2, :, :, :]) ./ 2.0f0
    
    return derivative
end

"""
    ∂x(field::AbstractArray{T,4}) where T

Berechnet horizontale Ableitung (∂/∂x) mittels zentraler Differenzen.

# Argumente
- `field`: Eingabefeld mit Shape [H, W, C, B]

# Rückgabe
- Horizontale Ableitung mit gleicher Shape [H, W, C, B]

# Methode
Zentrale Differenzen mit periodischem Padding:
∂f/∂x ≈ (f[j+1] - f[j-1]) / 2Δx

Für Randpunkte wird periodisches Padding verwendet:
- Links: f[W] wird verwendet
- Rechts: f[1] wird verwendet
"""
function ∂x(field::AbstractArray{T,4}) where T
    # field Shape: (H, W, C, B)
    H, W, C, B = size(field)
    
    # Periodisches Padding: füge letzte Spalte links und erste Spalte rechts hinzu
    # Resultierende Shape: (H, W+2, C, B)
    padded = cat(
        field[:, end:end, :, :],  # Letzte Spalte (links)
        field,                      # Original
        field[:, 1:1, :, :],        # Erste Spalte (rechts)
        dims=2
    )
    
    # Zentrale Differenzen: (f[j+1] - f[j-1]) / 2
    # padded[:, 3:end] entspricht f[j+1]
    # padded[:, 1:end-2] entspricht f[j-1]
    derivative = (padded[:, 3:end, :, :] .- padded[:, 1:end-2, :, :]) ./ 2.0f0
    
    return derivative
end

"""
    verify_divergence_free(velocities::AbstractArray{T,4}; tol=1e-5) where T

Verifiziert, dass das Geschwindigkeitsfeld divergenzfrei ist.

# Argumente
- `velocities`: Geschwindigkeitsfeld [H, W, 2, B] mit vx und vz
- `tol`: Toleranzschwelle für Divergenz (default: 1e-5)

# Rückgabe
- `(is_divergence_free::Bool, max_divergence::Float32)`

# Physik
Testet die Bedingung: ∇·v = ∂vx/∂x + ∂vz/∂z ≈ 0
"""
function verify_divergence_free(velocities::AbstractArray{T,4}; tol=1e-5) where T
    # Extrahiere vx und vz
    vx = velocities[:, :, 1:1, :]
    vz = velocities[:, :, 2:2, :]
    
    # Berechne Divergenz: ∂vx/∂x + ∂vz/∂z
    divergence = ∂x(vx) .+ ∂z(vz)
    
    # Maximale absolute Divergenz
    max_div = maximum(abs.(divergence))
    
    # Prüfe ob unter Toleranz
    is_divergence_free = max_div < tol
    
    return is_divergence_free, max_div
end

# =============================================================================
# TEST-FUNKTIONEN
# =============================================================================

"""
    test_stream_function_derivatives()

Testet die Ableitungsfunktionen mit einem bekannten analytischen Beispiel.
"""
function test_stream_function_derivatives()
    println("="^60)
    println("TEST: Stream Function Ableitungen")
    println("="^60)
    
    # Erzeuge einfache Testfunktion: ψ(x,z) = sin(2πx) * cos(2πz)
    H, W = 64, 64
    x = range(0, 1, length=W)
    z = range(0, 1, length=H)
    
    # Stream Function
    ψ = zeros(Float32, H, W, 1, 1)
    for i in 1:H, j in 1:W
        ψ[i, j, 1, 1] = sin(2π * x[j]) * cos(2π * z[i])
    end
    
    # Analytische Ableitungen
    vx_analytical = zeros(Float32, H, W, 1, 1)
    vz_analytical = zeros(Float32, H, W, 1, 1)
    for i in 1:H, j in 1:W
        vx_analytical[i, j, 1, 1] = -2π * sin(2π * x[j]) * sin(2π * z[i])  # ∂ψ/∂z
        vz_analytical[i, j, 1, 1] = -2π * cos(2π * x[j]) * cos(2π * z[i])  # -∂ψ/∂x
    end
    
    # Numerische Berechnung
    velocities = compute_velocities_from_stream(ψ)
    vx_numerical = velocities[:, :, 1:1, :]
    vz_numerical = velocities[:, :, 2:2, :]
    
    # Fehler berechnen (ohne Randpunkte, da dort Padding-Effekte auftreten)
    inner = 3:H-2
    error_vx = sqrt(mean((vx_numerical[inner, inner, 1, 1] .- vx_analytical[inner, inner, 1, 1]).^2))
    error_vz = sqrt(mean((vz_numerical[inner, inner, 1, 1] .- vz_analytical[inner, inner, 1, 1]).^2))
    
    println("RMS-Fehler vx: ", error_vx)
    println("RMS-Fehler vz: ", error_vz)
    
    # Divergenz-Test
    is_div_free, max_div = verify_divergence_free(velocities)
    println("Maximale Divergenz: ", max_div)
    println("Divergenzfrei (tol=1e-5): ", is_div_free)
    
    success = error_vx < 0.1 && error_vz < 0.1 && is_div_free
    println("="^60)
    println(success ? "✓ TEST BESTANDEN" : "✗ TEST FEHLGESCHLAGEN")
    println("="^60)
    
    return success
end

"""
    test_stream_function_batch()

Testet Stream Function Berechnung mit Batch-Verarbeitung.
"""
function test_stream_function_batch()
    println("\n" * "="^60)
    println("TEST: Stream Function Batch-Verarbeitung")
    println("="^60)
    
    H, W, B = 128, 128, 4
    
    # Zufällige Stream Functions
    ψ = randn(Float32, H, W, 1, B)
    
    # Berechne Geschwindigkeiten
    velocities = compute_velocities_from_stream(ψ)
    
    # Prüfe Shape
    expected_shape = (H, W, 2, B)
    shape_correct = size(velocities) == expected_shape
    
    println("Input Shape: ", size(ψ))
    println("Output Shape: ", size(velocities))
    println("Expected Shape: ", expected_shape)
    println("Shape korrekt: ", shape_correct)
    
    # Prüfe Divergenz für alle Batch-Elemente
    all_div_free = true
    max_divs = Float32[]
    
    for b in 1:B
        is_div_free, max_div = verify_divergence_free(velocities[:, :, :, b:b])
        push!(max_divs, max_div)
        all_div_free = all_div_free && is_div_free
    end
    
    println("Maximale Divergenzen: ", max_divs)
    println("Alle divergenzfrei: ", all_div_free)
    
    success = shape_correct && all_div_free
    println("="^60)
    println(success ? "✓ TEST BESTANDEN" : "✗ TEST FEHLGESCHLAGEN")
    println("="^60)
    
    return success
end

# Automatischer Test beim Laden
println("Stream Function Modul geladen!")
println("Verfügbare Funktionen:")
println("  - compute_velocities_from_stream(ψ)")
println("  - verify_divergence_free(velocities)")
println("  - test_stream_function_derivatives()")
println("  - test_stream_function_batch()")
println("")
println("Führe Tests aus mit: test_stream_function_derivatives()")