# Datei: fno_layers.jl
# 2D Fourier Neural Operator Layer-Bausteine
# Daten-Layout: (nx, nz, C, B) – kompatibel mit Flux Conv
# SpectralConv2D   FFT → Low-Mode Multiply → iFFT (global path)
# FNOBlock         spectral + 1×1 Conv (local path) + activation


# Architektur

# Input x: (nx, nz, C, B) Float32
#          │
#     ┌────┴────┐
#     │         │
# SpectralConv2D   Conv((1,1))
#     │  rfft        │
#     │  W·x̂         │  (lokaler Pfad)
#     │  irfft       │
#     └────┬────┘
#          │  + (elementweise Addition)
#          │
#       gelu(·)
#          │
# Output y: (nx, nz, C, B) Float32
# Implementierte Komponenten
# Komponente	                Was sie tut
# complex_weight_init	        ComplexF32 Gewichte aus N(0, scale²)
# SpectralConv2D	            Struct mit W1, W2 ∈ ℂ^(Cout×Cin×mx×mz) für 2 Mode-Blöcke
# spectral_mul	                Batched Matmul: out[i,j,o,b] = Σ_c W[o,c,i,j]·x̂[i,j,c,b] via NNlib.batched_mul
# Forward-Pass	                rfft(x,(1,2)) → Low-Mode Extract → spectral_mul → Zero-Pad → irfft
# FNOBlock	                    gelu(spectral(x) + conv1x1(x))

# Zentrale Design-Entscheidungen
# rfft statt fft – spart ~50% Speicher, dim 1 wird auf nx÷2+1 reduziert → nur 2 Gewichtsblöcke (pos/neg z-Frequenzen), nicht 4
# Zygote-kompatibel – kein Array-Mutation im Forward-Pass, Spektrum wird mit cat + zeros_like zusammengebaut
# batched_mul – Moden als Batch-Dimension, effizient über alle (mx×mz) Moden gleichzeitig
# zeros_like via similar + fill! – CPU/GPU-transparent (wenn CUDA geladen ist, erzeugt es CuArrays)
# @functor – Flux findet automatisch W1, W2 und Conv-Parameter für Training

module FNOLayers

using Flux
using Functors: @functor
using FFTW
using NNlib: batched_mul, gelu
using Zygote: @ignore

export SpectralConv2D,
       FNOBlock

# =============================================================================
# Helpers
# =============================================================================

"""
Erzeugt einen Null-Tensor mit gleichem Array-Typ wie `x` (CPU/GPU-kompatibel).
@ignore: Zero-Padding hat per Definition keine Gradienten.
"""
zeros_like(x::AbstractArray, dims...) = @ignore fill!(similar(x, dims...), zero(eltype(x)))

"""
    complex_weight_init(Cout, Cin, mx, mz; scale=0.02f0)

Erzeugt komplexe Gewichte W ∈ ℂ^(Cout×Cin×mx×mz) mit kleiner Normalverteilung.
Die kleine Standardabweichung σ = 0.02 sorgt dafür, dass die Spektralgewichte an-
fangs nahe Null liegen. Damit dominiert zu Beginn des Trainings der lokale Pfad (1 × 1-
Faltung), und der spektrale Pfad wird graduell angelernt — ein implizites Warmup der
globalen Features.
"""
function complex_weight_init(Cout::Int, Cin::Int, mx::Int, mz::Int;
                             scale::Float32 = 0.02f0)
    re = randn(Float32, Cout, Cin, mx, mz) .* scale
    im = randn(Float32, Cout, Cin, mx, mz) .* scale
    return complex.(re, im)
end

# =============================================================================
# F4 – SpectralConv2D Struct
# =============================================================================

"""
    SpectralConv2D(Cin, Cout, modes_x, modes_z)

Spectral Convolution Layer für 2D FNO (Li et al., 2020).

Intern werden nur die niedrigen Fourier-Moden gelernt:
- W1 für positive z-Frequenzen (1:mz)
- W2 für negative z-Frequenzen (nz-mz+1:nz)

Nutzt `rfft` entlang (x,z) → dim 1 wird auf nx÷2+1 reduziert,
daher reichen 2 Gewichtsblöcke (statt 4 bei vollem FFT).

Input/Output: reell `(nx, nz, C, B)` Float32.
"""
struct SpectralConv2D{A <: AbstractArray{ComplexF32, 4}}
    W1::A       # (Cout, Cin, modes_x, modes_z) – positive z-Frequenzen
    W2::A       # (Cout, Cin, modes_x, modes_z) – negative z-Frequenzen
    modes_x::Int
    modes_z::Int
end

@functor SpectralConv2D

function SpectralConv2D(Cin::Int, Cout::Int, modes_x::Int, modes_z::Int;
                        init_scale::Float32 = 0.02f0)
    W1 = complex_weight_init(Cout, Cin, modes_x, modes_z; scale=init_scale)
    W2 = complex_weight_init(Cout, Cin, modes_x, modes_z; scale=init_scale)
    return SpectralConv2D(W1, W2, modes_x, modes_z)
end

function Base.show(io::IO, l::SpectralConv2D)
    Cout, Cin = size(l.W1, 1), size(l.W1, 2)
    print(io, "SpectralConv2D($Cin => $Cout, modes=($(l.modes_x), $(l.modes_z)))")
end

# =============================================================================
# F6 – Spectral Multiply (batched matmul über Moden)
# =============================================================================

"""
    spectral_mul(x̂, W)

Komplexe Multiplikation im Fourier-Raum.

- `x̂`: (mx, mz, Cin, B) complex – extrahierte Low-Modes
- `W`:  (Cout, Cin, mx, mz) complex – gelernte Gewichte
- Output: (mx, mz, Cout, B) complex

Für jeden Mode (i,j): out[i,j,o,b] = Σ_c W[o,c,i,j] · x̂[i,j,c,b]
Implementiert via `NNlib.batched_mul` mit Moden als Batch-Dimension.
"""
function spectral_mul(x̂::AbstractArray{<:Complex, 4},
                      W::AbstractArray{<:Complex, 4})
    mx, mz, Cin, B = size(x̂)
    Cout = size(W, 1)
    M = mx * mz

    # x̂: (mx, mz, Cin, B) → (M, Cin, B) → (Cin, B, M)
    x_flat = permutedims(reshape(x̂, M, Cin, B), (2, 3, 1))

    # W: (Cout, Cin, mx, mz) → (Cout, Cin, M)
    W_flat = reshape(W, Cout, Cin, M)

    # batched_mul: (Cout, Cin, M) × (Cin, B, M) → (Cout, B, M)
    out_flat = batched_mul(W_flat, x_flat)

    # → (M, Cout, B) → (mx, mz, Cout, B)
    return reshape(permutedims(out_flat, (3, 1, 2)), mx, mz, Cout, B)
end

# =============================================================================
# F5 – SpectralConv2D Forward Pass
# =============================================================================

function (layer::SpectralConv2D)(x::AbstractArray{<:Real, 4})
    nx, nz, Cin, B = size(x)
    mx, mz = layer.modes_x, layer.modes_z
    Cout = size(layer.W1, 1)

    @assert Cin == size(layer.W1, 2) "Cin mismatch: input hat $Cin, Gewichte erwarten $(size(layer.W1, 2))"
    @assert mx ≤ nx ÷ 2 + 1 "modes_x=$mx zu groß für nx=$nx"
    @assert mz ≤ nz ÷ 2 "modes_z=$mz zu groß für nz=$nz"

    # ── Forward FFT (real → complex, dim 1 halved) ──
    x̂ = rfft(x, (1, 2))   # (nx÷2+1, nz, Cin, B)
    nk = size(x̂, 1)       # nx÷2+1

    # ── Low-mode multiply ──
    # Block 1: positive z-Frequenzen [1:mz]
    ŷ1 = spectral_mul(x̂[1:mx, 1:mz, :, :], layer.W1)

    # Block 2: negative z-Frequenzen [nz-mz+1:nz]
    ŷ2 = spectral_mul(x̂[1:mx, (nz-mz+1):nz, :, :], layer.W2)

    # ── Assemble output spectrum (Zygote-kompatibel, keine Mutation) ──
    z_gap = nz - 2mz
    if z_gap > 0
        z_pad = zeros_like(ŷ1, mx, z_gap, Cout, B)
        top_row = cat(ŷ1, z_pad, ŷ2; dims=2)      # (mx, nz, Cout, B)
    else
        top_row = cat(ŷ1, ŷ2; dims=2)
    end

    x_gap = nk - mx
    if x_gap > 0
        x_pad = zeros_like(ŷ1, x_gap, nz, Cout, B)
        ŷ = cat(top_row, x_pad; dims=1)            # (nk, nz, Cout, B)
    else
        ŷ = top_row
    end

    # ── Inverse FFT (complex → real) ──
    return irfft(ŷ, nx, (1, 2))   # (nx, nz, Cout, B) Float32
end

# =============================================================================
# F8 – FNOBlock (ein kompletter FNO-Layer)
# =============================================================================

"""
    FNOBlock(width, modes_x, modes_z; activation=gelu)

Ein FNO-Block bestehend aus:
- **Spectral path**: `SpectralConv2D` (globale Fourier-Features)
- **Local path**: `Conv((1,1))` (lokale/punktweise Features)
- **Activation**: `gelu` (default) oder beliebige Funktion

Forward: `y = σ(spectral(x) + local(x))`

Input/Output Shape: `(nx, nz, width, B)`.
"""
struct FNOBlock{S, C, F}
    spectral::S
    local_conv::C
    activation::F
end

@functor FNOBlock

function FNOBlock(width::Int, modes_x::Int, modes_z::Int;
                  activation = gelu)
    spectral   = SpectralConv2D(width, width, modes_x, modes_z)
    local_conv = Conv((1, 1), width => width)
    return FNOBlock(spectral, local_conv, activation)
end

function Base.show(io::IO, b::FNOBlock)
    print(io, "FNOBlock($(b.spectral), $(b.local_conv), $(b.activation))")
end

function (block::FNOBlock)(x::AbstractArray{<:Real, 4})
    return block.activation.(block.spectral(x) .+ block.local_conv(x))
end

end # module
