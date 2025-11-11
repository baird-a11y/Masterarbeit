module DataGenerationPsi

using Random
using JLD2
using Printf
using ..LaMEMInterface: run_sinking_crystals
using Statistics
"""
    build_input_channels(phase)

Erstellt Eingabekanäle für das U-Net.
Aktuell: nur Kristallmaske (phase == 1) als ein Kanal.
"""
function build_input_channels(phase)
    nx, nz = size(phase)
    input = Array{Float32}(undef, nx, nz, 1)
    input[:, :, 1] .= phase .== 1
    return input
end

"""
    normalize_psi(ψ)

Skaliert ψ über den mittleren Zehnerexponenten, um numerisch stabile Werte zu erhalten.
Gibt (ψ_norm, scale) zurück mit ψ_norm = ψ * scale.
"""
function normalize_psi(ψ)
    absψ = abs.(ψ)
    mask = absψ .> 0
    if !any(mask)
        return ψ, 1.0
    end
    exponents = log10.(absψ[mask])
    p_mean    = mean(exponents)
    scale     = 10.0^(-p_mean)
    ψ_norm    = ψ .* scale
    return ψ_norm, scale
end

"""
    generate_psi_sample(rng)

Erzeugt ein einzelnes Sample mit zufälliger Kristallposition.
Rückgabe:
    input   :: Array{Float32,3}  (nx, nz, channels)
    ψ_norm  :: Array{Float64,2}
    scale   :: Float64
    meta    :: NamedTuple (cx, cz, R)
"""
function generate_psi_sample(rng::AbstractRNG)
    nx, nz = 256, 256

    cx = rand(rng, -0.7:0.01:0.7)
    cz = rand(rng, -0.7:0.01:0.7)
    R  = 0.1   # in km

    sim = run_sinking_crystals(; nx=nx, nz=nz,
                                centers_2D=[(cx, cz)],
                                radii=[R])

    input = build_input_channels(sim.phase)
    ψ_norm, scale = normalize_psi(sim.ψ)

    meta = (; cx, cz, R)
    return input, ψ_norm, scale, meta
end

"""
    generate_dataset(outdir; n_train=100, rng)

Erzeugt n_train Samples und speichert sie als einzelne .jld2-Dateien im Ordner `outdir`.
"""
function generate_dataset(outdir::String; n_train::Int=100, rng=Random.default_rng())
    mkpath(outdir)
    for i in 1:n_train
        input, ψ_norm, scale, meta = generate_psi_sample(rng)
        filename = joinpath(outdir, @sprintf("psi_sample_%04d.jld2", i))
        @save filename input ψ_norm scale meta
        @info "Sample $i gespeichert → $filename"
    end
end

end # module
