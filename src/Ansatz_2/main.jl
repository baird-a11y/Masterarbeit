#!/usr/bin/env julia
# ================================
# main.jl – Steuerung für Ansatz 2
# ================================

using Random
using Dates
using Printf

# Eigene Dateien (alle im selben Ordner)
include("streamfunction_poisson.jl")
include("lamem_interface.jl")
include("data_generation_psi.jl")
include("dataset_psi.jl")
include("unet_psi.jl")
include("training_psi.jl")


# Module verfügbar machen
using .StreamFunctionPoisson
using .LaMEMInterface
using .DataGenerationPsi
using .DatasetPsi
using .UNetPsi
using .TrainingPsi


# ================================
# Konfiguration
# ================================

# mode:
#   "debug_single"   → eine Simulation + ψ-Plot
#   "generate_data"  → viele Samples als .jld2 speichern
mode = "generate_data"          # für Server z.B. auf "generate_data" setzen

seed = 42
rng  = MersenneTwister(seed)

# Datengenerierung
n_train   = 10                 # Anzahl der Samples für generate_data
outdir    = "data_psi"         # Zielordner für .jld2-Dateien

@info "Starte main.jl im Modus: $mode"

# ================================
# Ablaufsteuerung
# ================================

if mode == "generate_data"
    mkpath(outdir)
    @info "Erzeuge $n_train Trainings-Samples in Ordner: $outdir"
    DataGenerationPsi.generate_dataset(outdir; n_train=n_train, rng=rng)
    @info "Datengenerierung abgeschlossen."

elseif mode == "debug_single"
    # Nur für lokalen Test – braucht GLMakie
    @info "Erzeuge ein einzelnes Sample und plotte ψ_norm."

    using GLMakie

    input, ψ_norm, scale, meta = DataGenerationPsi.generate_psi_sample(rng)

    nx, nz, _ = size(input)
    @info "Gridgröße: ($nx, $nz), Normierungsfaktor = $scale"
    @info "Meta-Daten: $(meta)"

    fig = Figure(resolution = (800, 600))
    ax  = Axis(fig[1, 1], title = "ψ_norm", xlabel = "x-Index", ylabel = "z-Index")

    # Transponiert für "normale" Darstellung (x horizontal, z vertikal)
    hm  = heatmap!(ax, ψ_norm')
    Colorbar(fig[1, 2], hm, label = "ψ_norm")

    timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
    filename  = "psi_debug_$timestamp.png"
    save(filename, fig)

    display(fig)
    @info "Plot gespeichert als $filename"

else
    error("Unbekannter Modus: $mode")
end
