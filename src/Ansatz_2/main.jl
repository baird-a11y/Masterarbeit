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
include("evaluate_psi.jl")


# Module verfügbar machen
using .StreamFunctionPoisson
using .LaMEMInterface
using .DataGenerationPsi
using .DatasetPsi
using .UNetPsi
using .TrainingPsi
using .EvaluatePsi



# ================================
# Konfiguration
# ================================

# mode:
#   "debug_single"   → eine Simulation + ψ-Plot
#   "generate_data"  → viele Samples als .jld2 speichern
#   "train"          → U-Net auf Daten trainieren
mode          = "eval_single"

# Zufall
seed = 42
rng  = MersenneTwister(seed)

# Datengenerierung
n_train   = 4              # nur benutzt, wenn mode == "generate_data"
outdir    = "data_psi"      # Ordner für .jld2-Samples

# Training
epochs        = 2
batch_size    = 2
learning_rate = 1e-4
model_path    = "unet_psi.bson"

# Eval
eval_sample_idx = 1
eval_prefix     = "eval_psi"


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

elseif mode == "train"

    mkpath(outdir)
    @info "Erzeuge $n_train Trainings-Samples in Ordner: $outdir"
    DataGenerationPsi.generate_dataset(outdir; n_train=n_train, rng=rng)
    @info "Datengenerierung abgeschlossen."

    @info "Starte Training auf Datensatz in $outdir"
    TrainingPsi.train_unet(; data_dir=outdir,
                            epochs=epochs,
                            batch_size=batch_size,
                            lr=learning_rate,
                            rng=rng,
                            save_path=model_path)

elseif mode == "eval_single"

    @info "Evaluiere ein einzelnes Sample aus $outdir mit Modell $model_path"
    EvaluatePsi.evaluate_single(; data_dir=outdir,
                                model_path=model_path,
                                sample_idx=eval_sample_idx,
                                out_prefix=eval_prefix)


else
    error("Unbekannter Modus: $mode")
end
