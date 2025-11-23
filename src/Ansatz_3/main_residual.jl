#!/usr/bin/env julia

using Random
using Dates
using Printf



include("streamfunction_poisson.jl")
include("lamem_interface.jl")
include("data_generation_residual.jl")
include("dataset_residual.jl")
include("unet_psi.jl")           # U-Net-Architektur wiederverwenden
include("training_residual.jl")
include("evaluate_residual.jl")

using .StreamFunctionPoisson
using .LaMEMInterface
using .DataGenerationResidual
using .DatasetResidual
using .UNetPsi
using .TrainingResidual
using .EvaluateResidual

# ================================
# Konfiguration
# ================================

# mode:
#   "generate_data"  → viele Samples als .jld2 speichern
#   "train"          → U-Net auf Daten trainieren
#   "eval_dataset"   → gesamten Datensatz auswerten (Statistik je Kristallanzahl)
mode          = "eval_dataset"   # z.B. zum Testen


# Zufall
seed = 42
rng  = MersenneTwister(seed)

# --- Geometrie-Parameter für die Datengenerierung ---
min_crystals = 1      # minimum Anzahl Kristalle pro Sample
max_crystals = 1      # maximum Anzahl Kristalle pro Sample

radius_mode  = :fixed # :fixed oder :range

R_fixed      = 0.1    # wird nur benutzt, wenn radius_mode == :fixed
R_min        = 0.02   # wird nur benutzt, wenn radius_mode == :range
R_max        = 0.05


# Datengenerierung
n_train   = 1000                # nur benutzt, wenn mode == "generate_data"
outdir    = "data_psi"          # Ordner für .jld2-Samples

# Training
epochs        = 100
batch_size    = 2
learning_rate = 1e-4
model_path    = "unet_psi._1000_100_8_1e4.bson"

# Eval
eval_sample_idx = 1
eval_prefix     = "eval_psi"
plots_save      = "eval_plots_phys"
psi_denorm = true                       # ob im physikalischen ψ-Raum ausgewertet wird


@info "Starte main.jl im Modus: $mode"

if mode == "generate_data"

    mkpath(outdir)
    @info "Erzeuge $n_train Residual-Samples in Ordner: $outdir"
    DataGenerationResidual.generate_dataset(
        outdir;
        n_train     = n_train,
        rng         = rng,
        nx          = 256,
        nz          = 256,
        η           = 1e20,
        Δρ          = 200.0,
        min_crystals = min_crystals,
        max_crystals = max_crystals,
        radius_mode  = radius_mode,
        R_fixed      = R_fixed,
        R_min        = R_min,
        R_max        = R_max,
    )

    @info "Residual-Datengenerierung abgeschlossen."

elseif mode == "train"

    mkpath(outdir)
    @info "Erzeuge $n_train Residual-Samples in Ordner: $outdir"
    DataGenerationResidual.generate_dataset(
        outdir;
        n_train     = n_train,
        rng         = rng,
        nx          = 256,
        nz          = 256,
        η           = 1e20,
        Δρ          = 200.0,
        min_crystals = min_crystals,
        max_crystals = max_crystals,
        radius_mode  = radius_mode,
        R_fixed      = R_fixed,
        R_min        = R_min,
        R_max        = R_max,
    )

    @info "Starte Residual-Training auf Datensatz in $outdir"
    TrainingResidual.train_unet_residual(; data_dir=outdir,
                                         epochs=epochs,
                                         batch_size=batch_size,
                                         lr=learning_rate,
                                         rng=rng,
                                         save_path=model_path)


elseif mode == "eval_dataset"
    outdir     = "data_residual"
    

    EvaluateResidual.evaluate_dataset_residual(
        data_dir      = outdir,
        model_path    = model_path,
        out_prefix    = "eval_residual_dataset",
        save_plots    = true,              # Bilder erzeugen
        plot_dir      = "eval_residual_plots",
        denorm_residual = true,
    )


else
    error("Unbekannter Modus: $mode")
end
