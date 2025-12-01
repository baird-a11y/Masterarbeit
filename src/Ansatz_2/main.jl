#!/usr/bin/env julia
# ================================
# main.jl – Steuerung für Ansatz 2
# ================================

using Random
using Dates
using Printf
using CUDA

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
n_train   = 2000                # nur benutzt, wenn mode == "generate_data"
outdir    = "/local/home/baselt/src/Daten/data_psi"          # Ordner für .jld2-Samples

# Training
epochs        = 800
batch_size    = 8
learning_rate = 5e-5
model_path    = "unet_psi.bson"

# Eval
eval_sample_idx = 1
eval_prefix     = "eval_psi"
plots_save      = "eval_plots_phys_validation"
psi_denorm = true                       # ob im physikalischen ψ-Raum ausgewertet wird


@info "Starte main.jl im Modus: $mode"

# ================================
# Ablaufsteuerung
# ================================

if mode == "generate_data"

    mkpath(outdir)
    @info "Erzeuge $n_train Trainings-Samples in Ordner: $outdir"
    DataGenerationPsi.generate_dataset(
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

    @info "Datengenerierung abgeschlossen."

elseif mode == "train"

    # mkpath(outdir)
    # @info "Erzeuge $n_train Trainings-Samples in Ordner: $outdir"
    # DataGenerationPsi.generate_dataset(
    # outdir;
    # n_train     = n_train,
    # rng         = rng,
    # nx          = 256,
    # nz          = 256,
    # η           = 1e20,
    # Δρ          = 200.0,
    # min_crystals = min_crystals,
    # max_crystals = max_crystals,
    # radius_mode  = radius_mode,
    # R_fixed      = R_fixed,
    # R_min        = R_min,
    # R_max        = R_max,
    # )

    # @info "Datengenerierung abgeschlossen."

    @info "Starte Training auf Datensatz in $outdir"
    TrainingPsi.train_unet(; data_dir=outdir,
                            epochs=epochs,
                            batch_size=batch_size,
                            lr=learning_rate,
                            rng=rng,
                            save_path=model_path)

    @info "Training abgeschlossen."

    # @info "Evaluiere gesamten Datensatz in $outdir mit Modell $model_path"
    # EvaluatePsi.evaluate_dataset(; data_dir   = outdir,
    #                              model_path  = model_path,
    #                              out_prefix  = eval_prefix,
    #                              save_plots  = true,
    #                              plot_dir    = plots_save,
    #                              denorm_psi  = psi_denorm)

    


elseif mode == "eval_dataset"

    @info "Evaluiere gesamten Datensatz in $outdir mit Modell $model_path"
    EvaluatePsi.evaluate_dataset(; data_dir   = outdir,
                                 model_path  = model_path,
                                 out_prefix  = eval_prefix,
                                 save_plots  = true,
                                 plot_dir    = plots_save,
                                 denorm_psi  = psi_denorm)


else
    error("Unbekannter Modus: $mode")
end
