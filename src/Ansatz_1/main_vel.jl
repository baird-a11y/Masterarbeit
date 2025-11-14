#!/usr/bin/env julia

using Random
using Dates
using Printf
using CUDA

include("streamfunction_poisson.jl")  # brauchst du hier eigentlich nicht, aber schadet nicht
include("lamem_interface.jl")
include("data_generation_vel.jl")
include("dataset_vel.jl")
include("unet_psi.jl")       # U-Net wird geteilt!
include("training_vel.jl")
include("evaluate_vel.jl")

using .StreamFunctionPoisson
using .LaMEMInterface
using .DataGenerationVel
using .DatasetVel
using .UNetPsi
using .TrainingVel
using .EvaluateVel

mode = "train_vel"   # "generate_data_vel", "train_vel", "eval_dataset_vel"

seed = 42
rng  = MersenneTwister(seed)

min_crystals = 1
max_crystals = 10

radius_mode  = :fixed
R_fixed      = 0.04
R_min        = 0.02
R_max        = 0.05

n_train   = 10
outdir    = "data_vel"

epochs        = 4
batch_size    = 2
learning_rate = 1e-5
model_path    = "unet_vel.bson"

@info "Starte main_vel.jl im Modus: $mode"

if mode == "generate_data_vel"

    mkpath(outdir)
    @info "Erzeuge $n_train Vel-Samples in Ordner: $outdir"
    DataGenerationVel.generate_dataset(
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
    @info "Datengenerierung (Ansatz 1) abgeschlossen."

elseif mode == "train_vel"

    mkpath(outdir)
    @info "Erzeuge $n_train Vel-Trainings-Samples in Ordner: $outdir"
    DataGenerationVel.generate_dataset(
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

    @info "Starte Training von Ansatz 1 auf Datensatz in $outdir"
    TrainingVel.train_unet_vel(; data_dir=outdir,
                               epochs=epochs,
                               batch_size=batch_size,
                               lr=learning_rate,
                               rng=rng,
                               save_path=model_path)

elseif mode == "eval_dataset_vel"

    @info "Evaluiere Ansatz 1 auf Datensatz in $outdir mit Modell $model_path"
    EvaluateVel.evaluate_dataset(; data_dir   = outdir,
                                 model_path  = model_path,
                                 denorm_vel  = true)

else
    error("Unbekannter Modus: $mode")
end
