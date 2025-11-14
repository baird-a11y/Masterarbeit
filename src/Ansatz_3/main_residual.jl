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
# (später ggf. evaluate_residual.jl)

using .StreamFunctionPoisson
using .LaMEMInterface
using .DataGenerationResidual
using .DatasetResidual
using .UNetPsi
using .TrainingResidual

mode = "train_residual"   # oder "generate_residual"

seed = 42
rng  = MersenneTwister(seed)

min_crystals = 1
max_crystals = 10

radius_mode  = :fixed
R_fixed      = 0.04
R_min        = 0.02
R_max        = 0.05

n_train   = 10
outdir    = "data_residual"

epochs        = 4
batch_size    = 2
learning_rate = 1e-5
model_path    = "unet_residual.bson"

@info "Starte main_residual.jl im Modus: $mode"

if mode == "generate_residual"

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

elseif mode == "train_residual"

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

else
    error("Unbekannter Modus: $mode")
end
