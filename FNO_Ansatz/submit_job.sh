#!/bin/bash

#SBATCH --job-name=Paul_FNO
#SBATCH --time='72:00:00'
#SBATCH --ntasks=1


# =============================================
# Schritt 1: Daten generieren
# =============================================

# Training-Daten (500 Samples, 1-4 Kristalle)
#/opt/julia/bin/julia generate_data.jl --n 1000 --out data_train --n_crystals_range 10:10 --seed 37
#/opt/julia/bin/julia generate_data.jl --n 1000 --out data_train --n_crystals_range 20:20 --seed 26
#/opt/julia/bin/julia generate_data.jl --n 1000 --out data_train --n_crystals_range 15:15 --seed 32
#/opt/julia/bin/julia generate_data.jl --n 1000 --out data_train --n_crystals_range 25:25 --seed 23


# Validation-Daten (100 Samples, anderer Seed)
#/opt/julia/bin/julia generate_data.jl --n 10 --out data_eval --n_crystals_range 1:1 --seed 12
#/opt/julia/bin/julia generate_data.jl --n 10 --out data_eval --n_crystals_range 2:2 --seed 13
#/opt/julia/bin/julia generate_data.jl --n 10 --out data_eval --n_crystals_range 3:3 --seed 24
#/opt/julia/bin/julia generate_data.jl --n 10 --out data_eval --n_crystals_range 4:4 --seed 1
#/opt/julia/bin/julia generate_data.jl --n 10 --out data_eval --n_crystals_range 10:10 --seed 1
#/opt/julia/bin/julia generate_data.jl --n 10 --out data_eval --n_crystals_range 20:20 --seed 1

# Evaluierungsdaten (10 Samples)
# /opt/julia/bin/julia generate_data.jl --n 10 --out data_eval --n_crystals_range 5:5 --seed 12
# /opt/julia/bin/julia generate_data.jl --n 10 --out data_eval --n_crystals_range 6:6 --seed 13
# /opt/julia/bin/julia generate_data.jl --n 10 --out data_eval --n_crystals_range 7:7 --seed 78
# /opt/julia/bin/julia generate_data.jl --n 10 --out data_eval --n_crystals_range 8:8 --seed 1
# /opt/julia/bin/julia generate_data.jl --n 10 --out data_eval --n_crystals_range 11:11 --seed 1
# /opt/julia/bin/julia generate_data.jl --n 10 --out data_eval --n_crystals_range 50:50 --seed 1
# /opt/julia/bin/julia generate_data.jl --n 10 --out data_eval --n_crystals_range 60:100 --seed 1

# =============================================
# Schritt 2: Training
# =============================================

/opt/julia/bin/julia run_training.jl --epochs 50 --batch 16 --train_dir data_train --val_dir data_val

# =============================================
# Schritt 3: Evaluation (separat, nutzt best_model)
# =============================================

/opt/julia/bin/julia run_eval.jl --checkpoint checkpoints/best_model.jld2 --data_dir data_eval --out eval_output_5
