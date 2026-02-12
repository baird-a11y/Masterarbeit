#!/bin/bash

#SBATCH --job-name=Paul_FNO
#SBATCH --time='72:00:00'
#SBATCH --ntasks=1


# =============================================
# Schritt 1: Daten generieren
# =============================================

# # Training-Daten (500 Samples, 1-4 Kristalle)
 /opt/julia/bin/julia generate_data.jl --n 500 --out data_train --n_crystals_range 1:1 --seed 38
 #/opt/julia/bin/julia generate_data.jl --n 500 --out data_train --n_crystals_range 2:2 --seed 23
 #/opt/julia/bin/julia generate_data.jl --n 500 --out data_train --n_crystals_range 3:3 --seed 23
 #/opt/julia/bin/julia generate_data.jl --n 500 --out data_train --n_crystals_range 4:4 --seed 23


# # Validation-Daten (100 Samples, anderer Seed)
# /opt/julia/bin/julia generate_data.jl --n 100 --out data_val --n_crystals_range 1:1 --seed 123
# /opt/julia/bin/julia generate_data.jl --n 100 --out data_val --n_crystals_range 2:2 --seed 123
# /opt/julia/bin/julia generate_data.jl --n 100 --out data_val --n_crystals_range 3:3 --seed 123
# /opt/julia/bin/julia generate_data.jl --n 100 --out data_val --n_crystals_range 4:4 --seed 123

# =============================================
# Schritt 2: Training
# =============================================

/opt/julia/bin/julia run_training.jl --epochs 500 --batch 16 --train_dir data_train --val_dir data_val

# =============================================
# Schritt 3: Evaluation (separat, nutzt best_model)
# =============================================

/opt/julia/bin/julia run_eval.jl --checkpoint checkpoints/best_model.jld2 --data_dir data_val --out eval_output_2
