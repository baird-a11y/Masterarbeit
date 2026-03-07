#!/bin/bash

#SBATCH --job-name=Paul_FNO
#SBATCH --time='72:00:00'
#SBATCH --ntasks=1


# =============================================
# Schritt 1: Daten generieren
# =============================================

# Training-Daten (1000 Samples, 1-4 Kristalle)
#/opt/julia/bin/julia generate_data.jl --n 1000 --out data_train --n_crystals_range 5:5 --seed 37
#/opt/julia/bin/julia generate_data.jl --n 1000 --out data_train --n_crystals_range 6:6 --seed 26
#/opt/julia/bin/julia generate_data.jl --n 1000 --out data_train --n_crystals_range 7:7 --seed 32
#/opt/julia/bin/julia generate_data.jl --n 1000 --out data_train --n_crystals_range 8:8 --seed 23
#/opt/julia/bin/julia generate_data.jl --n 1000 --out data_train --n_crystals_range 9:9 --seed 23


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
# /opt/julia/bin/julia generate_data.jl --n 10 --out data_eval --n_crystals_range 100:100 --seed 1
# /opt/julia/bin/julia generate_data.jl --n 10 --out data_eval --n_crystals_range 60:100 --seed 1

# =============================================
# Experiment 2 und 3 
# =============================================

# =============================================
# Training
# =============================================

# /opt/julia/bin/julia run_training.jl --epochs 100 --batch 16 --train_dir /local/home/baselt/src/Ansatz/data_train/n01 --val_dir /local/home/baselt/src/Ansatz/data_val/n01 --lr 1e-3 --save_dir checkpoints_exp1_16_1 --history_csv history_exp1_16_1.csv
# /opt/julia/bin/julia run_eval.jl --checkpoint checkpoints_exp1_16_1/best_model.jld2 --data_dir /local/home/baselt/src/Ansatz/data_eval/n01 --out eval_output_exp1_16_1 --history_csv history_exp1_16_1.csv

# /opt/julia/bin/julia run_training.jl --epochs 100 --batch 16 --train_dir /local/home/baselt/src/Ansatz/data_train/n01 --val_dir /local/home/baselt/src/Ansatz/data_val/n01 --lr 5e-3 --save_dir checkpoints_exp1_16_2 --history_csv history_exp1_16_2.csv
# /opt/julia/bin/julia run_eval.jl --checkpoint checkpoints_exp1_16_2/best_model.jld2 --data_dir /local/home/baselt/src/Ansatz/data_eval/n01 --out eval_output_exp1_16_2 --history_csv history_exp1_16_2.csv

# /opt/julia/bin/julia run_training.jl --epochs 100 --batch 16 --train_dir /local/home/baselt/src/Ansatz/data_train/n01 --val_dir /local/home/baselt/src/Ansatz/data_val/n01 --lr 1e-4 --save_dir checkpoints_exp1_16_3 --history_csv history_exp1_16_3.csv
# /opt/julia/bin/julia run_eval.jl --checkpoint checkpoints_exp1_16_3/best_model.jld2 --data_dir /local/home/baselt/src/Ansatz/data_eval/n01 --out eval_output_exp1_16_3 --history_csv history_exp1_16_3.csv

/opt/julia/bin/julia run_training.jl --epochs 100 --batch 16 --modes 32 --train_dir /local/home/baselt/src/Ansatz/data_train/n01 --val_dir /local/home/baselt/src/Ansatz/data_val/n01 --lr 5e-4 --save_dir checkpoints_exp1_16_5 --history_csv history_exp1_16_5.csv
/opt/julia/bin/julia run_eval.jl --checkpoint checkpoints_exp1_16_5/best_model.jld2 --data_dir /local/home/baselt/src/Ansatz/data_eval/n01 --out eval_output_exp1_16_5 --history_csv history_exp1_16_5.csv


# /opt/julia/bin/julia run_training.jl --epochs 100 --batch 8 --train_dir /local/home/baselt/src/Ansatz/data_train/n01 --val_dir /local/home/baselt/src/Ansatz/data_val/n01 --lr 1e-3 --save_dir checkpoints_exp1_8_1 --history_csv history_exp1_8_1.csv
# /opt/julia/bin/julia run_eval.jl --checkpoint checkpoints_exp1_8_1/best_model.jld2 --data_dir /local/home/baselt/src/Ansatz/data_eval/n01 --out eval_output_exp1_8_1 --history_csv history_exp1_8_1.csv

# /opt/julia/bin/julia run_training.jl --epochs 100 --batch 8 --train_dir /local/home/baselt/src/Ansatz/data_train/n01 --val_dir /local/home/baselt/src/Ansatz/data_val/n01 --lr 5e-3 --save_dir checkpoints_exp1_8_2 --history_csv history_exp1_8_2.csv
# /opt/julia/bin/julia run_eval.jl --checkpoint checkpoints_exp1_8_2/best_model.jld2 --data_dir /local/home/baselt/src/Ansatz/data_eval/n01 --out eval_output_exp1_8_2 --history_csv history_exp1_8_2.csv

# /opt/julia/bin/julia run_training.jl --epochs 100 --batch 8 --train_dir /local/home/baselt/src/Ansatz/data_train/n01 --val_dir /local/home/baselt/src/Ansatz/data_val/n01 --lr 1e-4 --save_dir checkpoints_exp1_8_3 --history_csv history_exp1_8_3.csv
# /opt/julia/bin/julia run_eval.jl --checkpoint checkpoints_exp1_8_3/best_model.jld2 --data_dir /local/home/baselt/src/Ansatz/data_eval/n01 --out eval_output_exp1_8_3 --history_csv history_exp1_8_3.csv

# /opt/julia/bin/julia run_training.jl --epochs 100 --batch 8 --train_dir /local/home/baselt/src/Ansatz/data_train/n01 --val_dir /local/home/baselt/src/Ansatz/data_val/n01 --lr 5e-4 --save_dir checkpoints_exp1_8_4 --history_csv history_exp1_8_4.csv
# /opt/julia/bin/julia run_eval.jl --checkpoint checkpoints_exp1_8_4/best_model.jld2 --data_dir /local/home/baselt/src/Ansatz/data_eval/n01 --out eval_output_exp1_8_4 --history_csv history_exp1_8_4.csv

# =============================================
# Experiment 2 und 3 
# =============================================

# =============================================
# Training
# =============================================

# /opt/julia/bin/julia run_training.jl --epochs 100 --batch 16 --train_dir /local/home/baselt/src/Ansatz/data_train_n01_n10 --val_dir /local/home/baselt/src/Ansatz/data_val --lr 1e-3 --save_dir checkpoints_exp2 --history_csv history_exp2.csv

# =============================================
# Evaluation (separat, nutzt best_model)
# =============================================

# /opt/julia/bin/julia run_eval.jl --checkpoint checkpoints_exp2/best_model.jld2 --data_dir /local/home/baselt/src/Ansatz/data_eval_n01_n25 --out eval_output_exp2 --history_csv history_exp2.csv

# =============================================
# Training
# =============================================

# /opt/julia/bin/julia run_training.jl --epochs 100 --batch 16 --train_dir /local/home/baselt/src/Ansatz/data_train --val_dir /local/home/baselt/src/Ansatz/data_val --lr 1e-3 --save_dir checkpoints_exp3 --history_csv history_exp3.csv

# =============================================
# Evaluation (separat, nutzt best_model)
# =============================================

# /opt/julia/bin/julia run_eval.jl --checkpoint checkpoints_exp3/best_model.jld2 --data_dir /local/home/baselt/src/Ansatz/data_eval --out eval_output_exp3 --history_csv history_exp3.csv

