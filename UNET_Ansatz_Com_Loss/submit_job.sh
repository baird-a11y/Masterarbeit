#!/bin/bash

#SBATCH --job-name=Paul_UNET
#SBATCH --time='54:00:00'
#SBATCH --ntasks=1

# =============================================
# Experiment 1 – Single-crystal generalization
# Training: n = 1
# Eval:     n=1.
#          
#
# LR: 1e-3, 5e-3. 1e-4,5e-4
# Batch size: 8, 16
# =============================================

#

# =============================================
# Experiment 2 – Multi-crystal generalization
# Training: n in {1,...,10}
# Eval:     n=1..10 (in-distribution)
#           n=11..25 (out-of-distribution)
#
# LR: beste aus Experiment 1 eintragen!
# =============================================

/opt/julia/bin/julia run_training.jl \
     --epochs 180 \
     --batch 8 \
     --lr 5e-3 \
     --train_dir /local/home/baselt/src/Ansatz/data_train_n01_n10 \
     --val_dir   /local/home/baselt/src/Ansatz/data_val \
     --save_path exp2_8/unet_psi.bson \
     --history_csv exp2_8/training_history.csv

# --- Evaluation in-distribution (n=1..10) ---

 /opt/julia/bin/julia run_eval.jl \
     --model    exp2_8/unet_psi_best.bson \
     --data_dir /local/home/baselt/src/Ansatz/data_eval_n01_n25 \
     --out      exp2_8/eval_psi_indist \
     --plot_dir exp2_8/eval_plots_indist \
     --save_plots \
     --denorm


# =============================================
# Experiment 3 – Stress-testing generalization limits
# Training: n in {1,...,25}
# Eval:     n=1..25 (in-distribution)
#           n=26..100 (out-of-distribution)
#
# LR: beste aus Experiment 1 eintragen!
# =============================================

# /opt/julia/bin/julia run_training.jl \
#     --epochs 180 \
#     --batch 16 \
#     --lr 1e-3 \
#     --train_dir /local/home/baselt/src/Ansatz/data_train \
#     --val_dir   /local/home/baselt/src/Ansatz/data_val \
#     --save_path exp3/unet_psi.bson \
#     --history_csv exp3/training_history.csv

# --- Evaluation in-distribution (n=1..25) ---

# /opt/julia/bin/julia run_eval.jl \
#     --model    exp3/unet_psi_best.bson \
#     --data_dir /local/home/baselt/src/Ansatz/data_eval\
#     --out      exp3/eval_psi_indist \
#     --plot_dir exp3/eval_plots_indist \
#     --save_plots \
#     --denorm




# =============================================
# Experiment 4 – Crystal size generalization
# Training: n in {1,...,10}  (gleich wie Exp 2)
# Eval:     n=1..10 mit anders skalierten Kristallen
#           (Faktor bis 10 in beide Richtungen)
#
# LR: beste aus Experiment 1 eintragen!
# =============================================

# #/opt/julia/bin/julia run_training.jl \
#     --epochs 180 \
#     --batch 16 \
#     --lr 1e-3 \
#     --train_dir /local/home/baselt/src/Ansatz/data_train/n01_n10 \
#     --val_dir   /local/home/baselt/src/Ansatz/data_val/n01_n10 \
#     --save_path exp4/unet_psi.bson \
#     --history_csv exp4/training_history.csv

# # --- Evaluation auf anders skalierten Kristallen ---

# #/opt/julia/bin/julia run_eval.jl \
#     --model    exp4/unet_psi_best.bson \
#     --data_dir /local/home/baselt/src/Ansatz/data_eval/n01_n10_scaled \
#     --out      exp4/eval_psi_scaled \
#     --plot_dir exp4/eval_plots_scaled \
#     --save_plots \
#     --denorm
