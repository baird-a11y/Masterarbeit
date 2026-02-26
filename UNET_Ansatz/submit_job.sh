#!/bin/bash


#Hier vergeben wir dem Auftrag einen Namen. Das ist für die Schlange wichtig, damit andere sehen, wem das genau gehört
#SBATCH --job-name=Paul_UNET
# Hier legen wir fest, wie lange maximal der Code laufen soll.
#SBATCH --time='54:00:00'
#SBATCH --ntasks=1

# Pfad des Skriptes. Muss immer im selben Ordner wie das Skript liegen. Am besten das Skript selbst start.jl nennen.
/opt/julia/bin/julia main.jl

/opt/julia/bin/julia run_eval.jl \
    --model    /home/pbaselt/Documents/Masterarbeit/Ergebnisse/UNET_Ergebnisse/One_Crystal/exp_one/unet_psi_best.bson \
    --data_dir /home/pbaselt/Documents/Masterarbeit/data_eval/n01 \
    --out      /home/pbaselt/Documents/Masterarbeit/Ergebnisse/UNET_Ergebnisse/One_Crystal/exp_one/eval_psi \
    --plot_dir /home/pbaselt/Documents/Masterarbeit/Ergebnisse/UNET_Ergebnisse/One_Crystal/exp_one/eval_plots \
    --denorm