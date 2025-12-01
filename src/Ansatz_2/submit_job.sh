#!/bin/bash


#Hier vergeben wir dem Auftrag einen Namen. Das ist für die Schlange wichtig, damit andere sehen, wem das genau gehört
#SBATCH --job-name=Paul_UNET
# Hier legen wir fest, wie lange maximal der Code laufen soll.
#SBATCH --time='48:00:00'
#SBATCH --ntasks=1

# Pfad des Skriptes. Muss immer im selben Ordner wie das Skript liegen. Am besten das Skript selbst start.jl nennen.
/opt/julia/bin/julia main.jl

