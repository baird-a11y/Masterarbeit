#!/bin/bash


#Hier vergeben wir dem Auftrag einen Namen. Das ist für die Schlange wichtig, damit andere sehen, wem das genau gehört
#SBATCH --job-name=Kitty_Paul
# Hier legen wir fest, wie lange maximal der Code laufen soll. Im Code selbst sagen wir ihm wie viele Schritte er maximal rechnen soll, hier wie lang er maximal dafür Zeit hat.
#SBATCH --time='10:00:00'
#SBATCH --ntasks=1

# Pfad des Skriptes. Muss immer im selben Ordner wie das Skript liegen. Am besten das Skript selbst start.jl nennen.
/opt/julia/bin/julia Multi_crystal.jl

