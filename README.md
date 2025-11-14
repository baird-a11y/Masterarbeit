U-Net für Strömungsvorhersagen aus LaMEM-Simulationen

Masterarbeit – Computational Sciences (Geowissenschaften)

Dieses Repository enthält den vollständigen Code für drei Modelle zur Vorhersage von Strömungsgrößen in viskosen Medien (z. B. sinkende Kristalle in Magmen). Grundlage sind LaMEM-Simulationen, aus denen Trainingsdaten generiert und anschließend mit einem U-Net in Julia / Flux.jl ausgewertet werden.

Der Fokus liegt auf der Stromfunktion ψ, den Geschwindigkeitsfeldern und einem Residualansatz (analytisch vs. numerisch).

Inhalt

Datengenerierung (LaMEM Interface, zufällige Kristallgeometrien, Normalisierung)

U-Net-Architektur (gemeinsam für alle Ansätze)

Trainingspipelines (ψ-Modell, Geschwindigkeitsmodell)

Evaluierung mit Fehlerstatistiken & Heatmaps

Unterstützung für GPU / CPU (CUDA, fallback-safe)

Modelle / Ansätze
Ansatz 1: Geschwindigkeiten direkt lernen

Output: 
v_x
v_z
	​
Ziel: Direkte Approximation der LaMEM-Geschwindigkeitsfelder (Baseline).
Ort im Code:

data_generation_vel.jl

training_vel.jl

evaluate_vel.jl

Ansatz 2: Stromfunktion ψ vorhersagen

Output: 1 Kanal → ψ(x, z)
Ziel: Glattes, physikalisch konsistentes Feld (divergenzfrei).
Ort im Code:

data_generation_psi.jl

training_psi.jl

evaluate_psi.jl

Ansatz 3: Residuen lernen (analytisch – numerisch)

Output: Residualfeld r = ψ_analytic – ψ_LaMEM
Ziel: Modell lernt nur die Abweichungen zur analytischen Lösung → stabiler, besser generalisierbar.
Dokumentation: Ansatz 3.md
(Implementierung im Aufbau)

Projektstruktur
src/
 ├─ data_generation_psi.jl     # Sample-Generierung für ψ
 ├─ data_generation_vel.jl     # Sample-Generierung für v
 ├─ dataset_psi.jl             # Dataset-Loader für ψ
 ├─ dataset_vel.jl             # Dataset-Loader für v
 ├─ lamem_interface.jl         # Direkte Anbindung an LaMEM
 ├─ streamfunction_poisson.jl  # Poisson-Löser (ψ aus ω)
 ├─ unet_psi.jl                # U-Net-Architektur
 ├─ training_psi.jl            # Training für ψ
 ├─ training_vel.jl            # Training für v
 ├─ evaluate_psi.jl            # Evaluierung ψ
 ├─ evaluate_vel.jl            # Evaluierung v
 ├─ main.jl                    # Hauptskript: generate/train/eval (ψ)
 ├─ main_vel.jl                # Hauptskript: generate/train/eval (v)

Installation
Voraussetzungen

Julia ≥ 1.10

LaMEM installiert & im PATH

Abhängigkeiten:

using Pkg
Pkg.add(["Flux", "JLD2", "BSON", "CairoMakie", "Functors", "CUDA"])

Verwendung
1) Trainingsdaten erzeugen

Im Hauptskript main.jl:

mode = "generate_data"
julia main.jl


Legt .jld2-Samples in data_psi/ an.

2) ψ-Modell trainieren
mode = "train"
julia main.jl


Trainiert das U-Net auf den generierten Daten
→ speichert Modell: unet_psi.bson

3) Datensatz evaluieren
mode = "eval_dataset"
julia main.jl


MSE & relative L2-Errors

optional: Heatmaps mit Kristallumrissen

Gruppiert nach Kristallanzahl (1–10)

4) Geschwindigkeitsmodell (Ansatz 1)
mode = "train_vel"
julia main_vel.jl

Datengenerierung (LaMEM)

Jedes Sample enthält:

Kristallgeometrie (Zentren & Radien, 1–10 Kristalle)

Phasefeld (Maske)

ψ oder v-Felder

normierte Zielgröße

scalings (dynamic normalisation)

Metadaten (physikalische Koordinaten usw.)

Die Felder werden grundsätzlich auf einem 256×256-Gitter erzeugt.

U-Net Architektur

Encoder–Decoder mit Skip-Connections

4 Ebenen Downsampling / Upsampling

Conv → BatchNorm → ReLU

1 Output-Kanal (ψ) oder 2 Kanäle (v)

Architektur definiert in:

unet_psi.jl

Evaluierung

Für jedes Sample werden berechnet:

MSE

relative L2-Norm

Δψ oder Δv-Heatmaps

Kristall-Overlays

Gruppierung nach Kristallanzahl

Unterordner pro Kristallanzahl:

eval_plots_phys/
 ├─ n_01/
 ├─ n_02/
 └─ ...

Features

realistische Strömungsdaten via LaMEM

vollständige Training- und Evaluierungspipeline

GPU-Beschleunigung (mit Fallback)

sample-spezifische Normalisierung (ψ & v)

plots in physikalischen Koordinaten

modular aufgebaut (leicht erweiterbar)

Lizenz

Wird von mir irgendwann ergänzt (MIT / GPL / Proprietary).

Autor

Paul Baselt
Masterstudent Computational Sciences (Geowissenschaften)
Johannes Gutenberg-Universität Mainz