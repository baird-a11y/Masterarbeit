Ansatz 2 – U-Net auf der Stromfunktion ψ(x, z)

Bei Ansatz 2 steht die Stromfunktion ψ(x, z) im Zentrum. Aus ihr lassen sich die Geschwindigkeitskomponenten rekonstruieren, sodass das Modell implizit ein physikalisch konsistentes Strömungsfeld lernt.

1. Grundidee

Input des U-Nets

Geometrie / Materialinformation auf dem 256×256-Gitter.

Aktuell:

Kristallmaske (phase == 1)

Signed Distance Field (SDF) zum nächsten Kristall
→ zusammen als 2 Eingabekanäle (Maske + SDF).

Perspektivisch erweiterbar um:

Viskosität

Randbedingungen

weitere physikalische Felder.

Output des U-Nets

Ein Skalarfeld ψ(x, z) auf demselben Gitter.

Implementierung: ein Kanal (nx, nz, 1).

Trainingsziel

Das Netz approximiert die von LaMEM abgeleitete Stromfunktion ψ_LaMEM (bzw. deren normalisierte Version ψ_norm).

Evaluierung (konzeptionell)

Vergleich von ψ_pred (U-Net) und ψ_ref (ψ_LaMEM bzw. Poisson-Lösung).

Optional: Rekonstruktion der Geschwindigkeiten aus ψ_pred und Vergleich mit Vx, Vz aus LaMEM.

2. Datenpipeline für Ansatz 2
2.1 Datengenerierung in LaMEM

Für jede Stichprobe:

LaMEM-Simulation einer Kristallkonfiguration im 2D-Schnitt.

Kristall-Setup:

Anzahl: 1…n Kristalle.

Positionen zufällig.

Radien: fest oder zufällig in einem Intervall.

Die Funktion run_sinking_crystals liefert:

Phasefeld (Kristall / Matrix),

Vx, Vz,

Gradienten → daraus Wirbelstärke ω,

ψ (über Poisson-Gleichung berechnet).

Alle Simulationen werden seriell ausgeführt (siehe Threading-Abschnitt unten).

2.2 Berechnung von ω und ψ

Wirbelstärke:

ω = ∂Vz/∂x − ∂Vx/∂z, aus den von LaMEM ausgegebenen Geschwindigkeitsgradienten.

Stromfunktion ψ:

Lösung der Poisson-Gleichung

Δ
𝜓
=
−
𝜔
Δψ=−ω

Implementiert in poisson2D(omega, dx, dz) mit Dirichlet-Randbedingung ψ = 0 am Rand.

2.3 Vorbereitung der Trainingsdaten

Inputkanäle

build_input_channels(phase, x_vec_1D, z_vec_1D, centers_2D) erzeugt ein Array (nx, nz, 2) mit:

Kristallmaske ∈ {0,1},

Signed Distance Field (SDF):

außen: positive Distanz,

innen: negative Distanz,

normiert auf ungefähr [−1, 1].

Zielgröße

ψ wird als 2D-Feld (nx, nz) berechnet.

ψ wird normalisiert zu ψ_norm (siehe Normalisierung).

Speicherung: ψ_norm + zugehöriger scale.

Speicherung pro Sample

generate_psi_sample erzeugt .jld2-Dateien mit:

input → (nx, nz, 2),

ψ_norm,

scale,

meta (n_crystals, Zentren, Radien, Gitterkoordinaten etc.).

Datensatz-Aufbau

PsiDataset speichert die Dateipfade.

get_sample lädt input und ψ_norm, formt sie zu (nx, nz, C) und gibt x, y zurück.

Splits (Train/Val/Test): über Verzeichnisse/Unterordner der .jld2-Files realisierbar.

3. Normalisierung von ψ
3.1 Motivation

Die ψ-Werte aus der Poisson-Lösung sind sehr klein

typischerweise in der Größenordnung 10⁻¹²–10⁻¹⁵.

Direkte Verwendung wäre numerisch ungünstig:

schlechte Konditionierung der Loss-Funktion,

langsame oder instabile Konvergenz.

3.2 Dynamische Skalierung pro Sample

In normalize_psi(ψ):

Berechnung der Exponenten:

exponents = log10.(absψ[mask]) (nur über nicht-null Werte).

Bestimmung des mittleren Exponenten:

p_mean = mean(exponents).

Skalierungsfaktor:

scale = 10.0^(-p_mean).

Normierte Stromfunktion:

ψ_norm = ψ * scale.

Rückgabe: (ψ_norm, scale).

Vorteil: ψ_norm liegt typischerweise im Bereich O(1) → stabileres Training.

4. Training des U-Nets
4.1 Architektur

U-Net aus UNetPsi.build_unet(in_channels, out_channels):

Input-Kanäle: 2 (Maske + SDF).

Output-Kanäle: 1 (ψ_norm).

Encoder:

Convolution-Blocks mit 3×3-Kernen, BatchNorm + ReLU.

Downsampling via strided Conv, keine MaxPool-Layer mehr
→ reduziert Grid-/Checkerboard-Artefakte.

Decoder:

ConvTranspose zum Upsampling,

Skip-Connections durch Concatenation,

finaler 1×1-Conv auf 1 Kanal.

4.2 Loss-Funktion & Optimierung

Basis-Loss:

Huber-Loss statt reinem MSE:

robust gegenüber Ausreißern,

glatter Übergang zwischen L2- und L1-Verhalten.

Optional:

Weighted MSE:

Kristallregionen werden höher gewichtet als Matrix,

Ziel: bessere Auflösung der Strukturen direkt an den Kristallgrenzen.

Optimierung:

manuelles SGD-Update:

p .= p .- lr * grad


GPU/CPU:

optionales CUDA-Training mit sicherer Fallback-Logik.

Geräteselektion: use_gpu = nothing | true | false.

Modell-Parameter werden mit fmap konsistent auf das Zielgerät verschoben.

5. Evaluierung
5.1 Evaluierungsskript

evaluate_dataset:

lädt Modell (aus .bson),

iteriert über alle Samples im Datensatz,

berechnet:

ψ_pred (normalisiert oder de-normalisiert),

Fehlermaße für ψ, ψ_x, ψ_z,

gruppiert nach Kristallanzahl.

Optionen:

denorm_psi = false:

Evaluierung im normalisierten Raum (ψ_norm).

denorm_psi = true:

Evaluierung im physikalischen Raum (ψ = ψ_norm / scale).

5.2 Metriken

Für ψ:

MSE(ψ):

mittlere quadratische Abweichung.

Relative L2-Norm:

∥
𝜓
pred
−
𝜓
true
∥
2
∥
𝜓
true
∥
2
∥ψ
true
	​

∥
2
	​

∥ψ
pred
	​

−ψ
true
	​

∥
2
	​

	​


Pixelweise Relativfehler:

Anteil der Pixel mit Fehler > 1 %, 5 %, 10 %:

ε₀₁, ε₀₅, ε₁₀.

Für Ableitungen ψ_x, ψ_z:

MSE(ψ_x), MSE(ψ_z),

relative L2-Normen für ψ_x und ψ_z.

Alles wird pro Kristallanzahl n gesammelt und in einer CSV
<out_prefix>_by_n.csv abgelegt.

5.3 Plots

Pro Sample (optional):

ψ-Figuren (3 Panels):

ψ_true,

ψ_pred,

Δψ = ψ_pred − ψ_true,

Gradienten-Figur (2×3 Panels):

ψ_x true / pred / Fehler,

ψ_z true / pred / Fehler,

Plots in physikalischen Koordinaten (km),

Kristallumrisse (Kreise) werden überlagert.

6. Threading-Problematik bei der Datengenerierung
6.1 Ausgangssituation

Ursprünglich wurde versucht, die LaMEM-Simulationen für Ansatz 2 mit Threads zu parallelisieren:

Threads.@threads for i in 1:N
    run_sinking_crystals(...)
end


Das führte zu zwei grundlegenden Problemen:

6.2 LaMEM ist nicht thread-safe (PETSc/MPI)

Alle Threads teilen sich denselben Prozessraum:

gleicher PETSc-Kontext,

gleicher MPI-State.

Bei parallelen Aufrufen von run_sinking_crystals:

mehrere PETSc-Initialisierungen im selben Prozess,

kollidierende MPI-Kommunikatoren.

Folge:

nach einigen erfolgreichen Samples:

Prozessabbrüche (Exitcode 83),

nicht deterministisches Auftreten → klassische Race-Condition.

6.3 Race-Conditions beim Schreiben von LaMEM-Ausgabedateien

Alle Threads schrieben in dieselben LaMEM-Output-Dateien:

FS_vel_gradient.0000.vtr

FS_vel_gradient.info

FS_vel_gradient.0000.pvtr

Da LaMEM hierfür nicht thread-sicher ausgelegt ist, entstanden:

teilweise korrupt erzeugte Dateien,

Einlesen führte u. a. zu:

XMLParseError: premature end of data,

unvollständigen Arrays (Vorticity, Gradients).

Charakteristisch:

Die Fehler traten erst nach 30–80 Samples auf,

→ typisches Verhalten von Race-Conditions.

7. Versuche und finale Lösung der Datengenerierung
7.1 Idee: Multi-Prozess (Distributed)

Konzept: mehrere Prozesse statt Threads:

jeder Prozess mit eigenem Speicher,

eigener PETSc/MPI-Kontext,

eigenen Output-Verzeichnissen.

Theoretisch die saubere Lösung (LaMEM mag Prozesse mehr als Threads).

Aber:

hätte bedeutet:

kompletten Umbau der Datengenerierung,

Interprozess-Kommunikation / Queueing,

Verwaltung von getrennten Output-Pfaden (run01/, run02/, …).

Für die Masterarbeit zeitlich zu aufwendig.

7.2 Endgültige Entscheidung: Serielle Ausführung

Alle LaMEM-Simulationen werden jetzt seriell ausgeführt.

Vorteile:

100 % stabil,

keine Datei-Race-Conditions,

keine PETSc/MPI-Konflikte,

deterministisches Verhalten.

Performance (Daumenregel):

ca. 8–10 Sekunden pro Sample,

1 000 Samples ≈ 2.2–3 Stunden,

10 000 Samples ≈ 24–30 Stunden,

→ für den Umfang der Masterarbeit akzeptabel.

8. Architektur- und Trainingsverbesserungen nach Stabilisierung

Nachdem die Datengenerierung stabil war, wurden mehrere Änderungen umgesetzt, um die Qualität der Vorhersagen deutlich zu verbessern.

8.1 Erweiterter Input: Maske + Signed Distance Field (SDF)

Vorher:

nur Kristallmaske (0/1) als Eingabekanal.

Jetzt:

2 Kanäle:

Kristallmaske,

SDF zum nächsten Kristallzentrum (innen negativ, außen positiv, normiert).

Effekt:

glattere Inputs,

weniger harte Kanten,

deutlich weniger vertikale/horizontale Artefakte in ψ_pred.

8.2 U-Net-Redesign: keine MaxPools mehr

Vorher:

Downsampling via MaxPool,

typisch für Grid- und Checkerboard-Artefakte,

in den Fehlerplots sichtbar als vertikale/horizontale Linien.

Jetzt:

Downsampling über strided Convs (Conv mit stride=2),

weiterhin 3×3-Convs + BatchNorm + ReLU,

Up-Sampling über ConvTranspose.

Effekt:

stabilere Gradienten,

glattere Vorhersagen,

deutlich weniger Artefakte entlang Grid-Grenzen.

8.3 Training & Evaluierung

Huber-Loss (statt reinem MSE) → robustere Regression von ψ.

optionaler gewichteter MSE → stärkere Fokussierung auf Kristallregionen.

GPU-Support:

in Training und Evaluierung einheitlich,

Fallback auf CPU, falls CUDA nicht verfügbar/initialisierbar ist.

Evaluierung:

Pixel-Fehlerstatistiken (ε₀₁, ε₀₅, ε₁₀),

Metriken für ψ, ψ_x, ψ_z,

Übersicht nach Kristallanzahl (CSV + Logging),

Plot-Output für ψ und Gradienten.

9. Kurz-Fazit für das Gespräch mit deinem Betreuer

Physikalische Idee:
Ansatz 2 lernt ψ(x, z) direkt; Geschwindigkeiten sind daraus ableitbar → physikalisch konsistentes Strömungsfeld.

Technische Basis:

LaMEM-Simulationen liefern Vx, Vz, ω, ψ.

ψ wird normalisiert → ψ_norm.

U-Net mit 2 Eingabekanälen (Maske + SDF) und 1 Output-Kanal (ψ_norm).

Wichtige Lessons Learned:

LaMEM/PETSc/MPI sind nicht thread-safe → keine Thread-Parallelisierung.

Serielle Datengenerierung ist der stabile Kompromiss.

Erweiterung des Inputs (SDF) und Umbau der Architektur (strided Convs statt MaxPool) haben die Linien-Artefakte in den Vorhersagen deutlich reduziert.