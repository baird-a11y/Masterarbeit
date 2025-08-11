# UNet für Geschwindigkeitsfeld-Vorhersage in Multi-Kristall Sedimentationssystemen

Dieses Repository implementiert eine UNet-Architektur zur Vorhersage von Strömungsfeldern um sinkende Kristalle in geowissenschaftlichen Systemen. Das Projekt ist in Julia entwickelt und nutzt Flux.jl als Deep Learning Framework in Verbindung mit LaMEM.jl für die physikalische Simulation.

## Projektübersicht

**Forschungsziel:** Untersuchung der Generalisierungsfähigkeit von UNet-Modellen zur Vorhersage von Geschwindigkeitsfeldern bei variierender Kristallanzahl.

**Kernfrage:** Kann ein auf 10 Kristallen trainiertes UNet-Modell zuverlässige Vorhersagen für Systeme mit 1-10 Kristallen erstellen?

### Input/Output

- **Input:** Phasenfeld (1 Kanal: 0=Matrix, 1-10=Kristall-IDs)
- **Output:** 2D-Geschwindigkeitsfelder (2 Kanäle: v_x, v_z)
- **Auflösung:** 256×256 Pixel (konsistent durch gesamte Pipeline)
- **Normalisierung:** Geschwindigkeiten skaliert mit Stokes-Geschwindigkeit

## Verzeichnisstruktur

```
├── batch_management.jl              # Adaptive Batch-Größen und GPU-Memory-Management
├── data_processing.jl               # Datenvorverarbeitung und Größenanpassung
├── lamem_interface.jl               # LaMEM-Integration für 1-10 Kristalle
├── main.jl                          # Hauptscript für Generalisierungsstudie
├── training.jl                      # Zygote-sicheres Training mit Early Stopping
├── unet_architecture.jl             # UNet-Implementierung
├── unet_config.jl                   # UNet-Konfigurations-Management
├── submit_job.sh                    # SLURM Job-Script
└── README.md                        # Diese Dokumentation
```

## Installation und Setup

### Julia-Abhängigkeiten installieren
```julia
using Pkg
Pkg.add(["LaMEM", "GeophysicalModelGenerator", "Flux", "CUDA", "Optimisers", "BSON", "Statistics", "Random"])
```

## Verwendung

### Schnellstart
```bash
# SLURM-Job einreichen (empfohlen für Server)
sbatch submit_job.sh

# ODER: Lokal ausführen
julia main.jl
```

### Konfiguration anpassen
Die Hauptparameter können in `main.jl` unter `SERVER_CONFIG` angepasst werden:

```julia
const SERVER_CONFIG = (
    target_crystal_count = 10,         # Trainingsziel: 10 Kristalle
    n_training_samples = 6,            # Anzahl Trainingssamples
    num_epochs = 20,                   # Training-Epochen
    learning_rate = 0.00005f0,         # Lernrate
    batch_size = 1,                    # Batch-Größe
    eval_crystal_range = 9:10,         # Evaluierungsbereich
    n_eval_samples_per_count = 1,      # Samples pro Kristallanzahl
    use_gpu = false,                   # Hardware-Einstellung
)
```

## Technische Details

### UNet-Architektur
```julia
struct SimplifiedUNet
    # Encoder: 3-stufig mit Skip-Connections
    enc1_conv1::Conv; enc1_conv2::Conv; enc1_pool::MaxPool
    enc2_conv1::Conv; enc2_conv2::Conv; enc2_pool::MaxPool
    enc3_conv1::Conv; enc3_conv2::Conv; enc3_pool::MaxPool
    
    # Bottleneck
    bottleneck_conv1::Conv; bottleneck_conv2::Conv
    
    # Decoder: 3-stufig mit Skip-Connections
    dec3_up::ConvTranspose; dec3_conv1::Conv; dec3_conv2::Conv
    dec2_up::ConvTranspose; dec2_conv1::Conv; dec2_conv2::Conv
    dec1_up::ConvTranspose; dec1_conv1::Conv; dec1_conv2::Conv
    
    # Output für Regression
    output_conv::Conv
end
```

### Training-Konfiguration
- **Optimizer:** Adam mit angepasster Lernrate
- **Verlustfunktion:** MSE für Regression
- **Normalisierung:** Stokes-Geschwindigkeit als Referenz
- **Early Stopping:** Verhindert Overfitting

### Multi-Kristall Simulation
- **Variable Kristallanzahl:** 1-10 Kristalle pro Simulation
- **Intelligente Platzierung:** Automatische Kollisionsvermeidung
- **Physikalisch realistische Parameter:** Variable Radien und Dichtedifferenzen
- **LaMEM-Integration:** Vollständige Navier-Stokes Lösung

## Experimenteller Aufbau

Das Experiment folgt einem systematischen Ansatz:

1. **Training:** Das UNet wird hauptsächlich auf 10-Kristall-Systemen trainiert (70% der Trainingsdaten)
2. **Evaluierung:** Das trainierte Modell wird auf Systemen mit 1-10 Kristallen getestet
3. **Metriken:** Mean Absolute Error (MAE) und R² zur Bewertung der Generalisierungsfähigkeit

## Output

Das Programm speichert alle Ergebnisse in `generalization_results.bson`:
- Trainiertes Modell
- Training- und Validation-Losses
- Evaluierungsmetriken für jede Kristallanzahl
- Konfigurationsparameter

### Erwartete Ergebnisse
```
ERGEBNISSE:
9 Kristalle: MAE = 0.003156, R² = 0.948
10 Kristalle: MAE = 0.004555, R² = 0.921
```

## Hardware-Anforderungen

- **CPU:** Multi-Core empfohlen
- **RAM:** Mindestens 8 GB
- **GPU:** Optional, CUDA-Support verfügbar
- **Speicher:** ~2-5 GB für komplette Studie

## SLURM Cluster-Ausführung

Das bereitgestellte `submit_job.sh` Script ist für SLURM-Cluster konfiguriert:

```bash
#SBATCH --job-name=X_UNET
#SBATCH --time='10:00:00'
#SBATCH --ntasks=1
```

## Hauptfunktionen

- `LaMEM_Multi_crystal()`: Generiert physikalische Simulationen für 1-10 Kristalle
- `create_simplified_unet()`: Erstellt Zygote-kompatible UNet-Architektur
- `train_velocity_unet_safe()`: Robustes Training mit Early Stopping
- `simple_evaluate_on_crystal_count()`: Evaluierung der Generalisierungsfähigkeit

## Validierung

Das System prüft automatisch:
- Kontinuitätsgleichung: ∂v_x/∂x + ∂v_z/∂z ≈ 0
- Physikalische Plausibilität der Geschwindigkeitsfelder
- Numerische Stabilität während des Trainings

## Zitation


**Verwandte Arbeiten:**
- LaMEM.jl: Kaus et al. (2016) - Lithospheric Modeling Environment
- U-Net: Ronneberger et al. (2015) - Convolutional Networks for Biomedical Image Segmentation
- Flux.jl: Innes et al. (2018) - Machine Learning Stack in Julia