# UNet für Geschwindigkeitsfeld-Vorhersage in Multi-Kristall Sedimentationssystemen

Dieses Repository implementiert eine UNet-Architektur zur Vorhersage von Strömungsfeldern um sinkende Kristalle in geowissenschaftlichen Systemen. Das Projekt ist in Julia entwickelt und nutzt Flux.jl als Deep Learning Framework in Verbindung mit LaMEM.jl für die physikalische Simulation.

## Projektübersicht

**Forschungsziel:** Untersuchung der Generalisierungsfähigkeit von UNet-Modellen zur Vorhersage von Geschwindigkeitsfeldern bei variierender Kristallanzahl mit Fokus auf LaMEM-Treue.

**Kernfrage:** Kann ein UNet-Modell zuverlässige LaMEM-konforme Vorhersagen für Systeme mit 1-15 Kristallen erstellen?

### Input/Output

- **Input:** Phasenfeld (1 Kanal: 0=Matrix, 1-10=Kristall-IDs)
- **Output:** 2D-Geschwindigkeitsfelder (2 Kanäle: v_x, v_z)
- **Auflösung:** 256×256 Pixel (konsistent durch gesamte Pipeline)
- **Normalisierung:** Geschwindigkeiten skaliert mit Stokes-Geschwindigkeit
- **Ziel:** Möglichst exakte Reproduktion der LaMEM-Simulationsergebnisse

## Verzeichnisstruktur

```
├── batch_management.jl              # Adaptive Batch-Größen und GPU-Memory-Management
├── data_processing.jl               # Datenvorverarbeitung und Größenanpassung
├── lamem_interface.jl               # LaMEM-Integration für 1-15 Kristalle
├── main.jl                          # Hauptscript für 10-Kristall Training
├── training.jl                      # Zygote-sicheres Training mit Early Stopping
├── unet_architecture.jl             # UNet-Implementierung
├── unet_config.jl                   # UNet-Konfigurations-Management
├── evaluate_model.jl                # LaMEM-Treue Evaluierung und Metriken
├── visualization.jl                 # 3-Panel Visualisierung (Phase|LaMEM|UNet)
├── submit_job.sh                    # SLURM Job-Script
└── README.md                        # Diese Dokumentation
```

## Installation und Setup

### Julia-Abhängigkeiten installieren
```julia
using Pkg
Pkg.add(["LaMEM", "GeophysicalModelGenerator", "Flux", "CUDA", "Optimisers", "BSON", "Statistics", "Random", "Plots"])
```

## Verwendung

### Training
```bash
# SLURM-Job einreichen (empfohlen für Server)
sbatch submit_job.sh

# ODER: Lokal ausführen
julia main.jl
```

### Evaluierung und Visualisierung
```julia
# LaMEM-Treue Evaluierung
include("evaluate_model.jl")
test_lamem_fidelity_evaluation()

# 3-Panel Visualisierung
include("visualization.jl")
test_visualization()

# Interaktive Kristall-Exploration
interactive_visualization()
```

### Konfiguration anpassen
Die Hauptparameter können in `main.jl` unter `SERVER_CONFIG` angepasst werden:

```julia
const SERVER_CONFIG = (
    target_crystal_count = 10,         # Trainingsziel: 10 Kristalle
    n_training_samples = 20,           # Anzahl Trainingssamples
    num_epochs = 30,                   # Training-Epochen
    learning_rate = 0.0005f0,          # Lernrate
    batch_size = 1,                    # Batch-Größe
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
- **Zygote-Kompatibilität:** Sichere Gradient-Berechnung

### Multi-Kristall Simulation
- **Variable Kristallanzahl:** 1-15 Kristalle pro Simulation
- **Intelligente Platzierung:** Automatische Kollisionsvermeidung
- **Physikalisch realistische Parameter:** Variable Radien und Dichtedifferenzen
- **LaMEM-Integration:** Vollständige Navier-Stokes Lösung

## Evaluierung und Metriken

### LaMEM-Treue Bewertung
Das Projekt verwendet ein umfassendes Evaluierungssystem, das LaMEM-Simulationen als Ground Truth behandelt:

#### Haupt-Qualitätsmetriken
- **MAE/RMSE:** Direkte Abweichung von LaMEM-Geschwindigkeitsfeldern
- **Strukturelle Korrelation:** Ähnlichkeit der Strömungsmuster mit LaMEM
- **Relative Fehler:** Bezogen auf LaMEM-Geschwindigkeitsbereiche

#### Physik-Konsistenz
- **Kontinuitätsgleichung:** ∂vx/∂x + ∂vz/∂z ≈ 0 für LaMEM und UNet
- **Divergenz-Ähnlichkeit:** Reproduktion der LaMEM-Physik

#### Bewertungsskala
```
🥇 Exzellent: MAE < 0.01, Korrelation > 0.95
🥈 Gut: MAE < 0.05, Korrelation > 0.85  
🥉 Akzeptabel: MAE < 0.1, Korrelation > 0.70
⚠️ Schwach: Korrelation > 0.50 (Struktur erkennbar)
❌ Unzureichend: Korrelation < 0.50
```

### 3-Panel Visualisierung
```
[Phasenfeld] | [LaMEM: v_z] | [UNet: v_z]
```
- Automatische Kristall-Zentren Erkennung (weiße/rote Punkte)
- Geschwindigkeits-Minima Markierung (gelbe Sterne)
- Koordinaten-Alignment Analyse
- Interaktive Kristallanzahl-Exploration (1-15 Kristalle)

## Aktueller Entwicklungsstand

### ✅ Vollständig Implementiert (Stand: 15. August 2025)
Das System ist vollständig funktional mit allen Hauptkomponenten:

- **✅ Zygote-sichere UNet-Architektur** mit stabiler Gradient-Berechnung
- **✅ 10-Kristall Trainingspipeline** mit intelligenter Kristall-Platzierung
- **✅ LaMEM-Treue Evaluierung** als wissenschaftlich fundierte Bewertungsmetrik
- **✅ 3-Panel Visualisierung** (Phasenfeld | LaMEM | UNet) mit koordinaten-korrekter Darstellung
- **✅ Interaktive Kristall-Exploration** für 1-15 Kristalle mit optimierten Grid-Layouts
- **✅ Koordinaten-Debugging-Tools** zur Transformation-Validierung

### 🎯 Erfolgreiche Validierung
**Koordinaten-Transformation:** Perfekt funktionsfähig (±1 Pixel Genauigkeit)
```
Test bestätigt: LaMEM [-1,1] → Pixel [1,256] Transformation korrekt
Kristall-Erkennung: Clustering-Algorithmus robust für 1-15 Kristalle
Visualisierung: Grid-Layouts optimiert für klare Multi-Kristall Darstellung
```

**Demonstrierte Funktionalität:**
- **2-Kristall System:** Links-Rechts Layout, GT Alignment: 6.4px, UNet: 2.6px
- **5-Kristall System:** Grid-Formation, GT Alignment: 19.6px, UNet: 1.2px
- **Physikalische Plausibilität:** Dipol-Strömungen und Multi-Partikel Interaktionen sichtbar

### ⚠️ Identifizierte Optimierungsbereiche
**Training-Performance:** 
- Aktuelles Modell zeigt suboptimale LaMEM-Treue (MAE: 0.488, Ziel: <0.05)
- Bias und Skalierungsprobleme in UNet-Ausgaben
- Training-Konfiguration benötigt Optimierung

**GPU-Kompatibilität:** 
- CPU-Training stabil und funktional
- GPU-Kernel-Compilation-Probleme bei komplexen Tensor-Operationen
- Backup-Lösung: CPU-Training für alle Experimente

### Nächste Entwicklungsschritte (Prioritäten für 16. August 2025)

#### 🚀 Hochpriorität: Training-Optimierung
```julia
# Empfohlene verbesserte Konfiguration:
OPTIMIZED_CONFIG = (
    n_training_samples = 100,     # 5x mehr Daten für bessere Generalisierung
    num_epochs = 50,              # Längeres Training für Konvergenz
    learning_rate = 0.001f0,      # Höhere Lernrate für effizientere Optimierung
    batch_size = 2,               # Stabilere Gradienten
    early_stopping_patience = 15  # Mehr Geduld für komplexe 10-Kristall Physik
)
```

#### 📊 Systematische Evaluierung
1. **LaMEM-Treue Verbesserung:** Ziel MAE < 0.05, Korrelation > 0.85
2. **Generalisierungstests:** Validation auf 1-15 Kristall-Systemen
3. **Physik-Konsistenz:** Kontinuitätsgleichung und Divergenz-Ähnlichkeit

#### 🔬 Erweiterte Analyse
- **Hyperparameter-Tuning:** Lernrate, Architektur-Größe, Regularisierung
- **Physik-Informed Loss:** Integration von ∂vx/∂x + ∂vz/∂z ≈ 0
- **Benchmark-Vergleiche:** Gegen lineare Interpolation und naive Baselines

## Output und Ergebnisse

Das Programm speichert alle Ergebnisse strukturiert:

### Training-Output
```
ten_crystal_checkpoints/
├── best_model.bson          # Bestes Modell basierend auf Validation Loss
├── final_model.bson         # Finales Modell nach Training
└── checkpoint_epoch_X.bson  # Zwischenstände alle 5 Epochen

ten_crystal_results/
├── ten_crystal_training_results.bson  # Komplette Training-Historie
└── ten_crystal_dataset.jls            # Generierte Trainingsdaten
```

### Evaluierung-Output
```
# LaMEM-Treue Metriken
test_results.bson            # Quantitative Bewertung

# Visualisierungen  
test_visualization.png       # 3-Panel Beispielplot
visualization_X_crystals.png # Kristallanzahl-spezifische Plots
```

### Erwartete Ergebnisse (bei optimiertem Training)
```
LAMEM-TREUE EVALUIERUNG:
🎯 HAUPT-QUALITÄTSMETRIKEN:
  MAE Total: < 0.05
  Korrelation Total: > 0.85

📊 STRUKTURELLE ÄHNLICHKEIT:
  Strömungsmuster-Erhaltung: ✓
  Dipol-Strukturen erkennbar: ✓

⚡ PHYSIK-KONSISTENZ:
  Kontinuitäts-Status: ✓ Physikalisch plausibel
```

## Hardware-Anforderungen

- **CPU:** Multi-Core empfohlen (aktuell stabile Option)
- **RAM:** Mindestens 8 GB, 16 GB empfohlen
- **GPU:** Optional, CUDA-Support verfügbar (in Entwicklung)
- **Speicher:** ~2-5 GB für komplette Studie

## SLURM Cluster-Ausführung

Das bereitgestellte `submit_job.sh` Script ist für SLURM-Cluster konfiguriert:

```bash
#SBATCH --job-name=Paul_UNET
#SBATCH --time='10:00:00'
#SBATCH --ntasks=1
```

## Hauptfunktionen

### Datengenerierung
- `LaMEM_Multi_crystal()`: Generiert physikalische Simulationen für 1-15 Kristalle
- `generate_ten_crystal_dataset()`: Spezialisierte 10-Kristall Datenerstellung

### Training
- `create_simplified_unet()`: Erstellt Zygote-kompatible UNet-Architektur
- `train_velocity_unet_safe()`: Robustes Training mit Early Stopping
- `run_ten_crystal_training()`: Komplette 10-Kristall Trainingspipeline

### Evaluierung
- `calculate_lamem_fidelity_metrics()`: Umfassende LaMEM-Treue Bewertung
- `test_lamem_fidelity_evaluation()`: Schnelle Modell-Evaluierung
- `print_lamem_fidelity_summary()`: Strukturierte Ergebnisdarstellung

### Visualisierung
- `create_three_panel_plot()`: Phasenfeld|LaMEM|UNet Visualisierung
- `interactive_visualization()`: Interaktive Kristallanzahl-Exploration
- `create_crystal_comparison_plots()`: Batch-Visualisierung für Vergleiche

## Validierung

Das System prüft automatisch:
- **LaMEM-Treue:** Korrelation und MAE zwischen UNet und LaMEM-Simulationen
- **Kontinuitätsgleichung:** ∂v_x/∂x + ∂v_z/∂z ≈ 0 für physikalische Konsistenz
- **Kristall-Erkennung:** Clustering-basierte Kristall-Zentren Identifikation
- **Numerische Stabilität:** Zygote-kompatible Gradient-Berechnung

## Wissenschaftlicher Beitrag

### Methodische Innovationen
- **LaMEM-Treue als Hauptmetrik:** Fokus auf physikalische Genauigkeit statt künstlicher Koordinaten-Metriken
- **Clustering-basierte Kristall-Erkennung:** Robuste Identifikation variabler Kristallanzahlen
- **Zygote-sichere UNet-Architektur:** Stabile Gradient-Berechnung für Geschwindigkeitsfeld-Regression
- **Interaktive Evaluierung:** Systematische Generalisierungstests über Kristallanzahl-Bereiche

### Anwendungsgebiete
- **Geowissenschaften:** Magma-Kristall Interaktionen, Sedimentationsprozesse
- **Strömungsmechanik:** Multi-Partikel Sedimentation, komplexe Fluid-Struktur Interaktionen  
- **Machine Learning:** Physics-Informed Neural Networks für PDEs, UNet-Regression für kontinuierliche Felder

## Zitation

Wenn du dieses Repository verwendest, zitiere bitte:

```bibtex
@software{unet_velocity_prediction,
  title={UNet für Geschwindigkeitsfeld-Vorhersage in Multi-Kristall Sedimentationssystemen},
  author={[Dein Name]},
  year={2025},
  url={[Repository URL]},
  note={Masterarbeit - Machine Learning für LaMEM-Strömungsfelder}
}
```

**Verwandte Arbeiten:**
- LaMEM.jl: Kaus et al. (2016) - Lithospheric Modeling Environment
- U-Net: Ronneberger et al. (2015) - Convolutional Networks for Biomedical Image Segmentation  
- Flux.jl: Innes et al. (2018) - Machine Learning Stack in Julia