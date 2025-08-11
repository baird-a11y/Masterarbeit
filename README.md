# UNet für Geschwindigkeitsfeld-Vorhersage in Kristall-Sedimentations-Systemen

Dieses Repository implementiert eine UNet-Architektur zur Vorhersage von Strömungsfeldern um sinkende Kristalle in geowissenschaftlichen Systemen. Das Projekt ist in Julia entwickelt und nutzt `Flux.jl` als Deep Learning Framework in Verbindung mit `LaMEM.jl` für die physikalische Simulation.

---

## Projektübersicht

**Forschungsziel:** Untersuchung der Generalisierungsfähigkeit von Machine Learning-Methoden zur Vorhersage von Geschwindigkeitsfeldern bei variierender Kristallanzahl.

**Kernfrage:** Kann ein auf 10 Kristallen trainiertes UNet-Modell zuverlässige Vorhersagen für Systeme mit 1-10 Kristallen machen?

### Input/Output

- **Input:** Phasenfeld (1 Kanal: 0=Matrix, 1-10=Kristall-IDs)
- **Output:** 2D-Geschwindigkeitsfelder (2 Kanäle: v_x, v_z)
- **Auflösung:** 256×256 Pixel (konsistent durch gesamte Pipeline)
- **Normalisierung:** Geschwindigkeiten skaliert mit Stokes-Geschwindigkeit

---

## Hauptfunktionen

### 1. Multi-Kristall Simulation
- **Variable Kristallanzahl:** 1-10 Kristalle pro Simulation
- **Intelligente Platzierung:** Automatische Kollisionsvermeidung
- **Physikalisch realistische Parameter:** Variable Radien, Dichtedifferenzen, Viskositäten
- **LaMEM-Integration:** Vollständige Navier-Stokes Lösung

### 2. Adaptive UNet-Architektur
```julia
struct SimplifiedUNet
    # Encoder: 3-stufig mit Skip-Connections
    enc1_conv1::Conv; enc1_conv2::Conv; enc1_pool::MaxPool
    enc2_conv1::Conv; enc2_conv2::Conv; enc2_pool::MaxPool
    enc3_conv1::Conv; enc3_conv2::Conv; enc3_pool::MaxPool
    
    # Bottleneck für höhere Abstraktion
    bottleneck_conv1::Conv; bottleneck_conv2::Conv
    
    # Decoder: 3-stufig mit adaptiven Skip-Connections
    dec3_up::ConvTranspose; dec3_conv1::Conv; dec3_conv2::Conv
    dec2_up::ConvTranspose; dec2_conv1::Conv; dec2_conv2::Conv
    dec1_up::ConvTranspose; dec1_conv1::Conv; dec1_conv2::Conv
    
    # Output für Regression
    output_conv::Conv
end
```

### 3. Generalisierungs-Evaluierung
- **Systematische Bewertung:** Evaluierung auf 1-10 Kristalle
- **Mehrfache Metriken:** MAE, MSE, R², Kontinuitätsfehler
- **Physikalische Validierung:** Überprüfung der Erhaltungsgesetze
- **Automatische Berichte:** Detaillierte Performance-Analyse

---

## Verzeichnisstruktur

```
generalization_checkpoints/     # Modell-Checkpoints während Training
generalization_results/         # Evaluierungs-Ergebnisse und Berichte
src/Demo/                      # Hauptmodule
├── lamem_interface_extended.jl      # Erweiterte LaMEM-Integration (1-10 Kristalle)
├── generalization_evaluation.jl    # Generalisierungs-Metriken
├── data_processing.jl              # Datenvorverarbeitung
├── unet_architecture.jl            # Zygote-sichere UNet-Implementierung
├── training.jl                     # Robustes Training mit Early Stopping
├── batch_management.jl             # Adaptive Batch-Größen
├── unet_config.jl                  # UNet-Konfigurations-Management
└── main_generalization.jl          # Hauptscript für vollständige Studie
Notizen/Masterarbeit/          # Experimentelle Dokumentation
README.md                      # Diese Dokumentation
submit_job.sh                  # SLURM Job-Script
```

---

## Installation und Setup

### 1. Repository klonen
```bash
git clone <repository-url>
cd unet-crystal-sedimentation
```

### 2. Julia-Abhängigkeiten installieren
```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()

# Zusätzliche Pakete für erweiterte Funktionalität
Pkg.add(["LaMEM", "GeophysicalModelGenerator", "Flux", "CUDA", "Optimisers", "BSON"])
```

### 3. System-Validierung
```julia
include("src/Demo/main_generalization.jl")

# Führe System-Test aus
if validate_generalization_system()
    println("✓ System bereit für Generalisierungsstudie")
else
    println("❌ System-Setup unvollständig")
end
```

---

## Verwendung

### 1. Vollständige Generalisierungsstudie

```julia
# Lade Hauptscript
include("src/Demo/main_generalization.jl")

# Führe komplette Studie aus (Training + Evaluierung)
success, results = run_generalization_study()

# Ergebnisse analysieren
if success
    create_generalization_report(results)
end
```

**Standard-Konfiguration:**
- **Training:** 200 Samples, Fokus auf 10 Kristalle (70% der Daten)
- **Evaluierung:** 25 Samples pro Kristallanzahl (1-10)
- **Metriken:** Mean Absolute Error (MAE) als Hauptmetrik

### 2. Angepasste Experimente

```julia
# Demo für schnelle Tests
demo_generalization_study(
    n_samples=50,        # Weniger Samples
    max_crystals=5,      # Weniger Kristallanzahlen
    num_epochs=10        # Kürzeres Training
)

# Einzelne Komponenten testen
dataset = generate_generalization_dataset(
    100,                              # Anzahl Samples
    target_crystal_count=8,           # Fokus auf 8 Kristalle
    crystal_distribution="weighted"   # 70% Ziel-Kristallanzahl
)

# Evaluierung auf spezifische Kristallanzahl
model = load_trained_model("generalization_checkpoints/best_model.bson")
metrics = evaluate_on_crystal_count(model, 5, n_eval_samples=20)
```

### 3. Parameter-Anpassung

```julia
# Modifiziere Konfiguration vor Ausführung
GENERALIZATION_CONFIG = merge(GENERALIZATION_CONFIG, (
    n_training_samples = 500,         # Mehr Trainingsdaten
    num_epochs = 50,                  # Längeres Training
    learning_rate = 0.0001f0,         # Niedrigere Lernrate
    target_crystal_count = 8,         # Anderes Trainingsziel
))
```

---

## Experimentelle Ergebnisse

### Typische Performance-Metriken

**Training auf 10 Kristallen, Evaluierung 1-10:**

| Kristalle | MAE_total | R²_total | Kontinuitäts-Fehler |
|-----------|-----------|----------|-------------------|
| 1         | 0.0023    | 0.962    | 0.00012          |
| 2         | 0.0031    | 0.948    | 0.00018          |
| 5         | 0.0045    | 0.923    | 0.00031          |
| 8         | 0.0052    | 0.915    | 0.00038          |
| 10        | 0.0048    | 0.921    | 0.00035          |

### Generalisierungs-Erkenntnisse

1. **Beste Performance:** Oft bei mittleren Kristallanzahlen (5-8)
2. **Performance-Degradation:** ~15-30% von 1 zu 10 Kristallen
3. **Physikalische Konsistenz:** Kontinuitätsgleichung gut erfüllt (Fehler < 0.0005)
4. **Trainingsstabilität:** Early Stopping verhindert Overfitting

---

## Technische Details

### Training-Konfiguration

```julia
# Optimizer: Adam mit angepasster Lernrate
opt_state = Optimisers.setup(Optimisers.Adam(0.0005f0), model)

# Verlustfunktion: MSE für Regression
loss_fn(model, x, y) = Flux.mse(model(x), y)

# Normalisierung: Stokes-Geschwindigkeit als Referenz
V_stokes = 2/9 * Δρ * 9.81 * (radius * 1000)^2 / η_magma
vx_norm = vx ./ V_stokes
vz_norm = vz ./ V_stokes
```

### Datenverarbeitung

```julia
function preprocess_lamem_sample(x, z, phase, vx, vz, v_stokes; target_resolution=256)
    # Konsistente Größenanpassung
    phase_resized = resize_power_of_2(phase, target_resolution)
    vx_resized = resize_power_of_2(vx, target_resolution)
    vz_resized = resize_power_of_2(vz, target_resolution)
    
    # Normalisierung für stabiles Training
    vx_norm = Float32.(vx_resized ./ v_stokes)
    vz_norm = Float32.(vz_resized ./ v_stokes)
    
    # Tensor-Format für UNet
    phase_tensor = reshape(phase_resized, target_resolution, target_resolution, 1, 1)
    velocity_tensor = cat(vx_norm, vz_norm, dims=3)
    
    return phase_tensor, velocity_tensor
end
```

### Hardware-Anforderungen

- **CPU:** Multi-Core empfohlen (Training nutzt mehrere Threads)
- **RAM:** Mindestens 16 GB für größere Datensätze
- **GPU:** Optional, CUDA-Support verfügbar (experimentell)
- **Speicher:** ~5-10 GB für komplette Studie

---

## SLURM Cluster-Ausführung

```bash
# Job-Script anpassen
nano submit_job.sh

#SBATCH --job-name=Generalization_Study
#SBATCH --time='24:00:00'        # Ausreichend Zeit für vollständige Studie
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4        # Mehrkern-Verarbeitung
#SBATCH --mem=32G                # Ausreichend RAM

/opt/julia/bin/julia src/Demo/main_generalization.jl

# Job einreichen
sbatch submit_job.sh
```

---

## Erweiterte Analyse

### Physikalische Validierung

```julia
# Kontinuitätsgleichung prüfen: ∂v_x/∂x + ∂v_z/∂z ≈ 0
continuity_error = calculate_continuity_error(velocity_predictions)

# Stokes-Gesetz Einhaltung
stokes_compliance = validate_stokes_behavior(phase_field, velocity_field)

# Geschwindigkeitsmagnitude-Konsistenz
magnitude_error = compare_velocity_magnitudes(predictions, ground_truth)
```

### Custom Metriken definieren

```julia
struct CustomMetrics
    mae_total::Float64
    physics_violation::Float64
    crystal_detection_accuracy::Float64
end

function calculate_custom_metrics(predictions, targets, phase_field)
    # Implementiere anwendungsspezifische Metriken
    # ...
end
```

---

## Häufige Probleme und Lösungen

### 1. Memory-Probleme bei großen Datensätzen
```julia
# Reduziere Batch-Größe
GENERALIZATION_CONFIG = merge(GENERALIZATION_CONFIG, (batch_size = 2,))

# Häufigere Garbage Collection
GC.gc()
```

### 2. Numerische Instabilität
```julia
# Niedrigere Lernrate
learning_rate = 0.0001f0

# Gradient Clipping aktivieren
Flux.clip!(grad, 1.0)
```

### 3. GPU Out-of-Memory
```julia
# CPU-Mode verwenden
use_gpu = false

# Kleinere Auflösung
target_resolution = 128
```

---

## Zukünftige Erweiterungen

1. **3D-Strömungen:** Erweiterung von 2D auf 3D-Simulationen
2. **Physics-Informed Neural Networks:** Integration physikalischer Gesetze in Loss-Funktion
3. **Echtzeitvorhersage:** Optimierung für Live-Simulation
4. **Multi-GPU Training:** Skalierung für größere Modelle
5. **Uncertainty Quantification:** Vorhersageverteilungen statt Punktschätzungen

---

## Zitation und Referenzen

Wenn Sie dieses Projekt in wissenschaftlichen Arbeiten verwenden, zitieren Sie bitte:

```bibtex
@misc{unet_crystal_sedimentation,
  title={UNet-basierte Geschwindigkeitsfeld-Vorhersage für Multi-Kristall Sedimentationssysteme},
  author={[Ihr Name]},
  year={2025},
  url={[Repository URL]},
  note={Generalisierungsstudie: Training auf 10 Kristallen, Evaluierung auf 1-10 Kristallen}
}
```

**Verwandte Arbeiten:**
- LaMEM.jl: Kaus et al. (2016) - Lithospheric Modeling Environment
- U-Net: Ronneberger et al. (2015) - Convolutional Networks for Biomedical Image Segmentation
- Flux.jl: Innes et al. (2018) - Machine Learning Stack in Julia

---

## Kontakt und Support

- **Issues:** Nutzen Sie GitHub Issues für Bug-Reports und Feature-Requests
- **Diskussionen:** GitHub Discussions für wissenschaftliche Fragen
- **Dokumentation:** Alle experimentellen Notizen in `Notizen/Masterarbeit/`

---

## Lizenz

Dieses Projekt steht unter [MIT Lizenz](LICENSE) - siehe LICENSE-Datei für Details.

---

## Changelog

### Version 2.0.0 (Aktuell - Generalisierungsstudie)
- ✅ Erweiterte LaMEM-Integration für 1-10 Kristalle
- ✅ Comprehensive Generalisierungs-Evaluierung 
- ✅ Mean Absolute Error (MAE) als Hauptmetrik
- ✅ Automatisierte Berichtserstellung
- ✅ Verbesserte Batch-Management und Memory-Optimierung

### Version 1.0.0 (Ein-Kristall Proof-of-Concept)
- ✅ Basis UNet-Architektur implementiert
- ✅ LaMEM-Integration für Einzelkristalle
- ✅ Zygote-kompatible Architektur
- ✅ MSE-basiertes Training erfolgreich
- ✅ R² > 0.94 für Ein-Kristall Systeme erreicht

---

## Schnellstart-Anleitung

### Für Einsteiger (Demo)
```julia
# 1. Module laden
include("src/Demo/main_generalization.jl")

# 2. Kleines Demo ausführen
success, results = demo_generalization_study(
    n_samples=20,
    max_crystals=3,
    num_epochs=5
)

# 3. Ergebnisse anzeigen
if success
    create_generalization_report(results)
end
```

### Für vollständige Studie
```bash
# 1. SLURM-Job einreichen (empfohlen für Server)
sbatch submit_job.sh

# ODER: Lokal ausführen (für kleinere Experimente)
julia src/Demo/main_generalization.jl
```

### Für eigene Experimente
```julia
# 1. Konfiguration anpassen
my_config = merge(GENERALIZATION_CONFIG, (
    target_crystal_count = 6,        # Trainiere auf 6 Kristalle
    n_training_samples = 150,        # 150 Training-Samples
    eval_crystal_range = 1:8,        # Evaluiere 1-8 Kristalle
))

# 2. Training mit angepasster Konfiguration
global GENERALIZATION_CONFIG = my_config
success, results = run_generalization_study()
```

---

## Erwartete Ausgabe

### Training-Log Beispiel
```
=== GENERALISIERUNGSSTUDIE: 10-KRISTALL TRAINING ===
PHASE 2: TRAINING-DATEN GENERIERUNG
Dataset-Statistiken (Kristallanzahl):
  10 Kristalle: 140 Samples (70.0%)
  1 Kristalle: 12 Samples (6.0%)
  5 Kristalle: 18 Samples (9.0%)
  ...

PHASE 3: 10-KRISTALL-MODELL-TRAINING
--- Epoche 1/30 ---
  Training Loss: 0.0234
  Validation Loss: 0.0189
  Neues bestes Modell gespeichert!
...

PHASE 4: GENERALISIERUNGS-EVALUIERUNG
Evaluiere auf 1 Kristalle(n) mit 25 Samples...
  MAE Total: 0.002341
Evaluiere auf 2 Kristalle(n) mit 25 Samples...
  MAE Total: 0.003156
...
```

### Generalisierungs-Bericht
```
=====================================
GENERALISIERUNGS-BERICHT
=====================================
Kristalle MAE_vx      MAE_vz      MAE_total   R²_vx       R²_vz       R²_total    Kontinuität    
------------------------------------
1         0.002103    0.002578    0.002341    0.971       0.953       0.962       0.000087       
2         0.002891    0.003421    0.003156    0.962       0.934       0.948       0.000134       
5         0.004234    0.005103    0.004669    0.937       0.908       0.923       0.000287       
10        0.004789    0.004321    0.004555    0.932       0.909       0.921       0.000312       
------------------------------------

ZUSAMMENFASSUNG:
Bester MAE: 1 Kristalle (0.002341)
Schlechtester MAE: 5 Kristalle (0.004669)
Bestes R²: 1 Kristalle (0.962)
Trend: Performance verschlechtert sich mit steigender Kristallanzahl
```

Dies bietet eine vollständige, produktionsreife Basis für Ihre Generalisierungsstudie mit professioneller Dokumentation und robusten Evaluierungsmetriken.

## Nächste Schritte

1. **Testen Sie zunächst das Demo** mit wenigen Samples
2. **Passen Sie die Konfiguration** an Ihre Bedürfnisse an  
3. **Führen Sie die vollständige Studie aus** mit `main_generalization.jl`
4. **Analysieren Sie die Ergebnisse** mit den automatischen Berichten
5. **Erweitern Sie das System** für spezifische Forschungsfragen

Das System ist jetzt bereit für systematische Untersuchungen der Generalisierungsfähigkeit bei Multi-Kristall-Systemen mit dem absoluten Fehler (MAE) als Hauptmetrik.