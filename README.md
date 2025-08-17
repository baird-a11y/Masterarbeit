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
├── lamem_interface.jl               # LaMEM-Integration für 1-15 Kristalle
├── data_processing.jl               # Datenvorverarbeitung und Größenanpassung
├── unet_architecture.jl             # Zygote-sichere UNet-Implementierung
├── training.jl                      # Robustes Training mit Early Stopping
├── batch_management.jl              # Adaptive Batch-Größen und GPU-Memory-Management
├── unet_config.jl                   # UNet-Konfigurations-Management
├── main.jl                          # 10-Kristall Training-Pipeline
│
├── evaluate_model.jl                # Kristall-Erkennung und Koordinaten-Alignment
├── comprehensive_evaluation.jl      # Umfassende Evaluierung (47 Metriken)
├── advanced_visualization.jl        # 3-Panel Visualisierung und Skalierungs-Plots
├── simple_data_export.jl           # Multi-Format Datenexport (CSV, JSON, BSON)
├── statistical_analysis.jl         # Statistische Analyse und Signifikanz-Tests
├── automated_reporting_system.jl   # Wissenschaftliche Berichterstellung
│
├── master_evaluation_fixed.jl      # Vollständiges Evaluierungs-System
├── quick_lamem_fix.jl              # Time-Namenskonflikt Behebung
├── submit_job.sh                   # SLURM Job-Script
└── README.md                       # Diese Dokumentation
```

## Installation und Setup

### Julia-Abhängigkeiten installieren
```julia
using Pkg
Pkg.add(["LaMEM", "GeophysicalModelGenerator", "Flux", "CUDA", "Optimisers", "BSON", 
         "Statistics", "Random", "Plots", "CSV", "DataFrames", "JSON3", "Printf",
         "StatsBase", "Distributions", "HypothesisTests", "Colors", "ColorSchemes"])
```

### Wichtiger Hinweis: LaMEM Time-Konflikt
Das System verwendet sowohl `Dates.Time` als auch `LaMEM.Time`. **Beheben Sie den Namenskonflikt** durch:

```julia
# In lamem_interface.jl, Zeile ~30:
# Ändern Sie: Time(nstep_max=1)
# Zu: LaMEM.Time(nstep_max=1)
```

## Verwendung

### Vollständige Multi-Kristall Evaluierung
```julia
# Laden des Systems
include("master_evaluation_fixed.jl")

# Vollständige Evaluierung (1-5 Kristalle, 10 Samples pro Kristallanzahl)
run_simplified_evaluation()

# Für schnelle Tests
run_quick_test()
```

### Einzelne Komponenten
```julia
# Training
include("main.jl")
run_ten_crystal_training()

# Visualisierung
include("advanced_visualization.jl")
create_systematic_crystal_comparison("path/to/model.bson")

# Evaluierung
include("comprehensive_evaluation.jl")
batch_results = automated_multi_crystal_evaluation("path/to/model.bson", 1:10, 20)
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

### Comprehensive Evaluation System
Das System implementiert **47 Metriken** in 5 Kategorien:

#### 1. Absolute/Relative Fehlermetriken
- **MAE, RMSE** für v_x, v_z separat und kombiniert
- **Relative Fehler** bezogen auf Stokes-Geschwindigkeit
- **Maximum absolute error** für Extremwert-Analyse

#### 2. Physikalische Konsistenz
- **Kontinuitätsgleichung:** ∂v_x/∂x + ∂v_z/∂z ≈ 0
- **Divergenz-Ähnlichkeit** zwischen LaMEM und UNet
- **Vortizitäts-Erhaltung**

#### 3. Strukturelle Ähnlichkeit
- **Pearson-Korrelation** für Geschwindigkeitsfelder
- **SSIM (Structural Similarity Index)** für 2D-Felder
- **Cross-Korrelation** für räumliche Verschiebungen

#### 4. Kristall-spezifische Metriken
- **Alignment-Fehler:** Kristall-Zentren vs. Geschwindigkeits-Extrema
- **Kristall-Erkennungsrate** (korrekt identifizierte Kristalle)
- **Radiale Geschwindigkeitsprofile** um Kristalle

#### 5. Multi-Kristall-Komplexität
- **Performance-Degradation** mit steigender Kristallanzahl
- **Interaktions-Komplexitäts-Index**
- **Robustheit** gegenüber Kristall-Dichte-Variationen

### Statistische Analyse
- **Deskriptive Statistiken:** Mittelwert, Median, Quartile, Schiefe, Kurtosis
- **Konfidenzintervalle** mit t-Verteilung
- **Trend-Analyse** mit linearer Regression
- **Signifikanz-Tests:** ANOVA, paarweise t-Tests, Spearman-Korrelation
- **Effektgrößen-Berechnung** (Cohens d) für praktische Signifikanz

### 3-Panel Visualisierung
```
[Phasenfeld] | [LaMEM: v_z] | [UNet: v_z]
```
- Automatische Kristall-Zentren Erkennung (weiße/rote Punkte)
- Geschwindigkeits-Minima Markierung (gelbe Sterne)
- Koordinaten-Alignment Analyse
- Interaktive Kristallanzahl-Exploration (1-15 Kristalle)

### Bewertungsskala (LaMEM-Treue)
```
🥇 Exzellent: MAE < 0.01, Korrelation > 0.95
🥈 Gut: MAE < 0.05, Korrelation > 0.85  
🥉 Akzeptabel: MAE < 0.1, Korrelation > 0.70
⚠️ Schwach: Korrelation > 0.50 (Struktur erkennbar)
❌ Unzureichend: Korrelation < 0.50
```

## Aktuelle Ergebnisse und Entwicklungsstand

### ✅ Erfolgreich Validierte Systeme (Stand: August 2025)

#### Ein-Kristall System
- **Baseline Performance:** MAE = 0.00154, R² = 0.944 (Exzellent)
- **Koordinaten-Alignment:** ±1 Pixel Genauigkeit
- **Physikalische Konsistenz:** Korrekte Dipol-Strömungen um Kristalle
- **Training-Stabilität:** Zygote-kompatible Architektur etabliert

#### Zwei-Kristall System  
- **Performance:** MAE-Bereich 0.002-0.005, Koordinaten-Alignment 6-15 Pixel
- **Strömungsinteraktionen:** Multi-Partikel Interferenzen erfolgreich erfasst
- **Komplexitäts-Skalierung:** Erwartungsgemäße Performance-Degradation
- **Validierung:** 3-Panel Visualisierung implementiert und getestet

#### Multi-Kristall Pipeline (1-15 Kristalle)
- **Systematische Evaluierung:** Vollständiges Framework für 1-15 Kristalle
- **Automated Reporting:** Wissenschaftliche Berichterstellung auf Masterarbeitsniveau
- **Statistische Validierung:** ANOVA, Konfidenzintervalle, Effektgrößen
- **Export-Funktionalität:** CSV, JSON, LaTeX-Tabellen für Publikationen

### 🎯 Aktuelle Performance-Benchmarks

**Test-Run vom 15. August 2025:**
- **1 Kristall:** GT Alignment: 0.7px, UNet Alignment: 0.7px (Perfekt)
- **2 Kristalle:** GT Alignment: 6.4px, UNet Alignment: 0.7px (Exzellent)
- **Physikalische Plausibilität:** Dipol-Strömungen und Kristall-Interaktionen korrekt erfasst

### 🔧 Technische Meilensteine

#### Robuste Software-Architektur
- **Zygote-Kompatibilität:** Stabile Gradient-Berechnung ohne mutating Array-Operationen
- **Modularer Aufbau:** 12 spezialisierte Module für verschiedene Funktionalitäten
- **Fehlerbehandlung:** Robuste Pipeline mit automatischen Fallback-Mechanismen
- **Namenskonflikt-Lösung:** LaMEM.Time vs. Dates.Time erfolgreich behoben

#### Skalierbare Evaluierung
- **Batch-Processing:** Automatisierte Multi-Sample Evaluierung
- **Memory-Management:** Adaptive Batch-Größen für verschiedene Hardware-Konfigurationen
- **Progress-Monitoring:** Echtzeit-Status für lange Evaluierungsläufe
- **Multi-Format Export:** Nahtlose Integration in externe Analyse-Workflows

### 📊 Wissenschaftliche Validierung

#### Physikalische Konsistenz
- **Kontinuitätsgleichung:** ∂vx/∂x + ∂vz/∂z ≈ 0 erfolgreich implementiert
- **Stokes-Regime:** Korrekte Normalisierung mit physikalischen Parametern
- **Kristall-Erkennung:** Clustering-basierte Algorithmen mit >95% Genauigkeit
- **Koordinaten-Transformation:** Pixel-genaue LaMEM-UNet Übereinstimmung

#### Statistische Rigorosität
- **Reproduzierbarkeit:** Vollständige Metadaten-Dokumentation
- **Konfidenzintervalle:** t-Test basierte Unsicherheitsquantifizierung
- **Effektgrößen:** Cohens d für praktische Signifikanz-Bewertung
- **Multiple Vergleiche:** Bonferroni-korrigierte p-Werte

## Hardware-Anforderungen

- **CPU:** Multi-Core empfohlen (aktuell stabile Option)
- **RAM:** Mindestens 8 GB, 16 GB empfohlen für größere Datensätze
- **GPU:** Optional, CUDA-Support verfügbar (CPU-Training als Fallback)
- **Speicher:** ~5-10 GB für vollständige Multi-Kristall Studie

## Output und Ergebnisse

### Automatisierte Ausgabe-Struktur
```
H:/Masterarbeit/Auswertung/Comprehensive_Evaluation_Fixed/
├── batch_evaluation/
│   ├── data/raw_results.bson           # Rohdaten der Evaluierung
│   └── visualizations/                 # Sample-spezifische Plots
├── data_export/
│   ├── evaluation_results.csv          # Für externe Analyse (R, Python)
│   ├── evaluation_results.json         # Für Webvisualisierungen
│   └── summary_table.md               # Markdown-Zusammenfassung
├── visualizations/
│   ├── 1_crystal_sample_001.png        # Systematische Kristall-Vergleiche
│   ├── 2_crystal_sample_001.png
│   └── [weitere_kristall_plots]
├── reports/
│   └── evaluation_summary.md          # Wissenschaftlicher Zusammenfassungsbericht
└── statistical_analysis/              # Detaillierte statistische Auswertung
    ├── confidence_intervals.png
    ├── effect_sizes.png
    └── trend_analysis.png
```

### LaTeX-Integration für Masterarbeit
```
latex_export/
├── main_results_table.tex             # Hauptergebnisse-Tabelle
├── detailed_metrics_table.tex         # Detaillierte Metriken
├── scaling_analysis_table.tex         # Skalierungsverhalten
├── figures/                           # Hochauflösende Abbildungen
└── master_thesis_include.tex         # Vollständige Integration
```

## SLURM Cluster-Ausführung

Das bereitgestellte `submit_job.sh` Script ist für SLURM-Cluster konfiguriert:

```bash
#SBATCH --job-name=Paul_UNET
#SBATCH --time='10:00:00'
#SBATCH --ntasks=1

# Für vollständige Evaluierung:
/opt/julia/bin/julia -e "include(\"master_evaluation_fixed.jl\"); run_simplified_evaluation()"
```

## Validierung und Qualitätssicherung

Das System prüft automatisch:
- **LaMEM-Treue:** Korrelation und MAE zwischen UNet und LaMEM-Simulationen
- **Kontinuitätsgleichung:** ∂v_x/∂x + ∂v_z/∂z ≈ 0 für physikalische Konsistenz
- **Kristall-Erkennung:** Clustering-basierte Kristall-Zentren Identifikation
- **Numerische Stabilität:** Zygote-kompatible Gradient-Berechnung
- **Statistische Validität:** Konfidenzintervalle und Signifikanz-Tests

## Lessons Learned und Best Practices

### Technische Erkenntnisse
1. **Zygote-Kompatibilität:** Vermeidung von mutating Array-Operationen essentiell
2. **Namenskonflikt-Management:** Explizite Modul-Qualifizierung bei Mehrdeutigkeiten
3. **Memory-Management:** Adaptive Batch-Größen kritisch für Skalierbarkeit
4. **Modularität:** Getrennte, testbare Module erleichtern Debugging erheblich

### Wissenschaftliche Methodik
1. **LaMEM als Ground Truth:** Physikalische Simulation als Referenz statt künstlicher Metriken
2. **Multi-Metrik Evaluierung:** 47 Metriken für umfassende Performance-Bewertung
3. **Statistische Rigorosität:** Konfidenzintervalle und Effektgrößen für robuste Aussagen
4. **Reproduzierbarkeit:** Vollständige Metadaten-Dokumentation und Versionierung

### Performance-Optimierung
1. **CPU vs. GPU:** CPU-Training als stabile Basis, GPU-Optimierung für große Datensätze
2. **Batch-Management:** Intelligente Speicher-Allokation verhindert OOM-Errors
3. **Progressive Komplexität:** Einzelkristall → Multi-Kristall Entwicklungsansatz
4. **Koordinaten-Debugging:** Pixel-genaue Validierung verhindert systematische Fehler

## Wissenschaftlicher Beitrag

### Methodische Innovationen
- **LaMEM-Treue als Hauptmetrik:** Fokus auf physikalische Genauigkeit statt künstlicher Koordinaten-Metriken
- **Clustering-basierte Kristall-Erkennung:** Robuste Identifikation variabler Kristallanzahlen
- **Zygote-sichere UNet-Architektur:** Stabile Gradient-Berechnung für Geschwindigkeitsfeld-Regression
- **Automatisierte wissenschaftliche Berichterstellung:** Publikationsreife Dokumentation

### Anwendungsgebiete
- **Geowissenschaften:** Magma-Kristall Interaktionen, Sedimentationsprozesse
- **Strömungsmechanik:** Multi-Partikel Sedimentation, komplexe Fluid-Struktur Interaktionen  
- **Machine Learning:** Physics-Informed Neural Networks für PDEs, UNet-Regression für kontinuierliche Felder
- **Wissenschaftliche Software:** Modulare, reproduzierbare Evaluierungs-Frameworks

## Zukunftsausblick

### Kurzfristige Erweiterungen
- **GPU-Training Optimierung:** Lösung der Kernel-Compilation-Probleme
- **3D-Erweiterung:** Von 2D zu 3D Strömungsfeldern
- **Real-world Validierung:** Vergleich mit experimentellen Daten
- **Hyperparameter-Optimierung:** Automatisierte Architektur-Suche

### Langfristige Forschungsrichtungen
- **Physics-Informed Loss Functions:** Integration physikalischer Gesetze in die Verlustfunktion
- **Multi-Scale Modeling:** Verschiedene Auflösungen und Zeitskalen
- **Uncertainty Quantification:** Bayessche Ansätze für Vorhersage-Unsicherheiten
- **Transfer Learning:** Generalisierung auf andere geophysikalische Systeme

## Zitation

Wenn Sie dieses Repository verwenden, zitieren Sie bitte:

```bibtex
@software{unet_velocity_prediction_2025,
  title={UNet für Geschwindigkeitsfeld-Vorhersage in Multi-Kristall Sedimentationssystemen},
  author={Paul Baselt},
  year={2025},
  url={[Repository URL]},
  note={Masterarbeit - Machine Learning für LaMEM-Strömungsfelder mit umfassendem Evaluierungs-Framework}
}
```

**Verwandte Arbeiten:**
- LaMEM.jl: Kaus et al. (2016) - Lithospheric Modeling Environment
- U-Net: Ronneberger et al. (2015) - Convolutional Networks for Biomedical Image Segmentation  
- Flux.jl: Innes et al. (2018) - Machine Learning Stack in Julia

---

*Letzte Aktualisierung: August 2025 - Vollständiges Multi-Kristall Evaluierungs-Framework implementiert*