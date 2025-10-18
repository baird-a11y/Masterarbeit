
## Kernaussage

DeepCFD verwendet CNNs (U-Net Architektur) um CFD-Simulationen für stationäre laminare Strömungen zu approximieren - **bis zu 3 Größenordnungen schneller** als klassische CFD bei niedrigen Fehlerraten.

## Hauptbeiträge

1. **Vollständige Lösung**: Liefert alle Geschwindigkeitskomponenten (Ux, Uy) UND Druckfeld (nicht nur Geschwindigkeit wie Vorgänger)
2. **U-Net mit separaten Decodern**: Bessere Performance als Autoencoder-Baseline
3. **Open Source**: Code & Datensatz auf GitHub verfügbar

## Methodik

- **Input**: Signed Distance Function (SDF) + Multi-Class Region Labels (Geometrieinformation)
- **Output**: Velocity fields (Ux, Uy) + Pressure field (p)
- **Training**: 700 Samples, 2D Kanalströmung um zufällige Hindernisse
- **Ground Truth**: OpenFOAM (simpleFoam Solver)

## Ergebnisse

### Performance

|Setup|Zeit|Speedup|
|---|---|---|
|CFD (CPU)|52.51s|-|
|DeepCFD (CPU)|0.035s|**~1500×**|
|DeepCFD (GPU, batch=100)|0.0001s|**~500,000×**|

### Genauigkeit

- **MSE Total**: 2.03 (vs. 3.29 Baseline)
- **Relative Errors**: Meiste Werte <10%
- Qualitativ sehr gute Übereinstimmung mit CFD (siehe Figures 7-9)

## Limitierungen

- Nur **2D**, nur **laminar**, nur **steady-state**
- Feste Randbedingungen
- Training auf spezifische Geometrieklassen beschränkt
- Generalisierung auf komplett andere Geometrien unklar

---

## Checkliste für die Arbeit

### Theoretischer Teil

- [ ] **DeepCFD Architektur** im Detail (U-Net, separate Decoder)
### Praktischer Teil

- [ ] **Forward Problem reproduzieren**:
    
    - Training auf Channel Flow Dataset
    - Vergleich mit Paper-Ergebnissen (Table 2, Figure 6)
- [ ] **Sensitivity Studies**:
    
    - [ ] Dateneffizienz: Training mit 100/300/700 Samples
    - [ ] Rauschen: Gaussisches Noise auf Inputs/Outputs
    - [ ] Architektur-Variationen (Anzahl Layer, Filter etc.)
- [ ] **Visualisierungen**:
    
    - [ ] Velocity/Pressure Heatmaps (wie Figure 7-9)
    - [ ] Error Maps (absolut + relativ)
    - [ ] Training Curves (MSE over Epochs)
    - [ ] Data Distribution Plots (wie Figure 10-11)
- [ ] **Erweiterungen (optional)**:
    
    - [ ] Inverse Problem: Geometrie aus Flow rekonstruieren
    - [ ] Transfer Learning: Auf neue Geometrien fine-tunen
    - [ ] Physics-informed Loss ergänzen (Navier-Stokes Residual)

### Diskussion

- [ ] **Rechenzeit-Analyse**:
    - Training Time vs. Inference Time
    - Break-even Point (ab wie vielen Simulationen lohnt sich ML?)
- [ ] **Limitierungen ehrlich benennen**:
    - Keine Turbulenz
    - Keine zeitabhängigen Flows
    - Schlechte Generalisierung außerhalb Trainingsdaten?
- [ ] **Praktische Anwendbarkeit**:
    - Wo macht DeepCFD Sinn? (Rapid Prototyping, Parameterstudien)
    - Wo NICHT? (Sicherheitskritische Anwendungen)

### Technische Details

- [ ] **Loss Functions** rechtfertigen:
    - MSE für Velocity
    - MAE für Pressure
    - Normalisierung wichtig!
- [ ] **Hyperparameter-Suche** dokumentieren:
    - Learning Rate: 1e-3
    - Kernel Size: 5
    - Filter Config: [8,16,32,32]
    - Keine Batch/Weight Norm

### Vergleiche

- [ ] **vs. Baseline (AE-3)**: Deine Ergebnisse sollten ähnlich Table 2 sein
- [ ] **vs. klassisches CFD**: Speedup reproduzieren (Table 3)
- [ ] **vs. PINNs (aus Literatur)**: Wenn Zeit - generische PINNs oft langsamer im Training

---

## Wichtige Formeln/Konzepte

**Navier-Stokes (steady, incompressible):**

```
∇·u = 0                           (Kontinuität)
ρ(u·∇)u = -∇p + μ∇²u + f        (Momentum)
```

**SDF (Signed Distance Function):**

```
SDF(x) = { +d(x,∂Ω)  if x ∈ Ω (fluid)
         { -d(x,∂Ω)  if x ∈ Ωᶜ (obstacle)
```

**Loss Function (DeepCFD):**

```
L_total = λ₁·MSE(Ux) + λ₂·MSE(Uy) + λ₃·MAE(p)
```

(λ weights für Normalisierung zwischen Variablen)

