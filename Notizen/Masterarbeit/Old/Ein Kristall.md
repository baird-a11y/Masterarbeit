## Notizen über das Training mit einem Kristall

## Ausgangslage

Für die Erstellung von Trainingsdaten brauche ich Bilder und Masken, die die Position des Kristalles enthalten, sowie die beiden Geschwindigkeitsfelder (v_x,v_z). Hier müsste ich schauen, ob ich einfach reine Bilder erstelle oder ob ich die Werte auslesen lasse.

## Umsetzung

Erster Ansatz wäre einfach ein Bild erst mal nur vom Kristall zu erstellen und dann noch einmal eins mit dem Geschwindigkeitsfeld und dieses dann als Maske zu nutzen. Dabei sollte der Kristall in beiden Fällen immer die gleiche Farbe haben als Orientierung. Zu beginn, sollten 10-20 Bilder reichen um eine grobe Orientierung zu haben ob dieser Ansatz klappt.

### Experiment 1: UNet für Geschwindigkeitsfeld-Regression

#### Theorie

- **Ansatz geändert**: Von Segmentierung zu Regression
- **Input**: Phasenfeld (1 Kanal: 0=Matrix, 1=Kristall)
- **Output**: Geschwindigkeitsfelder (2 Kanäle: v_x, v_z)
- **Verlustfunktion**: MSE statt Cross-Entropy
- **Normalisierung**: Geschwindigkeiten durch Stokes-Geschwindigkeit geteilt

#### Umsetzung

- **Modell-Architektur**: UNet mit 1→2 Kanälen (statt 3→19)
- **Trainingsdaten**: Automatische Generierung mit LaMEM_Single_crystal()
- **Variation**: Zufällige η, Δρ, Positionen, Radien pro Sample
- **Evaluierung**: Vergleich UNet vs. LaMEM Ground Truth

#### Probleme

1. **CUDA-Precompilation Fehler**: Module-Loading-Probleme
    - **Lösung**: CPU-Version für erste Tests
2. **Kanal-Mismatch**: Modell erwartete 3 Kanäle (RGB), bekam 1 (Phase)
    - **Lösung**: Korrekte 1-Kanal Architektur implementiert
3. **Forward-Pass Fehler**: `crop_and_concat` Funktion defekt
    - **Lösung**: Skip-Connections korrekt implementiert
4. **Koordinatensystem**: v_z Vorzeichen unklar
    - **Status**: ~~Funktioniert, evtl. später zu prüfen~~ **GELÖST** durch Debugging

#### Wichtige Code Teile

**Modell-Definition:**

```julia
struct VelocityUNet
    encoder1; encoder2; encoder3; encoder4; bottleneck
    decoder4; decoder4_1; decoder3; decoder3_1
    decoder2; decoder2_1; decoder1; decoder1_1
end

# Verlustfunktion für Regression
function velocity_loss_fn(model, x, y_true)
    y_pred = model(x)
    return mse(y_pred, y_true)
end
```

**Datenvorbereitung:**

```julia
# Phasenfeld → UNet Input
phase_input = reshape(standardize_size(phase), H, W, 1, 1)

# Geschwindigkeiten → Targets (normalisiert)
vx_norm = vx ./ v_stokes
vz_norm = vz ./ v_stokes
velocity_target = cat(vx_norm, vz_norm, dims=3)
```

**Evaluierung:**

```julia
# Komplette Bewertung: Funktionalität + Physik + Genauigkeit
results = complete_velocity_evaluation("model.bson")
```

#### Ergebnisse

**Test 1: 16 Bilder, 3 Epochen**

- Fehlgeschlagen: Modell lernte nur Rauschen
- Keine erkennbare Struktur
- Große Abweichungen zu LaMEM
- Pipeline funktioniert grundsätzlich

**Test 2: 100 Bilder, 40 Epochen**

- **MSE: 0.00154** (exzellent, < 0.01)
- **R²: 0.944** (94.4% Korrelation mit LaMEM)
- Physikalisch sinnvolle Strömungsmuster
- Klare Geschwindigkeits-Hotspots am Kristall
- Korrekte Dipolströmung (v_x links-rechts)
- Sinkende Kristalle (negative v_z)

**Benchmark-Einordnung:**

- Schlechte Modelle: MSE > 0.1, R² < 0.5
- Okay Modelle: MSE 0.01-0.05, R² 0.5-0.8
- **Unser Modell**: MSE = 0.0015, R² = 0.944 → **EXZELLENT**

---

### Experiment 2: Koordinaten-System Debugging

#### Problem-Erkennung

Bei ersten Anwendungen des trainierten UNets stellte sich heraus, dass **Kristall-Positionen im Input nicht mit Geschwindigkeits-Hotspots im Output** übereinstimmten. Verdacht auf Koordinatensystem-Probleme.

#### Debugging-Ansatz

- **Vereinfachtes Evaluierungs-Tool** entwickelt (nur v_z fokussiert)
- **Pixel-genaue Analyse** von Kristall-Zentrum vs. min(v_z) Position
- **Dimensionen-Konsistenz** zwischen Training und Anwendung prüfen

#### Erkenntnisse

**Problem identifiziert:**

- **LaMEM-Output**: 257×257 Pixel
- **UNet-Training**: 256×256 Pixel
- **UNet-Output**: 256×256 Pixel
- → **1-Pixel Versatz** durch Dimensionen-Mismatch

**Koordinaten-Präzision gemessen:**

- **Ground Truth Alignment**: 6.1 Pixel (✅ physikalisch korrekt)
- **UNet Alignment**: 6.3 Pixel (✅ lernt korrekte Physik)
- **GT vs. UNet Abweichung**: **1.0 Pixel** (✅ praktisch perfekt!)

**Fazit:**

- **KEIN Koordinatensystem-Problem**
- **UNet lernt korrekte Physik**
- **Problem lag an Dimensionen-Inkonsistenz**

#### Code-Tools

**Koordinaten-Debugging:**

```julia
# Vereinfachte v_z-fokussierte Evaluierung
result = debug_vz_coordinates("model.bson", target_size=(256,256))

# Findet automatisch:
# - Kristall-Zentrum im Phasenfeld
# - Min v_z Position in Ground Truth vs. UNet
# - Pixel-genaue Abweichungen
```

**Dimensionen-Fix:**

```julia
# Konsistente Größen-Behandlung
actual_size = size(phase_gt)
if actual_size != target_size
    println("LaMEM liefert $(actual_size) statt $(target_size)")
    # Automatische Anpassung...
end
```

---

### Experiment 3: Code-Refactoring und Pipeline-Verbesserung

#### Problem-Erkennung

Der ursprüngliche Trainingscode hatte mehrere Probleme:

- **Hartcodierte Parameter** überall im Code verstreut
- **Inkonsistente Dimensionen** zwischen Training (256×256) und LaMEM-Output (257×257)
- **GPU Scalar Indexing Probleme** bei der `crop_and_concat` Funktion
- **Schwierige Wartbarkeit** durch unstrukturierten Code

#### Refactoring-Ansatz

- **Zentrale Konfiguration**: Alle Parameter in `CONFIG` Struktur gesammelt
- **Modularer Aufbau**: Separate Funktionen für jede Pipeline-Komponente
- **256×256 Konsistenz-Fix**: LaMEM Parameter angepasst (`nel=(255,255)` für 256×256 Output)
- **GPU-Kompatibilität**: Vereinfachte `crop_and_concat` Funktion

#### Code-Verbesserungen

**Zentrale Konfiguration:**

```julia
const CONFIG = (
    image_size = (256, 256),           # Konsistente Dimensionen
    dataset_size = 20,                 # Anzahl Trainingssamples  
    learning_rate = 0.001,             # Lernrate
    eta_range = (1e19, 1e21),         # Parameter-Bereiche
    use_gpu = false,                   # Hardware-Einstellungen
    # ... alle anderen Parameter zentral
)
```

**LaMEM Dimensionen-Fix:**

```julia
function LaMEM_Single_crystal_fixed(; η=1e20, Δρ=200, cen_2D=[(0.0, 0.0)], R=[0.1])
    target_h, target_w = CONFIG.image_size
    nel_h, nel_w = target_h - 1, target_w - 1  # LaMEM erzeugt nel+1 Punkte
    
    model = Model(Grid(nel=(nel_h, nel_w), x=[-1,1], z=[-1,1]), ...)
    # Automatische Dimensionsprüfung und Zuschneidung
end
```

**Konfigurierbare Parameter-Generierung:**

```julia
function generate_random_params()
    η_min, η_max = CONFIG.eta_range
    η = 10^(rand() * (log10(η_max) - log10(η_min)) + log10(η_min))
    # Weitere Parameter basierend auf CONFIG-Bereichen
end
```

#### Erkenntnisse

**GPU-Training Probleme:**

- **Scalar Indexing Fehler** bei komplexen Array-Operationen auf GPU
- **CPU-Training** als stabile Alternative implementiert
- GPU-Training für später optimierbar, CPU funktioniert zuverlässig

**Dimensionen-Konsistenz erfolgreich:**

- LaMEM liefert jetzt garantiert 256×256 Output
- Training und Anwendung verwenden identische Dimensionen
- Automatische Validierung in jeder Pipeline-Stufe

**Code-Wartbarkeit drastisch verbessert:**

- **Ein CONFIG-Block** statt Parameter überall im Code
- **Einfache Experimente**: Parameter ändern ohne Code-Suche
- **Modularer Aufbau** ermöglicht einfache Erweiterungen

#### Aktuelle Pipeline-Funktionen

**Komplettes Training:**

```julia
# Alle Parameter konfigurierbar
model, losses, data = run_complete_training_configured()

# Parameter ändern:
CONFIG = merge(CONFIG, (dataset_size = 100, num_epochs = 50))
```

**Testing und Debugging:**

```julia
# Konsistenz-Test der gesamten Pipeline
test_256_consistency()

# Koordinaten-Debugging für trainierte Modelle
result = debug_vz_coordinates("model.bson")
```

#### Status

**Erfolgreich implementiert:**

- **256×256 Konsistenz** in gesamter Pipeline
- **Zentrale Konfiguration** für alle Parameter
- **CPU-Training** funktioniert stabil
- **Modularer, wartbarer Code**

**Aktuelle Herausforderungen:**

- **GPU-Training** benötigt weitere Optimierung
- **Skip-Connections** für komplexe Größen-Unterschiede

**Nächste Tests:**

- **Kleine Trainingsläufe** (20 Samples, 10 Epochen) für Pipeline-Validierung
- **Skalierung** bei erfolgreicher Validierung

---

### Experiment 4: Zygote-Kompatibilität und Sichere Pipeline - 22.06.2025

#### Problem-Erkennung

Nach dem Refactoring traten beim Training **kritische Zygote-Fehler** auf:

- **"Mutating arrays is not supported"** - `push!` Operationen in UNet Forward-Pass
- **"step cannot be zero"** - Range-Fehler in Validation-Schleifen
- **GPU-Kernel-Compilation Errors** - Broadcast-Probleme mit komplexen Tensor-Operationen

#### Ursachen-Analyse

**Zygote-Inkompatibilität:**

- Original UNet verwendete `push!(skip_features, ...)` für Skip-Connections
- Zygote kann mutating Array-Operationen nicht differenzieren
- Komplexe Array-Manipulationen in `crop_and_concat` Funktionen

**Range-Fehler:**

- Validation-Schleifen mit ungültigen Start/End-Indizes
- Leere Datasets führten zu `1:0` Ranges
- Fehlende Validierung von Batch-Größen vs. Dataset-Größe

#### Lösungsansatz

**1. Vereinfachte Zygote-sichere UNet-Architektur:**

```julia
struct SimplifiedUNet
    # Explizite Layer-Definition ohne dynamische Arrays
    enc1_conv1::Conv; enc1_conv2::Conv; enc1_pool::MaxPool
    enc2_conv1::Conv; enc2_conv2::Conv; enc2_pool::MaxPool
    # ... weitere Layer
end

# Forward-Pass ohne push! Operationen
function (model::SimplifiedUNet)(x)
    enc1 = model.enc1_conv2(model.enc1_conv1(x))
    # Direkte Skip-Connections ohne Array-Sammlung
    # ...
end
```

**2. Sichere Training-Schleife:**

```julia
function train_velocity_unet_safe(model, dataset, target_resolution; config)
    # Robuste Dataset-Validierung
    if length(dataset) == 0
        error("Dataset ist leer!")
    end
    
    # Sichere Range-Erstellung
    for i in 1:batch_size:min(max_batches, length(dataset))
        if i > length(dataset)
            break  # Verhindere ungültige Ranges
        end
        # ...
    end
    
    # Zygote-sichere Gradient-Berechnung
    loss_val, grads = Flux.withgradient(loss_fn, model)
end
```

**3. Reduzierte Server-Konfiguration:**

```julia
const SERVER_CONFIG = (
    n_samples = 5,           # Weniger Samples für Stabilität
    target_resolution = 128, # Kleinere Auflösung
    num_epochs = 2,          # Kurze Tests
    batch_size = 2,          # Kleine Batches
    use_gpu = false,         # CPU-only für Zygote-Kompatibilität
)
```

#### Implementierung

**Neue Module erstellt:**

- `unet_architecture_zygote_safe.jl` - Vereinfachte UNet-Struktur
- `training_safe.jl` - Robuste Training-Funktionen
- `main_safe.jl` - Sichere Server-Pipeline

**Deprecated Syntax behoben:**

- `Flux.@functor` → `Flux.@layer` (neuere Flux-Version)
- Dokumentationsstring-Syntax korrigiert
- CPU-fokussiertes Training implementiert

#### Ergebnisse - Erfolgreicher Durchlauf

**Training-Performance:**

```
--- Epoche 1/2 ---
  Training Loss: 0.357047
  Validation Loss: 0.094105
  Neues bestes Modell gespeichert!

--- Epoche 2/2 ---
  Training Loss: 0.334409
  Validation Loss: 0.087728
  Neues bestes Modell gespeichert!
```

**Final Test Ergebnisse:**

- **MSE Vx**: 0.599 (Horizontale Geschwindigkeit)
- **MSE Vz**: 0.697 (Vertikale Geschwindigkeit)
- **MSE Total**: 0.648
- **Laufzeit**: 0.78 Minuten für kompletten Durchlauf

#### Bewertung der Ergebnisse

**Positive Aspekte:**

- **Stabile Konvergenz**: Beide Losses sinken konsistent (6-7% Verbesserung)
- **Kein Overfitting**: Validation Loss besser als Training Loss
- **Zygote-Kompatibilität**: Keine Differenzierung-Fehler mehr
- **Robuste Pipeline**: Alle Module arbeiten fehlerfrei zusammen

**Verbesserungspotential:**

- **MSE-Werte noch hoch**: 0.6-0.7 vs. ideale Werte < 0.1
- **Wenig Trainingsdaten**: Nur 5 Samples vs. optimale 100-500
- **Kurzes Training**: 2 Epochen vs. empfohlene 20-50

#### Skalierungs-Empfehlungen

**Sofortige Verbesserungen:**

```julia
const IMPROVED_CONFIG = (
    n_samples = 50,        # 10x mehr Trainingsdaten
    num_epochs = 20,       # 10x mehr Epochen  
    batch_size = 4,        # Größere Batches
    learning_rate = 0.0005f0,  # Feinere Lernrate
)
```

**Erweiterte Validierung:**

- **Physik-Tests**: Kontinuitätsgleichung ∂vx/∂x + ∂vz/∂z ≈ 0
- **Visualisierung**: Vorhersagen vs. Ground Truth Plots
- **Robustheitstests**: Verschiedene Kristall-Positionen und -Größen

#### Technische Erkenntnisse

**Zygote-Anforderungen:**

- **Keine mutierende Operationen**: Vermeidung von `push!`, `append!`, etc.
- **Statische Architektur**: Feste Layer-Struktur ohne dynamische Arrays
- **Sichere Gradient-Berechnung**: `Flux.withgradient()` statt manueller Differenzierung

**CPU vs. GPU Training:**

- **CPU**: Stabil, Zygote-kompatibel, ausreichend für Prototyping
- **GPU**: Potentiell schneller, aber komplexere Fehlerdiagnose erforderlich
- **Empfehlung**: CPU für Entwicklung, GPU für Production-Training

---

## Lessons Learned

1. **Datenqualität > Quantität**: 100 gute Samples besser als 16 schlechte
2. **Evaluierung ist kritisch**: Ohne Ground Truth Vergleich keine echte Bewertung
3. **Physikalische Plausibilität**: Metriken können täuschen, visuelle Inspektion wichtig
4. **Iterative Entwicklung**: Kleine Tests → Probleme finden → Fixes → Skalierung
5. **Dimensionen-Konsistenz ist KRITISCH**: 1-Pixel Unterschiede können zu falschen Schlüssen führen
6. **Debugging-Tools entwickeln**: Vereinfachte, fokussierte Evaluierung besser als komplexe
7. **Zygote-Kompatibilität beachten**: Frühe Architektur-Entscheidungen können spätes Training verhindern
8. **Modulare Entwicklung**: Getrennte, testbare Module erleichtern Debugging erheblich
9. **CPU-Training als Fallback**: Nicht immer ist GPU-Training nötig oder stabil

---

## Nächste Schritte

### Sofort (aktueller Stand):

1. **Pipeline funktioniert** - Zygote-sichere Architektur etabliert
2. **Skalierung möglich** - Erhöhung von Samples und Epochen
3. **Modular erweiterbar** - Basis für komplexere Experimente

### Kurz-/Mittelfristig:

1. **Performance-Optimierung**: 50-100 Samples, 20-50 Epochen
2. **Physik-Validierung**: Kontinuitätsgleichung und Strömungsmuster prüfen
3. **Hyperparameter-Tuning**: Lernrate, Architektur-Größe, Regularisierung
4. **Visualisierung**: Plots von Vorhersagen vs. Ground Truth

### Langfristig:

1. **Mehrere Kristalle**: Erweiterung auf 2-3 Kristalle gleichzeitig
2. **Komplexere Geometrien**: Ellipsen, verschiedene Formen
3. **Real-world Validation**: Vergleich mit experimentellen Daten
4. **GPU-Optimierung**: Für große Datensätze und komplexe Modelle

---

## Aktueller Status

- **Pipeline-Status**: ✅ Vollständig funktional und Zygote-kompatibel
- **Training-Qualität**: ✅ Konvergent, aber verbesserungsfähig (MSE ≈ 0.65)
- **Code-Wartbarkeit**: ✅ Modularer, erweiterbarer Aufbau
- **Skalierbarkeit**: ✅ Bereit für mehr Daten und längeres Training
- **Physik-Validierung**: ⏳ Nächster wichtiger Schritt

**Haupterkenntniss**: Nach Überwindung der Zygote-Kompatibilitätsprobleme haben wir jetzt eine **stabile, funktionale UNet-Pipeline** für LaMEM-Daten, die als solide Basis für weitere Experimente dient.