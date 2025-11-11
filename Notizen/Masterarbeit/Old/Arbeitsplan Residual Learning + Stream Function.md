## Phase 1: Grundlagen schaffen

### 1.1 Analytische Stokes-Lösung implementieren

- Stokes-Geschwindigkeitsfeld für einzelne Kristalle berechnen
- Superposition für Multiple-Crystal-Szenarien
- Tests mit bekannten Fällen validieren

### 1.2 Kristallparameter-Extraktion

- Funktion zum Auslesen von Position, Radius, Dichte aus Phasenfeldern
- Automatische Kristallerkennung aus Phase-Files
- Datenstruktur für Crystal-Parameters definieren

### 1.3 Residuen-Analyse

- Vorberechnete Residuen zwischen LaMEM und Stokes visualisieren
- Typische Größenordnungen dokumentieren
- Räumliche Verteilung der Abweichungen analysieren

## Phase 2: Modell-Architektur entwickeln

### 2.1 Stream Function Integration

- Stream Function Layer implementieren (aus deinem bisherigen Code)
- Curl-Operator für divergenzfreies Geschwindigkeitsfeld
- Mit Residual Learning kombinieren

### 2.2 Hybrid-Architektur

```julia
# UNet → Stream Function (ψ) → Residual Velocity (Δv)
# v_total = v_stokes + curl(ψ)
```

- Input: Phasenfeld
- Output: Stream Function (statt direkt Geschwindigkeit)
- Automatische Massenerhaltung durch Konstruktion

### 2.3 Loss-Funktionen designen

- Velocity Loss (Hauptziel)
- Residual Penalty (kleine Korrekturen bevorzugen)
- Stream Function Regularisierung (Glattheit)
- Evtl. Gradienten-Loss für bessere Strukturen

## Phase 3: Daten-Pipeline

### 3.1 Trainingsdaten vorbereiten

- Phase-Fields laden und preprocessen
- Kristallparameter extrahieren
- LaMEM Ground Truth laden
- Stokes-Baseline berechnen
- Tatsächliche Residuen dokumentieren

### 3.2 Augmentation-Strategie

- Verschiedene Kristallkonfigurationen
- Skalierung/Translation (wenn möglich)
- Noise im Phasenfeld

### 3.3 Validation Split

- Test-Set mit anderen Kristallgrößen/Positionen
- Prüfung der Generalisierung

## Phase 4: Training

### 4.1 Baseline-Training

- Nur Residual Learning ohne Stream Function
- Learning Rate Schedule finden
- Erste Metriken sammeln

### 4.2 Hybrid-Training

- Residual Learning + Stream Function kombiniert
- Loss-Gewichte tunen
- Massenerhaltung überprüfen

### 4.3 Hyperparameter-Optimierung

- Netzwerk-Tiefe
- Regularisierungsgewichte
- Batch Size / Learning Rate

## Phase 5: Evaluation & Interpretation

### 5.1 Quantitative Metriken

- MSE gegen LaMEM
- Divergenz messen (sollte ~0 sein)
- Vergleich mit reinem UNet (Ansatz 1)

### 5.2 Qualitative Analyse

- Geschwindigkeitsfelder visualisieren
- Stokes vs. Residuum vs. Total separieren
- Fehleranalyse: Wo liegen die größten Abweichungen?

### 5.3 Generalisierungstest

- Neue Kristallgrößen
- Andere Positionskonfigurationen
- Extremfälle (sehr große/kleine Kristalle)

## Phase 6: Optimierungen (optional)

### 6.1 Effizienz

- Multi-Scale-Training
- Mixed Precision
- Batch-Inferenz optimieren

### 6.2 Erweiterte Physik

- Mehr Kristallparameter (Form, Orientierung)
- Temperaturabhängigkeit
- Zeitliche Evolution

### 6.3 Deployment

- Inferenz-Pipeline bauen
- Integration in bestehende Workflows
- Performance-Benchmarks

---

## Nächste konkrete Schritte

1. **Analytische Stokes-Funktion finalisieren** (hast du teilweise Code dafür?)
2. **Stream Function aus deinem Code extrahieren und testen**
3. **Erste Residuen visualisieren** (LaMEM - Stokes)
4. **Hybrid-Modell implementieren** (ResidualUNet + StreamFunction)
5. **Kleines Trainingsexperiment** auf Subset der Daten


# Benötigte Änderungen für Residual Learning + Stream Function

## Kritische Änderungen (Kern-Funktionalität)

### 1. **Neue Stokes-Lösung implementieren**

**Datei:** `src/stokes_analytical.jl` (NEU)

```julia
function compute_stokes_velocity(phase_field, crystal_params)
    # Analytische Stokes-Lösung für jeden Kristall
    # Superposition der Einzelfelder
end

function extract_crystal_params(phase_field)
    # Position, Radius, Dichte aus Phasenfeld extrahieren
end
```

**Status:**  Nicht vorhanden **Priorität:** HOCH - Kern des Ansatzes

---

### 2. **ResidualUNet Architektur erstellen**

**Datei:** `src/Testing/unet_architecture.jl` (ERWEITERN)

```julia
struct ResidualUNet
    unet::SimplifiedUNet
    use_stream_function::Bool  # Optional: Stream Function aktivieren
end

function (model::ResidualUNet)(phase_field, crystal_params)
    # 1. Stokes-Baseline berechnen
    v_stokes = compute_stokes_velocity(phase_field, crystal_params)
    
    # 2. UNet lernt Residuum (als Stream Function oder direkt)
    if model.use_stream_function
        ψ_residual = model.unet(phase_field)
        Δv = compute_velocities_from_stream(ψ_residual)
    else
        Δv = model.unet(phase_field)
    end
    
    # 3. Kombination
    v_total = v_stokes .+ Δv
    
    return v_total, v_stokes, Δv
end
```

**Status:** Nicht vorhanden **Priorität:** HOCH

---

### 3. **Stream Function Modul hinzufügen**

**Datei:** `src/stream_function.jl` (NEU)

```julia
function compute_velocities_from_stream(ψ::AbstractArray{T,4}) where T
    # ψ: [H, W, 1, batch]
    vx = ∂z(ψ)   # ∂ψ/∂z
    vz = -∂x(ψ)  # -∂ψ/∂x
    return cat(vx, vz, dims=3)
end

function ∂z(field)
    # Zentrale Differenzen (vertikal)
end

function ∂x(field)
    # Zentrale Differenzen (horizontal)
end
```

**Status:**  Teilweise in Dokumentation, nicht implementiert **Priorität:** MITTEL (wenn Stream Function gewünscht)

---

## Wichtige Änderungen (Loss & Training)

### 4. **Loss-Funktion anpassen**

**Datei:** `src/Testing/training.jl` (MODIFIZIEREN)

```julia
function loss_residual(model, phase, crystal_params, velocity_target; 
                      lambda_residual=0.01f0, lambda_sparsity=0.001f0)
    
    v_pred, v_stokes, Δv = model(phase, crystal_params)
    
    # Haupt-Loss: Gesamtgeschwindigkeit
    velocity_loss = mse(v_pred, velocity_target)
    
    # Regularisierung: Residuen klein halten
    residual_penalty = lambda_residual * mean(abs2, Δv)
    
    # Optional: Sparsity
    sparsity_loss = lambda_sparsity * mean(abs, Δv)
    
    # Optional: Divergenz-Loss (wenn keine Stream Function)
    if !model.use_stream_function
        divergence = compute_divergence(v_pred)
        physics_loss = mean(abs2, divergence)
        return velocity_loss + residual_penalty + sparsity_loss + 0.1f0 * physics_loss
    end
    
    return velocity_loss + residual_penalty + sparsity_loss
end
```

**Status:**  Existierende `physics_informed_loss` muss erweitert werden **Priorität:** HOCH

---

### 5. **Daten-Pipeline erweitern**

**Datei:** `src/Testing/lamem_interface.jl` (ERWEITERN)

```julia
function prepare_residual_training_data(phase_file, lamem_file)
    # Phasenfeld laden
    phase = load_phase_field(phase_file)
    
    # Kristall-Parameter extrahieren
    crystal_params = extract_crystal_params(phase)
    
    # LaMEM Ground Truth
    v_lamem = load_lamem_velocities(lamem_file)
    
    # Optional: Residuum vorberechnen zur Analyse
    v_stokes = compute_stokes_velocity(phase, crystal_params)
    residuum = v_lamem .- v_stokes
    
    return (phase, crystal_params, v_lamem, residuum)
end
```

**Status:**  Kristallparameter-Extraktion fehlt **Priorität:** HOCH

---

### 6. **Batch-Erstellung anpassen**

**Datei:** `src/Testing/lamem_interface.jl` (MODIFIZIEREN)

```julia
function create_residual_batch(samples, target_resolution)
    phases = []
    crystal_params_batch = []
    velocities = []
    
    for sample in samples
        phase, params, v_target = sample
        push!(phases, phase)
        push!(crystal_params_batch, params)
        push!(velocities, v_target)
    end
    
    phase_batch = cat(phases..., dims=4)
    velocity_batch = cat(velocities..., dims=4)
    
    return phase_batch, crystal_params_batch, velocity_batch
end
```

**Status:**  Existierende `create_adaptive_batch` muss angepasst werden **Priorität:** MITTEL

---

##  Optionale Verbesserungen

### 7. **UNet Output-Kanal anpassen**

**Datei:** `src/Testing/unet_architecture.jl`

**Aktuell:**

```julia
output_conv::Conv((1,1), base_filters => 2)  # 2 Kanäle (vx, vz)
```

**Für Stream Function:**

```julia
output_conv::Conv((1,1), base_filters => 1)  # 1 Kanal (ψ)
```

**Status:**  Muss konfigurierbar gemacht werden **Priorität:** NIEDRIG (nur wenn Stream Function verwendet wird)

---

### 8. **Visualisierung erweitern**

**Datei:** Neues Modul `src/visualization_residual.jl`

```julia
function visualize_residual_decomposition(phase, v_stokes, Δv, v_total, v_lamem)
    # 4-Panel Plot:
    # 1. Stokes-Baseline
    # 2. Gelerntes Residuum
    # 3. Gesamtvorhersage
    # 4. LaMEM Ground Truth
end
```

**Status:**  Nicht vorhanden **Priorität:** NIEDRIG (aber sehr nützlich für Interpretierbarkeit)

---

### 9. **Evaluation-Metriken anpassen**

**Datei:** `src/Testing/master_evaluation_fixed.jl` (ERWEITERN)

```julia
function evaluate_residual_model(model, test_data)
    # Zusätzliche Metriken:
    # - Mean Residual Magnitude: mean(abs, Δv)
    # - Stokes Contribution: mean(abs, v_stokes) / mean(abs, v_total)
    # - Residual Sparsity: count(abs.(Δv) < threshold) / total
end
```

**Status:**  Existierende Evaluation-Funktionen müssen erweitert werden **Priorität:** MITTEL

---

##  Zusammenfassung der Prioritäten

|Priorität|Änderung|Datei|Aufwand|
|---|---|---|---|
|**HOCH**|Stokes-Lösung|`stokes_analytical.jl` (NEU)|3-5 Tage|
|**HOCH**|ResidualUNet|`unet_architecture.jl`|1-2 Tage|
|**HOCH**|Loss-Funktion|`training.jl`|1 Tag|
|**HOCH**|Daten-Pipeline|`lamem_interface.jl`|2-3 Tage|
|**MITTEL**|Stream Function|`stream_function.jl` (NEU)|2 Tage|
|**MITTEL**|Batch-Erstellung|`lamem_interface.jl`|1 Tag|
|**MITTEL**|Evaluation|`master_evaluation_fixed.jl`|1-2 Tage|
|**NIEDRIG**|Visualisierung|`visualization_residual.jl` (NEU)|1-2 Tage|
|**NIEDRIG**|UNet-Konfiguration|`unet_architecture.jl`|0.5 Tage|

---

##  Empfohlene Implementierungsreihenfolge

### Phase 1: Residual Learning (ohne Stream Function)

1. Stokes-Lösung implementieren
2. ResidualUNet erstellen
3. Loss-Funktion anpassen
4. Daten-Pipeline erweitern
5. Erstes Training durchführen

### Phase 2: Stream Function hinzufügen

6. Stream Function Modul erstellen
7. ResidualUNet um Stream Function erweitern
8. Vergleichstraining durchführen

### Phase 3: Evaluation & Visualisierung

1. Evaluation-Metriken erweitern
2. Visualisierungs-Tools erstellen

Möchtest du bei einer dieser Änderungen detaillierteren Code sehen?