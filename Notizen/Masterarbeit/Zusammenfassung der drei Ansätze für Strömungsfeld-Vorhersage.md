

## 📊 Übersichtstabelle

|Aspekt|**Ansatz 1: Direkt**|**Ansatz 2: Stream Function**|**Ansatz 3: Residual**|
|---|---|---|---|
|**Output**|(vx, vz)|Stream Function a|v_stokes + Δv|
|**Massenerhaltung**|Soft Constraint|**Automatisch**|Soft Constraint*|
|**Lernaufgabe**|Schwer|Mittel|**Einfach**|
|**Physikalische Basis**|Keine|Mathematisch|**Analytisch**|
|**Implementierung**|**Einfach**|Mittel|**Einfach**|
|**Für Ihr Problem**|⚠️ Basis|✅ Gut|✅✅ **Best**|

---

## Ansatz 1: Direktes Lernen (Ihr aktueller Ansatz)

### Konzept

Das Netzwerk lernt **direkt** die vollständigen Geschwindigkeitsfelder (vx, vz) aus dem Phasenfeld.

### Architektur

```julia
struct SimplifiedUNet
    # Encoder: 3 Stufen
    enc1_conv1::Conv; enc1_conv2::Conv; enc1_pool::MaxPool
    enc2_conv1::Conv; enc2_conv2::Conv; enc2_pool::MaxPool
    enc3_conv1::Conv; enc3_conv2::Conv; enc3_pool::MaxPool
    
    # Bottleneck
    bottleneck_conv1::Conv; bottleneck_conv2::Conv
    
    # Decoder: 3 Stufen
    dec3_up::ConvTranspose; dec3_conv1::Conv; dec3_conv2::Conv
    dec2_up::ConvTranspose; dec2_conv1::Conv; dec2_conv2::Conv
    dec1_up::ConvTranspose; dec1_conv1::Conv; dec1_conv2::Conv
    
    # Output: 2 Kanäle für vx, vz
    output_conv::Conv  # Output: 2 channels
end
```

### Schritte

**1. Daten vorbereiten**

```julia
# Input: Phasenfeld (0=Matrix, 1=Kristall)
phase_field = load_phase_field(file)  # [H, W, 1, batch]

# Target: LaMEM Geschwindigkeiten
vx, vz = load_lamem_velocities(file)
velocity_target = cat(vx, vz, dims=3)  # [H, W, 2, batch]
```

**2. Training**

```julia
function loss_direct(model, phase, velocity_target)
    # Direkte Vorhersage
    velocity_pred = model(phase)
    
    # MSE Loss
    velocity_loss = Flux.mse(velocity_pred, velocity_target)
    
    # Optional: Massenerhaltung als Soft Constraint
    vx_pred = velocity_pred[:,:,1,:]
    vz_pred = velocity_pred[:,:,2,:]
    div_v = ∂x(vx_pred) + ∂z(vz_pred)
    mass_penalty = mean(abs2, div_v)
    
    return velocity_loss + 0.1f0 * mass_penalty
end
```

**3. Inferenz**

```julia
phase_new = load_new_phase_field()
velocity_pred = model(phase_new)
vx_pred = velocity_pred[:,:,1,:]
vz_pred = velocity_pred[:,:,2,:]
```

### ✅ Vorteile

- Einfachste Implementierung
- Standardisierte UNet-Architektur
- Direkter Vergleich mit Ground Truth

### ❌ Nachteile

- Massenerhaltung nur näherungsweise
- Lernt große absolute Werte (schwierig)
- Schlechte Generalisierung auf neue Kristallgrößen

---

## Ansatz 2: Stream Function (Agarwal et al.)

### Konzept

Das Netzwerk lernt eine **Stream Function** a(x,z), aus der die Geschwindigkeiten als Ableitungen berechnet werden:

- vx = ∂a/∂z
- vz = -∂a/∂x

Dies garantiert **automatisch** ∇·v = 0 (Massenerhaltung).

### Architektur

```julia
struct StreamFunctionUNet
    # Identisch wie Ansatz 1, ABER:
    output_conv::Conv  # Output: 1 channel für Stream Function
end
```

### Schritte

**1. Geschwindigkeitsberechnung implementieren**

```julia
function compute_velocities_from_stream(a::AbstractArray{T,4}) where T
    # a: [H, W, 1, batch]
    
    # Zentrale Differenzen
    function ∂z(field)  # Vertikale Ableitung
        padded = cat(field[end:end,:,:,:], field, 
                     field[1:1,:,:,:], dims=1)
        return (padded[3:end,:,:,:] - padded[1:end-2,:,:,:]) / 2.0f0
    end
    
    function ∂x(field)  # Horizontale Ableitung
        padded = cat(field[:,end:end,:,:], field, 
                     field[:,1:1,:,:], dims=2)
        return (padded[:,3:end,:,:] - padded[:,1:end-2,:,:]) / 2.0f0
    end
    
    # Geschwindigkeiten
    vx = ∂z(a)
    vz = -∂x(a)
    
    return cat(vx, vz, dims=3)  # [H, W, 2, batch]
end
```

**2. Forward Pass ändern**

```julia
function (model::StreamFunctionUNet)(x)
    # Encoder → Bottleneck → Decoder (wie vorher)
    # ...
    
    # Stream Function Output
    a = model.output_conv(dec1)  # [H, W, 1, batch]
    
    # Geschwindigkeiten berechnen
    velocities = compute_velocities_from_stream(a)
    
    return velocities  # [H, W, 2, batch]
end
```

**3. Training**

```julia
function loss_stream(model, phase, velocity_target)
    # Vorhersage (intern: a → v)
    velocity_pred = model(phase)
    
    # MSE Loss (Massenerhaltung automatisch erfüllt!)
    velocity_loss = Flux.mse(velocity_pred, velocity_target)
    
    # Optional: Stream Function regularisieren
    # (kein Zugriff auf 'a' nach compute_velocities)
    
    return velocity_loss
end
```

**4. Inferenz**

```julia
phase_new = load_new_phase_field()
velocity_pred = model(phase_new)  # Automatisch ∇·v = 0!
```

### ✅ Vorteile

- **Exakte Massenerhaltung** (mathematisch garantiert)
- Elegante physikalische Formulierung
- Bewährt in Mantelkonvektions-Simulationen

### ❌ Nachteile

- Komplexere Implementierung
- Ableitungsberechnung erfordert sorgfältiges Boundary-Handling
- Stream Function nicht direkt interpretierbar
- Debugging schwieriger

---

## Ansatz 3: Residual Learning (Empfohlen für Sie!)

### Konzept

Das Netzwerk lernt **nicht** die absolute Geschwindigkeit, sondern nur die **Abweichung** von der analytischen Stokes-Lösung:

**v_total = v_stokes(analytisch) + Δv(gelernt)**

### Architektur

```julia
# Gleiche UNet-Struktur wie Ansatz 1
# ABER: Lernt nur kleine Residuen statt großer absoluter Werte
```

### Schritte

**1. Analytische Stokes-Lösung implementieren**

```julia
function compute_stokes_velocity(phase_field, crystal_params)
    # crystal_params: [position, radius, density] für jeden Kristall
    
    vx_stokes = zeros(Float32, size(phase_field)[1:2])
    vz_stokes = zeros(Float32, size(phase_field)[1:2])
    
    # Für jeden Kristall: analytische Lösung
    for crystal in crystal_params
        vx_single, vz_single = analytical_stokes_field(
            crystal.x, crystal.z,   # Position
            crystal.r,              # Radius
            crystal.ρ               # Dichte
        )
        
        # Superposition der Einzelfelder
        vx_stokes .+= vx_single
        vz_stokes .+= vz_single
    end
    
    return cat(vx_stokes, vz_stokes, dims=3)
end

function analytical_stokes_field(x_crystal, z_crystal, radius, density)
    # Stokes-Lösung für kugelsymmetrischen Kristall
    # (Vereinfachte 2D-Projektion)
    
    # Für jeden Punkt (x,z) im Gitter:
    # - Abstand zum Kristall berechnen
    # - Stokes-Geschwindigkeitsfeld auswerten
    
    # Details siehe Geophysik-Literatur oder Ihre LaMEM-Implementierung
end
```

**2. Forward Pass mit Residuum**

```julia
struct ResidualUNet
    unet::SimplifiedUNet
end

function (model::ResidualUNet)(phase_field, crystal_params)
    # 1. Analytische Basis
    v_stokes = compute_stokes_velocity(phase_field, crystal_params)
    
    # 2. Gelerntes Residuum
    Δv = model.unet(phase_field)
    
    # 3. Gesamtgeschwindigkeit
    v_total = v_stokes .+ Δv
    
    return v_total, v_stokes, Δv
end
```

**3. Training**

```julia
function loss_residual(model, phase, crystal_params, velocity_target)
    v_pred, v_stokes, Δv = model(phase, crystal_params)
    
    # Haupt-Loss: Gesamtgeschwindigkeit
    velocity_loss = Flux.mse(v_pred, velocity_target)
    
    # Regularisierung: Residuen klein halten
    residual_penalty = 0.01f0 * mean(abs2, Δv)
    
    # Optional: Sparsity (viele Nullen in Δv)
    sparsity_loss = 0.001f0 * mean(abs, Δv)
    
    return velocity_loss + residual_penalty + sparsity_loss
end
```

**4. Daten vorbereiten**

```julia
function prepare_residual_training_data(phase_file, lamem_file)
    # Phasenfeld
    phase = load_phase_field(phase_file)
    
    # Kristall-Parameter extrahieren
    crystal_params = extract_crystal_params(phase_file)
    # → [position, radius, density] für jeden Kristall
    
    # LaMEM Ground Truth
    v_lamem = load_lamem_velocities(lamem_file)
    
    # Optional: Residuum vorberechnen zur Visualisierung
    v_stokes = compute_stokes_velocity(phase, crystal_params)
    residuum = v_lamem - v_stokes
    
    return (phase, crystal_params, v_lamem, residuum)
end
```

**5. Inferenz**

```julia
phase_new = load_new_phase_field()
crystal_params_new = extract_crystal_params(phase_new)

v_pred, v_stokes, Δv = model(phase_new, crystal_params_new)

println("Stokes-Beitrag: $(mean(abs, v_stokes))")
println("Gelerntes Residuum: $(mean(abs, Δv))")
println("Gesamt: $(mean(abs, v_pred))")
```

### ✅ Vorteile (für Ihr Problem!)

- **Einfachere Lernaufgabe**: Kleine Residuen statt große absolute Werte
- **Physikalische Basis**: 90% der Physik bereits eingebaut
- **Beste Generalisierung**: Stokes-Lösung skaliert automatisch mit Kristallgröße
- **Interpretierbar**: Kann Stokes vs. Interaktionseffekte separieren
- **Einfache Implementierung**: Nur analytische Funktion hinzufügen

### ❌ Nachteile

- Benötigt analytische Stokes-Lösung (aber Sie haben diese bereits!)
- Massenerhaltung weiterhin Soft Constraint (kann mit Stream Function kombiniert werden)

---

## 🎯 Empfehlung für Ihre Masterarbeit

### Phase 1: Residual Learning (2-3 Wochen)

1. Implementieren Sie `compute_stokes_velocity()`
2. Modifizieren Sie Ihr bestehendes UNet minimal
3. Trainieren und vergleichen mit Ansatz 1
4. **Erwartung**: 50-70% bessere Generalisierung auf 2-15 Kristalle

### Phase 2 (Optional): Hybrid-Ansatz (2-3 Wochen)

Kombinieren Sie Residual + Stream Function:

```julia
# Beste aller Welten:
v_total = v_stokes(analytisch) + compute_velocities_from_stream(a_residual)
#          ↑                      ↑
#     Physikalische Basis    Gelernte Interaktionen mit ∇·v=0
```

### Warum Residual für Sie optimal ist:

1. ✅ **Minimal-invasiv**: Nutzt Ihr bestehendes UNet
2. ✅ **Schnelle Iteration**: Keine komplexe neue Architektur
3. ✅ **Starke Story**: "Von Physik inspiriert, durch ML verfeinert"
4. ✅ **Skalierbar**: Funktioniert für 1-15 Kristalle
5. ✅ **Interpretierbar**: Ideal für wissenschaftliche Publikation

Der reine Stream Function Ansatz wäre **akademisch elegant**, aber der Residual-Ansatz ist **praktisch überlegen** für Kristallsedimentation.