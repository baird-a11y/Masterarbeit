_(Baseline-Modell für Vergleich mit ψ-Ansatz und Residual-Ansatz)_

Bei diesem Ansatz wird das **Geschwindigkeitsfeld selbst** vorhergesagt, also  
**v(x, z) = (vₓ, v_z)** auf einem 256×256-Gitter.  
Das Modell soll lernen, wie Kristallgeometrie → Geschwindigkeitsverteilung führt.

---

## **Grundidee**

- **Input des U-Nets:**
    
    - Geometrieinformation, aktuell:
        
        - **Kristallmaske**: phase==1 → 1, sonst 0  
            (1 Kanal, Form: (nx, nz, 1))
            
    - Optional künftig:
        
        - Randbedingungen (z.B. Freier Rand oben/unten)
            
        - Viskosität, Materialparameter  
            → Viele Optionen sind erweiterbar, aber momentan fokussiert sich Ansatz 1 auf die Maske als stärkstes Signal.
            
- **Output des U-Nets:**
    
    - Zwei Kanäle:
        
        - **vₓ_norm(x, z)** – normierte horizontale Geschwindigkeit
            
        - **v_z_norm(x, z)** – normierte vertikale Geschwindigkeit  
            (Form: (nx, nz, 2))
            
- **Ziel des Trainings:**
    
    - Das Modell approximiert die von LaMEM simulierten Geschwindigkeitsfelder.
        
- **Evaluierung:**
    
    - Vergleich v_pred vs. v_LaMEM  
        → wahlweise:
        
        - **im normalisierten Raum** (für direkte Vergleichbarkeit im Training)
            
        - **im physikalischen Raum** (durch Rückskalierung mit scale_v)
            

---

# **Datenpipeline für Ansatz 1**

Ich strukturiere sie in denselben Schritten wie bei Ansatz 2.

---

## **1. Datengenerierung in LaMEM**

- Für jede Probe wird eine LaMEM-Simulation mit 1–n Kristallen durchgeführt:
    
    - **Positionen zufällig**
        
    - **Radien zufällig oder fix** (einstellbar)
        
    - Materialparameter konstant (η, Δρ)
        

Die Funktion  
**`run_sinking_crystals(centers_2D, radii, nx, nz, η, Δρ)`**  
liefert:

- Phasefeld (Kristalle vs. Matrix)
    
- Geschwindigkeiten Vx, Vz
    
- Vorticity ω
    
- Streamfunktion ψ (aktuell nicht genutzt, aber optional für spätere Vergleiche)
    

**Status:**

- LaMEM-Setup funktioniert vollständig für 1–10 Kristalle.
    

---

## **2. Normierung der Geschwindigkeit**

Die Rohgeschwindigkeiten aus LaMEM (in cm/year) liegen über verschiedene Größenordnungen.  
Zur Stabilisierung des Trainings wird ein **dynamischer, sample-spezifischer Skalierungsfaktor** verwendet.

### Algorithmus (`normalize_velocity`):

1. Berechne Betraggeschwindigkeit:
    
    `vmag = sqrt(Vx^2 + Vz^2)`
    
2. Bestimme Exponenten:
    
    `exponents = log10.(vmag[vmag .> 0])`
    
3. mittlerer Exponent:
    
    `p_mean = mean(exponents)`
    
4. Skalierung:
    
    `scale_v = 10^(-p_mean)`
    
5. normierte Felder:
    
    `Vx_norm = Vx * scale_v Vz_norm = Vz * scale_v`
    

**Wichtig:**  
Die Normierung wird in jeder .jld2-Datei gespeichert – für spätere Rekonstruktion.

---

## **3. Vorbereitung der Trainingsdaten**

- **Input (Kristallmaske)**: `build_input_channels(phase)`  
    → ergibt Array (nx, nz, 1)
    
- **Targets**:
    
    - v_norm: Tensor (nx, nz, 2)
        
- **Meta-Infos, die gespeichert werden:**
    
    - n_crystals
        
    - Kristallzentren & Radien
        
    - x_vec_1D, z_vec_1D (physikalische Koordinaten)
        
    - scale_v
        
- **Datensatz-Aufbau:**
    
    - `VelDataset(files::Vector{String})`
        
    - Durch `get_sample` erhält man:
        
        - **x**: (256, 256, 1)
            
        - **y**: (256, 256, 2)
            
- **Batches:**
    
    - `make_batches` erzeugt 4D-Tensoren:
        
        - x_batch: (nx, nz, 1, B)
            
        - y_batch: (nx, nz, 2, B)
            

---

## **4. U-Net Architektur (Ansatz 1)**

Identisch zu Ansatz 2, nur die **Output-Kanalzahl = 2**.

`U-Net( in_channels = 1,        out_channels = 2 )`

Siehe Datei `unet_psi.jl` (gemeinsame Architektur).

Encoder–Decoder-Struktur mit:

- ConvBlöcken (3×3, BatchNorm, ReLU)
    
- MaxPool
    
- ConvTranspose in Decoder
    
- Skip-Connections
    
- Finaler 1×1-Conv → 2 Ausgabekanäle
    

---

## **5. Training des Modells**

Implementiert in `training_vel.jl`.

### Details:

- Loss:
    
    `mse_loss(y_pred, y_true)`
    
- Optimierung:
    
    - Manuelles SGD:
        
        `p .= p .- lr * grad`
        
- Jede Epoche speichert ein CPU-Checkpoint:
    
    `unet_vel.bson`
    

### GPU/CPU-Handling:

Automatische Logik:

- use_gpu = nothing → auto
    
- CUDA wird nur genutzt, wenn verfügbar
    
- Fallback auf CPU, falls CUDA nicht funktioniert
    

---

## **6. Evaluierung**

Implementiert in `evaluate_vel.jl`.

### Optionen:

- **denorm_vel=false**  
    → Fehler im Trainingsraum (v_norm)
    
- **denorm_vel=true**  
    → Fehler im physikalischen Raum (mit scale_v zurückskaliert)
    

### Abläufe je Sample:

1. x, v_true_norm laden
    
2. Modellvorhersage y_pred_norm
    
3. scale_v + meta aus der `.jld2` laden
    
4. Falls gewünscht:
    
    `v_pred_phys = v_pred_norm ./ scale_v`
    
5. Metriken berechnen:
    
    - MSE
        
    - relative L2-Norm
        
6. Ergebnisse werden nach **Kristallanzahl** gruppiert
    

### Beispiel-Statistik:

`n_crystals=3 | N=42 | MSE: 1.23e-5 ± 3.4e-6 | relL2: 3.0e-2 ± 1.8e-2`

Damit bekommst du eine klare Übersicht über Generalisierungsfähigkeit vs. Geometriekomplexität.

---

# **Vergleich mit Ansatz 2 (ψ-Ansatz)**

|Aspekt|Ansatz 1 (v)|Ansatz 2 (ψ)|
|---|---|---|
|Output|2 Kanäle (Vx, Vz)|1 Kanal (ψ)|
|Physikalische Konsistenz|schwächer (Divergenz kann ≠0 sein)|stark (Geschwindigkeiten aus ψ automatisch divergenzfrei)|
|Lernschwierigkeit|höher (2 Felder, komplizierter)|niedriger (glatte Potentialfunktion)|
|Numerical scaling|eigene Normierung nötig|ψ braucht zwingend exponentielle Normierung|
|Vorteil|direkt interpretierbare Geschwindigkeiten|physikalisch eleganter, glatter|

Der ψ-Ansatz ist typischerweise stabiler und physikalisch sauberer.  
Ansatz 1 dient als wichtige **Baseline**, um genau zu sehen, wie viel Vorteil der ψ-Ansatz bringt.