Hier steht die **Stromfunktion ψ(x, z)** im Zentrum. Aus ihr lassen sich die Geschwindigkeitskomponenten rekonstruieren, sodass das Modell implizit ein physikalisch konsistentes Strömungsfeld lernt.

### Grundidee

- **Input des U-Net:**
    
    - Geometrie und Material/Phaseninformation auf dem **256×256-Gitter**, aktuell:
        
        - Kristallmaske (phase == 1) als einzelner Kanal (siehe `build_input_channels`).
        
    - Später erweiterbar um weitere Kanäle (z. B. Viskosität, Randbedingungen).
    
- **Output des U-Net:**
    
    - Ein **Skalarfeld ψ(x, z)** auf dem gleichen 256×256-Gitter  
        → in der Implementierung als ein Kanal mit Form (nx, nz, 1).
    
- **Ziel des Trainings:**
    
    - Das Netz soll die von LaMEM abgeleitete ψ-Funktion (bzw. eine daraus normalisierte Version) approximieren.
    
- **Evaluierung (konzeptionell):**
    
    1. Vergleich zwischen **ψ_pred** (vom U-Net) und **ψ_LaMEM** (bzw. ψ aus der Poisson-Lösung).
        
    2. Optional: Rekonstruktion der Geschwindigkeiten aus ψ_pred und Vergleich mit den LaMEM-Geschwindigkeiten **Vx, Vz**.
    

---

## Datenpipeline für Ansatz 2

Die Pipeline ist im Prinzip schon angelegt, ich formuliere sie hier einmal klar durch – inkl. aktuellem Status.

### 1. Datengenerierung in LaMEM

- Für jede Stichprobe:
    
    - LaMEM-Simulation für eine Kristallkonfiguration im 2D-Schnitt.
    
    - Aktuell: **ein Kristall** mit zufälliger Position (cx, cz) und festem Radius R = 0.1 km.
    
- Die Funktion **`run_sinking_crystals`** erzeugt: Phasefeld, Vx, Vz, Wirbelstärke ω und daraus ψ über die Poisson-Gleichung.
    

**Status (13.11):** LaMEM-Setup und Datengenerierung funktionieren für **1 Kristall**.

---

### 2. Berechnung von ω und ψ

- ω wird aus den Geschwindigkeitsgradienten berechnet:  
    **ω = dVz/dx − dVx/dz** aus dem von LaMEM ausgegebenen Geschwindigkeitsgradienten.
    
- ψ wird anschließend mit der Poisson-Routine bestimmt:
    
    - **Poisson-Gleichung:**
    
        Δψ=−ω\Delta \psi = -\omegaΔψ=−ω
    - Umsetzung in `poisson2D(omega, dx, dz)` mit Dirichlet-Randbedingung ψ = 0 am Rand.
    

---

### 3. Vorbereitung der Trainingsdaten

- Aus dem Phasefeld wird der **Inputkanal** erzeugt:
    
    - `build_input_channels(phase)` erzeugt ein Array (nx, nz, 1) mit einer **Binärmaske für die Kristalle**.
    
- Zielgröße:
    
    - ψ wird als 2D-Feld (nx, nz) berechnet und danach normalisiert (siehe unten) und als ψ_norm gespeichert.
    
- Speicherung:
    
    - `generate_psi_sample` erzeugt für jedes Sample:  
        **input, ψ_norm, scale, meta**  
        und `generate_dataset` schreibt diese in einzelne `.jld2`-Dateien.
    
- Datensatz-Aufbau:
    
    - `PsiDataset` speichert nur die Dateipfade.
        
    - `get_sample` lädt `input` und `ψ_norm`, formt sie in `(nx, nz, 1)` um und gibt sie als **x, y** zurück.
    
- **Splits (konzeptionell):**
    
    - Train/Val/Test werden über verschiedene Verzeichnisse bzw. Unterordner realisiert (noch umzusetzen)
    

---

### 4. Normalisierung von ψ

- Problem: ψ-Werte aus der Poisson-Lösung sind **sehr klein** (typisch Größenordnung 10⁻¹²–10⁻¹⁵).
    
- Lösung: **dynamische Skalierung pro Sample** in `normalize_psi(ψ)`:
    
    - Berechnung des mittleren Exponenten über die nicht-null Werte:
        
        - `exponents = log10.(absψ[mask])`

        - `p_mean = mean(exponents)`

    - Skalierungsfaktor:

        - `scale = 10.0^(-p_mean)`

    - Normierte Stromfunktion:
        
        - **ψ_norm = ψ * scale**
            
    - Rückgabe: `(ψ_norm, scale)`.
        

**Vorteil:**  
ψ_norm liegt im numerisch angenehmen Bereich (typisch O(1)), was Training und Konvergenz verbessert.

---

### 5. Training des U-Net mit ψ als Target

- Architektur: U-Net aus `UNetPsi.build_unet(1, 1)` mit:
    
    - einem Eingabekanal (Kristallmaske),
        
    - einem Ausgabekanal (ψ_norm).
        
- Optimierung:
    
    - Loss-Funktion: **MSE** zwischen vorhergesagter ψ_norm und referenz ψ_norm:
        
        - `mse_loss(y_pred, y_true) = mean((y_pred .- y_true).^2)`.
            
    - Einfaches manuelles SGD-Update in `train_unet`.
        
- GPU/CPU:
    
    - Optionaler CUDA-Pfad mit Fallback auf CPU.
        

---

### 6. Evaluierung (aktuell)

- `evaluate_single` lädt:
    
    - `x, y` (ψ_norm als Ziel),
        
    - das gespeicherte Modell.
        
- Es berechnet:
    
    - `y_pred = model(x_batch)` als ψ_norm_pred,
        
    - bildet ψ_true, ψ_pred und **err = ψ_pred − ψ_true** (alles im **normalisierten** Raum),
        
    - erstellt drei Heatmaps (ψ_true, ψ_pred, Fehler) mit gemeinsamer Farbskala.
        

Bis hierhin findet die gesamte Evaluierung noch **auf den normalisierten ψ-Feldern** statt.

---

## Nächste Schritte (konkret ausgearbeitet)

### 1. Mehrere Kristalle + flexible Radien

**Ziel:** Datengenerierung so erweitern, dass 1…n Kristalle mit zufälligen Positionen und wählbaren Radien erzeugt werden.

**Konzept:**

- In `generate_psi_sample` nicht nur **ein** Zentrum `(cx, cz)` und einen Radius `R`, sondern:
    
    - Anzahl `n_crystals` festlegen (z. B. zufällig in [1, 10] oder als Parameter aus `main.jl`),
        
    - Vektor `centers_2D = [(cx₁, cz₁), …, (cxₙ, czₙ)]`,
        
    - Vektor `radii = [R₁, …, Rₙ]`,
        
        - entweder alle gleich,
            
        - oder für jeden Kristall per Zufall aus einem Intervall (z. B. [R_min, R_max]).
            
- Das Interface `run_sinking_crystals` unterstützt das bereits über die Argumente `centers_2D` und `radii`.
    

**Einbindung in `main.jl`:**

- Oben in `main.jl`:
    
    - Parameter wie **`min_crystals`, `max_crystals`, `R_min`, `R_max`** definieren.
        
- In `generate_psi_sample(rng, ...)`:
    
    - `n_crystals` aus `[min_crystals, max_crystals]` ziehen,
        
    - für jeden Kristall: zufälliges Zentrum und Radius im Bereich,
        
    - an `run_sinking_crystals` übergeben.
        

So kann in `main.jl` die Spannweite der Geometrien kontrolliert werden, ohne den Modulcode jedes Mal zu ändern.

---

### 2. Evaluierung nach Kristallanzahl + tabellarische Übersicht

**Ziel:** Eine Evaluierung, die z. B. pro Kristallanzahl (1–10) zusammenfasst, wie gut das Modell ist.

**Vorschlag:**

- Bei der Datengenerierung sicherstellen, dass im `meta`-Tuple pro Sample gespeichert wird:
    
    - Kristallanzahl,
        
    - ggf. Radii, Positionen etc. (das machst du ja schon für 1 Kristall mit `cx, cz, R`).
        
- Eine neue Funktion, z. B. `evaluate_dataset`, die:
    
    1. Alle Samples im Datensatz durchläuft,
        
    2. Für jedes Sample ψ_pred und ψ_true berechnet,
        
    3. **Fehlermetriken** sammelt, gruppiert nach Kristallanzahl (oder Radiusintervall).
        

**Sinnvolle Größen für die Tabelle:**

Für **ψ**:

- **MSE(ψ)** pro Sample → Mittelwert & Standardabweichung pro Kristallanzahl.
    
- **MAE(ψ)** oder maximale Abweichung:
    
    - z. B. `max(abs.(ψ_pred - ψ_true))`.
        
- **Relative L2-Norm**:
    
    ∥ψpred−ψtrue∥2∥ψtrue∥2\frac{\| \psi_{\text{pred}} - \psi_{\text{true}} \|_2} {\| \psi_{\text{true}} \|_2}∥ψtrue​∥2​∥ψpred​−ψtrue​∥2​​

Optional für die aus ψ rekonstruierten **Geschwindigkeiten**:

- **MSE/MAE der Geschwindigkeitskomponenten** Vx, Vz.
    
- Relative Fehlernorm im Geschwindigkeitsfeld.
    

Damit eine Tabelle aufgebaut werden, z. B.:

|Kristallanzahl|N Samples|MSE(ψ) Mittel|MSE(ψ) Std|rel. L2-Fehler ψ|MSE(V) (optional)|
|---|---|---|---|---|---|

Das beantwortet sehr klar, **wie gut** das System je nach Komplexität der Geometrie funktioniert.

---

### 3. Normalisierung vs. Evaluierung – muss ψ zurückskaliert werden?

Kurz: **Ja, wenn du „physikalische“ Größen vergleichen oder interpretieren willst, musst du die Normalisierung nach der Vorhersage wieder umkehren.**  
Für reine Trainingsmetriken ist der Vergleich im normalisierten Raum aber völlig okay.

**Aktuelle Situation:**

- In den `.jld2`-Dateien speicherst du `input, ψ_norm, scale, meta`.
    
- In `DatasetPsi.get_sample` lädst du nur `input` und `ψ_norm`, **nicht** `scale`.
    
- Training und `evaluate_single` arbeiten ausschließlich mit ψ_norm.
    

**Was heißt das?**

- Das Modell lernt eine Funktion im **normalisierten Raum**:  
    ψ_norm_pred ≈ ψ_norm_true.
    
- Solange du MSE usw. auf ψ_norm berechnest, ist alles konsistent.
    

**Wenn du reale ψ-Werte brauchst (oder Geschwindigkeiten daraus rekonstruierst):**

- Du musst **denselben scale-Faktor verwenden, der beim Sample berechnet wurde**:
    
    - ψ_true_phys = ψ_norm_true / scale
        
    - ψ_pred_phys = ψ_norm_pred / scale
        
- Dafür solltest du bei der Evaluierung:
    
    - entweder `scale` mit aus der Datei laden,
        
    - oder einen zweiten Datensatz-Loader schreiben, der `scale` zusätzlich ausgibt.
        

**Gibt es Probleme durch die Normalisierung bei der Vorhersage?**

- Mathematisch ist die Normalisierung nur eine **lineare Skalierung**:
    
    - ψ_norm = ψ * scale
        
    - ψ = ψ_norm / scale
        
- Das bedeutet:
    
    - Die Loss-Funktion im normalisierten Raum entspricht der Loss-Funktion im physikalischen Raum bis auf einen konstanten Skalierungsfaktor (pro Sample).
        
    - Wenn du bei der Evaluierung sauber zurück skalierst, gibt es **keine prinzipiellen Probleme**.
        
- Wichtig ist nur:
    
    - Du musst darauf achten, dass **ψ_pred und ψ_true mit demselben scale zurück skaliert werden**.
        
    - Für Geschwindigkeiten gilt: wenn du `velocity_from_streamfunction` auf ψ_norm statt auf ψ laufen lässt, bekommst du auch Geschwindigkeit in „normierten“ Einheiten (vergrößert oder verkleinert). Für physikalische Vx/Vz solltest du also mit rückskaliertem ψ arbeiten.
        

**Fazit zur Normalisierung:**

- Training auf ψ_norm: **gut und sinnvoll**.
    
- Evaluierung:
    
    - Für reine Modellvergleiche → normalisierter Raum reicht.
        
    - Für physikalische Interpretation, Plots in realistischen Größenordnungen, Vergleich mit LaMEM-Vx/Vz → ψ nach der Vorhersage zurück skalieren.