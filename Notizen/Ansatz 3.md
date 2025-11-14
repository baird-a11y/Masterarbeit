**Ziel:**  
Nicht ψ selbst lernen, sondern **den Fehler zwischen einer analytischen Referenz-Lösung und der LaMEM-Lösung**:

r(x,z)=ψanalytisch(x,z)−ψLaMEM(x,z)r(x,z) = \psi_{\text{analytisch}}(x,z) - \psi_{\text{LaMEM}}(x,z)r(x,z)=ψanalytisch​(x,z)−ψLaMEM​(x,z)

Das U-Net sagt also **nur das Residuum** voraus.  
Die korrigierte Stromfunktion erhält man anschließend aus:

ψkorrigiert=ψLaMEM+rpred\psi_{\text{korrigiert}} = \psi_{\text{LaMEM}} + r_{\text{pred}}ψkorrigiert​=ψLaMEM​+rpred​

Dieser Ansatz ist nützlich, wenn man:

- bereits eine grobe physikalische Näherung kennt (→ analytisch),
    
- nur die Abweichungen (z. B. durch Randbedingungen, Kristallinteraktion) lernen möchte,
    
- ein stabileres, fehlerreduziertes Modell erhalten will.
    

---

# **Modellidee**

### **Input des Modells**

- Geometrie / Materialinformationen auf dem **256×256-Gitter**
    
- Aktuell: **Binärmaske** der Kristalle als ein Kanal `(nx, nz, 1)`  
    → `build_input_channels(phase)`
    

### **Output des Modells**

- Ein **Skalarfeld** `(nx, nz, 1)`
    
- Dieses Feld ist **residual_norm**, also das _normalisierte Residuum_ r_norm.
    

### **Trainingsziel**

rpred,norm≈rtrue,normr_{\text{pred,norm}} \approx r_{\text{true,norm}}rpred,norm​≈rtrue,norm​

Das Netzwerk lernt also **Differenzen**, nicht absolute Werte.

---

# **Datenpipeline für Ansatz 3**

## **1. Analytische Referenzlösung (Stokes-Sinkgeschwindigkeit)**

Für einen Kreis/Kristall, der in viskoser Flüssigkeit sinkt, ist die analytische Sinkgeschwindigkeit:

U=Δρ g R24 ηU = \frac{\Delta\rho\,g\,R^2}{4\,\eta}U=4ηΔρgR2​

Eine passende (vereinfachte) analytische Stromfunktion lautet:

ψanalytisch(x,z)=U⋅xphys\psi_{\text{analytisch}}(x,z) = U \cdot x_{\text{phys}}ψanalytisch​(x,z)=U⋅xphys​

Damit gilt:

- vx=0v_x = 0vx​=0
    
- vz=−Uv_z = -Uvz​=−U
    

→ Eine sehr einfache, aber physikalisch korrekte Grundströmung.

### Eigenschaften:

- basiert direkt auf den Materialparametern (η, Δρ, R),
    
- linear in x,
    
- unabhängig von z,
    
- liefert ein gutes „Baseline“-Strömungsfeld für Residual-Netze.
    

---

## **2. LaMEM-Simulation (numerische Lösung)**

Analog zu Ansatz 2:

- `run_sinking_crystals` liefert:
    
    - Phasefeld
        
    - LaMEM-ψ
        
    - Gitterkoordinaten
        
- Diese LaMEM-ψ bildet die **Referenz für den numerischen Anteil**.
    

---

## **3. Berechnung des Residuals**

Für jedes Sample:

r=ψanalytisch−ψLaMEMr = \psi_{\text{analytisch}} - \psi_{\text{LaMEM}}r=ψanalytisch​−ψLaMEM​

Dieses Residuum enthält genau die Strukturdifferenzen, die nicht durch die einfache analytische Lösung erklärt werden können.

Typisch:

- Störungen durch die Kristallgeometrie
    
- Rotation, Streuung, Randbedingungen
    
- Mehrere Kristalle: komplexe Interaktionsstrukturen
    

---

## **4. Normalisierung des Residuals**

Analog zu Ansatz 2, aber für r:

- Berechnung des mittleren Zehnerexponenten:
    
    pmean=mean(log⁡10∣r∣)p_{\text{mean}} = \mathrm{mean}(\log_{10}|r|)pmean​=mean(log10​∣r∣)
- Skalierung:
    
    rnorm=r⋅10−pmeanr_{\text{norm}} = r \cdot 10^{-p_{\text{mean}}}rnorm​=r⋅10−pmean​

→ Normalisierte Residuen sind O(1) → stabileres Training.

Der Scale wird pro Sample gespeichert.

---

## **5. Datenspeicherung**

Jedes `.jld2`-Sample enthält:

- `input` → Kristallmaske (nx, nz, 1)
    
- `residual_norm`
    
- `scale`
    
- `meta` (Anzahl Kristalle, Positionen, Radien, U_stokes usw.)
    
- `ψ_lamem`
    
- `ψ_analytic`
    

So bleiben alle Schritte reproduzierbar und evaluierbar.

---

# **Training des U-Net (Ansatz 3)**

- Architektur: gleich wie in Ansatz 2  
    → `build_unet(1, 1)`
    
- Loss: **MSE** zwischen predicted und true `residual_norm`
    
- Optimierung: manuelles SGD (wie im vorhandenen Code)
    
- GPU-Unterstützung: optional über CUDA, identisch implementiert zu Ansatz 2
    

Das Modell lernt also nur das **Differenzfeld**.

---

# **Evaluierung**

1. Lade ein Sample:
    
    - residual_norm, scale
        
    - ψ_lamem, ψ_analytic
        
2. Modellvorhersage:
    
    rpred,norm=model(x)r_{\text{pred,norm}} = \text{model}(x)rpred,norm​=model(x)
3. Rückskalierung:
    
    rpred=rpred,norm/scaler_{\text{pred}} = r_{\text{pred,norm}} / scalerpred​=rpred,norm​/scale
4. Rekonstruktion der Stromfunktion:
    
    ψkorrigiert=ψLaMEM+rpred\psi_{\text{korrigiert}} = \psi_{\text{LaMEM}} + r_{\text{pred}}ψkorrigiert​=ψLaMEM​+rpred​
5. Evaluierung:
    
    - Fehler in r:
        
        rpred−rr_{\text{pred}} - rrpred​−r
    - Fehler in ψ_korrigiert:
        
        ψkorrigiert−ψanalytisch\psi_{\text{korrigiert}} - \psi_{\text{analytisch}}ψkorrigiert​−ψanalytisch​
    - Vergleich mit Ansatz 2 (direkte ψ-Vorhersage)
        
6. Visualisierung:
    
    - Heatmaps der Felder (ψ_LaMEM, ψ_analytic, ψ_corrected)
        
    - Residual-Heatmaps
        
    - MSE/MAE pro Sample
        
    - Statistiken nach Kristallanzahl
        

---

# **Vorteile von Ansatz 3**

✔ **Physikalisch motiviert:** nutzt analytische Lösung als „Grundmodell“  
✔ **Stabil:** Modell lernt nur Abweichungen, nicht komplette ψ  
✔ **Bessere Generalisierung:** Fehler sind oft glatter/schwächer ausgeprägt  
✔ **Effizient für Multi-Kristall-Fälle**  
✔ **Numerisch einfacher:** Residuen sind meist kleiner und strukturierter