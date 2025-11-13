Ziel:  
Eine **übersichtliche, interaktive Ergebnisdarstellung**, die mehrere Modell-Runs vergleicht und Fehler nach Kristallanzahl zeigt.

Die Auswertung besteht aus **3 Bausteinen**:

---

# **1️⃣ Ergebnis-Export aus Julia (JSON)**

Für jeden Evaluationslauf (`eval_dataset`) sollen die wichtigsten Informationen in eine JSON-Datei geschrieben werden.  
Diese JSON dient später als Datenquelle für die Website.

### Inhalt eines gespeicherten Runs:

- Metadaten:
    
    - Run-ID
        
    - Datum/Uhrzeit
        
    - Modellname (z. B. „U-Net ψ“)
        
    - Trainingsparameter (Epochs, Learning Rate, Batchgröße)
        
    - Dataset-Name / Ordner
        
    - Ob im **normalisierten** oder **physikalischen** ψ-Raum evaluiert wurde
        
    - Notizen (optional)
        
- Fehler pro Kristallanzahl (aus `eval_dataset`):
    
    - Kristallanzahl _n_
        
    - Anzahl Samples _N_
        
    - MSE (Mittel + Std)
        
    - relativer L2-Fehler (Mittel + Std)
        

### Beispiel-Struktur:

```json
[
  {
    "id": "psi_unet_run_001",
    "date": "2025-01-13T14:52:10",
    "model": "U-Net ψ",
    "epochs": 70,
    "learning_rate": 1e-4,
    "batch_size": 2,
    "data_dir": "data_psi_eval_1_10",
    "denorm_psi": true,
    "notes": "Baseline ψ-Modell, 1–5 Kristalle",
    "metrics": [
      {
        "n_crystals": 1,
        "N": 20,
        "mse_mean": 1.2e-6,
        "mse_std": 3e-7,
        "rel_l2_mean": 0.015,
        "rel_l2_std": 0.004
      }
      // ...
    ]
  }
]
```

### Was später noch umgesetzt wird:

- Kleine Julia-Funktion, die nach `evaluate_dataset` automatisch so ein Run-Objekt als JSON abspeichert (Anhängen an `results_psi.json`).
    

So entsteht eine **chronologische Sammlung von Modell-Runs**.

---

# **2️⃣ Interaktive Ergebnis-Website (Single-File HTML)**

Eine einzige Datei `index.html`, die:

- `results_psi.json` lädt
    
- alle Modell-Runs in einem **Dropdown** anzeigt
    
- filterbare **Metriken** zeigt (MSE / relL2)
    
- aus den Daten
    
    - eine **Tabelle** und
        
    - einen **Fehlerplot** (Chart.js)  
        rendert
        

### Features der Website:

- Run-Auswahl (Dropdown)
    
- Metrik-Auswahl (MSE oder relL2)
    
- Übersicht mit „Pills“:
    
    - Modellname
        
    - Epochs / Learning Rate / Batch Size
        
    - Dataset
        
    - normalisiert vs. physikalisch
        
    - Datum
        
    - Notizen
        
- Tabelle:
    
    - Fehler pro Kristallanzahl
        
- Interaktiver Plot:
    
    - Fehler(y) vs. Kristallanzahl(x)
        

### Verwendung:

- `index.html` und `results_psi.json` ins gleiche Verzeichnis legen.
    
- Datei einfach im Browser öffnen.
    
- Fertig.
    

Damit bekommst du eine professionelle, **interaktive Ergebnisübersicht**, die du auch für die Masterarbeit verwenden kannst.

---

# **3️⃣ Workflow später (wenn du es umsetzt)**

### Schritt 1 — Eval-Run durchführen

```julia
errors = EvaluatePsi.evaluate_dataset(
    data_dir="data_psi_eval_1_10",
    model_path="unet_psi.bson",
    denorm_psi=true
)
```

### Schritt 2 — Run als JSON abspeichern

_(Mit der später eingebauten Julia-Funktion)_

```julia
save_eval_run_json(
    "results_psi.json";
    run_id="psi_unet_run_001",
    model_name="U-Net ψ",
    epochs=70,
    lr=1e-4,
    batch_size=2,
    data_dir="data_psi_eval_1_10",
    denorm_psi=true,
    notes="Baseline ψ-Modell",
    errors_by_n=errors
)
```

### Schritt 3 — Website anzeigen

Einfach `index.html` im Browser öffnen → alles wird geladen und angezeigt.

---

# **4️⃣ Warum das sinnvoll für die Masterarbeit ist**

- Reproduzierbare, klar strukturierte Ergebnisübersicht
    
- Vergleich verschiedener Modelle (ψ vs. Residuen vs. v-Baseline)
    
- Vergleich verschiedener Hyperparameter
    
- Fehleranalyse pro Kristallanzahl
    
- Profi-Level Auswertung, die du in die Thesis einbauen kannst
    
- Zero-setup: reine Browser-Datei (keine Server)
    

---

# Fertig zum späteren Aufgreifen ✔️

Diese Zusammenfassung kannst du als **Projektplan** benutzen, sobald du bereit bist, den Auswertungsteil auszubauen.

Sag einfach Bescheid, wann du das angehen willst — dann setzen wir **Schritt für Schritt** erst  
🟦 JSON-Export, dann  
🟩 HTML-Seite, dann  
🟨 Filter & Extras.