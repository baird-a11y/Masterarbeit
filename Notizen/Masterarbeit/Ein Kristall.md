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
    - **Status**: Funktioniert, evtl. später zu prüfen

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

#### Ausblick

**Sofortige nächste Schritte:**

- **Mission erfolgreich**: UNet kann LaMEM-Geschwindigkeitsfelder vorhersagen
- **Robustheits-Tests**: Extreme Parameter, Physik-Constraints prüfen
- **Skalierung**: 500-1000 Bilder für noch bessere Generalisierung

**Weitere Entwicklung:**

- **Mehrere Kristalle**: 2-3 Kristalle gleichzeitig
- **Komplexere Geometrien**: Ellipsen, verschiedene Formen
- **Real-world Validation**: Vergleich mit experimentellen Daten
- **Parameterstudien**: Automatische Exploration von η, Δρ, R-Räumen

**Offene Fragen für nächste Session:**

- Funktioniert das Modell bei extremen Parametern?
- Lernt es echte Physik oder nur Pattern-Matching?
- Wie robust ist es gegenüber ungesehenen Konfigurationen?

---

## Lessons Learned

1. **Datenqualität > Quantität**: 100 gute Samples besser als 16 schlechte
2. **Evaluierung ist kritisch**: Ohne Ground Truth Vergleich keine echte Bewertung
3. **Physikalische Plausibilität**: Metriken können täuschen, visuelle Inspektion wichtig
4. **Iterative Entwicklung**: Kleine Tests → Probleme finden → Fixes → Skalierung