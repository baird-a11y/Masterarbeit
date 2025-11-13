# U-Net zur Vorhersage von Strömungsgrößen aus LaMEM-Daten

## Kurzfassung

Ziel dieser Arbeit ist die Entwicklung und der Vergleich dreier Ansätze zur Vorhersage von Strömungsgrößen in viskosen Medien mithilfe eines **U-Net** in **Julia**. Untersucht werden dabei Strömungen im Kontext von **Kristallbewegungen in viskosen Magmen**, wie sie beispielsweise in **Magmakammern** auftreten. Die Modelle basieren auf numerischen Simulationen mit **LaMEM** und sollen unterschiedliche Repräsentationen des Strömungsfeldes – Geschwindigkeiten, Stromfunktion und Residuen – erlernen und miteinander verglichen werden.

  

---
## 1. Ziel der Masterarbeit

Ziel der Arbeit ist die Entwicklung und der Vergleich von drei Ansätzen zur Vorhersage von Strömungsgrößen in viskosen Strömungen mithilfe eines U-Net in Julia. Grundlage sind numerische Simulationen mit **LaMEM** für verschiedene Kristall-Konfigurationen.  

Vorhergesagte Größen sind:


- Geschwindigkeitsfeld **v(x, z) = (vₓ, v_z)**

- **Stromfunktion ψ(x, z)**

- **Differenz** zwischen analytischer und numerischer Lösung (Residualfeld)  

Alle Modelle sollen auf einem **256×256-Gitter** trainiert werden.


---

## 2. Datengrundlage

### 2.1 Geometrie-Setups
  

- Simulationsdomäne wird auf ein 256×256-Gitter diskretisiert.

- In der Domäne befinden sich **1 bis 10 Kristalle**, deren:

  - Anzahl variiert (**1–10**),

  - Position zufällig variiert (für die Trainingsdaten),

  - Materialparameter und Randbedingungen **konstant** gehalten werden.

- Für die **Evaluierung** werden wieder 1–10 Kristalle verwendet, **aber:**

  - die Geometrien sind fix (immer dieselben Positionen),

  - damit sind die Evaluierungsdaten für alle Modelle identisch und vergleichbar.

  

### 2.2 Motivation für dieses Setup

  

- Durch die Variation der Kristallanzahl kann geprüft werden, **ob eine Generalisierung stattfindet**, d. h. ob das Modell robuste Vorhersagen für verschiedene Kristallzahlen machen kann.

- Gefahr: Overfitting auf bestimmte Kristallanzahlen (z. B. gute Vorhersage nur, wenn Trainings- und Testanzahl identisch sind).

- Ziel: **Robustheit gegenüber Anzahl und Position der Kristalle**, nicht nur das reine Merken des Trainings-Setups.

  

---
## 3. Modellarchitektur

  

- Verwendung eines **U-Net** in Julia (z. B. auf Basis von Flux.jl):

  - Eingabe: Felder auf einem 256×256-Gitter (z. B. Material-/Phasenfeld, Randbedingungen, ggf. zusätzliche Skalare).

  - Ausgabe: Feld gleicher Auflösung (Geschwindigkeit, ψ oder Fehlerfeld).

- Architektur über alle Ansätze hinweg möglichst konstant halten, um vergleichbare Ergebnisse zu ermöglichen.

- Unterschiede der Ansätze liegen primär in:

  - der gewählten Zielgröße (Target),

  - der Vorverarbeitung (z. B. ψ-Berechnung, Residuenbildung),

  - der Normalisierung.

  

---
## 4. Ansatz 1 – Direkte Vorhersage der Geschwindigkeiten (Baseline)

  

*(Zunächst optional, dient als Referenzmodell)*


- **Input:** Geometrie/Materialfeld aus LaMEM (z. B. Kristallmaske, Viskosität, Randbedingungen).

- **Output:** Geschwindigkeitskomponenten **vₓ(x, z)** und **v_z(x, z)** direkt aus den LaMEM-Daten gelernt.

- Besonderheit:

  - Modell hat **2 Eingabekanäle (oder mehr)**, je nach Geometrierepräsentation.

  - Output hat **2 Kanäle (vₓ und v_z)**.

- Vorteil:

  - Konzeptionell am einfachsten, schnell umzusetzen.

- Nachteil:

  - Kann physikalisch inkonsistente Felder produzieren (z. B. Divergenz ≠ 0).

  - Vermutlich weniger elegant als ψ-basierte Ansätze.

  

---
## 5. Ansatz 2 – Lernen der Stromfunktion ψ

Hier steht im Zentrum die **Stromfunktion ψ**, aus der sich Geschwindigkeiten rekonstruieren lassen.

 ### 5.1 Grundidee


- **Input:** Geometrie / Material-/Phasenfelder als Gitter (wie bei Ansatz 1).

- **Output:** Feld **ψ(x, z)** auf dem 256×256-Gitter.

- Evaluierung:

  - Vergleich zwischen **vorhergesagter ψ** und **aus LaMEM abgeleiteter ψ**.

  - Optional: Rekonstruktion der Geschwindigkeiten aus ψ und Vergleich mit LaMEM-Geschwindigkeiten.

### 5.2 Datenpipeline (Schritte)

1. **Datengenerierung in LaMEM**

   - Durchführung von LaMEM-Simulationen für verschiedene Kristallkonfigurationen (1–10 Kristalle, zufällige Positionen).  

   - Nutzung des existierenden Skripts, das **ψ direkt aus den LaMEM-Ausgaben** bestimmt.

2. **Lösen der Poisson-Gleichung (falls nötig)**

   - Falls ψ nicht direkt vorliegt, Berechnung über die Poisson-Gleichung  

     $$\Delta \psi = -\omega$$

  

3. **Berechnung von Ω (Vorticity)**

   - Berechnung von **ω (Omega)** aus Geschwindigkeitsfeldern oder ψ (je nach Formulierung).

4. **Vorbereitung der Trainingsdaten**

   - Speicherung der Eingabekanäle (z. B. Kristallmaske, Materialparameter) als 256×256-Tensoren.

   - Speicherung der Zielgröße **ψ(x, z)** als 256×256-Tensor.

   - **Split in Trainings-, Validierungs- und Testdaten**.


5. **Normalisierung von ψ**

   - ψ-Werte sind **sehr klein** (z. B. Größenordnung 10⁻¹² bis 10⁻¹⁵).

   - Dynamische Skalierung pro Datensatz:  

     Bestimmung des mittleren Exponentenbereichs und Skalierung mit dem Mittelwert der Potenzen, um ψ-Werte in einen numerisch stabilen Bereich zu bringen.  

     $$\psi_{\text{norm}} = \psi \cdot 10^{-p_{\text{mean}}}$$

   - Dies verbessert die numerische Stabilität und Konvergenz im Training.

6. **Training des U-Net mit ψ als Target**

   - Loss-Funktion: MSE zwischen vorhergesagter ψ_norm und Referenz ψ_norm.


7. **Evaluierung**

   - Direkter Vergleich ψ_pred vs. ψ_LaMEM.

   - Optional: Rekonstruktion der Geschwindigkeiten aus ψ_pred.

  

---
## 6. Ansatz 3 – Lernen der Differenz (Residual) zwischen analytischer und numerischer Lösung

  

### 6.1 Grundidee

  

Das U-Net soll **nicht das gesamte Feld**, sondern nur die **Differenz** zwischen analytischer und numerischer Lösung der Stromfunktion ψ lernen.  

  

Formell:

  

$$r(x,z) = \psi_{\text{analytisch}}(x,z) - \psi_{\text{LaMEM}}(x,z)$$

  

Im Einsatz:

  

$$\psi_{\text{korrigiert}}(x,z) = \psi_{\text{LaMEM}}(x,z) + r_{\text{pred}}(x,z)$$

  

### 6.2 Datenpipeline

  

1. **Berechnung der analytischen Lösung**

2. **LaMEM-Simulationen für dieselben Setups**

3. **Bildung des Residualfeldes**

4. **Training des U-Net auf r**

5. **Evaluierung**

   - Vergleich ψ_korrigiert vs. ψ_analytisch und ψ_LaMEM vs. ψ_analytisch.

  

---
## 7. Evaluierungskonzept (für alle Ansätze)

  

1. **Datensplits**

   - Training: zufällige Kristallpositionen und -anzahlen (1–10).

   - Validation: andere zufällige Setups.

   - Test/Evaluierung: **feste** 1–10 Kristall-Setups (identische Geometrien für Vergleichbarkeit).

  

2. **Metriken**

   - MSE, MAE, relative Fehlernormen.

   - Optional: Divergenzfreiheit, Erhaltung physikalischer Größen.

  

3. **Generalisationstests**

   - Fehler als Funktion der Kristallanzahl (1–10).

   - Tests mit neuen Kristallverteilungen.

  

4. **Vergleich der Ansätze**

   - Ansatz 1: v direkt lernen (Baseline)

   - Ansatz 2: ψ lernen

   - Ansatz 3: Residuum lernen

  

---
## 8. Umsetzungsplan (praktische Schritte)

  

1. **Vorbereitung**

   - Ordnerstruktur für Daten (Train/Val/Test).

   - Die Daten sollen bei jedem Run **neu generiert** werden, um Generalisierung zu fördern und Speicherplatz zu sparen.

  

2. **Schritt 1:** Datengenerierung + ψ-Extraktion  

3. **Schritt 2:** Normalisierung + Dataset-Klassen  

4. **Schritt 3:** U-Net-Implementierung in Julia  

5. **Schritt 4:** Training für Ansatz 2  

6. **Schritt 5:** Implementierung Ansatz 3  

7. **Schritt 6:** Ansatz 1 als Baseline  

8. **Schritt 7:** Auswertung & Dokumentation