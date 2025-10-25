## Überblick

Das Paper stellt **Physics-Informed Neural Networks (PINNs)** für die Simulation inkompressibler Strömungen vor - von laminaren bis zu turbulenten Kanälen bei Re_τ ≈ 1000.

## Zwei Formulierungen

1. **VP-NSFnet** (Velocity-Pressure):
    
    - Input: Raum- und Zeitkoordinaten (t, x, y, z)
    - Output: Geschwindigkeit (u, v, w) + Druck (p)
    - Druck ist "hidden state" - wird über Inkompressibilitätsbedingung bestimmt
2. **VV-NSFnet** (Vorticity-Velocity):
    
    - Input: Raum- und Zeitkoordinaten
    - Output: Geschwindigkeit + Vortizität (ωx, ωy, ωz)
    - Druckterm eliminiert, Inkompressibilität exakt erfüllt

## Kernmerkmale

- **Automatische Differentiation** für alle Ableitungen in den Navier-Stokes-Gleichungen
- **Unüberwachtes Lernen**: Nur Rand-/Anfangsbedingungen als Daten, keine Druckdaten benötigt
- **Dynamische Gewichte** in der Loss-Funktion zur Verbesserung der Konvergenz

## Getestete Strömungen

### Laminare Strömungen:

- 2D Kovasznay-Strömung (steady)
- 2D Zylinderumströmung (unsteady)
- 3D Beltrami-Strömung

**Ergebnis**: VV-NSFnet erreicht höhere Genauigkeit als VP-NSFnet für laminare Fälle

### Turbulente Strömung:

- Kanalströmung bei Re_τ ≈ 1000
- Simulation über mehrere konvektive Zeiteinheiten
- **Ergebnis**: NSFnets können Turbulenz aufrechterhalten mit <10% Fehler für Geschwindigkeit

## Besondere Anwendungen

NSFnets können Probleme lösen, die für klassische CFD schwierig/unmöglich sind:

1. **Fehlende Randbedingungen** (ill-posed problems)
2. **Verrauschte Daten**
3. **Inverse Probleme** (z.B. unbekannte Reynolds-Zahl bestimmen)
4. **Transfer Learning** beschleunigt Training für ähnliche Problemstellungen

## Relevanz für die Masterarbeit

Besonders interessant könnten sein:

- Residual Learning Ansatz (ähnlich wie v_total = v_stokes + Δv)
- Umgang mit dynamischen Gewichten in der Loss-Funktion
- Multi-scale Probleme (Turbulenz hat verschiedene Skalen, wie mein Multi-Kristall-System)
- Automatische Differentiation für physikalische Gleichungen