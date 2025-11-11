# Kristall Geschwindigkeitsfeld Vorhersage mit UNet

## Projektübersicht

**Ziel:** Vorhersage des Geschwindigkeitsfeldes (v_x, v_z) eines sinkenden Kristalls mittels UNet oder ähnlicher Architektur.

### Input/Output

- **Input:** Phase des Kristalls
- **Output:** Geschwindigkeitskomponenten v_x und v_z

## Aufgabenstellung

### Hauptziel

Geschwindigkeitsfeld in x- und z-Richtung eines sinkenden Kristalls vorhersagen

### Aktueller Fokus

- [ ] Einzelner Kristall mit variierender Position und Radius
- [ ] Später: Erweiterung auf zwei Kristalle

## Technische Anforderungen

### Normalisierung

- **Skalar mean:** 0
- **Standardabweichung/Varianz:** 1
- ❓ Im UNet implementieren (zu klären)

### Physikalische Constraints

**Kontinuitätsgleichung:** ∂V_x/∂x + ∂V_z/∂z = 0

- [ ] Algorithmus bestrafen wenn Bedingung verletzt wird
- [ ] Eventuell zusätzliches Training erforderlich

### Grid-Spezifikationen

- **Auflösung:** Vielfaches von 2^n
- **Empfohlene Werte:** 256 oder 512
- **Radius:** Fester Wert von 0.05 (initial)

> ⚠️ **Wichtig:** Auflösung und Radius müssen aufeinander abgestimmt sein. Zu kleine Auflösung mit zu kleinem Radius kann Probleme verursachen.

## Nächste Schritte

### Kurzfristig

- [x] Mail für Spezialisierung formulieren
    - Bewusst vage halten bezüglich konkreter Methoden
    - Flexibilität für Systemänderungen bewahren
- [ ] Training mit einem [[Ein Kristall]] starten
- [ ] Ggf. Erweiterung auf zwei Kristalle

### Experimenteller Aufbau

1. **Phase 1:** Einzelkristall, fixer Radius (0.05), variable Position
2. **Phase 2:** Validation der physikalischen Constraints
3. **Phase 3:** Erweiterung auf Mehrkristall-Systeme

## Offene Fragen

- [ ] Implementierung der Normalisierung im UNet
- [ ] Optimale Methode zur Durchsetzung der Kontinuitätsgleichung
- [ ] Validierung der Grid-Radius-Kombination

## Referenzen und Links

<!-- Hier können später Links zu Papers, Code-Repositories oder anderen relevanten Materialien eingefügt werden -->

---

**Tags:** #machine-learning #unet #fluid-dynamics #crystal-sedimentation #physics-informed-ml