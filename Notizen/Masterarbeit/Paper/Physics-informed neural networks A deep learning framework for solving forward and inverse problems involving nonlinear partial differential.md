## Überblick

Diese wegweisende Arbeit von Raissi, Perdikaris und Karniadakis (2019) stellt **Physics-Informed Neural Networks (PINNs)** vor – eine neue Klasse von neuronalen Netzen, die physikalische Gesetze in Form von partiellen Differentialgleichungen (PDEs) direkt in den Lernprozess integrieren.

## Kernkonzept

PINNs kombinieren:

- **Deep Learning** (universelle Funktionsapproximatoren)
- **Automatische Differentiation** (zur Berechnung von Ableitungen)
- **Physikalische Gesetze** (als Regularisierung)

Die Grundidee: Ein neuronales Netz approximiert die Lösung u(t,x) einer PDE, während ein zweites "physics-informed" Netz f(t,x) sicherstellt, dass die physikalischen Gesetze erfüllt werden.

## Zwei Hauptanwendungen

### 1. **Data-driven Lösung von PDEs**

- Gegeben: Wenige Datenpunkte + physikalische Gleichung
- Gesucht: Vollständige Lösung im gesamten Raum-Zeit-Bereich
- Vorteil: Sehr dateneffizient durch Einbeziehung physikalischer Constraints

### 2. **Data-driven Entdeckung von PDEs**

- Gegeben: Messdaten des Systems
- Gesucht: Unbekannte Parameter der zugrunde liegenden Gleichung
- Beispiel: Bestimmung von Viskosität oder Reaktionsraten

## Methodische Ansätze

### Kontinuierliche Zeit-Modelle

- Loss-Funktion: MSE = MSE_u + MSE_f
- MSE_u: Anpassung an Mess-/Randdaten
- MSE_f: Erfüllung der PDE an Kollokationspunkten

### Diskrete Zeit-Modelle

- Integration von **Runge-Kutta Zeitschrittverfahren**
- Ermöglicht **sehr große Zeitschritte** (z.B. 500+ Stufen)
- Beispiel: Burgers-Gleichung von t=0.1 bis t=0.9 in **einem einzigen Schritt**

## Beispielanwendungen

1. **Schrödinger-Gleichung**: Komplexe quantenmechanische Systeme mit periodischen Randbedingungen
    
2. **Allen-Cahn-Gleichung**: Phasenseparation in Legierungen, mit scharfen Gradienten
    
3. **Navier-Stokes-Gleichungen**: Inkompressible Strömungen um Zylinder (Kármán-Wirbelstraße)
    
    - Bemerkenswert: Rekonstruktion des **gesamten Druckfelds** ohne direkte Druckmessungen
4. **Korteweg-de Vries (KdV) Gleichung**: Flachwasserwellen mit Dispersionseffekten
    
5. **Burgers-Gleichung**: Prototyp für nichtlineare Schockbildung
    

## Wichtige Ergebnisse

- **Hohe Genauigkeit**: Relative L2-Fehler oft < 10^-3
- **Robustheit gegenüber Rauschen**: Funktioniert auch mit 1-10% Messrauschen
- **Dateneffizienz**: Gute Ergebnisse mit nur 50-200 Datenpunkten
- **Parameteridentifikation**: Präzise Bestimmung unbekannter physikalischer Parameter

## Technische Details

**Netzwerk-Architektur**:

- Typisch: 4-9 Schichten mit 20-200 Neuronen pro Schicht
- Aktivierungsfunktion: tanh
- Keine zusätzliche Regularisierung (L1/L2, Dropout) nötig

**Optimierung**:

- Kleine Datensätze: L-BFGS (Quasi-Newton)
- Große Datensätze: Adam oder andere SGD-Varianten

**Automatische Differentiation**:

- Berechnung von ∂u/∂t, ∂²u/∂x², etc. durch Backpropagation
- Gleicher Mechanismus wie beim Training, aber bzgl. Eingabe-Koordinaten

## Vorteile

6. **Physikalische Konsistenz**: Lösungen respektieren Erhaltungsgesetze
7. **Mesh-frei**: Keine Diskretisierung des Raum-Zeit-Bereichs nötig
8. **Flexibilität**: Anwendbar auf verschiedenste PDEs
9. **Inverse Probleme**: Gleichzeitige Lösung und Parameteridentifikation
10. **Kleine Daten**: Funktioniert auch bei stark limitierten Messdaten

## Limitierungen & Offene Fragen

- Keine theoretischen Konvergenzgarantien
- Netzwerk-Architektur muss empirisch gewählt werden
- Trainingsdynamik nicht vollständig verstanden
- Skalierung auf höherdimensionale Probleme unklar
- Variabilität in den Ergebnissen je nach Setup

## Bedeutung

Diese Arbeit etabliert eine **neue Paradigma** für wissenschaftliches Rechnen, das:

- Deep Learning mit mathematischer Physik verbindet
- Klassische numerische Methoden (Runge-Kutta) mit modernem ML kombiniert
- Den Weg für datengetriebene, physikalisch konsistente Modellierung ebnet

Die Methode hat weitreichende Anwendungen in Strömungsmechanik, Materialwissenschaften, Quantenmechanik und vielen anderen Bereichen der Computational Science.



## Checkliste für die Arbeit

### Was ich machen könnte:

- [ ]  PINNs vs. klassische Methoden vergleichen
- [ ]  Sensitivity Studies (Daten, Rauschen, Architektur)
- [ ]  Visualisierungen wie im Paper (Heatmaps, Snapshots, Vergleiche)
- [ ]  Sowohl Forward als auch Inverse Problem zeigen
- [ ]  Rechenzeit und Dateneffizienz diskutieren
- [ ]  Limitierungen ehrlich erwähnen