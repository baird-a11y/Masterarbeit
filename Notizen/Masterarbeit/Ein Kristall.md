
Notizen über das Training mit einem Kristall

Beginnen Trainingsdaten zu erstellen mit erst einmal einem Kristall an verschiedenen Positionen und diese strukturiert abspeichern für späteres Training.

## Ausgangslage

Für die Erstellung von Trainingsdaten brauche ich Bilder und Masken, die die Position des Kristalles enthalten, sowie die beiden Geschwindigkeitsfelder (v_x,v_z).

Hier müsste ich schauen, ob ich einfach reine Bilder erstelle oder ob ich die Werte auslesen lasse.

## Umsetzung

Erster Ansatz wäre einfach ein Bild erst mal nur vom Kristall zu erstellen und dann noch einmal eins mit dem Geschwindigkeitsfeld und dieses dann als Maske zu nutzen. Dabei sollte der Kristall in beiden Fällen immer die gleiche Farbe haben als Orientierung.

Zu beginn, sollten 10-20 Bilder reichen um eine grobe Orientierung zu haben ob dieser Ansatz klappt.

### Experiment 1

Ich nehme als ersten Datensatz einfach die Bilder ich per SinkingSphere_LaMEM.jl erstelle als Trainingsbilder.


Theorie
Umsetzung
Probleme
Wichtige Code Teile
Ergebnisse
Ausblick