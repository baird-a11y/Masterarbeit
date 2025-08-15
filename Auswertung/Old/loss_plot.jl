using Plots
using DelimitedFiles

# Daten definieren (aus Ihrer Datei)
learning_rates = [0.001, 0.0001, 0.0001, 0.00001, 0.000001]
lr_labels = ["0.001", "0.0001", "0.0001", "0.00001", "0.000001"]

# Loss-Daten für jede Lernrate
loss_data = [
    # LR 0.001
    [114.27193, 105.69452, 104.02271, 102.305, 102.76305, 102.209, 102.18109, 101.17594, 100.38804, 101.58008, 100.85787, 100.40553, 99.91543, 100.24317, 99.898026, 100.36048, 99.90484, 99.73067, 99.747375, 100.52446, 99.423645, 99.19807, 99.29912, 99.38355, 99.16368, 99.50646, 98.61688, 99.04977, 98.89799, 99.28998, 98.84672, 98.90957, 100.87528, 99.26875, 98.60429, 98.34442, 98.20122, 98.17732, 98.21575, 98.01108],
    
    # LR 0.0001
    [110.76494, 104.76683, 103.22501, 101.73743, 101.06946, 100.817024, 100.10096, 100.33682, 100.20386, 99.92931, 99.81411, 99.737526, 99.26319, 99.14252, 99.05326, 98.847855, 98.79031, 98.57224, 98.44957, 98.41456, 98.47076, 98.35497, 98.13642, 98.05589, 97.91163, 97.681335, 97.54658, 97.58754, 97.80139, 97.71316, 97.66887, 97.53166, 97.19124, 97.336296, 97.17356, 96.97161, 96.837204, 96.77293, 96.85009, 96.73103, 96.33688, 96.391945, 96.23451, 96.24377, 96.346405, 96.23491, 96.20153, 96.22911, 95.98245, 95.95312, 95.842224, 95.87049, 96.01949, 95.90624, 95.981514, 95.829544, 95.97368, 95.74474, 95.69406, 95.55398, 95.752304, 95.453835, 95.37662, 95.45198, 95.31463, 95.33952, 95.31595, 95.4147, 95.44998, 95.61402],
    
    # LR 0.0001 (zweite Spalte)
    [111.59001, 104.807976, 102.74609, 101.901276, 101.14665, 100.67665, 100.20994, 99.95926, 99.72258, 99.487656, 99.3598, 99.184616, 99.0521, 98.82778, 98.7524, 98.62144, 98.57313, 98.57242, 98.49208, 98.24889, 98.17086, 98.16899, 98.301346, 97.932556, 97.66026, 97.658455, 97.61104, 97.446785, 97.322296, 97.27461, 96.95251, 97.1554, 97.10045, 97.0382, 97.16658, 96.89252, 96.613106, 97.42867, 96.90678, 96.50509],
    
    # LR 0.00001
    [115.60013, 114.13953, 111.07326, 110.04098, 108.15566, 106.649475, 105.90178, 105.05805, 104.34326, 103.779, 103.25347, 102.92891, 102.53587, 102.22712, 101.90452, 101.673996, 101.40688, 101.36227, 101.09671, 101.07431, 100.88128, 100.73557, 100.48435, 100.44914, 100.48559, 100.514854, 100.445175, 100.19924, 100.12667, 100.12685, 100.08719, 99.94527, 99.896126, 99.82747, 99.78317, 99.63339, 99.631485, 99.52601, 99.49924, 99.38069],
    
    # LR 0.000001
    [120.59029, 132.06792, 130.17447, 127.66563, 125.38408, 124.04262, 122.62452, 120.9682, 119.78976, 118.505005, 117.44937, 116.778275, 116.14664, 115.52773, 114.99539, 114.45889, 113.937935, 113.60342, 113.33457, 112.977844, 112.53668, 112.31247, 111.70987, 111.66034, 111.343254, 111.00992, 110.67369, 110.36374, 110.112495, 109.73943, 109.54539, 109.18452, 108.8794, 108.65381, 108.427284, 108.21206, 107.9766, 107.74949, 107.53199, 107.3267]
]

# Plot erstellen
p = plot(size=(1000, 600), dpi=300)

# Farben für verschiedene Lernraten
colors = [:red, :blue, :green, :orange, :purple]
linestyles = [:solid, :dash, :dot, :dashdot, :dashdotdot]

# Für jede Lernrate eine Linie plotten
for (i, (lr_label, losses)) in enumerate(zip(lr_labels, loss_data))
    epochs = 1:length(losses)
    plot!(p, epochs, losses, 
          label="LR = $lr_label", 
          color=colors[i],
          linestyle=linestyles[i],
          linewidth=2,
          marker=:circle,
          markersize=3)
end

# Plot-Eigenschaften
plot!(p,
    title="Training Loss über Epochen für verschiedene Lernraten",
    xlabel="Epoche",
    ylabel="Loss",
    legend=:topright,
    grid=true,
    gridwidth=1,
    gridcolor=:lightgray,
    background_color=:white,
    foreground_color=:black,
    titlefontsize=14,
    labelfontsize=12,
    legendfontsize=10)

# Y-Achse begrenzen für bessere Lesbarkeit
ylims!(p, (90, 135))

# Mean IoU Daten (finale Evaluierung)
mean_iou_data = [
    ("0.001", 0.3226),      # 001.bson
    ("0.0001 (70 Epochen)", 0.6262), # 0001_70.bson
    ("0.0001", 0.5355),     # 0001.bson  
    ("0.00001", 0.3542),    # 00001.bson
    ("0.000001", 0.2083)    # 000001.bson
]

# Zweiter Plot für Mean IoU
p2 = plot(size=(800, 500), dpi=300)

# Lernraten und IoU-Werte extrahieren
lr_names = [data[1] for data in mean_iou_data]
iou_values = [data[2] for data in mean_iou_data]

# Balkendiagramm für Mean IoU
bar!(p2, lr_names, iou_values,
     color=[:red, :blue, :green, :orange, :purple],
     alpha=0.7,
     title="Mean IoU nach Lernrate",
     xlabel="Lernrate",
     ylabel="Mean IoU",
     legend=false,
     grid=true,
     gridwidth=1,
     gridcolor=:lightgray,
     titlefontsize=14,
     labelfontsize=12)

# IoU-Werte auf den Balken anzeigen
for (i, (lr, iou)) in enumerate(mean_iou_data)
    annotate!(p2, i, iou + 0.02, text(string(round(iou, digits=4)), :center, 10))
end

# Y-Achse auf sinnvollen Bereich begrenzen
ylims!(p2, (0, 0.7))

# Kombinierter Plot mit beiden Diagrammen nebeneinander
combined_plot = plot(p, p2, layout=(1, 2), size=(1400, 600))

# Titel für den gesamten Plot hinzufügen
plot!(combined_plot, plot_title="Julia U-Net Training: Lernraten-Analyse und Performance-Evaluation", 
      plot_titlefontsize=16)

# Nur den kombinierten Plot anzeigen
display(combined_plot)

# Optional: Plots speichern
# savefig(p, "training_loss_comparison.png")
# savefig(p2, "mean_iou_comparison.png")
# savefig(combined_plot, "complete_training_analysis.png")

println("Plots erstellt!")
println("\n=== TRAINING LOSS ANALYSE ===")
println("- LR 0.0001: Beste Konvergenz und niedrigste finale Verluste")
println("- LR 0.001: Schnelle anfängliche Konvergenz, aber höhere finale Verluste")
println("- LR 0.00001: Langsamere aber stetige Konvergenz")
println("- LR 0.000001: Sehr langsame Konvergenz, höchste Verluste")

println("\n=== MEAN IoU ERGEBNISSE ===")
for (lr, iou) in mean_iou_data
    println("- LR $lr: Mean IoU = $(round(iou, digits=4))")
end

println("\n=== WICHTIGSTE ERKENNTNISSE ===")
println("✓ Beste Performance: LR 0.0001 mit 70 Epochen (IoU: 0.6262)")
println("✓ Längeres Training verbessert deutlich die Performance (0.5355 → 0.6262)")
println("✓ Zu hohe/niedrige Lernraten führen zu schlechteren Ergebnissen")
println("✓ Klare Korrelation zwischen Loss-Konvergenz und finaler IoU-Performance")