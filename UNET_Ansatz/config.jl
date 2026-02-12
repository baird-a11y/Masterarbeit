# Datei: config.jl
# Zentrale Konfiguration für UNet-ψ
# Hyperparameter (nx, nz, base_channels, depth, lr, batch, epochs)
# Pfade (data_root, train/val/test dirs)
# Schalter: use_gpu, save_plots, gauge_fix, norm_strategy
# Output: Config struct + load_config()

# Definition des Config-Structs
Base.@kwdef struct Config
    # Grid-Parameter
    nx::Int            # Gittergröße in x-Richtung
    nz::Int            # Gittergröße in z-Richtung

    # Modell-Architektur
    base_channels::Int # Anzahl der Kanäle in der ersten Schicht (z.B. 32)
    depth::Int         # Tiefe des U-Net (Anzahl der Down/Up-Blöcke)

    # Training-Hyperparameter
    lr::Float64        # Lernrate
    batch_size::Int    # Batchgröße
    epochs::Int        # Anzahl der Trainings-Epochen
    max_grad_norm::Float64  # Gradient Clipping Norm

    # Loss-Gewichte
    α_grad::Float32    # Gewicht für Gradienten-Loss
    α_bnd::Float32     # Gewicht für Rand-Loss
    warmup_epochs::Int # Anzahl Epochen für α_grad Warmup

    # Pfade
    train_dir::String  # Verzeichnis für Trainingsdaten
    val_dir::String    # Verzeichnis für Validierungsdaten

    # Schalter
    use_gpu::Bool      # Ob GPU verwendet werden soll (true/false)
    save_plots::Bool   # Ob Plots gespeichert werden sollen (true/false)
    gauge_fix::Bool    # Ob die Eichkorrektur angewendet werden soll (true/false)
    norm_strategy::String  # Normalisierungsstrategie ("standard", "minmax", "none")

    # Dataset-Optionen
    use_coords::Bool   # Ob Koordinatenkanäle (x, z) hinzugefügt werden sollen
    coord_range::Symbol # :pm1 für [-1,1], :phys für physikalische Koordinaten
end

# Funktion zum Laden der Konfiguration
function load_config()
    # Hier können die Werte entweder hartkodiert oder aus einer Datei (z.B. JSON, YAML) geladen werden
    return Config(
        # Grid
        nx=256,
        nz=256,

        # Modell
        base_channels=32,
        depth=4,

        # Training
        lr=5e-5,
        batch_size=8,
        epochs=300,
        max_grad_norm=1.0,

        # Loss-Gewichte
        α_grad=0.1f0,
        α_bnd=0.01f0,
        warmup_epochs=5,

        # Pfade (Standard - können über CLI überschrieben werden)
        train_dir="/local/home/baselt/src/Daten/data_psi_train",
        val_dir="/local/home/baselt/src/Daten/data_psi_val",

        # Schalter
        use_gpu=true,
        save_plots=true,
        gauge_fix=false,
        norm_strategy="standard",

        # Dataset
        use_coords=true,
        coord_range=:pm1
    )
end
