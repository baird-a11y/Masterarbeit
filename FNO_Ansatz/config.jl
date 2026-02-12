# Datei: config.jl
# zentrale Hyperparameter (nx=nz=256, modes, width, depth, lr, batch, epochs)
# Pfade (data_root, train/val/test dirs)
# Schalter: use_gpu, save_plots, gauge_fix, norm_strategy
# Output: Config struct + load_config()

# Definition des Config-Structs
Base.@kwdef struct Config
    # Hyperparameter
    nx::Int # Gittergröße in x-Richtung
    nz::Int # Gittergröße in z-Richtung
    modes::Int # Anzahl der Fourier-Moden
    width::Int # Breite des Modells
    depth::Int  # Tiefe des Modells
    lr::Float64 # Lernrate
    batch_size::Int # Batchgröße
    epochs::Int # Anzahl der Trainings-Epochen

    # Pfade
    train_dir::String # Verzeichnis für Trainingsdaten
    val_dir::String # Verzeichnis für Validierungsdaten


    # Schalter
    use_gpu::Bool # Ob GPU verwendet werden soll (true/false)
    save_plots::Bool # Ob Plots gespeichert werden sollen (true/false)
    gauge_fix::Bool # Ob die Eichkorrektur angewendet werden soll (true/false)
    norm_strategy::String # Normalisierungsstrategie
end

# Funktion zum Laden der Konfiguration
function load_config()
    # Hier können die Werte entweder hartkodiert oder aus einer Datei (z.B. JSON, YAML) geladen werden
    return Config(
        nx=256,
        nz=256,
        modes=16,
        width=64,
        depth=4,
        lr=0.001,
        batch_size=32,
        epochs=100,
        train_dir= "/local/home/baselt/src/FNO_BASIS/data_train",
        val_dir="/local/home/baselt/src/FNO_BASIS/data_val",
        use_gpu=true,
        save_plots=true,
        gauge_fix=false,
        norm_strategy="standard"
    )
end