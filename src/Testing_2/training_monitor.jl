# =============================================================================
# TRAINING MONITOR - KORRIGIERTE VERSION
# =============================================================================

using Plots
using BSON
using Statistics
using Dates
using Printf
using Flux
using NNlib

# Server-Modus aktivieren
ENV["GKSwstype"] = "100"
gr()

const CHECKPOINT_DIR = "ten_crystal_checkpoints_optimized"
const PLOT_DIR = "training_plots"

# Vereinfachte Version ohne komplexe Typen
function simple_monitor()
    println("=== EINFACHER TRAINING MONITOR ===")
    
    mkpath(PLOT_DIR)
    
    while true
        try
            # Finde neueste Checkpoint-Datei
            files = filter(f -> endswith(f, ".bson"), readdir(CHECKPOINT_DIR, join=true))
            if isempty(files)
                println("Keine Checkpoints gefunden")
                sleep(30)
                continue
            end
            
            # Sortiere nach Änderungszeit
            sort!(files, by=mtime, rev=true)
            latest = files[1]
            
            println("\nLade: $(basename(latest))")
            
            # Versuche zu laden
            data = try
                BSON.load(latest, @__MODULE__)
            catch e
                println("Fehler beim Laden: $e")
                sleep(30)
                continue
            end
            
            # Extrahiere Losses
            train_losses = get(data, :train_losses, Float32[])
            val_losses = get(data, :val_losses, Float32[])
            physics_losses = get(data, :physics_losses, Float32[])
            
            if isempty(train_losses)
                println("Keine Training-Daten gefunden")
                sleep(30)
                continue
            end
            
            # Status ausgeben
            println("\n" * "="^50)
            println("TRAINING STATUS")
            println("="^50)
            println("Epochen: $(length(train_losses))")
            
            if !isempty(train_losses)
                last_train = train_losses[end]
                println("Letzter Train Loss: $(isfinite(last_train) ? round(last_train, digits=6) : "Inf")")
            end
            
            if !isempty(val_losses)
                last_val = val_losses[end]
                println("Letzter Val Loss: $(isfinite(last_val) ? round(last_val, digits=6) : "Inf")")
                
                # Bestes Modell
                finite_vals = filter(isfinite, val_losses)
                if !isempty(finite_vals)
                    best_val = minimum(finite_vals)
                    println("Bester Val Loss: $(round(best_val, digits=6))")
                end
            end
            
            if !isempty(physics_losses)
                last_physics = physics_losses[end]
                println("Letzter Physics Loss: $(isfinite(last_physics) ? round(last_physics, digits=6) : "Inf")")
            end
            
            # Einfacher Plot
            println("\nErstelle Plot...")
            
            epochs = 1:length(train_losses)
            
            # Ersetze Inf mit NaN für Plotting
            plot_train = replace(x -> isfinite(x) ? x : NaN, train_losses)
            plot_val = replace(x -> isfinite(x) ? x : NaN, val_losses)
            plot_physics = replace(x -> isfinite(x) ? x : NaN, physics_losses)
            
            # Erstelle Plot
            p = plot(epochs, plot_train, 
                    label="Train", 
                    linewidth=2,
                    title="Training Progress",
                    xlabel="Epoch",
                    ylabel="Loss",
                    legend=:topright)
            
            if !isempty(plot_val)
                plot!(p, epochs[1:length(plot_val)], plot_val, 
                     label="Validation", 
                     linewidth=2)
            end
            
            if !isempty(plot_physics)
                plot!(p, epochs[1:length(plot_physics)], plot_physics, 
                     label="Physics", 
                     linewidth=2,
                     yscale=:log10)
            end
            
            # Speichere Plot
            plot_file = joinpath(PLOT_DIR, "training_progress.png")
            savefig(p, plot_file)
            println("Plot gespeichert: $plot_file")
            
            # CSV Export
            csv_file = joinpath(PLOT_DIR, "metrics.csv")
            open(csv_file, "w") do f
                println(f, "Epoch,Train,Val,Physics")
                for i in epochs
                    t = i <= length(train_losses) ? train_losses[i] : ""
                    v = i <= length(val_losses) ? val_losses[i] : ""
                    p = i <= length(physics_losses) ? physics_losses[i] : ""
                    println(f, "$i,$t,$v,$p")
                end
            end
            println("CSV gespeichert: $csv_file")
            
            println("="^50)
            
        catch e
            println("Fehler: $e")
        end
        
        println("\nWarte 30 Sekunden...")
        sleep(30)
    end
end

# Starte Monitor
simple_monitor()