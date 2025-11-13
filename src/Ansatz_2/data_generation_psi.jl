module DataGenerationPsi

using Random
using JLD2
using Printf
using Statistics
using ..LaMEMInterface: run_sinking_crystals

"""
    build_input_channels(phase)

Erstellt Eingabekanäle für das U-Net.
Aktuell: nur Kristallmaske (phase == 1) als ein Kanal.
"""
function build_input_channels(phase)
    nx, nz = size(phase)
    input = Array{Float32}(undef, nx, nz, 1)
    input[:, :, 1] .= phase .== 1
    return input
end

"""
    normalize_psi(ψ)

Skaliert ψ über den mittleren Zehnerexponenten, um numerisch stabile Werte zu erhalten.
Gibt (ψ_norm, scale) zurück mit ψ_norm = ψ * scale.
"""
function normalize_psi(ψ)
    absψ = abs.(ψ)
    mask = absψ .> 0
    if !any(mask)
        return ψ, 1.0
    end
    exponents = log10.(absψ[mask])
    p_mean    = mean(exponents)
    scale     = 10.0^(-p_mean)
    ψ_norm    = ψ .* scale
    return ψ_norm, scale
end

"""
    sample_center_nonoverlapping(rng, centers, radii, r;
                                 x_min=-0.5, x_max=0.5,
                                 z_min=-0.5, z_max=0.5,
                                 max_tries=1000)

Zieht eine zufällige Kristallposition (cx, cz), so dass

  1. der Kristall komplett im Rechteck [x_min, x_max] × [z_min, z_max] liegt
  2. kein Überlappen mit bestehenden Kreisen in `centers` und `radii` auftritt.

Wir verlangen: Distanz^2 ≥ (r + r_j)^2 für alle vorhandenen j.
"""
function sample_center_nonoverlapping(rng,
                                      centers::Vector{Tuple{Float64,Float64}},
                                      radii::Vector{Float64},
                                      r::Float64;
                                      x_min::Float64 = -0.5,
                                      x_max::Float64 =  0.5,
                                      z_min::Float64 = -0.5,
                                      z_max::Float64 =  0.5,
                                      max_tries::Int = 1000)

    # Damit der Kristall vollständig im Fenster liegt:
    inner_x_min = x_min + r
    inner_x_max = x_max - r
    inner_z_min = z_min + r
    inner_z_max = z_max - r

    @assert inner_x_min < inner_x_max "Kristallradius zu groß für x-Domäne"
    @assert inner_z_min < inner_z_max "Kristallradius zu groß für z-Domäne"

    for attempt in 1:max_tries
        cx = rand(rng) * (inner_x_max - inner_x_min) + inner_x_min
        cz = rand(rng) * (inner_z_max - inner_z_min) + inner_z_min

        ok = true
        for ((c_old_x, c_old_z), r_old) in zip(centers, radii)
            dx = cx - c_old_x
            dz = cz - c_old_z
            # Vergleich in Quadraten (schneller, vermeidet sqrt)
            if dx*dx + dz*dz < (r + r_old)^2
                ok = false
                break
            end
        end

        if ok
            return (cx, cz)
        end
    end

    error("Konnte nach $max_tries Versuchen keinen nicht-überlappenden Kristall platzieren (r = $r).")
end

"""
    generate_psi_sample(rng;
                        nx=256, nz=256,
                        η=1e20, Δρ=200.0,
                        min_crystals=1,
                        max_crystals=1,
                        radius_mode::Symbol = :fixed,  # :fixed oder :range
                        R_fixed=0.1,
                        R_min=0.05,
                        R_max=0.15)

Erzeugt ein einzelnes Sample mit 1–n Kristallen.

Rückgabe:
    input   :: Array{Float32,3}  (nx, nz, channels)
    ψ_norm  :: Array{Float64,2}
    scale   :: Float64
    meta    :: NamedTuple (n_crystals, centers_2D, radii)
"""
function generate_psi_sample(rng;
                             nx::Int = 256,
                             nz::Int = 256,
                             η::Float64 = 1e20,
                             Δρ::Float64 = 200.0,
                             min_crystals::Int = 1,
                             max_crystals::Int = 1,
                             radius_mode::Symbol = :fixed,
                             R_fixed::Float64 = 0.1,
                             R_min::Float64 = 0.05,
                             R_max::Float64 = 0.15)

    @assert min_crystals ≥ 1 "min_crystals muss ≥ 1 sein"
    @assert max_crystals ≥ min_crystals "max_crystals muss ≥ min_crystals sein"

    # --- Anzahl Kristalle für dieses Sample ---
    n_crystals = rand(rng, min_crystals:max_crystals)

    centers_2D = Vector{Tuple{Float64,Float64}}(undef, n_crystals)
    radii      = Vector{Float64}(undef, n_crystals)

    for i in 1:n_crystals
        # 1) Radius für diesen Kristall festlegen
        r_i = if radius_mode == :fixed
            R_fixed
        elseif radius_mode == :range
            rand(rng) * (R_max - R_min) + R_min
        else
            error("Unbekannter radius_mode = $radius_mode. Erlaubt: :fixed oder :range")
        end
        radii[i] = r_i

        # 2) Zentrum so ziehen, dass kein Überlappen mit vorherigen Kreisen auftritt
        #    centers_2D[1:(i-1)] und radii[1:(i-1)] sind die bisherigen Kristalle.
        existing_centers = i == 1 ? Tuple{Float64,Float64}[] : centers_2D[1:i-1]
        existing_radii   = i == 1 ? Float64[]                : radii[1:i-1]

        cx, cz = sample_center_nonoverlapping(
            rng,
            existing_centers,
            existing_radii,
            r_i;
            x_min = -0.5,  # wie bisherige Domäne
            x_max =  0.5,
            z_min = -0.5,
            z_max =  0.5,
        )

        centers_2D[i] = (cx, cz)
    end

    # --- LaMEM-Simulation aufrufen ---
    result = run_sinking_crystals(; nx=nx,
                                   nz=nz,
                                   η=η,
                                   Δρ=Δρ,
                                   centers_2D=centers_2D,
                                   radii=radii)

    phase = result.phase
    ψ     = result.ψ

    # Eingabekanäle bauen (aktuell: Kristallmaske)
    input = build_input_channels(phase)

    # ψ normalisieren
    ψ_norm, scale = normalize_psi(ψ)

    # Meta-Info: Anzahl, Zentren, Radien
    meta = (
        n_crystals = n_crystals,
        centers_2D = centers_2D,
        radii      = radii,
    )

    return input, ψ_norm, scale, meta
end

"""
    generate_dataset(outdir;
                     n_train=100,
                     rng=Random.default_rng(),
                     nx=256, nz=256,
                     η=1e20, Δρ=200.0,
                     min_crystals=1,
                     max_crystals=1,
                     radius_mode::Symbol = :fixed,
                     R_fixed=0.1,
                     R_min=0.05,
                     R_max=0.15)

Erzeugt n_train Samples und speichert sie als einzelne .jld2-Dateien im Ordner `outdir`.
"""
function generate_dataset(outdir::String;
                          n_train::Int = 100,
                          rng = Random.default_rng(),
                          nx::Int = 256,
                          nz::Int = 256,
                          η::Float64 = 1e20,
                          Δρ::Float64 = 200.0,
                          min_crystals::Int = 1,
                          max_crystals::Int = 1,
                          radius_mode::Symbol = :fixed,
                          R_fixed::Float64 = 0.1,
                          R_min::Float64 = 0.05,
                          R_max::Float64 = 0.15)

    mkpath(outdir)
    for i in 1:n_train
        input, ψ_norm, scale, meta = generate_psi_sample(
            rng;
            nx=nx, nz=nz,
            η=η, Δρ=Δρ,
            min_crystals=min_crystals,
            max_crystals=max_crystals,
            radius_mode=radius_mode,
            R_fixed=R_fixed,
            R_min=R_min,
            R_max=R_max,
        )

        filename = joinpath(outdir, @sprintf("psi_sample_%04d.jld2", i))
        @save filename input ψ_norm scale meta
        @info "Sample $i gespeichert → $filename (n_crystals=$(meta.n_crystals))"
    end
end

end # module
