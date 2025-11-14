module DataGenerationVel

using Random
using JLD2
using Printf
using Statistics

using ..LaMEMInterface: run_sinking_crystals

"""
    build_input_channels(phase)

Wie bei Ansatz 2: aktuell nur Kristallmaske (phase == 1) als ein Kanal.
"""
function build_input_channels(phase)
    nx, nz = size(phase)
    input = Array{Float32}(undef, nx, nz, 1)
    input[:, :, 1] .= phase .== 1
    return input
end

"""
    normalize_velocity(Vx, Vz)

Skaliert Vx, Vz über den mittleren Zehnerexponenten der
Betragsgeschwindigkeit |v|, um numerisch stabile Werte zu bekommen.

Gibt (Vx_norm, Vz_norm, scale) zurück mit:

    Vx_norm = Vx * scale
    Vz_norm = Vz * scale
"""
function normalize_velocity(Vx, Vz)
    # Betraggeschwindigkeit (z.B. in cm/year)
    vmag = sqrt.(Vx.^2 .+ Vz.^2)
    mask = vmag .> 0

    if !any(mask)
        return Vx, Vz, 1.0
    end

    exponents = log10.(vmag[mask])
    p_mean    = mean(exponents)
    scale     = 10.0^(-p_mean)

    Vx_norm = Vx .* scale
    Vz_norm = Vz .* scale

    return Vx_norm, Vz_norm, scale
end

"""
    generate_vel_sample(rng; ...)

Erzeugt ein einzelnes Sample für Ansatz 1.

Rückgabe:
    input    :: Array{Float32,3}  (nx, nz, in_channels)   – Kristallmaske
    v_norm   :: Array{Float32,3}  (nx, nz, 2)             – (Vx_norm, Vz_norm)
    scale_v  :: Float64                                    – Normierung
    meta     :: NamedTuple mit Geometrie-Infos
"""
function generate_vel_sample(rng;
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

    # Wir nutzen deine bestehende Logik aus Ansatz 2 → am besten
    # denselben Platziercode aus DataGenerationPsi übernehmen ODER
    # du kopierst die sample_center_nonoverlapping-Funktion hierher.
    # Hier nehmen wir an, du kopierst sie 1:1 nach DataGenerationVel
    # und rufst sie genauso auf:
    for i in 1:n_crystals
        r_i = if radius_mode == :fixed
            R_fixed
        elseif radius_mode == :range
            rand(rng) * (R_max - R_min) + R_min
        else
            error("Unbekannter radius_mode = $radius_mode. Erlaubt: :fixed oder :range")
        end
        radii[i] = r_i

        existing_centers = i == 1 ? Tuple{Float64,Float64}[] : centers_2D[1:i-1]
        existing_radii   = i == 1 ? Float64[]                : radii[1:i-1]

        cx, cz = sample_center_nonoverlapping(
            rng,
            existing_centers,
            existing_radii,
            r_i;
            x_min = -0.5,
            x_max =  0.5,
            z_min = -0.5,
            z_max =  0.5,
        )

        centers_2D[i] = (cx, cz)
    end

    # --- LaMEM-Simulation (wie bei Ansatz 2) ---
    result = run_sinking_crystals(; nx=nx,
                                   nz=nz,
                                   η=η,
                                   Δρ=Δρ,
                                   centers_2D=centers_2D,
                                   radii=radii)

    x_vec_1D = result.x_vec_1D
    z_vec_1D = result.z_vec_1D
    phase    = result.phase
    Vx       = result.Vx
    Vz       = result.Vz

    # Eingabekanäle (Kristallmaske)
    input = build_input_channels(phase)

    # Geschwindigkeiten normalisieren
    Vx_norm, Vz_norm, scale_v = normalize_velocity(Vx, Vz)

    # Ziel als (nx, nz, 2)-Tensor
    nxg, nzg = size(Vx_norm)
    v_norm = Array{Float32}(undef, nxg, nzg, 2)
    v_norm[:, :, 1] .= Vx_norm
    v_norm[:, :, 2] .= Vz_norm

    meta = (
        n_crystals = n_crystals,
        centers_2D = centers_2D,
        radii      = radii,
        x_vec_1D   = x_vec_1D,
        z_vec_1D   = z_vec_1D,
    )

    return input, v_norm, scale_v, meta
end

"""
    generate_dataset(outdir; ...)

Erzeugt n_train Samples und speichert sie als .jld2-Dateien
für Ansatz 1 (Geschwindigkeiten).
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
        input, v_norm, scale_v, meta = generate_vel_sample(
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

        filename = joinpath(outdir, @sprintf("vel_sample_%04d.jld2", i))
        @save filename input v_norm scale_v meta
        @info "Vel-Sample $i gespeichert → $filename (n_crystals=$(meta.n_crystals))"
    end
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

end # module
