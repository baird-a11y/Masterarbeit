
module DataGenerationPsi

using Random
using JLD2
using Printf
using Statistics
using ..LaMEMInterface: run_sinking_crystals

"""
    build_input_channels(phase, x_vec_1D, z_vec_1D, centers_2D)

Erstellt Eingabekanäle für das U-Net:

- Kanal 1: Kristallmaske (phase == 1) ∈ {0,1}
- Kanal 2: Signed Distance Field (SDF) zum nächsten Kristallzentrum, ca. in [-1, 1]

x_vec_1D, z_vec_1D und centers_2D sind in km.
"""
function build_input_channels(phase, x_vec_1D, z_vec_1D, centers_2D)
    nx, nz = size(phase)
    input = Array{Float32}(undef, nx, nz, 2)

    # Kanal 1: Binärmaske
    @inbounds input[:, :, 1] .= phase .== 1

    # Kanal 2: Distanz zum nächsten Kristallzentrum (normiert) → SDF
    dist = Array{Float32}(undef, nx, nz)

    @inbounds for ix in 1:nx
        x = x_vec_1D[ix]
        for iz in 1:nz
            z = z_vec_1D[iz]

            dmin2 = typemax(Float32)
            for (cx, cz) in centers_2D
                dx = x - cx
                dz = z - cz
                d2 = dx*dx + dz*dz
                d2 < dmin2 && (dmin2 = d2)
            end
            dist[ix, iz] = sqrt(dmin2)
        end
    end

    # Normierung auf [0,1]
    maxdist = maximum(dist)
    dist_norm = dist ./ maxdist

    # Signed Distance: innen negativ, außen positiv
    sdf = similar(dist_norm)
    @inbounds for ix in 1:nx, iz in 1:nz
        if phase[ix, iz] == 1
            sdf[ix, iz] = -dist_norm[ix, iz]  # innen: [-1, 0]
        else
            sdf[ix, iz] =  dist_norm[ix, iz]  # außen: [0, 1]
        end
    end

    input[:, :, 2] .= sdf

    return input
end



# Globale Skala für ψ (ggf. nachjustieren)
const GLOBAL_PSI_SCALE = 1e13  # z.B. 10^13 → ψ_norm ~ O(1)

"""
    normalize_psi(ψ)

Skaliert ψ mit einem GLOBAL_PSI_SCALE-Faktor, der für
den gesamten Datensatz gilt. Gibt (ψ_norm, scale) zurück
mit ψ_norm = ψ * scale.
"""
function normalize_psi(ψ)
    ψ_norm = ψ .* GLOBAL_PSI_SCALE
    scale  = GLOBAL_PSI_SCALE
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
            x_min = -0.9,  # wie bisherige Domäne
            x_max =  0.9,
            z_min = -0.9,
            z_max =  0.9,
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

    x_vec_1D = result.x_vec_1D
    z_vec_1D = result.z_vec_1D

    phase = result.phase
    ψ     = result.ψ

    # Eingabekanäle bauen: Maske + Distanzkanal
    input = build_input_channels(phase, x_vec_1D, z_vec_1D, centers_2D)


    # ψ normalisieren
    ψ_norm, scale = normalize_psi(ψ)

    # Meta-Info: Anzahl, Zentren, Radien
    meta = (
        n_crystals = n_crystals,
        centers_2D = centers_2D,
        radii      = radii,
        x_vec_1D   = x_vec_1D,
        z_vec_1D   = z_vec_1D,
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

    # ---------------------------------------------------------
    # NEU: Letzten bestehenden Index im Ordner finden
    # ---------------------------------------------------------
    existing = filter(f -> occursin(r"psi_sample_\d{4}\.jld2", f),
                      readdir(outdir))

    last_index = 0
    if !isempty(existing)
        # extrahiere die Nummern
        indices = map(existing) do fname
            m = match(r"psi_sample_(\d{4})\.jld2", fname)
            parse(Int, m.captures[1])
        end
        last_index = maximum(indices)
    end

    start_index = last_index + 1
    end_index   = last_index + n_train

    @info "Beginne Datengenerierung bei Index = $start_index (insgesamt $n_train Samples)."

    # ---------------------------------------------------------
    # Generierung wie vorher, aber mit neuem Nummernbereich
    # ---------------------------------------------------------
    Threads.@threads for k in 0:(n_train-1)
        i = start_index + k

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


"""
    estimate_global_psi_scale(; n_samples=20, rng=Random.default_rng(),
                               nx=256, nz=256,
                               η=1e20, Δρ=200.0,
                               min_crystals=1, max_crystals=1,
                               radius_mode::Symbol = :fixed,
                               R_fixed=0.1, R_min=0.05, R_max=0.15)

Führt n_samples zufällige LaMEM-Simulationen aus (wie generate_psi_sample),
sammelt alle |ψ|-Werte und schätzt daraus einen globalen Skalenfaktor:

    GLOBAL_PSI_SCALE ≈ 10^(-p_mean_global)

mit p_mean_global = Mittelwert der log10(|ψ|) über alle Samples.
"""
function estimate_global_psi_scale(; n_samples::Int = 20,
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

    

    exponents_all = Float64[]

    for k in 1:n_samples
        # --- Zufällige Konfiguration wie in generate_psi_sample ---
        n_crystals = rand(rng, min_crystals:max_crystals)

        centers_2D = Vector{Tuple{Float64,Float64}}(undef, n_crystals)
        radii      = Vector{Float64}(undef, n_crystals)

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
                x_min = -0.9,
                x_max =  0.9,
                z_min = -0.9,
                z_max =  0.9,
            )

            centers_2D[i] = (cx, cz)
        end

        # --- LaMEM laufen lassen ---
        result = run_sinking_crystals(; nx=nx,
                                       nz=nz,
                                       η=η,
                                       Δρ=Δρ,
                                       centers_2D=centers_2D,
                                       radii=radii)

        ψ = result.ψ
        absψ = abs.(ψ)
        mask = absψ .> 0

        if any(mask)
            exponents = log10.(absψ[mask])
            append!(exponents_all, exponents)
        end

        @info "Sample $k / $n_samples für GLOBAL_PSI_SCALE ausgewertet."
    end

    if isempty(exponents_all)
        error("Keine nicht-null ψ-Werte gefunden – GLOBAL_PSI_SCALE kann nicht geschätzt werden.")
    end

    p_mean_global = mean(exponents_all)
    global_scale  = 10.0^(-p_mean_global)

    @info "Geschätzter mittlerer Exponent p_mean_global = $p_mean_global"
    @info "Vorschlag: GLOBAL_PSI_SCALE ≈ $global_scale"

    return global_scale, p_mean_global
end

end # module
