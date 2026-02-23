# Datei: generate_data.jl
# Daten-Generierung für ψ-FNO Training
#
# Erzeugt N Samples mit zufälligen Kristallkonfigurationen:
#   1. Zufällige Kristalle (Positionen, Radien, Anzahl)
#   2. LaMEM-Simulation → (phase, Vx, Vz, ω, ψ)
#   3. Normalisierung (Gauge-Fix + Scale)
#   4. Input-Kanäle bauen (Mask + SDF)
#   5. Als JLD2 speichern
#
# Nutzung:
#   julia generate_data.jl              # 500 Samples mit Defaults
#   julia generate_data.jl --n 100      # 100 Samples
#
# Oder interaktiv:
#   include("generate_data.jl")
#   generate_all(; n_samples=500, out_dir="data_train")

using Random, Statistics, Printf
using JLD2

# ── Projekt laden ──
include("FNOPsi.jl")
using .FNOPsi
using .FNOPsi.LaMEMInterface: run_sinking_crystals
using .FNOPsi.Normalization: gauge_fix_psi, compute_scale, normalize_psi
using .FNOPsi.GridFDUtils: grid_spacing, normalize_coords

# =============================================================================
# Hilfsfunktionen für Kristall-Konfigurationen
# =============================================================================

"""
    random_crystal_config(n_crystals; domain=(-0.7, 0.7), r_range=(0.05, 0.15),
                          min_gap=0.02, max_attempts=200)

Erzeugt zufällige Kristallkonfiguration mit `n_crystals` Kristallen.
Kristalle überlappen nicht und haben Mindestabstand `min_gap`.

Returns: `(centers_2D, radii)`.
"""
function random_crystal_config(n_crystals::Int;
                               domain::Tuple{Float64,Float64} = (-0.8, 0.8),
                               r_range::Tuple{Float64,Float64} = (0.05, 0.05),
                               min_gap::Float64 = 0.02,
                               max_attempts::Int = 200000)
    lo, hi = domain
    r_min, r_max = r_range
    centers = Tuple{Float64,Float64}[]
    radii = Float64[]

    for ci in 1:n_crystals
        placed = false
        for _ in 1:max_attempts
            r = r_min + rand() * (r_max - r_min)
            cx = lo + r + rand() * (hi - lo - 2r)
            cz = lo + r + rand() * (hi - lo - 2r)

            # Überlappungscheck
            ok = true
            for (j, (ox, oz)) in enumerate(centers)
                dist = sqrt((cx - ox)^2 + (cz - oz)^2)
                if dist < r + radii[j] + min_gap
                    ok = false
                    break
                end
            end

            if ok
                push!(centers, (cx, cz))
                push!(radii, r)
                placed = true
                break
            end
        end

        if !placed
            @warn "Konnte Kristall $ci nicht platzieren nach $max_attempts Versuchen"
        end
    end

    return centers, radii
end

"""
    compute_sdf(x_vec, z_vec, centers, radii)

Berechnet Signed Distance Function (SDF) für Kreise auf dem Gitter.
- Negativ innerhalb der Kristalle
- Positiv außerhalb
"""
function compute_sdf(x_vec::AbstractVector, z_vec::AbstractVector,
                     centers::Vector{Tuple{Float64,Float64}},
                     radii::Vector{Float64})
    nx, nz = length(x_vec), length(z_vec)
    sdf = fill(Inf, nx, nz)

    for i in 1:nx, j in 1:nz
        for (k, (cx, cz)) in enumerate(centers)
            d = sqrt((x_vec[i] - cx)^2 + (z_vec[j] - cz)^2) - radii[k]
            sdf[i, j] = min(sdf[i, j], d)
        end
    end

    return Float32.(sdf)
end

"""
    compute_nearest_radius(x_vec, z_vec, centers, radii)

Berechnet für jeden Gitterpunkt den Radius des nächstgelegenen Kristalls.
Fluidpunkte weit von allen Kristallen erhalten den Radius des nächsten Kristalls
(gemessen am Abstand zum Mittelpunkt).
"""
function compute_nearest_radius(x_vec::AbstractVector, z_vec::AbstractVector,
                                centers::Vector{Tuple{Float64,Float64}},
                                radii::Vector{Float64})
    nx, nz = length(x_vec), length(z_vec)
    R = Array{Float32}(undef, nx, nz)

    for i in 1:nx, j in 1:nz
        best_dist = Inf
        best_r    = 0.0
        for (k, (cx, cz)) in enumerate(centers)
            d = sqrt((x_vec[i] - cx)^2 + (z_vec[j] - cz)^2)
            if d < best_dist
                best_dist = d
                best_r    = radii[k]
            end
        end
        R[i, j] = Float32(best_r)
    end

    return R
end

"""
    build_input_tensor(phase, x_vec, z_vec, centers, radii)

Baut den Input-Tensor (nx, nz, 3) aus:
- Kanal 1: Phasen-Maske (1 = Kristall, 0 = Fluid)
- Kanal 2: Signed Distance Function (SDF)
- Kanal 3: Radius des nächstgelegenen Kristalls (R)
"""
function build_input_tensor(phase::AbstractMatrix,
                            x_vec::AbstractVector, z_vec::AbstractVector,
                            centers::Vector{Tuple{Float64,Float64}},
                            radii::Vector{Float64})
    nx, nz = size(phase)
    mask = Float32.(phase .== 1)  # 1 = Kristall
    sdf  = compute_sdf(x_vec, z_vec, centers, radii)
    R    = compute_nearest_radius(x_vec, z_vec, centers, radii)

    input = Array{Float32}(undef, nx, nz, 3)
    input[:, :, 1] .= mask
    input[:, :, 2] .= sdf
    input[:, :, 3] .= R
    return input
end

# =============================================================================
# Einzelnes Sample generieren
# =============================================================================

"""
    generate_one_sample(n_crystals; nx=256, nz=256, η=1e20, Δρ=200,
                        gauge_mode=:global_mean, scale_strategy=:powmean)

Generiert ein Sample:
1. Zufällige Kristallkonfiguration
2. LaMEM-Simulation
3. ψ-Normalisierung
4. Input-Tensor + Meta

Returns: `(; input, ψ_norm, scale, meta)` – fertig zum Speichern als JLD2.
"""
function generate_one_sample(n_crystals::Int;
                             nx::Int = 256, nz::Int = 256,
                             η::Real = 1e20, Δρ::Real = 200,
                             gauge_mode::Symbol = :global_mean,
                             scale_strategy::Symbol = :powmean)
    # 1. Zufällige Kristalle
    centers, radii = random_crystal_config(n_crystals)

    # 2. LaMEM-Simulation
    res = run_sinking_crystals(; nx=nx, nz=nz, η=η, Δρ=Δρ,
                                 centers_2D=centers, radii=radii)

    # 3. ψ-Normalisierung
    ψ_fixed, offset = gauge_fix_psi(res.ψ; phase=res.phase, mode=gauge_mode)
    scale, norm_meta = compute_scale(ψ_fixed; strategy=scale_strategy)
    ψ_norm = normalize_psi(ψ_fixed, scale)

    # 4. Input-Tensor
    input = build_input_tensor(res.phase, res.x_vec_1D, res.z_vec_1D,
                               centers, radii)

    # 5. Meta-Daten
    meta = (
        n_crystals   = n_crystals,
        centers_2D   = centers,
        radii        = radii,
        η            = Float64(η),
        Δρ           = Float64(Δρ),
        nx           = nx,
        nz           = nz,
        x_vec_1D     = res.x_vec_1D,
        z_vec_1D     = res.z_vec_1D,
        psi_gauge_offset = offset,
        norm_scale   = scale,
        norm_strategy = scale_strategy,
    )

    return (; input, ψ_norm=Float32.(ψ_norm), scale, meta)
end

# =============================================================================
# Batch-Generierung
# =============================================================================

"""
    generate_all(; n_samples=500, out_dir="data_train",
                   n_crystals_range=1:4, nx=256, nz=256,
                   η=1e20, Δρ=200, seed=42)

Generiert `n_samples` Samples mit zufälliger Kristallanzahl.
Speichert in Unterordnern: out_dir/n01/, out_dir/n02/, ...

Jedes Sample wird als JLD2 gespeichert mit Keys:
  `input`, `ψ_norm`, `scale`, `meta`
"""
function generate_all(; n_samples::Int = 500,
                        out_dir::AbstractString = "data_train",
                        n_crystals_range::AbstractVector{Int} = collect(1:1),
                        nx::Int = 256, nz::Int = 256,
                        η::Real = 1e20, Δρ::Real = 200,
                        seed::Int = 42)
    Random.seed!(seed)
    mkpath(out_dir)

    # Unterordner anlegen + bestehende Samples zählen (Append-Modus)
    counters = Dict{Int,Int}()
    for nc in n_crystals_range
        subdir = joinpath(out_dir, @sprintf("n%02d", nc))
        mkpath(subdir)
        existing = count(f -> endswith(f, ".jld2"), readdir(subdir))
        counters[nc] = existing
    end

    n_existing = sum(values(counters))
    if n_existing > 0
        @info "Append-Modus: $n_existing bestehende Samples gefunden, starte ab dort"
    end

    n_success = 0
    n_fail = 0

    @info "Starte Generierung: $n_samples Samples → $out_dir"
    @info "Kristallanzahlen: $n_crystals_range, Grid: $(nx)×$(nz)"

    for i in 1:n_samples
        # Zufällige Kristallanzahl
        nc = rand(n_crystals_range)

        try
            sample = generate_one_sample(nc; nx=nx, nz=nz, η=η, Δρ=Δρ)

            # Dateiname
            counters[nc] += 1
            subdir = @sprintf("n%02d", nc)
            fname = @sprintf("sample_%06d.jld2", counters[nc])
            fpath = joinpath(out_dir, subdir, fname)

            # Speichern
            jldsave(fpath;
                    input  = sample.input,
                    ψ_norm = sample.ψ_norm,
                    scale  = sample.scale,
                    meta   = sample.meta)

            n_success += 1

            if i % max(1, n_samples ÷ 20) == 0
                @info @sprintf("[%d/%d] ✓ nc=%d  ψ_norm: [%.2f, %.2f]  scale=%.2e",
                               i, n_samples, nc,
                               minimum(sample.ψ_norm), maximum(sample.ψ_norm),
                               sample.scale)
            end

        catch e
            n_fail += 1
            @warn "Sample $i fehlgeschlagen (nc=$nc): $e"
            if n_fail > n_samples ÷ 5
                @error "Zu viele Fehler ($n_fail) – Abbruch"
                break
            end
        end
    end

    # Zusammenfassung
    println("\n" * "=" ^ 60)
    println("Generierung abgeschlossen")
    println("=" ^ 60)
    @printf("  Erfolg: %d / %d\n", n_success, n_samples)
    @printf("  Fehler: %d\n", n_fail)
    for nc in sort(collect(keys(counters)))
        @printf("  n_crystals=%d: %d Samples\n", nc, counters[nc])
    end
    println("  Ausgabe: $out_dir")
    println("=" ^ 60)

    return (; n_success, n_fail, counters)
end

# =============================================================================
# CLI Entry Point
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    local n = 500
    local out = "data_train"
    local nc_range = collect(1:4)
    local seed = 42

    # Argument-Parsing
    for i in eachindex(ARGS)
        if (ARGS[i] == "--n" || ARGS[i] == "--n_samples") && i < length(ARGS)
            n = parse(Int, ARGS[i+1])
        elseif (ARGS[i] == "--out" || ARGS[i] == "--out_dir") && i < length(ARGS)
            out = ARGS[i+1]
        elseif ARGS[i] == "--n_crystals_range" && i < length(ARGS)
            # z.B. "1:4" oder "1:1"
            parts = split(ARGS[i+1], ":")
            nc_range = collect(parse(Int, parts[1]):parse(Int, parts[end]))
        elseif ARGS[i] == "--seed" && i < length(ARGS)
            seed = parse(Int, ARGS[i+1])
        end
    end

    generate_all(; n_samples=n, out_dir=out, n_crystals_range=nc_range, seed=seed)
end
