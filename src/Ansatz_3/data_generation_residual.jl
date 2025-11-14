module DataGenerationResidual

using Random
using JLD2
using Printf
using Statistics

using ..LaMEMInterface: run_sinking_crystals

const g_gravity = 9.81  # m/s²

# ------------------------------------------------------------
# Hilfsfunktionen (analog zu Ansatz 2, aber lokal kopiert)
# ------------------------------------------------------------

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
    sample_center_nonoverlapping(rng, centers, radii, r; ...)

Zieht eine zufällige Kristallposition (cx, cz), so dass

  1. der Kristall komplett im Rechteck [x_min, x_max] × [z_min, z_max] liegt
  2. kein Überlappen mit bestehenden Kreisen in `centers` und `radii` auftritt.
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

# ------------------------------------------------------------
# Analytische Referenz (Stokes-Sinkgeschwindigkeit)
# ------------------------------------------------------------

"""
    analytic_streamfunction_uniform(x_vec_1D, z_vec_1D, radii; Δρ, η, g=9.81)

Erzeugt eine einfache analytische Stromfunktion für eine
gleichförmige vertikale Sinkbewegung mit Stokes-Geschwindigkeit.

- nutzt mittleren Radius R_mean (aus `radii`)
- Radii (km) → Meter
- U = Δρ g R_mean^2 / (4η)   [m/s]
- ψ_analytisch(x,z) = U * x_phys

Rückgabe:
    ψ_analytic :: Array{Float64,2}  (nx, nz)
    U          :: Float64           (Stokes-Geschwindigkeit in m/s)
"""
function analytic_streamfunction_uniform(x_vec_1D::AbstractVector,
                                         z_vec_1D::AbstractVector,
                                         radii::AbstractVector;
                                         Δρ::Real,
                                         η::Real,
                                         g::Real = g_gravity)

    @assert !isempty(radii) "radii darf nicht leer sein"

    radii_m = Float64.(radii) .* 1000.0
    R_mean  = mean(radii_m)

    U = Δρ * g * R_mean^2 / (4 * η)   # m/s

    x_m = Float64.(x_vec_1D) .* 1000.0
    nx = length(x_m)
    nz = length(z_vec_1D)

    ψ_analytic = Array{Float64}(undef, nx, nz)
    for i in 1:nx
        ψ_analytic[i, :] .= U * x_m[i]
    end

    return ψ_analytic, U
end

# ------------------------------------------------------------
# Sample- und Dataset-Generierung für Ansatz 3
# ------------------------------------------------------------

"""
    generate_residual_sample(rng; ...)

Erzeugt ein einzelnes Sample für Ansatz 3:

- input          :: (nx, nz, 1)  Float32  (Kristallmaske)
- residual_norm  :: (nx, nz)     Float64  (normiertes Residuum)
- scale          :: Float64      (Skalierung für das Residuum)
- meta           :: NamedTuple   (inkl. n_crystals, centers_2D, radii, x_vec_1D, z_vec_1D, U_stokes)
- ψ_lamem        :: (nx, nz)     Float64  (numerische ψ aus LaMEM)
- ψ_analytic     :: (nx, nz)     Float64  (analytische ψ)
"""
function generate_residual_sample(rng;
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

    # --- Anzahl Kristalle ---
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
            x_min = -0.5,
            x_max =  0.5,
            z_min = -0.5,
            z_max =  0.5,
        )
        centers_2D[i] = (cx, cz)
    end

    # --- LaMEM-Simulation ---
    result = run_sinking_crystals(; nx=nx,
                                   nz=nz,
                                   η=η,
                                   Δρ=Δρ,
                                   centers_2D=centers_2D,
                                   radii=radii)

    x_vec_1D = result.x_vec_1D   # km
    z_vec_1D = result.z_vec_1D   # km

    phase    = result.phase
    ψ_lamem  = result.ψ          # numerische ψ (physikalisch)

    # --- Eingabe (Kristallmaske) ---
    input = build_input_channels(phase)  # (nx, nz, 1), Float32

    # --- Analytische Stromfunktion ---
    ψ_analytic, U_stokes = analytic_streamfunction_uniform(
        x_vec_1D, z_vec_1D, radii;
        Δρ = Δρ,
        η  = η,
    )

    # --- Residuum ---
    residual = ψ_analytic .- ψ_lamem

    residual_norm, scale = normalize_psi(residual)

    meta = (
        n_crystals = n_crystals,
        centers_2D = centers_2D,
        radii      = radii,
        x_vec_1D   = x_vec_1D,
        z_vec_1D   = z_vec_1D,
        U_stokes   = U_stokes,
    )

    return input, residual_norm, scale, meta, ψ_lamem, ψ_analytic
end

"""
    generate_dataset(outdir; ...)

Erzeugt n_train Residual-Samples und speichert sie als .jld2:

Jede Datei enthält:
- input         (nx, nz, 1)
- residual_norm (nx, nz)
- scale         :: Float64
- meta          :: NamedTuple
- ψ_lamem       (nx, nz)
- ψ_analytic    (nx, nz)
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
        input, residual_norm, scale, meta, ψ_lamem, ψ_analytic =
            generate_residual_sample(
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

        filename = joinpath(outdir, @sprintf("residual_sample_%04d.jld2", i))
        @save filename input residual_norm scale meta ψ_lamem ψ_analytic

        @info "Residual-Sample $i gespeichert → $filename (n_crystals=$(meta.n_crystals))"
    end
end

end # module
