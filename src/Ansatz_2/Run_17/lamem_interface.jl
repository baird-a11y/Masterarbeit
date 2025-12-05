module LaMEMInterface

using LaMEM, GeophysicalModelGenerator
using ..StreamFunctionPoisson: poisson2D

"""
    run_sinking_crystals(; nx=256, nz=256, η=1e20, Δρ=200,
                           centers_2D=[(0.0, 0.0)], radii=[0.1])

Führt ein LaMEM-Modell mit 1–n Kristallen aus und liefert:
- Gitterkoordinaten x_vec_1D, z_vec_1D (in km)
- phase (nx × nz)
- Vx, Vz (nx × nz)
- ω (nx × nz)
- ψ (nx × nz)
"""
function run_sinking_crystals(; nx::Int=256, nz::Int=256,
                               η::Real=1e20, Δρ::Real=200,
                               centers_2D::Vector{Tuple{Float64,Float64}}=[(0.0, 0.0)],
                               radii::Vector{Float64}=[0.1])

    η_crystal = 1e4 * η
    ρ_magma   = 2700.0

    model = Model(
    Grid(nel = (nx - 1, nz - 1), x = [-1, 1], z = [-1, 1]),
    Time(nstep_max = 1),
    Output(out_strain_rate   = 1,
           out_vel_gr_tensor = 1,
           out_file_name     = "FS_vel_gradient")
    )

    matrix  = Phase(ID=0, Name="matrix",  eta=η,         rho=ρ_magma)
    crystal = Phase(ID=1, Name="crystal", eta=η_crystal, rho=ρ_magma + Δρ)

    add_phase!(model, crystal, matrix)

    @assert length(centers_2D) == length(radii)
    for i in 1:length(centers_2D)
        cx, cz = centers_2D[i]
        add_sphere!(model,
            cen    = (cx, 0.0, cz),   # y = 0, 2D-Schnitt
            radius = radii[i],
            phase  = ConstantPhase(1),
        )
    end

    # LaMEM laufen lassen
    run_lamem(model, 1)

    # Ergebnis einlesen
    data, _ = read_LaMEM_timestep(model, 1)

    x_vec_1D = data.x.val[:, 1, 1]     # in km
    z_vec_1D = data.z.val[1, 1, :]     # in km

    phase    = data.fields.phase[:, 1, :]
    Vx       = data.fields.velocity[1][:, 1, :]    # [cm/year]
    Vz       = data.fields.velocity[3][:, 1, :]    # [cm/year]

    vel_grad = data.fields.vel_gr_tensor
    dVx_dx   = vel_grad[1][:, 1, :]
    dVx_dz   = vel_grad[3][:, 1, :]
    dVz_dx   = vel_grad[7][:, 1, :]
    dVz_dz   = vel_grad[9][:, 1, :]

    # Gitterabstand in m
    dx = (x_vec_1D[2] - x_vec_1D[1]) * 1000.0
    dz = (z_vec_1D[2] - z_vec_1D[1]) * 1000.0

    # Wirbelstärke ω = dVz/dx - dVx/dz (2D)
    ω = dVz_dx .- dVx_dz

    # Streamfunktion
    ψ = poisson2D(ω, dx, dz)

    return (; x_vec_1D, z_vec_1D, phase, Vx, Vz, ω, ψ)
end

end # module
