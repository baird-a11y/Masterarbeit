using LaMEM, GeophysicalModelGenerator, GLMakie
using SparseArrays, LinearAlgebra

"""
    poisson2D(omega, dx, dy)

Solve ∇²ψ = ω on a uniform grid with Dirichlet (ψ = 0) boundary conditions.
- `omega` : 2D array (nx × ny), source term (vorticity)
- `dx`, `dy`: grid spacings

Returns ψ (2D array, same size as omega).
"""
function poisson2D(omega::Array{<:Real,2}, dx::Real, dy::Real)
    nx, ny = size(omega)
    N = nx * ny

    # --- 1D second-derivative matrices ---
    ex = ones(nx)
    ey = ones(ny)
    Tx = spdiagm(-1 => ex[1:end-1], 0 => -2*ex, 1 => ex[1:end-1])
    Ty = spdiagm(-1 => ey[1:end-1], 0 => -2*ey, 1 => ey[1:end-1])

    # --- 2D Laplacian via Kronecker sums ---
    # (replaces old speye with modern sparse identities)
    L = kron(spdiagm(0 => ones(ny)), Tx) / dx^2 +
        kron(Ty, spdiagm(0 => ones(nx))) / dy^2

    # Make matrix mutable for boundary modifications
    L = copy(L)
    rhs = -vec(omega)

    # --- Helper to flatten (i,j) → 1D index ---
    idx(i, j) = (j - 1) * nx + i

    # --- Apply Dirichlet ψ=0 at all boundaries ---
    for i in 1:nx, j in 1:ny
        if i == 1 || i == nx || j == 1 || j == ny
            k = idx(i, j)
            L[k, :] .= 0.0
            L[k, k] = 1.0
            rhs[k] = 0.0
        end
    end

    # --- Solve the sparse linear system ---
    ψvec = L \ rhs
    ψ = reshape(ψvec, nx, ny)

    # Remove arbitrary constant offset
    ψ .-= ψ[1,1]
    return ψ
end


function velocity_from_streamfunction(ψ, dx, dz)
    # Compute velocity components from streamfunction
    # Vx = dψ/dz, Vz = -dψ/dx
    nx, nz = size(ψ)
    Vx = zeros(nx, nz)
    Vz = zeros(nx, nz)

    for i in 2:nx-1, j in 2:nz-1
        Vx[i,j] = (ψ[i,j+1] - ψ[i,j-1]) / (2*dz)
        Vz[i,j] = -(ψ[i+1,j] - ψ[i-1,j]) / (2*dx)
    end

    return Vx, Vz
end


function LaMEM_Single_crystal(; nx=64, nz=64, η=1e20, Δρ=200, cen_2D=[(0.0, 0.0)], R=[0.1])
    # Create a model with a single crystal phase
    # nx, nz: number of grid points in x and z direction
    # η: viscosity of matrix in Pa.s
    # Δρ: density contrast in kg/m^3
    # cen_2D: center of sphere in km
    # R: radius of sphere in km

    # Define the model parameters
    η_crystal = 1e4*η       # viscosity of crystal in Pa.s
    ρ_magma   = 2700

    model     = Model(Grid(nel=(nx,nz), x=[-1,1], z=[-1,1]), Time(nstep_max=1), Output(out_strain_rate=1, out_vel_gr_tensor=1, out_file_name="FS_vel_gradient") )
    matrix    = Phase(ID=0,Name="matrix", eta=η,        rho=ρ_magma);
    crystal   = Phase(ID=1,Name="crystal",eta=η_crystal,rho=ρ_magma+Δρ);
    add_phase!(model, crystal, matrix)

    for i =1:length(cen_2D)
        # Add a sphere with the crystal phase
        add_sphere!(model, cen=(cen_2D[i][1], 0.0, cen_2D[i][2]), radius=R[i],  phase=ConstantPhase(1))
    end
   
    # Run LaMEM
    run_lamem(model,1)

    # Read results back into julia
    data, _ = read_LaMEM_timestep(model, 1)

    # Extract the data we need 
    x_vec_1D = data.x.val[:,1,1]
    z_vec_1D = data.z.val[1,1,:]

    phase = data.fields.phase[:,1,:]
    Vx    = data.fields.velocity[1][:,1,:]          # velocity in x [cm/year]
    Vz    = data.fields.velocity[3][:,1,:]          # velocity in z [cm/year]
    Exx   = data.fields.strain_rate[1][:,1,:]       # strainrate in x (=dVx/dx) in [1/s]
    Ezz   = data.fields.strain_rate[9][:,1,:]       # strainrate in z (=dVz/dz) in [1/s]
    rho   = data.fields.density[:,1,:]              # density in kg/m^3    
    log10eta  = data.fields.visc_creep[:,1,:]       # log10(viscosity)
    vel_grad = data.fields.vel_gr_tensor            # velocity gradient tensor
    dVx_dx = vel_grad[1][:,1,:]                      # dVx/dx -> to be checked
    dVx_dz = vel_grad[3][:,1,:]                      # dVx/dz -> to be checked
    dVz_dx = vel_grad[7][:,1,:]                      # dVz/dx -> to be checked
    dVz_dz = vel_grad[9][:,1,:]                      # dVz/dz -> to be checked

    # compute streamfunction from the LaMEM output
    dx      = (x_vec_1D[2]-x_vec_1D[1])*1000
    dz      = (z_vec_1D[2]-z_vec_1D[1])*1000
    ω       = dVz_dx - dVx_dz                    # vorticity in x/z plane (in units of 1/s)
    psi     =  poisson2D(ω, dx, dz)      # stream function


    V_stokes =  2/9*Δρ*9.81*(R[1]*1000)^2/(η)            # Stokes velocity in m/s
    V_stokes_cm_year = V_stokes * 100 * (3600*24*365.25) # convert to cm/year


    return x_vec_1D, z_vec_1D, phase, Vx, Vz, Exx, Ezz, V_stokes_cm_year, psi
end


# Compute model

# Setup 2D model
nx, nz = 256, 256
cen_2D = [(0.4, -0.4)]      # center of sphere
R      = [0.1]             # radius of sphere in km    


x, z, phase, Vx, Vz, Exx, Ezz, V_stokes_cm_year, psi = LaMEM_Single_crystal(; nx=nx, nz=nz, cen_2D=cen_2D, R=R)


fig = Figure()
ax1 = Axis(fig[1,1], title="ψ", xlabel="x (km)", ylabel="z (km)")

hm1 = heatmap!(ax1,x, z, 1e13*psi, colormap=:viridis)
Colorbar(fig[:, 2], hm1)  

display(fig)

