using LaMEM, GeophysicalModelGenerator, GLMakie


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

    model     = Model(Grid(nel=(nx,nz), x=[-1,1], z=[-1,1]), Time(nstep_max=1), Output(out_strain_rate=1) )
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


    V_stokes =  2/9*Δρ*9.81*(R[1]*1000)^2/(η)            # Stokes velocity in m/s
    V_stokes_cm_year = V_stokes * 100 * (3600*24*365.25) # convert to cm/year


    return x_vec_1D, z_vec_1D, phase, Vx, Vz, Exx, Ezz, V_stokes_cm_year
end


# Compute model

# Setup 2D model
nx, nz = 512, 512
cen_2D = [(-0.7, 0.8)]      # center of sphere
R      = [0.05]             # radius of sphere in km    


x, z, phase, Vx, Vz, Exx, Ezz, V_stokes_cm_year = LaMEM_Single_crystal(; nx=nx, nz=nz, cen_2D=cen_2D, R=R)

# plot Vz & phase
fig, ax, hm = heatmap(x, z, (Vz/V_stokes_cm_year), colormap=:viridis)

contour!(x, z, phase, levels=[0.5], color=:white, linewidth=2)

Colorbar(fig[:, 2], hm)  
ax.xlabel = "x (km)"
ax.ylabel = "z (km)"
ax.title = "Vz, minimum(Vz/Vz_stokes)=$(minimum(Vz/V_stokes_cm_year))"
display(fig)