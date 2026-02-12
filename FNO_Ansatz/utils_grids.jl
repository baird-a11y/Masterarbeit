# Datei: utils_grids.jl
# dx/dz + Grid Checks (Uniformität, Einheiten)
# meshgrid + normierte Koordinatenkanäle (für FNO-Inputs)
# Ableitungen ddx/ddz (für ω, div, grad-loss, etc.)
# Divergenz + Vorticity als Utility (Sanity-Checks, Loss)
module GridFDUtils

using Statistics

# =============================================================================
# Conventions
# =============================================================================
# - Field arrays are (nx, nz) with x along dim=1 and z along dim=2
# - dx, dz are spacings in meters
# - Default BC for derivatives: :onesided (robust, works with Dirichlet ψ=0 Poisson)
# =============================================================================

export grid_spacing,
       check_uniform_grid,
       normalize_coords,
       laplacian,
       divergence,
       vorticity,
       velocity_from_streamfunction,
       stats,
       interior_mask

# -----------------------------------------------------------------------------
# Basic grid helpers
# -----------------------------------------------------------------------------

"""
    grid_spacing(x_vec_1D, z_vec_1D)

Return (dx, dz) assuming 1D coordinate vectors.
No unit conversion is done here (you must pass meters if you want meters).
"""
function grid_spacing(x_vec_1D::AbstractVector, z_vec_1D::AbstractVector)
    @assert length(x_vec_1D) ≥ 2 "x_vec_1D too short"
    @assert length(z_vec_1D) ≥ 2 "z_vec_1D too short"
    dx = x_vec_1D[2] - x_vec_1D[1]
    dz = z_vec_1D[2] - z_vec_1D[1]
    @assert dx != 0 && dz != 0 "dx/dz must be non-zero"
    return dx, dz
end

"""
    check_uniform_grid(x_vec_1D, z_vec_1D; rtol=1e-10, atol=0.0)

Return (is_uniform_x, is_uniform_z). Use this to sanity-check your LaMEM grids.
"""
function check_uniform_grid(x_vec_1D::AbstractVector, z_vec_1D::AbstractVector; rtol=1e-10, atol=0.0)
    dxs = diff(x_vec_1D)
    dzs = diff(z_vec_1D)
    ux = maximum(abs.(dxs .- first(dxs))) ≤ atol + rtol*abs(first(dxs))
    uz = maximum(abs.(dzs .- first(dzs))) ≤ atol + rtol*abs(first(dzs))
    return ux, uz
end

"""
    meshgrid(x_vec_1D, z_vec_1D)

Return (X, Z) as (nx, nz) matrices.
"""
function meshgrid(x_vec_1D::AbstractVector, z_vec_1D::AbstractVector)
    nx = length(x_vec_1D)
    nz = length(z_vec_1D)
    X = repeat(reshape(x_vec_1D, nx, 1), 1, nz)
    Z = repeat(reshape(z_vec_1D, 1, nz), nx, 1)
    return X, Z
end

"""
    normalize_coords(x_vec_1D, z_vec_1D; range=:pm1)

Create normalized coordinate grids (Xn, Zn) as (nx, nz).
- range=:pm1 -> [-1, 1]
- range=:unit -> [0, 1]
"""
function normalize_coords(x_vec_1D::AbstractVector, z_vec_1D::AbstractVector; range::Symbol = :pm1)
    xmin, xmax = extrema(x_vec_1D)
    zmin, zmax = extrema(z_vec_1D)
    @assert xmax > xmin && zmax > zmin "Degenerate coordinate ranges"

    if range == :pm1
        xnorm(v) = 2*(v - xmin)/(xmax - xmin) - 1
        znorm(v) = 2*(v - zmin)/(zmax - zmin) - 1
    elseif range == :unit
        xnorm(v) = (v - xmin)/(xmax - xmin)
        znorm(v) = (v - zmin)/(zmax - zmin)
    else
        error("Unknown range=$range (use :pm1 or :unit)")
    end

    X, Z = meshgrid(x_vec_1D, z_vec_1D)
    return xnorm.(X), znorm.(Z)
end

# -----------------------------------------------------------------------------
# Derivatives (1st order)
# -----------------------------------------------------------------------------

"""
    ddx(A, dx; bc=:onesided)

Compute ∂A/∂x for A(nx,nz), x along dim=1.

bc options:
- :onesided  (default) forward/backward at boundaries, central inside
"""
function ddx(A::AbstractMatrix, dx::Real; bc::Symbol = :onesided)
    nx, nz = size(A)
    @assert nx ≥ 2 "nx must be ≥ 2"
    out = similar(A, Float64)

    @inbounds for j in 1:nz
        out[1, j] = (A[2, j] - A[1, j]) / dx
        for i in 2:nx-1
            out[i, j] = (A[i+1, j] - A[i-1, j]) / (2dx)
        end
        out[nx, j] = (A[nx, j] - A[nx-1, j]) / dx
    end
    return out
end

"""
    ddz(A, dz; bc=:onesided)

Compute ∂A/∂z for A(nx,nz), z along dim=2.
"""
function ddz(A::AbstractMatrix, dz::Real; bc::Symbol = :onesided)
    nx, nz = size(A)
    @assert nz ≥ 2 "nz must be ≥ 2"
    out = similar(A, Float64)

    @inbounds for i in 1:nx
        out[i, 1] = (A[i, 2] - A[i, 1]) / dz
        for j in 2:nz-1
            out[i, j] = (A[i, j+1] - A[i, j-1]) / (2dz)
        end
        out[i, nz] = (A[i, nz] - A[i, nz-1]) / dz
    end
    return out
end

# -----------------------------------------------------------------------------
# Second derivatives + Laplacian
# -----------------------------------------------------------------------------

"""
    d2dx2(A, dx; bc=:onesided)

Compute ∂²A/∂x².
Note: for :onesided boundaries this uses a simple 2nd order one-sided stencil.
"""
function d2dx2(A::AbstractMatrix, dx::Real; bc::Symbol = :onesided)
    nx, nz = size(A)
    @assert nx ≥ 4 "nx must be ≥ 4 for second derivative"
    out = similar(A, Float64)

    @inbounds for j in 1:nz
        # 2nd order one-sided stencils
        out[1, j]  = (2A[1, j] - 5A[2, j] + 4A[3, j] - A[4, j]) / (dx^2)
        for i in 2:nx-1
            out[i, j] = (A[i+1, j] - 2A[i, j] + A[i-1, j]) / (dx^2)
        end
        out[nx, j] = (2A[nx, j] - 5A[nx-1, j] + 4A[nx-2, j] - A[nx-3, j]) / (dx^2)
    end
    return out
end

"""
    d2dz2(A, dz; bc=:onesided)

Compute ∂²A/∂z².
"""
function d2dz2(A::AbstractMatrix, dz::Real; bc::Symbol = :onesided)
    nx, nz = size(A)
    @assert nz ≥ 4 "nz must be ≥ 4 for second derivative"
    out = similar(A, Float64)
    @inbounds for i in 1:nx
        out[i, 1]  = (2A[i, 1] - 5A[i, 2] + 4A[i, 3] - A[i, 4]) / (dz^2)
        for j in 2:nz-1
            out[i, j] = (A[i, j+1] - 2A[i, j] + A[i, j-1]) / (dz^2)
        end
        out[i, nz] = (2A[i, nz] - 5A[i, nz-1] + 4A[i, nz-2] - A[i, nz-3]) / (dz^2)
    end
    return out
end

"""
    laplacian(A, dx, dz; bc=:onesided)

Compute ΔA = ∂xx A + ∂zz A.
"""
function laplacian(A::AbstractMatrix, dx::Real, dz::Real; bc::Symbol = :onesided)
    return d2dx2(A, dx; bc=bc) .+ d2dz2(A, dz; bc=bc)
end

# -----------------------------------------------------------------------------
# Derived flow operators
# -----------------------------------------------------------------------------

"""
    velocity_from_streamfunction(ψ, dx, dz; bc=:onesided)

Compute:
- Vx =  ∂ψ/∂z
- Vz = -∂ψ/∂x
"""
function velocity_from_streamfunction(ψ::AbstractMatrix, dx::Real, dz::Real; bc::Symbol = :onesided)
    Vx = ddz(ψ, dz; bc=bc)
    Vz = -ddx(ψ, dx; bc=bc)
    return Vx, Vz
end

"""
    divergence(Vx, Vz, dx, dz; bc=:onesided)

Compute ∂x Vx + ∂z Vz.
"""
function divergence(Vx::AbstractMatrix, Vz::AbstractMatrix, dx::Real, dz::Real; bc::Symbol = :onesided)
    @assert size(Vx) == size(Vz)
    return ddx(Vx, dx; bc=bc) .+ ddz(Vz, dz; bc=bc)
end

"""
    vorticity(Vx, Vz, dx, dz; bc=:onesided)

Compute ω = ∂x Vz - ∂z Vx.
"""
function vorticity(Vx::AbstractMatrix, Vz::AbstractMatrix, dx::Real, dz::Real; bc::Symbol = :onesided)
    @assert size(Vx) == size(Vz)
    return ddx(Vz, dx; bc=bc) .- ddz(Vx, dz; bc=bc)
end

# -----------------------------------------------------------------------------
# Masks + stats
# -----------------------------------------------------------------------------

"""
    interior_mask(nx, nz; width=1)

Boolean mask true on interior cells (excluding a boundary band of `width`).
Useful if you want to ignore derivative boundary artifacts in metrics/losses.
"""
function interior_mask(nx::Int, nz::Int; width::Int=1)
    @assert width ≥ 0
    mask = trues(nx, nz)
    if width > 0
        mask[1:width, :] .= false
        mask[end-width+1:end, :] .= false
        mask[:, 1:width] .= false
        mask[:, end-width+1:end] .= false
    end
    return mask
end

"""
    stats(name, A)

Print quick diagnostics (min/max/mean/std + NaN/Inf counts).
"""
function stats(name::AbstractString, A::AbstractArray)
    n_nan = count(isnan, A)
    n_inf = count(isinf, A)
    amin, amax = extrema(A)
    μ = mean(Float64.(A))
    σ = std(Float64.(A))
    println("[stats] $name | min=$amin max=$amax mean=$μ std=$σ NaN=$n_nan Inf=$n_inf")
    return nothing
end



end # module
