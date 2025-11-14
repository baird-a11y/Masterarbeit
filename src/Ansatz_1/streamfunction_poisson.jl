module StreamFunctionPoisson

using SparseArrays, LinearAlgebra

"""
    poisson2D(omega, dx, dz)

Löst ∇²ψ = ω auf einem regulären Gitter mit Dirichlet-Randbedingungen ψ = 0.
Gibt ψ als 2D-Array zurück (gleiche Größe wie `omega`).
"""
function poisson2D(omega::AbstractMatrix{<:Real}, dx::Real, dz::Real)
    nx, nz = size(omega)

    ex = ones(nx)
    ez = ones(nz)

    Tx = spdiagm(-1 => ex[1:end-1], 0 => -2 .* ex, 1 => ex[1:end-1])
    Tz = spdiagm(-1 => ez[1:end-1], 0 => -2 .* ez, 1 => ez[1:end-1])

    Ix = spdiagm(0 => ones(nx))
    Iz = spdiagm(0 => ones(nz))

    # Diskretisierung des Laplace-Operators
    L = kron(Iz, Tx) / dx^2 .+ kron(Tz, Ix) / dz^2

    rhs = -vec(omega)
    idx(i, j) = (j - 1) * nx + i

    # Dirichlet-Randbedingungen: ψ = 0
    for i in 1:nx, j in 1:nz
        if i == 1 || i == nx || j == 1 || j == nz
            k = idx(i, j)
            L[k, :] .= 0.0
            L[k, k] = 1.0
            rhs[k] = 0.0
        end
    end

    ψvec = L \ rhs
    ψ    = reshape(ψvec, nx, nz)

    # Konstantenoffset entfernen (damit Werte "klein" bleiben)
    ψ .-= ψ[1, 1]

    return ψ
end


"""
    velocity_from_streamfunction(ψ, dx, dz)

Berechnet:
- vx =  ∂ψ/∂z
- vz = -∂ψ/∂x
"""
function velocity_from_streamfunction(ψ::AbstractMatrix{<:Real}, dx::Real, dz::Real)
    nx, nz = size(ψ)
    vx = zeros(Float64, nx, nz)
    vz = zeros(Float64, nx, nz)

    for i in 2:nx-1, j in 2:nz-1
        vx[i, j] = (ψ[i, j+1] - ψ[i, j-1]) / (2 * dz)
        vz[i, j] = -(ψ[i+1, j] - ψ[i-1, j]) / (2 * dx)
    end

    return vx, vz
end

end # module
