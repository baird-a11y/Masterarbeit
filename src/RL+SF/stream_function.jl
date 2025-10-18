function compute_velocities_from_stream(ψ::AbstractArray{T,4}) where T
    # ψ: [H, W, 1, batch]
    vx = ∂z(ψ)   # ∂ψ/∂z
    vz = -∂x(ψ)  # -∂ψ/∂x
    return cat(vx, vz, dims=3)
end

function ∂z(field)
    # Zentrale Differenzen (vertikal)
end

function ∂x(field)
    # Zentrale Differenzen (horizontal)
end