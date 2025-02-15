using GLMakie
using Colors

# Erzeuge ein 3D-Gitter
xs = range(-20, 20, length = 100)
ys = range(-20, 20, length = 100)
zs = range(-20, 20, length = 100)

# Dichtefeld initialisieren
density = zeros(Float32, length(xs), length(ys), length(zs))

# Kugelparameter
radius = 10.0  # Radius der Kugel
center = (0.0, 0.0, 0.0)  # Mittelpunkt der Kugel

# Dichtefeld mit konstanten Werten definieren
density_inside = 5000.0  # Konstante Dichte innerhalb der Kugel
density_outside = 3000.0  # Konstante Dichte außerhalb der Kugel

for i in eachindex(xs), j in eachindex(ys), k in eachindex(zs)
    x, y, z = xs[i], ys[j], zs[k]
    r = sqrt((x - center[1])^2 + (y - center[2])^2 + (z - center[3])^2)
    if r <= radius
        density[i, j, k] = density_inside  # Konstante Dichte innerhalb
    else
        density[i, j, k] = density_outside  # Konstante Dichte außerhalb
    end
end

fig = Figure(resolution = (800, 600))

# 3D-Plot
ax = LScene(fig[1, 1], scenekw=(show_axis=true, ))

# Visualisierung des Volumens
volume!(ax, density; algorithm = :mip, colormap = :viridis)

# Farbskala hinzufügen
limits = (minimum(density), maximum(density))  # Wertebereich des Dichtefelds
Colorbar(fig[1, 2], colormap=:viridis, limits=limits, label="Dichte")

fig
