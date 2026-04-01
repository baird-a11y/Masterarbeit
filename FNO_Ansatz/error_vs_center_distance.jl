#!/usr/bin/env julia

# Script: error_vs_center_distance.jl (FNO)
# Analysis of the relationship between ψ error and crystal position.
# - Any number of crystals per sample
# - Distance of crystal centers from the domain center
# - Distances to the four corners of the domain
#
# CSV + plots are saved in out_dir.

using Printf
using Statistics
using CairoMakie
using Flux
using JLD2
using Base.Filesystem: mkpath

include("FNOPsi.jl")
using .FNOPsi
using .FNOPsi.FNOModel: build_fno
using .FNOPsi.DatasetPsi: load_dataset, get_sample

# ------------------------------------------------------------
# Main function: error vs. distances (per crystal)
# ------------------------------------------------------------
function compute_error_vs_center_distance(;
        data_dir::String,
        checkpoint_path::String,
        out_dir::String = joinpath(data_dir, "error_vs_distance"),
        out_csv_name::String = "error_vs_center_distance.csv",
        denorm_psi::Bool = true)

    mkpath(out_dir)
    out_csv = joinpath(out_dir, out_csv_name)

    # --- Load model from checkpoint ---
    @info "Loading model from $checkpoint_path"
    ckpt_data    = JLD2.load(checkpoint_path)
    model_config = get(ckpt_data, "model_config", nothing)

    if model_config !== nothing
        model = build_fno(model_config.Cin;
                          width   = model_config.width,
                          depth   = model_config.depth,
                          modes_x = model_config.modes_x,
                          modes_z = model_config.modes_z,
                          Cout    = model_config.Cout)
    else
        @warn "No model_config in checkpoint – using config.jl"
        cfg = FNOPsi.load_config()
        ds_tmp = load_dataset(data_dir; use_coords=true)
        X1, _, _ = get_sample(ds_tmp, 1)
        Cin = size(X1, 3)
        model = build_fno(Cin; width=cfg.width, depth=cfg.depth,
                          modes_x=cfg.modes, modes_z=cfg.modes, Cout=1)
    end

    Flux.loadmodel!(model, ckpt_data["model_state"])
    epoch = get(ckpt_data, "epoch", 0)
    @info "Model loaded (epoch $epoch)"

    # --- Load dataset ---
    ds = load_dataset(data_dir; use_coords=true)
    n_samples = length(ds)
    @info "Dataset with $n_samples samples loaded from $data_dir"

    results = NamedTuple[]

    Flux.testmode!(model)

    for i in 1:n_samples
        X, Y, smeta = get_sample(ds, i)
        nx, nz, _   = size(X)

        # Forward pass
        Xb    = reshape(X, nx, nz, size(X, 3), 1)
        ŷ_raw = model(Xb)
        y_pred_norm = dropdims(cpu(ŷ_raw); dims=(3, 4))
        y_true_norm = dropdims(Y; dims=3)

        # Metadata
        scale  = hasproperty(smeta, :scale) ? Float64(smeta.scale) : 1.0
        inner  = hasproperty(smeta, :meta) ? smeta.meta : smeta
        filepath = smeta.filepath

        n_crystals = if hasproperty(inner, :n_crystals)
            Int(inner.n_crystals)
        elseif hasproperty(inner, :centers_2D)
            length(inner.centers_2D)
        else
            -1
        end

        centers_2D = hasproperty(inner, :centers_2D) ? inner.centers_2D : []

        # Physical coordinates (km)
        xcoords = hasproperty(inner, :x_vec_1D) ? inner.x_vec_1D : collect(1:nx)
        zcoords = hasproperty(inner, :z_vec_1D) ? inner.z_vec_1D : collect(1:nz)

        # Domain center and corners
        x_center = (xcoords[1] + xcoords[end]) / 2
        z_center = (zcoords[1] + zcoords[end]) / 2
        x1, xN   = xcoords[1], xcoords[end]
        z1, zN   = zcoords[1], zcoords[end]

        # Choose ψ space
        if denorm_psi
            y_true_eval = y_true_norm ./ scale
            y_pred_eval = y_pred_norm ./ scale
        else
            y_true_eval = y_true_norm
            y_pred_eval = y_pred_norm
        end

        # ψ error (global per sample)
        mse_psi    = mean((y_pred_eval .- y_true_eval).^2)
        num_psi    = sqrt(sum((y_pred_eval .- y_true_eval).^2))
        denom_psi  = sqrt(sum(y_true_eval.^2)) + eps()
        rel_l2_psi = num_psi / denom_psi

        # One entry per crystal
        for (k, c) in enumerate(centers_2D)
            cx, cz = c

            dist_center = sqrt((cx - x_center)^2 + (cz - z_center)^2)
            dist_c1     = sqrt((cx - x1)^2 + (cz - z1)^2)  # bottom left
            dist_c2     = sqrt((cx - xN)^2 + (cz - z1)^2)  # bottom right
            dist_c3     = sqrt((cx - x1)^2 + (cz - zN)^2)  # top left
            dist_c4     = sqrt((cx - xN)^2 + (cz - zN)^2)  # top right

            push!(results, (;
                sample_idx    = i,
                crystal_idx   = k,
                n_crystals    = n_crystals,
                cx            = cx,
                cz            = cz,
                x_center      = x_center,
                z_center      = z_center,
                dist_center   = dist_center,
                dist_corner_1 = dist_c1,
                dist_corner_2 = dist_c2,
                dist_corner_3 = dist_c3,
                dist_corner_4 = dist_c4,
                mse_psi       = mse_psi,
                rel_l2_psi    = rel_l2_psi,
            ))
        end

        if i % max(1, n_samples ÷ 10) == 0 || i == n_samples
            @info @sprintf("  [%d/%d]  rel_l2=%.4f  mse=%.4e", i, n_samples, rel_l2_psi, mse_psi)
        end
    end

    Flux.trainmode!(model)

    # Write CSV (one row per crystal)
    open(out_csv, "w") do io
        println(io,
            "sample_idx,crystal_idx,n_crystals," *
            "cx,cz,x_center,z_center," *
            "dist_center,dist_corner_1,dist_corner_2,dist_corner_3,dist_corner_4," *
            "mse_psi,relL2_psi"
        )
        for r in results
            println(io, @sprintf(
                "%d,%d,%d,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e",
                r.sample_idx, r.crystal_idx, r.n_crystals,
                r.cx, r.cz,
                r.x_center, r.z_center,
                r.dist_center,
                r.dist_corner_1, r.dist_corner_2, r.dist_corner_3, r.dist_corner_4,
                r.mse_psi,
                r.rel_l2_psi,
            ))
        end
    end

    @info "Results written to $out_csv (n=$(length(results)) crystals across all samples)."
    return results, out_dir
end

# ------------------------------------------------------------
# Plot: rel. L2 error vs. distance (per crystal, colored by N)
# ------------------------------------------------------------
function plot_error_vs_distance_byN(
        results;
        out_dir::String,
        kind::Symbol = :center,
        outfile_name::Union{Nothing,String} = nothing)

    mkpath(out_dir)

    dist_accessor, x_label, title =
        if kind == :center
            (r -> r.dist_center,
             "Distance from domain center (km)",
             "Error vs. Distance from Center (per crystal)")
        elseif kind == :corner1
            (r -> r.dist_corner_1,
             "Distance to Corner 1 (bottom left) (km)",
             "Error vs. Distance to Corner 1")
        elseif kind == :corner2
            (r -> r.dist_corner_2,
             "Distance to Corner 2 (bottom right) (km)",
             "Error vs. Distance to Corner 2")
        elseif kind == :corner3
            (r -> r.dist_corner_3,
             "Distance to Corner 3 (top left) (km)",
             "Error vs. Distance to Corner 3")
        elseif kind == :corner4
            (r -> r.dist_corner_4,
             "Distance to Corner 4 (top right) (km)",
             "Error vs. Distance to Corner 4")
        else
            error("Unknown kind = $kind. Allowed: :center, :corner1..:corner4")
        end

    if outfile_name === nothing
        suffix = kind == :center ? "center" :
                 kind == :corner1 ? "corner1" :
                 kind == :corner2 ? "corner2" :
                 kind == :corner3 ? "corner3" : "corner4"
        outfile_name = "error_vs_$(suffix)_distance_byN.png"
    end

    outfile = joinpath(out_dir, outfile_name)

    all_N = sort(unique([r.n_crystals for r in results]))

    fig = Figure(resolution = (800, 500))
    ax  = Axis(fig[1, 1];
        xlabel = x_label,
        ylabel = "rel. L2 Error ψ",
        title  = title,
    )

    for N in all_N
        group = filter(r -> r.n_crystals == N, results)
        d   = [dist_accessor(r) for r in group]
        err = [r.rel_l2_psi    for r in group]

        scatter!(ax, d, err; label = "N = $N")
    end

    axislegend(ax; position = :rb)

    save(outfile, fig)
    @info "Plot saved: $outfile"

    return fig
end

# ------------------------------------------------------------
# Entry point when called as a script
# ------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    @info "Running error_vs_center_distance.jl (FNO) as script."
    @info "Working directory: $(pwd())"

    data_dir        = "../data_eval"
    checkpoint_path = "checkpoints/best_model.jld2"

    results, out_dir = compute_error_vs_center_distance(;
        data_dir        = data_dir,
        checkpoint_path = checkpoint_path,
        denorm_psi      = true,
    )

    plot_error_vs_distance_byN(results; out_dir = out_dir, kind = :center)
    plot_error_vs_distance_byN(results; out_dir = out_dir, kind = :corner1)
    plot_error_vs_distance_byN(results; out_dir = out_dir, kind = :corner2)
    plot_error_vs_distance_byN(results; out_dir = out_dir, kind = :corner3)
    plot_error_vs_distance_byN(results; out_dir = out_dir, kind = :corner4)
end
