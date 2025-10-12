# Physics-Informed UNet for Flow Field Prediction in Multi-Crystal Sedimentation Systems

**Master's Thesis:** Physics-Based Machine Learning for Predicting Flow Fields in Multi-Crystal Sedimentation Systems  
**Author:** Paul Baselt  
**Framework:** Julia + Flux.jl + LaMEM.jl

---

## Project Overview

This repository implements a physics-informed UNet architecture for predicting flow fields around sinking crystals in geoscientific systems. The project combines deep learning with physical constraints for physically consistent modeling of complex multi-crystal sedimentation processes.

### Research Objectives

Development and validation of physics-informed neural networks for physically consistent prediction of velocity fields in multi-crystal sedimentation systems (1-15 crystals).

### Core Research Questions

- Can a physics-informed UNet model produce physically consistent predictions conforming to LaMEM simulations?
- How does performance scale with increasing crystal count?
- What role do physical constraints play in model generalization?

### Input/Output Specifications

| Component | Description |
|-----------|-------------|
| Input | Phase field (1 channel: 0=matrix, 1-15=crystal IDs) |
| Output | 2D velocity fields (2 channels: v_x, v_z) |
| Resolution | 256×256 pixels (consistent throughout pipeline) |
| Normalization | Robust Z-score with percentile-based outlier clipping |
| Physics | Continuity equation (∇·v ≈ 0) as regularization |

---

## Recent Developments (December 2024)

### Physics-Informed Training

- Continuity equation: Divergence minimization as loss component
- Adaptive weighting: Warm-up strategy (λ: 0.01 → 0.15 over 15 epochs)
- GPU-compatible computation: Efficient divergence calculation

### Optimized Training Pipeline

- 750 training samples (effective 1500+ through augmentation)
- Batch normalization for improved convergence stability
- Robust normalization with 99.5th percentile clipping
- Automatic GPU/CPU fallback with intelligent error handling

### Evaluation Framework

- 47 metrics across 5 categories
- Statistical significance tests (ANOVA, confidence intervals, effect sizes)
- Automated report generation (CSV, JSON, LaTeX, HTML)
- Multi-format export for publication

### Technical Improvements

- Zygote compatibility: Eliminated mutating arrays
- Adaptive batch sizes: GPU memory optimized
- Modular architecture: 12 specialized modules
- SLURM integration: Cluster-ready with submit_job.sh

---

## Project Structure

```
├── Core Modules
│   ├── lamem_interface.jl              # LaMEM integration (1-15 crystals)
│   ├── data_processing.jl              # Robust normalization & preprocessing
│   ├── unet_architecture.jl            # UNet with batch normalization
│   ├── training.jl                     # Physics-informed training loop
│   ├── batch_management.jl             # GPU-optimized batch management
│   ├── gpu_utils.jl                    # GPU management & error handling
│   └── main.jl                         # Optimized 10-crystal pipeline
│
├── Evaluation & Analysis
│   ├── comprehensive_evaluation.jl     # 47-metric evaluation system
│   ├── statistical_analysis.jl         # Significance tests & effect sizes
│   ├── advanced_visualization.jl       # Visualization suite
│   ├── automated_reporting_system.jl   # Scientific report generation
│   └── master_evaluation_fixed.jl      # Complete evaluation framework
│
├── Utilities
│   ├── simple_data_export.jl           # Multi-format data export
│   ├── data_management_system.jl       # LaTeX table generator
│   └── submit_job.sh                   # SLURM job script
│
└── Documentation
    ├── README.md                       # This file
    └── Arbeit/                         # LaTeX thesis
```

---

## Installation

### System Requirements

**Minimum:**
- Julia 1.9+
- 4 CPU cores
- 8 GB RAM
- ~8h training time (CPU)

**Recommended:**
- Julia 1.10+
- 8+ CPU cores
- 16 GB RAM
- NVIDIA GPU with 8+ GB VRAM
- ~1-2h training time (GPU)

### Dependencies

```julia
using Pkg

# Core dependencies
Pkg.add(["LaMEM", "GeophysicalModelGenerator", "Flux", "CUDA", "Optimisers"])

# Data processing & statistics
Pkg.add(["Statistics", "StatsBase", "Random", "Distributions", "HypothesisTests"])

# I/O & serialization
Pkg.add(["BSON", "CSV", "DataFrames", "JSON3", "Printf"])

# Visualization
Pkg.add(["Plots", "Colors", "ColorSchemes"])
```

### Configuration

**1. Resolve LaMEM Time conflict:**

```julia
# In lamem_interface.jl, change:
Time(nstep_max=1)
# To:
LaMEM.Time(nstep_max=1)
```

**2. Verify GPU setup (optional):**

```julia
include("gpu_utils.jl")
check_gpu_availability()
```

---

## Usage

### Standard Training (10 crystals)

```julia
include("main.jl")

# Automatic configuration in SERVER_CONFIG:
# - 750 samples with augmentation
# - Physics-informed loss (λ: 0.01→0.15)
# - Batch size = 4, learning rate = 0.0005
# - 50 epochs with early stopping (patience=15)
# - Automatic GPU detection
```

### Comprehensive Evaluation (1-5 crystals)

```julia
include("master_evaluation_fixed.jl")
run_simplified_evaluation()

# Generates:
# - 47 metrics per sample
# - Statistical analyses (ANOVA, confidence intervals)
# - LaTeX tables for publication
# - Visualizations (training curves, divergence, etc.)
```

---

## Technical Implementation

### Physics-Informed Loss Function

```julia
function physics_informed_loss(prediction, velocity_batch; lambda_physics=0.1f0)
    data_loss = mse(prediction, velocity_batch)
    divergence = compute_divergence(prediction)
    physics_loss = mean(abs2, divergence)
    total_loss = data_loss + lambda_physics * physics_loss
    return total_loss, data_loss, physics_loss
end
```

### Robust Normalization

```julia
function robust_normalize(data; percentile_clip=99.5)
    lower_bound = percentile(vec(data), 100 - percentile_clip)
    upper_bound = percentile(vec(data), percentile_clip)
    data_clipped = clamp.(data, lower_bound, upper_bound)
    μ = mean(data_clipped)
    σ = std(data_clipped) + 1f-8
    data_normalized = (data_clipped .- μ) ./ σ
    return Float32.(data_normalized), Float32(μ), Float32(σ)
end
```

---

## Performance Benchmarks

### Current Performance (September 2025)

| Metric | Best Value | Average | Target | Status |
|--------|------------|---------|--------|--------|
| MAE | 0.2546 | 0.4755 | < 0.05 | Improvement needed |
| Correlation (v_z) | 0.778 | ~0.65 | > 0.85 | Improvement needed |
| Physics Loss | N/A | N/A | < 0.001 | In progress |
| Alignment | N/A | N/A | < 10 px | To be evaluated |

**Latest Model:** Modell_9 (trained on 750 samples, 10 crystals)

### Performance Evolution

```
Baseline (July 2024):
- MAE: 0.488
- Correlation: ~0.70
- Training: CPU-only
- Physics: Implicit only

Current (September 2025):
- MAE: 0.2546 (best), 0.4755 (average) - 50% improvement on best case
- Correlation: 0.778 (best), ~0.65 (average) - 11% improvement on best case
- Training: CPU-based (50 samples)
- Physics: Explicit continuity constraint implemented
- Scaling: 1-10 crystals tested

Identified Issues:
- Insufficient training data (50 samples)
- High variance between best and average performance
- Targets not yet achieved, requires optimization
```

### Training Times

| Setup | Samples | Epochs | Time | Notes |
|-------|---------|--------|------|-------|
| CPU (current) | 500 | N/A | ~1-2h | Current models (Modell_1-9) |
| CPU (planned) | 500 | 50 | ~8h | Target configuration |
| CPU (planned) | 750 | 50 | ~12h | Extended dataset |
| GPU (planned) | 500 | 50 | ~1.5h | With CUDA optimization |

### Model Comparison (Current Results - September 2025)

| Model | Samples | Crystal Range | Best MAE | Correlation | Average MAE | Status |
|-------|---------|---------------|----------|-------------|-------------|--------|
| Modell_9 | 50 | 1:10 | 0.2546 | 0.778 | 0.4755 | Best performance |
| Modell_8 | 50 | 1:10 | 0.2574 | 0.722 | 0.4820 | Second best |
| Modell_1 | 50 | 1:10 | 0.2741 | 0.687 | 0.4968 | Baseline |
| Modell_3 | 49 | 1:10 | 0.2916 | 0.710 | 0.5005 | - |
| Modell_5 | 50 | 1:10 | 0.2919 | 0.669 | 0.5051 | - |
| Modell_7 | 50 | 1:10 | 0.3014 | 0.635 | 0.5099 | - |
| Modell_4 | 48 | 1:10 | 0.3116 | 0.632 | 0.4977 | - |
| Modell_2 | 50 | 1:10 | 0.3351 | 0.623 | 0.5479 | Worst performance |
| Modell_6 | 50 | 1:10 | 0.3714 | 0.576 | 0.5392 | - |

**Key Observation:** Large gap between best-case and average performance indicates inconsistent generalization across crystal counts. Further optimization required.

---

## Evaluation Framework

### Metric Categories (47 total)

**1. Error Metrics (8 metrics)**
- MAE, RMSE (total, v_x, v_z)
- Relative errors (L1, L2)
- Maximum absolute error

**2. Physical Consistency (12 metrics)**
- Divergence (mean, max, 90th percentile)
- Continuity violation
- Vorticity
- Boundary condition fulfillment
- Momentum balance

**3. Structural Similarity (9 metrics)**
- SSIM (total, per channel)
- Pearson correlation
- Spearman correlation
- Cosine similarity
- R² score

**4. Crystal-Specific (10 metrics)**
- Alignment error (mean, max, std)
- Crystal detection rate
- False positives/negatives
- Peak velocity accuracy
- Near-field flow error

**5. Multi-Crystal (8 metrics)**
- Scaling behavior (MAE vs. N)
- Interaction accuracy
- Coverage rate
- Performance degradation

### Statistical Validation

```julia
# Confidence intervals (95%)
confidence_intervals = compute_confidence_intervals(results)

# ANOVA for multi-crystal comparisons
anova_results = perform_anova_analysis(batch_results)

# Effect sizes (Cohen's d)
effect_sizes = calculate_cohens_d(group1, group2)

# Bonferroni correction for multiple tests
bonferroni_threshold = 0.05 / n_comparisons
```

---

## Output Structure

```
optimized_results/
│
├── ten_crystal_checkpoints_optimized/
│   ├── best_model.bson
│   ├── checkpoint_epoch_X.bson
│   └── final_model.bson
│
├── ten_crystal_results_optimized/
│   ├── ten_crystal_dataset.jls
│   └── training_results.bson
│
├── evaluation_results/
│   ├── comprehensive_metrics.csv
│   ├── statistical_analysis.json
│   ├── aggregated_statistics.csv
│   └── latex_tables/
│       ├── main_results_table.tex
│       ├── detailed_metrics_table.tex
│       └── scaling_analysis_table.tex
│
└── visualizations/
    ├── training_curves.png
    ├── divergence_comparison.png
    ├── performance_scaling.png
    └── crystal_comparisons/
```

---

## Known Issues and Solutions

### GPU Kernel Compilation Errors

**Issue:** GPUCompiler.KernelError for non-bitstype arguments

**Solution:** Automatic CPU fallback implemented. GPU training functional for simple operations.

**Status:** Limited GPU support; full optimization in progress.

### LaMEM Time Namespace Conflict

**Issue:** UndefVarError: Time not defined

**Solution:** Use explicit module qualification: `LaMEM.Time(nstep_max=1)`

**Status:** Resolved.

### GPU Memory Overflow

**Issue:** OOM errors with large batches

**Solution:** Adaptive batch sizing based on resolution; automatic garbage collection.

**Status:** Resolved.

### Zygote Mutating Arrays

**Issue:** Mutating arrays not supported in automatic differentiation

**Solution:** Replaced mutable operations with immutable tuple-based approach.

**Status:** Resolved.

---

## Scientific Validation

### Physical Consistency

- Continuity equation: ∇·v ≈ 0 (implementation complete, validation pending)
- Stokes regime: Low Reynolds number implementation verified
- Boundary conditions: No-slip at crystal surfaces (in training pipeline)
- Momentum balance: Physics-informed loss integrated
- Coordinate alignment: To be evaluated in next iteration

### Current Validation Status

**Completed:**
- Physics-informed loss implementation
- Training pipeline with 1-10 crystal range
- Basic performance metrics (MAE, correlation)
- Model comparison across 9 variants

**In Progress:**
- Target performance achievement (MAE < 0.05)
- Extended dataset generation (50 → 500+ samples)
- Comprehensive physical consistency validation
- Statistical significance testing

**Planned:**
- Divergence analysis and quantification
- Boundary condition verification
- Benchmark validation (DKT, Richardson-Zaki)
- Cross-validation and robustness testing

### Benchmark Comparisons

| Benchmark | Description | Status |
|-----------|-------------|--------|
| Single-crystal baseline | Target: MAE<0.01, R²>0.90 | Planned |
| Current best (Modell_9) | MAE=0.2546, Corr=0.778 | Achieved |
| Multi-crystal (1-10) | Average MAE=0.48, Corr=0.65 | In progress |
| DKT sequence | Qualitative two-crystal dynamics | Planned |
| Richardson-Zaki | Hindered settling scaling | Planned |
| LBM-DEM | High-fidelity comparison | Future work |

**Performance Gap Analysis:**
- Current best MAE (0.2546) is 5× above target (0.05)
- Training data insufficient (50 vs. target 500+ samples)
- Inconsistent performance across crystal counts requires investigation

---

## Development Roadmap

### Immediate Priorities (Q4 2025)

- Increase training dataset to 500+ samples (currently 50)
- Optimize physics loss weighting (λ-parameter tuning)
- Extended training duration with early stopping
- Achieve target performance: MAE < 0.05, Correlation > 0.85
- Comprehensive evaluation across 1-15 crystal range

### Short-term (Q1 2026)

- GPU kernel optimization for full support
- Hyperparameter tuning with Optuna
- K-fold cross-validation
- Detailed performance analysis per crystal count
- Richardson-Zaki benchmark validation

### Medium-term (Q2-Q3 2026)

- 3D extension for volumetric flow fields
- Uncertainty quantification (Bayesian NNs)
- Transfer learning to other systems
- Real-world experimental validation
- ArXiv pre-print publication

### Long-term (2026+)

- Multi-scale modeling (micro-macro coupling)
- LaMEM integration as surrogate model
- Industrial application (magma chamber modeling)
- Online learning capabilities
- Hybrid ML-numerical solver

---

## References

### Primary Literature

1. Ronneberger et al. (2015): U-Net Architecture
2. Raissi et al. (2019): Physics-Informed Neural Networks
3. Peng et al. (2021): Deep Fluids (CFD with ML)
4. Lagaris et al. (1998): Neural Networks for PDEs
5. Pellegrin et al. (2022): Transfer Learning in Earth Sciences

### Software Documentation

- [LaMEM.jl](https://github.com/JuliaGeodynamics/LaMEM.jl)
- [Flux.jl](https://fluxml.ai/Flux.jl/stable/)
- [CUDA.jl](https://cuda.juliagpu.org/stable/)

---

## Contact

**Author:** Paul Baselt  

### Citation

```bibtex
@mastersthesis{baselt2025physics,
  title={Physics-Based Machine Learning for Predicting Flow Fields 
         in Multi-Crystal Sedimentation Systems},
  author={Baselt, Paul},
  year={2025},
  school={[University Name]},
  type={Master's thesis}
}
```

---

## License

[To be specified]

---

*Last updated: September 2025 - Physics-Informed Training implemented, performance optimization in progress*