# Presentation Script – Master's Thesis Defense
**Paul Baselt · JGU Mainz · April 2026**
*Talk: ~30 minutes · Q&A afterwards*

> Note on numbers: all figures below match the submitted thesis (Abstract, Tables 7.1, 7.4–7.6, 7.8–7.10). A few values that were rounded differently on earlier slides have been aligned to the thesis (e.g. FNO Exp 1 ψ-error 1.6–2.5 %; U-Net Exp 2 in-distribution 17–26 % on ψ; U-Net small-crystal 10.8 %).

---

## Schedule (20 slides)

| # | Slide | Time |
|---|---|---|
| 1 | Title | 0:45 |
| 2 | Agenda | 0:45 |
| 3 | §01 divider – Motivation | 0:10 |
| 4 | Geoscientific context | 2:15 |
| 5 | Research question & hypotheses | 2:15 |
| 6 | §02 divider – Physics | 0:10 |
| 7 | Stokes flow & stream function | 2:45 |
| 8 | §03 divider – Methodology | 0:10 |
| 9 | Methodological workflow | 2:00 |
| 10 | Architecture comparison | 2:45 |
| 11 | §04 divider – Experiments | 0:10 |
| 12 | Experimental design | 2:00 |
| 13 | Exp 1 – single-crystal baseline | 2:30 |
| 14 | Exp 2 & 3 – multi-crystal | 2:45 |
| 15 | Exp 4 – size generalization | 2:30 |
| 16 | §05 divider – Discussion | 0:10 |
| 17 | Discussion: causes & implications | 2:30 |
| 18 | Conclusion & answers | 2:00 |
| 19 | Outlook | 1:00 |
| 20 | Thanks & questions | 0:25 |
| | **Total** | **~30 min** |

*(These same scripts are embedded as speaker notes in `Verteidigung_Baselt_EN_v2.pptx`, visible in Presenter View.)*

---

## Slide 1 – Title (0:45)

Good afternoon. My name is Paul Baselt, and thank you all for being here. I am pleased to present my Master's thesis in Computational Sciences, titled *"Data-Driven Stokes Flow Prediction in Multi-Crystal Sedimentation – A Systematic Comparison of Spatial and Spectral Neural Surrogate Architectures."*

The work was supervised by Professor Kaus at the Institute of Geosciences and Professor Wand at the Institute of Computer Science. I will walk you through the key ideas in about thirty minutes and look forward to your questions afterwards.

---

## Slide 2 – Agenda (0:45)

Here is the structure of the talk. I will begin with the motivation and the underlying geophysical problem, and derive the research question from it. Then a brief look at the physics and the mathematical formulation we rely on. After that, the methodology: the two architectures I compare and the experimental design. I will then present the results of four experiments, and close with a discussion, the answers to my hypotheses, and an outlook on future work.

---

## Slide 3 – §01 Divider (0:10)

Let me start with the motivation: why do we need surrogate models for crystal sedimentation at all?

---

## Slide 4 – Geoscientific Context (2:15)

One branch of geophysics deals with simulating geological processes, to understand how structures or sedimentary deposits form. My focus is the settling of crystals in a magma chamber.

As the melt cools, crystals nucleate and grow, and begin to sink due to the density contrast between solid and melt. This sedimentation controls the chemical differentiation of the rock and produces the layered structures preserved in plutonic bodies.

The numerical challenge is two-fold. We must simultaneously resolve the microscopic boundary layers around each individual crystal and the long-range hydrodynamic interactions that span the whole domain. The interaction cost scales roughly with the square of the particle number, so for more than ten crystals the systems become very large, with runtimes of hours to days per simulation on a fixed 256×256 grid. Systematic parameter studies are therefore practically infeasible.

Everything happens in the Stokes regime – Reynolds number far below one – where viscous forces dominate and inertia is negligible. That is the motivation: can we train a fast surrogate that predicts these flow fields in milliseconds instead of hours?

---

## Slide 5 – Research Question & Hypotheses (2:15)

The central research question is: *to what extent can a U-Net and a Fourier Neural Operator, trained on configurations with up to a fixed maximum number of crystals, generalize to unseen scenarios with varying particle numbers and arrangements – and which factors fundamentally limit this generalization?*

This is directly relevant to geoscience: natural sedimentation rarely involves a fixed crystal count or regular geometry, so a useful surrogate must work across a broad range of configurations.


Before the experiments, let me briefly explain the physics and why we formulate the problem this way.

---

## Slide 6 – §02 Divider (0:10)

First the physics – Stokes flow and the stream-function formulation.

---

## Slide 7 – Stokes Flow & Stream Function (2:45)

**Step 1 — From Navier-Stokes to Stokes.**

The full incompressible Navier-Stokes equation is:

  ρ (∂u/∂t + (u·∇)u) = −∇p + η∇²u + f

Here the left-hand side contains the inertial terms: the local acceleration and the convective acceleration. The right-hand side contains the pressure gradient, the viscous term, and the body force f (in our case, buoyancy ρg from the density contrast between crystal and melt).

Crystal settling in magma occurs at Reynolds numbers Re = ρ U L / η typically between 10⁻⁸ and 10⁻⁴. At Re ≪ 1, the ratio of inertial to viscous forces is negligible — the left-hand side vanishes — and the flow reaches a quasi-static equilibrium instantaneously. We also have a steady state (no time dependence), so ∂u/∂t = 0 as well. What remains are the incompressible Stokes equations:

  −∇p + η∇²u + ρg = 0   (momentum balance)
  ∇·u = 0                 (mass conservation / incompressibility)

The slide shows the homogeneous form without the body-force term; the buoyancy ρg is the actual driver of settling and is implicitly present in all LaMEM training data.

**Step 2 — Why not predict the velocity directly?**

The incompressibility constraint ∇·u = ∂uₓ/∂x + ∂u_z/∂z = 0 must hold at every grid point. If a network predicts u directly, it must learn this global constraint purely from data — and in practice that leads to small but systematic divergence errors. We take a different route that enforces incompressibility analytically.

**Step 3 — Introducing the stream function ψ.**

In 2D, any divergence-free vector field can be written as the curl of a scalar potential. We define the stream function ψ(x, z) by:

  uₓ = +∂ψ/∂z
  u_z = −∂ψ/∂x

Let us verify immediately that this is divergence-free. The divergence of u is:

  ∇·u = ∂uₓ/∂x + ∂u_z/∂z
       = ∂(∂ψ/∂z)/∂x + ∂(−∂ψ/∂x)/∂z
       = ∂²ψ/∂x∂z − ∂²ψ/∂z∂x
       = 0

The last step follows from Schwarz's theorem: for any sufficiently smooth ψ, the mixed partial derivatives commute, so the two terms cancel identically. This is not a soft constraint the network must learn — it is an algebraic identity. Every prediction ψ̂, no matter how inaccurate, yields a velocity field that is exactly divergence-free.

**Step 4 — Deriving the Poisson equation for ψ.**

To reconstruct ψ from the LaMEM velocity output, we use vorticity as an intermediate. In 2D the vorticity is the scalar:

  ω = ∂u_z/∂x − ∂uₓ/∂z

Substituting the stream-function definitions:

  ω = ∂(−∂ψ/∂x)/∂x − ∂(∂ψ/∂z)/∂z
    = −∂²ψ/∂x² − ∂²ψ/∂z²
    = −∇²ψ

Rearranging: **∇²ψ = −ω** (the Poisson equation for ψ).

This is the equation we solve numerically. We first compute ω from the LaMEM velocity field by finite differences, then solve this Poisson problem. We impose homogeneous Dirichlet boundary conditions, ψ|∂Ω = 0, for two reasons: physically, this makes the domain boundary a single streamline — consistent with the free-slip walls in LaMEM, which permit no normal flow; mathematically, it fixes the gauge, removing the freedom to add any constant to ψ, and makes the Poisson problem well-posed with a unique solution. This ψ is the learning target for both networks.

A remark for context: taking the curl of the Stokes momentum equation eliminates the pressure gradient and yields ∇²ω = 0, i.e. the vorticity is harmonic in the absence of body forces. Combined with ∇²ψ = −ω, this implies ∇⁴ψ = 0 — the biharmonic equation — which is the governing PDE for the stream function of Stokes flow. We do not solve this directly; instead, we compute ω from simulation output and solve the simpler Poisson step. But it is good to know that ψ satisfies a well-posed elliptic PDE, which justifies expecting smooth, well-behaved solutions.

**Step 5 — Recovering the velocity from a predicted ψ̂.**

From any predicted ψ̂ we recover the velocity field by numerical differentiation:

  ûₓ ≈ (ψ̂[x, z+1] − ψ̂[x, z−1]) / (2Δz)
  û_z ≈ −(ψ̂[x+1, z] − ψ̂[x−1, z]) / (2Δx)

This step does introduce additional error. Numerical differentiation is a high-pass filter: if the residual ψ̂ − ψ contains a high-frequency component of the form ε sin(kx), the corresponding velocity error scales as kε. The higher the spatial frequency of the prediction error, the more it is amplified in the velocity. This is why velocity errors are systematically larger than ψ errors — roughly 2× for the FNO and 3× for the U-Net without the composite loss. I will return to this when discussing the results.

Now to the two architectures.

---

## Slide 8 – §03 Divider (0:10)

Now the methodology – the two architectures, trained and evaluated on identical data.

---

## Slide 9 – Methodological Workflow (2:00)

The pipeline is as follows. LaMEM produces a steady-state Stokes solution on the 256×256 finite-difference grid. From the velocity we reconstruct the stream function via the Poisson solve with Dirichlet boundary conditions. The geometry is then encoded into a four-channel input tensor, the model predicts ψ̂, and we evaluate with relative L² error, mean absolute error, and the divergence.

The four input channels are: (1) a binary crystal mask marking occupancy; (2) a signed distance field giving the distance to the nearest crystal surface; and (3, 4) the normalized x- and z-coordinates, which give the network absolute spatial position.

Crucially, both architectures receive exactly the same input encoding and output target, and are judged by the same metrics. This makes it a controlled, like-for-like comparison – any performance difference is attributable to the architecture, not the data.

---

## Slide 10 – Architecture Comparison (2:45)

I compare two fundamentally different neural network architectures. Both receive the same input – a 256×256 image encoding the crystal geometry – and output a predicted stream function ψ̂. But they are structured in completely different ways, and that structural difference is what the experiments probe.

---

**The U-Net** was originally developed for biomedical image segmentation – think of it as a network that was designed to look at a scan and outline which pixels belong to a tumour. The name comes from the shape of its architecture when drawn on paper: it looks like the letter U.

The left side of the U is the **encoder**: it progressively compresses the 256×256 input, halving the spatial resolution at each step while extracting more abstract features. Think of it as zooming out to see the bigger picture – the network starts by looking at fine local details, like the edges of a single crystal, and gradually builds up to coarser, more global patterns.

The right side is the **decoder**: it reverses this process and reconstructs the full-resolution output. But here is the clever part – **skip connections** directly connect each encoder level to its mirror level in the decoder, handing fine spatial detail that would otherwise be lost during compression directly to the reconstruction stage.

The key limitation is that global context – understanding how a crystal on the left of the domain affects the flow on the right – only emerges *indirectly*, through many successive pooling steps. The network has to piece together long-range interactions from local building blocks.

In terms of technical choices: the U-Net uses Batch Normalization to stabilize training, resize-convolutions instead of transposed convolutions to avoid checkerboard artifacts in the output, and a composite loss function that penalizes not only the error in ψ itself, but also errors in its spatial gradients and at the domain boundary.

---

**The Fourier Neural Operator** – or FNO – takes a fundamentally different approach. Rather than learning local patterns that it assembles step by step, it works directly in the *frequency domain*.

Here is the intuition: any spatial field, including a flow field on a grid, can be decomposed into a sum of sinusoidal waves of different frequencies. A fast Fourier transform does this decomposition. The FNO applies this transform to its input, then learns which frequencies to amplify or suppress – these are the learnable weights, operating on the 16 lowest Fourier modes. An inverse transform then returns the result to physical space. Running in parallel, a simple pointwise operation acts directly on the spatial signal, so the network can also capture features the spectral path might miss. Both paths are summed and passed through a nonlinearity.

The decisive advantage: a single Fourier layer already has a **global receptive field**. Every point in the output depends on every point in the input after just one pass – not after many pooling steps like the U-Net, but immediately. This is structurally well matched to the Stokes equations, which are *elliptic*: a small perturbation anywhere in the domain – say, a single crystal – propagates its influence throughout the entire flow field. An architecture with a global receptive field can capture this naturally.

---

So in short: the U-Net builds up global understanding gradually from local operations; the FNO sees the whole domain at once through spectral decomposition. This structural difference is the central hypothesis of the thesis, and the experiments test whether – and under what conditions – it actually matters.

Now to the experiments.

---

## Slide 11 – §04 Divider (0:10)

With these two architectures I ran four experiments, from a single crystal up to crystal-size generalization.

---

## Slide 12 – Experimental Design (2:00)

The four experiments are structured systematically.

- **Experiment 1** is the single-crystal baseline with exactly one crystal; here I ran a hyperparameter sweep over four learning rates and two batch sizes, to establish a reference for each architecture in the simplest setting.
- **Experiment 2** is multi-crystal generalization: both models are trained on 1–10 crystals and evaluated per crystal count, including unseen higher counts up to 25.
- **Experiment 3** is a stress test: training on up to 25 crystals and evaluating on much denser scenes, up to 84 crystals, to see whether the generalization range scales with the breadth of the training distribution.
- **Experiment 4** changes the variable from number to size: a model trained only on radius r = 0.05 is applied to very small crystals (r = 0.005) and very large ones (r = 0.5). I additionally train an FNO on three radii at once, to quantify the effect of size variability in the training data.

All data come from LaMEM and are stored as pre-computed JLD2 files. Let us look at the results.

---

## Slide 13 – Experiment 1: Single-Crystal Baseline (2:30)

In the single-crystal baseline the FNO has a clear advantage. The best FNO configurations reach mean relative L² errors of **1.6–2.5 %** on the stream function ψ, against **5.5–6.5 %** for the best U-Net – roughly a factor of three. On the derived velocity field the FNO is at about **3.0–4.2 %**, the U-Net at roughly **8.5–12.7 %**.

An important point concerns training dynamics. The FNO converges smoothly and monotonically. The U-Net shows pronounced oscillations in the validation loss, especially at higher learning rates and at the larger batch size of 16. We attribute this to Batch Normalization, which makes the effective learning rate uneven across layers – this connects to recent work by Mehmeti-Göpel and Wand (ICML 2024).

On the loss: the composite loss – MSE plus a gradient and a boundary term – reduces the U-Net's velocity-to-ψ error ratio from about 3× to about 1.5–2× at batch size 8. At batch size 16 it can backfire, which highlights how sensitive multi-term losses are to the training configuration.

Now to the multi-crystal scenarios.

---

## Slide 14 – Experiments 2 & 3: Multi-Crystal Generalization (2:45)

In the multi-crystal experiments the FNO again leads clearly. Within the training distribution, in Experiment 2 the FNO is at **4–9 %** on ψ, the U-Net at roughly **17–26 %** – again a factor of 3–4. In Experiment 3 the in-distribution means are about **6.8 %** for the FNO and about **23 %** for the U-Net.

Physically this makes sense. Stokes flow is linear, so the total field is approximately a superposition of the single-crystal disturbances. Representing such superpositions accurately needs global correlations across the whole grid – exactly what the FNO does naturally through its spectral operations. The U-Net has to build these global relationships indirectly through its pooling cascade.

Out of distribution: in Experiment 2, trained on up to 10 crystals, the FNO error first exceeds 10 % at n = 20 (≈ 10.1 %), while the U-Net sits around 22 % on ψ. In Experiment 3, trained on up to 25 crystals, the FNO reaches 12.4 % at n = 50, the U-Net 21.5 %.

The key finding: the out-of-distribution reach scales with the breadth of the training distribution. The broader Experiment 3 model stays below 10 % up to about n = 35, considerably further than Experiment 2's reach of about n = 20. At very high counts (n = 67–84) the error varies widely, depending on whether an arrangement happens to resemble the training data.

The final experiment shifts perspective: what if it is not the number but the size of the crystals that changes?

---

## Slide 15 – Experiment 4: Crystal Size Generalization (2:30)

Experiment 4 gives the most surprising result of the thesis: **the architectural ranking reverses.**

For small out-of-distribution crystals (r = 0.005 vs. the training radius 0.05), both models are comparable: the FNO at **12.1 ± 6.4 %** on ψ, the U-Net at **10.8 ± 6.8 %**. Small crystals produce weak, diffuse perturbations that both can partly represent.

For large crystals (r = 0.5) the picture flips. The FNO reaches **71.3 ± 4.8 %** – essentially unusable – while the U-Net is at **44.9 ± 8.2 %**: still high, but markedly better.

The reason is the fixed spectral truncation at 16 Fourier modes. For large crystals the boundary layer at the surface is high-frequency relative to the domain, and those frequencies lie above the mode cutoff, so the FNO simply cannot represent them. The U-Net's local convolutional filters operate at fixed grid scales and remain geometrically similar at the crystal surface regardless of crystal size.

The remedy is striking: an FNO trained on three radii (r ∈ {0.02, 0.05, 0.08}) reduces the mean out-of-distribution error from 71 % to about **10.4 %**. So the spectral limitation is not a fundamental barrier – it can be overcome through training-data diversity, without any architectural change.

Let me pull everything together.

---

## Slide 16 – §05 Divider (0:10)

Now the discussion: causes, implications, and limitations.

---

## Slide 17 – Discussion: Causes & Implications (2:30)

Four points.

First, **FNO stability vs. U-Net oscillations**: Batch Normalization in the U-Net widens the spread of effective learning rates across layers, increasing sensitivity to learning rate and batch size. The FNO, without Batch Normalization, converges monotonically.

Second, the **spectral truncation at k_max = 16** is precisely what causes the large-crystal failure: boundary-layer structures need higher Fourier frequencies. The remedy is either broader training variation or adaptive Fourier modes.

Third, the **stream function as inductive bias**: predicting ψ guarantees exact incompressibility – the measured divergence RMS is on the order of 10⁻²⁶, numerically zero – with no soft penalty, in contrast to direct velocity prediction.

Fourth, **velocity error > ψ error**, because numerical differentiation amplifies high-frequency residuals; the ratio is about 2× for the FNO, about 3× for the U-Net without the composite loss, and 1.5–2× with it.

And the payoff in efficiency: the FNO needs about **687 ms** end-to-end per sample – roughly a **12× speedup** over LaMEM's 8–9 s – while the pure U-Net forward pass takes only **5.5 ms**, over 1500× faster. This is what makes systematic studies and long-duration simulations tractable.

---

## Slide 18 – Conclusion & Answers (2:00)

To the hypotheses:
- **H1 — confirmed for the FNO:** sufficient sampling and a broad training distribution enable OOD generalization; for the U-Net it holds only weakly.
- **H2 — confirmed:** the ψ-formulation reduces task complexity and guarantees exact incompressibility for every prediction.
- **H3 — conditionally confirmed:** the FNO dominates in crystal-count generalization, but there is a rank reversal for crystal-size generalization, where the U-Net is better for large crystals.

From this I derive practical guidelines: prefer the FNO when generalization across crystal counts matters; prefer the U-Net when robustness to geometric scaling is required; for size generalization, broadening the training data beats changing the architecture; and always prefer the stream function over direct velocity prediction.

I should be open about the limitations. All simulations are 2D and steady-state; the datasets are relatively small; and the FNO in Experiments 2 and 3 was trained for fewer epochs than the U-Net (100 vs. 180), so a fully fair comparison would extend FNO training, which would likely improve its numbers further.

---

## Slide 19 – Outlook (1:00)

Several directions follow. Adaptive Fourier modes or a Geo-FNO would give robust size generalization without discrete training radii. Hybrid architectures such as the U-FNO (Wen et al. 2022) combine local U-Net processing with global spectral layers. The natural extension is to three dimensions and time-dependent flows for realistic geophysical applications.

Because Stokes flow is quasi-static, the surrogate can be embedded as a building block in a time-stepping scheme – milliseconds per step instead of seconds – enabling long-duration sedimentation simulations. Finally, uncertainty quantification through Bayesian methods or ensembles would give confidence estimates for the predictions.

---

## Slide 20 – Thanks & Questions (0:25)

This brings me to the end. I sincerely thank Professor Kaus and Professor Wand for their excellent supervision, and the examination committee for your time and attention. I am happy to take your questions.

---
---

# Q&A Preparation

*Grouped by theme. The mathematical/formulation block is the one most likely to be probed by the mathematics examiner.*

## A. Mathematical formulation

**Is the stream function ψ uniquely determined?**
In general ψ is defined only up to an additive constant. We fix this gauge with the homogeneous Dirichlet condition ψ|∂Ω = 0, so the boundary is a single streamline — consistent with the free-slip walls. The resulting Poisson problem ∇²ψ = −ω with Dirichlet data is well-posed and has a unique solution.

**Why does predicting ψ guarantee ∇·u = 0?**
With u = (∂ψ/∂z, −∂ψ/∂x), the divergence is ∂²ψ/∂x∂z − ∂²ψ/∂z∂x. For any sufficiently smooth ψ the mixed partials are equal (Schwarz/Clairaut), so this is identically zero — including for the network's output. That is what "incompressible by construction" means: it is an algebraic identity, not something learned.

**The momentum equation on Slide 7 has no body-force term — is that an omission?**
The slide shows the homogeneous Stokes operator. The full balance is 0 = −∇p + η∇²u + ρg; the buoyancy term ρg from the density contrast is the actual driver of settling. In the vorticity formulation, taking the curl eliminates the pressure gradient and the buoyancy enters as a source. Since I reconstruct ψ from the *simulated* velocity field, the driving force is implicitly present in the training data.

**Why solve a Poisson equation for ψ rather than integrating u directly?**
Path-integration of u to recover ψ is path-dependent and accumulates error whenever the discrete velocity field is not exactly divergence-free. The Poisson solve ∇²ψ = −ω is a global, well-posed projection that is robust and respects the boundary conditions. Importantly, this step introduces no new physics — it is an auxiliary transformation of the reference solution.

**How is the relative L² error defined?**
‖ψ̂ − ψ‖₂ / ‖ψ‖₂, the discrete L² (Frobenius) norm over the grid, reported as mean ± standard deviation over the evaluation samples in each stratum.

**Why an MSE loss and not a Sobolev / H¹ loss, since the velocity is a derivative of ψ?**
The primary target is ψ, so MSE is the natural base loss. Because the velocity is a derivative, it is sensitive to high-frequency residuals — that is exactly why the U-Net uses a *composite* loss with a gradient-matching term (an H¹-type penalty). The FNO needed only MSE, because its spectral parametrization already limits high-frequency content. A full Sobolev-norm loss is a sensible refinement for future work.

**Does the stream-function trick generalize to 3D?**
Not directly — a scalar ψ is specific to 2D. In 3D one enforces ∇·u = 0 with a vector potential A via u = ∇×A, which is a different (and richer) formulation. This is one reason the 3D extension is non-trivial and listed under future work.

## B. FNO-specific

**Why k_max = 16 modes?**
It is the number of retained low Fourier modes per dimension — a trade-off between expressiveness and cost/overfitting. Sixteen was a good accuracy/cost balance for r = 0.05 crystals. It is exactly this fixed cutoff that the large-crystal case in Experiment 4 exposes.

**Isn't the FNO supposed to be resolution-invariant? Then why does crystal size matter?**
Resolution-invariance refers to the *grid*: the operator is defined on Fourier modes in function space, so it can be evaluated at different discretizations. But a fixed k_max caps the representable frequency band regardless of grid resolution. Large crystals push spectral energy above that band, so the truncation — not the grid — is the limiting factor.

**What about universal approximation?**
Neural operators including the FNO have universal-approximation results for continuous operators (Kovachki et al., 2021/2023). That motivates the architecture but says nothing about finite-mode, finite-sample generalization — which is precisely what the experiments measure.

**The domain isn't periodic, but the FFT assumes periodicity — aliasing?**
Correct, this is a known FNO caveat. The periodic assumption can create boundary artifacts. They are mitigated by the parallel 1×1 spatial bypass, by the Dirichlet-consistent target (ψ = 0 on the boundary), and by the explicit coordinate channels that let the model localize features.

## C. Physics & generalization

**Is the total flow exactly the sum of the single-crystal fields?**
Stokes flow is linear in u for a fixed geometry and boundary conditions, so disturbances are long-ranged and additive at leading order (Happel & Brenner). But with several rigid inclusions, the presence of each particle modifies the others' boundary conditions, so the exact field is *not* a naive sum. "Approximately superposition" captures the leading-order, long-range additivity — and explains why global correlations (the FNO's strength) help.

**How are the crystals modeled — as no-slip boundaries?**
As high-viscosity circular inclusions in LaMEM (a viscosity/phase-contrast formulation), effectively rigid relative to the melt — not as explicit no-slip boundaries. The domain walls use free-slip conditions. Collision and lubrication forces are not modeled.

**Why does a broader training distribution help OOD?**
It increases coverage of the configuration space — more diverse wake and interaction patterns — so unseen test configurations lie closer to training examples. The operator then interpolates rather than extrapolates. This is why Experiment 3 (n ≤ 25) reaches further than Experiment 2 (n ≤ 10).

**Are the ± values statistically robust given small samples?**
Errors are means over 10 held-out samples per crystal-count stratum (more for several counts), with standard deviations reported. A caveat: for n = 1 the 1000 training samples densely cover the 2D configuration space, so "OOD" there is closer to interpolation than to genuine extrapolation.

**Why 256×256 and not higher resolution?**
A trade-off between accuracy and trainability. The LaMEM discretization error at 256×256 is already well below the observed 1–10 % model errors, so resolution is not the bottleneck. Higher resolution would make data generation and training substantially more expensive.

## D. Practical scope

**Could the surrogate replace the solver in a time-stepping loop in real time?**
Yes — that is the natural next step. Stokes flow is quasi-static (no time memory), so the surrogate can be embedded directly in a time-stepping scheme. Error accumulation across many steps still needs to be validated.

**Why not Physics-Informed Neural Networks (PINNs)?**
PINNs optimize per instance at inference time (slow) and scale poorly across many configurations, making them impractical for the systematic parameter studies this work targets.

**What does this mean for real magma chambers?**
Results transfer directly to any parameter combination with Re ≪ 1 and the same geometric setup. Qualitative screening of crystal arrangements is immediately feasible; quantitative trajectory calculations over many timesteps require higher accuracy and a study of error accumulation.
