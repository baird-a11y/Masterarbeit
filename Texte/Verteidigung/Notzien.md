# Presentation Script – Master's Thesis Defense
**Paul Baselt · JGU Mainz · April 2026**
*Total duration: ~30 minutes · Q&A: ~15 minutes*

---

## Schedule

| Section | Slides | Time |
|---|---|---|
| Title slide | 1 | 1 min |
| Agenda | 2 | 1 min |
| 01 Motivation | 3–4 | 4 min |
| 02 Physics | 5 | 3 min |
| 03 Methodology | 6–7 | 6 min |
| 04 Experiments & Results | 8–11 | 11 min |
| 05 Discussion & Conclusion | 12–14 | 4 min |
| Thanks & Questions | 15 | 1 min |
| **Total** | | **~31 min** |

---

## Slide 1 – Title Slide (1 min)

Good afternoon, my name is Paul Baselt. I am pleased to present my master's thesis to you today, which carries the title:

*"Data-Driven Stokes Flow Prediction in Multi-Crystal Sedimentation – A Systematic Comparison of Spatial and Spectral Neural Surrogate Architectures"*

The thesis was supervised by Professor Kaus at the Institute of Geosciences and Professor Wand at the Institute of Computer Science. I will try to guide you through the key aspects in roughly thirty minutes and look forward to your questions afterwards.

---

## Slide 2 – Agenda (1 min)

Briefly on the structure: I will start with the motivation and the central research question, then briefly cover the physical background, explain the two architectures and the methodology, present the four experiments with their most important findings, and close with discussion and conclusion.

---

## Slide 3 – Geoscientific Context (2 min)

When magma chambers cool or their chemical composition changes, minerals begin to crystallize. Due to the density contrast between the crystals and the surrounding melt, these crystals sink toward the chamber floor – a process known as crystal sedimentation.

Simulating this is computationally expensive, particularly when large numbers of crystals with varying sizes and shapes are involved. The simulation must resolve not only the flow around each individual crystal, but also the hydrodynamic interactions between them – interactions that can propagate over surprisingly large distances through the melt.

[INSERT COST EXAMPLE – e.g. "A single LaMEM solve for n=50 crystals on a 256×256 grid takes approximately 9 seconds."]

---

## Slide 4 – Research Question (2 min)

From this problem I derived the central research question: to what extent is it possible to predict the velocity fields of multi-crystal Stokes flow using a neural surrogate model – and how accurate can such predictions be, and where are the limits?

To answer this, I designed four experiments that test the models under systematically increasing difficulty. Throughout all experiments I compare two fundamentally different architectures: the U-Net as a spatial convolutional approach, and the Fourier Neural Operator as a spectral approach.


---

## Slide 5 – Physical Background (3 min)

In the Stokes regime – i.e. at very low Reynolds number, as is the case in magma – the Navier-Stokes equations simplify considerably. The inertia term vanishes, and we obtain the Stokes equations: negative pressure gradient plus viscous terms equals zero, combined with the incompressibility condition.

The key idea of my thesis: instead of predicting the velocity components ux and uz directly, I predict the stream function ψ. The connection is straightforward: ux is the partial derivative of ψ with respect to z, and uz is minus the partial derivative with respect to x. This means any smooth function ψ automatically yields a divergence-free velocity field – incompressibility is guaranteed by construction, without any penalty term in the loss function.

I reconstruct ψ from the LaMEM simulations by solving the Poisson equation ∇²ψ = −ω, where ω is the vorticity. This gives me a physically consistent scalar learning target on the 256×256 grid.

---

## Slide 6 – Architecture Overview (3 min)

I compare two architectures that approach the problem from fundamentally different angles.

## Slide 6.1 – U-Net

The U-Net is an encoder-decoder architecture originally developed for biomedical image segmentation. The encoder path progressively reduces spatial resolution while increasing feature depth; the decoder path restores resolution step by step. The distinguishing feature are skip connections that bridge corresponding encoder and decoder levels – they allow the network to combine high-level abstract features with fine-grained spatial detail.

In my implementation, the input is the binary crystal mask on a 256×256 grid. Each convolutional block is followed by Batch Normalization, which – as we will see – has implications for training stability.

## Slide 6.2 – FNO

The Fourier Neural Operator by Li et al. 2021 works fundamentally differently. Instead of local convolutions in physical space, it applies learned linear transformations in frequency space. The core operation is: transform the input to Fourier space via FFT, multiply the lowest kmax modes by learned complex weights, transform back, and add a local linear bypass in physical space. Stacking these layers yields a network whose parameters are in principle decoupled from the input resolution – however, Gao et al. 2025 show that this does not guarantee consistent performance across resolutions in practice, as discretization mismatch errors accumulate across Fourier layers. The FNO captures long-range dependencies efficiently through global spectral operations.

In my implementation, kmax=16 modes are retained, and – crucially – no Batch Normalization is used.

---

## Slide 7 – Training Setup (3 min)

The training data is generated with LaMEM: for each sample, crystal positions and sizes are drawn randomly, LaMEM solves the Stokes equations, and I reconstruct the stream function ψ by solving the Poisson equation ∇²ψ = −ω on the 256×256 grid.

Both architectures are trained with the same composite loss: MSE on ψ plus a gradient penalty on the derived velocity components ux and uz. The gradient penalty is necessary because numerical differentiation amplifies high-frequency residuals in ψ – without it, the velocity errors would be disproportionately large relative to the ψ errors.

Optimization uses Adam with a cosine learning rate schedule. The hyperparameter sweep in Experiment 1 covers four learning rates and two batch sizes to establish the best configuration for both architectures.

---

## Slide 8 – Experimental Design (2 min)

I conducted four experiments. Experiment 1 is the single-crystal baseline with a hyperparameter sweep over four learning rates and two batch sizes. Experiment 2 tests multi-crystal generalization with training on n from 1 to 10 crystals. Experiment 3 is the stress test with training on up to 25 crystals and OOD testing at n=50. Experiment 4 investigates crystal size generalization: training on radius r=0.05, testing on r=0.005 (small) and r=0.5 (large).

---

## Slide 9 – Exp. 1: Results (2 min)

In the single-crystal baseline, the FNO is clearly superior. The best FNO achieves 1.6 to 2.3 percent relative L² error on ψ, compared to 5.5 to 6.5 percent for the best U-Net – roughly a factor of three in favor of the FNO.

The U-Net shows pronounced validation loss oscillations at high learning rates, particularly with batch size 16. This is consistent with the theoretical prediction of the Mehmeti-Göpel paper on effective learning rate spread through Batch Normalization.

An important technical observation: the composite loss – MSE plus a gradient penalty on the derived velocities – reduces the ratio of velocity error to ψ error in the U-Net from approximately 3× to approximately 1.5–2×. This is because numerical differentiation amplifies high-frequency residuals in ψ; the gradient penalty forces the network to suppress these.

---

## Slide 10 – Exp. 2 & 3: Multi-Crystal Generalization (4 min)

In the multi-crystal experiments the FNO advantage continues, but OOD generalization is challenging.

In Experiment 2, trained on up to 10 crystals, the FNO achieves in-distribution errors of 4 to 7 percent on ψ. The U-Net reaches 17 to 29 percent – the factor of 3 to 4 is thus maintained. In the OOD regime, at n=20, twice as many as during training, both errors clearly exceed the 10 percent mark.

The decisive finding comes in Experiment 3: when I extend the training distribution to n ≤ 25, the FNO can hold errors below 10 percent up to n=35 – substantially further than in Experiment 2. This shows: the breadth of the training distribution is the decisive factor for OOD reach. This is consistent with findings from the literature on multi-obstacle flow modeling.

It is also noteworthy that the U-Net shows a flatter error profile across crystal numbers – it does not degrade as abruptly as the FNO outside the training distribution, but its absolute accuracy is consistently worse.

---

## Slide 11 – Exp. 4: Rank Reversal (3 min)

Experiment 4 yields the most surprising result. For small OOD crystals with r=0.005, both architectures perform similarly at around 11 to 12 percent on ψ.

For large crystals with r=0.5, however, the FNO fails dramatically: 71.3 percent error on ψ. The U-Net reaches 44.9 percent – also far from training performance, but substantially better than the FNO. This is a complete reversal of the architecture ranking.

The cause is clear: large crystals produce boundary layer structures requiring high spatial frequencies. The FNO truncates at kmax=16 Fourier modes – it simply cannot represent these structures. The U-Net with its local convolutions and skip connections is more robust for sharp gradients at novel scales.

The solution: an FNO trained on three discrete radii reduces the OOD error for large crystals to 10.4 percent. The spectral constraint is therefore not a fundamental architectural barrier – it can be overcome through variability in the training data, without modifying the architecture.

---

## Slide 12 – Discussion (2 min)

Four key points for discussion:

First, FNO stability versus U-Net oscillations: this can be traced directly to Batch Normalization dynamics. Without BN, the FNO converges monotonically and robustly.

Second, spectral truncation at kmax=16: this is the Achilles' heel of the FNO for size-OOD. Adaptive Fourier modes, as in Geo-FNO, would be a natural next step.

Third, the stream function representation: divergence RMS values on the order of 10⁻²⁶ confirm that exact incompressibility is guaranteed for all predictions. This is a clear advantage over direct velocity prediction.

Fourth, the consistent finding that velocity errors exceed ψ errors: numerical differentiation amplifies high-frequency residuals – this is an inherent issue of the stream function approach, but it can be substantially mitigated through the composite loss.

---

## Slide 13 – Conclusion (2 min)

Summarizing the three hypotheses:

Hypothesis 1 – confirmed for the FNO: OOD generalization is possible, but scales with the breadth of the training distribution. Only weakly confirmed for the U-Net, as it shows high errors even in-distribution.

Hypothesis 2 – fully confirmed: the ψ formulation enforces exact incompressibility and simplifies the learning task.

Hypothesis 3 – conditionally confirmed: the FNO dominates in Experiments 1 through 3, but size-OOD in Experiment 4 produces a rank reversal. This shows: the "better" architecture is context-dependent.

The practical guidelines I derive: prefer the FNO when generalization over crystal numbers is required. Use the U-Net when geometric scale robustness is more important. And always: stream function over direct velocity prediction.

---

## Slide 14 – Outlook (2 min)

Five directions for future work:

Adaptive Fourier modes would architecturally resolve the size-OOD limitation of the FNO. Hybrid U-FNO architectures could combine the best of both worlds. Extension to 3D Stokes flow would be the next step toward geophysically realistic applications. The surrogate model is intended as a building block for a time-stepping framework: per time step the surrogate takes milliseconds instead of eight seconds. And finally, uncertainty quantification would substantially increase confidence in predictions for geophysical applications.

---

## Slide 15 – Thank You (1 min)

That was an overview of my work. The key takeaways: the FNO is the stronger model for multi-crystal Stokes flow – except when crystal size varies. The stream function formulation is a sound design decision. And breadth of training distribution beats architectural complexity for OOD generalization.

The code is publicly available on GitLab. I look forward to your questions.

---

## Potential Examiner Questions & Answers

**Q: Why exactly does Batch Normalization cause instability?**
BN increases the effective learning rate spread between early and late layers. Mehmeti-Göpel & Wand 2024 show theoretically that this leads to heightened sensitivity to the global learning rate – particularly when batch size and learning rate are scaled jointly, since BN depends on batch statistics.

**Q: Why did you choose kmax=16?**
This is the standard value from the FNO literature (Li et al. 2021). For the training domain r=0.05 this is sufficient; the dominant spectral features of the flow lie within this range. Experiment 4 then shows where this choice reaches its limits.

**Q: How was the Poisson equation for ψ solved numerically?**
With a 5-point stencil on the 256×256 grid, Kronecker product assembly of the discrete Laplace operator, homogeneous Dirichlet boundary conditions ψ|∂Ω = 0. This fixes the gauge and is consistent with the free-slip wall conditions of LaMEM.

**Q: Why are the datasets so small (1000 samples)?**
This is a central limitation that I discuss explicitly. For n=1 the configuration space is two-dimensional and 1000 samples cover it well. For higher crystal counts the space grows exponentially, and the samples represent only a slice. Larger datasets would likely improve generalization.

**Q: Did you measure the computational speedup?**
Yes – Section 7.5 of the thesis. FNO inference is orders of magnitude faster than a LaMEM solve (~8s vs. milliseconds), making long-time simulations feasible in the first place.

**Q: Why no PINNs?**
PINNs optimize separately for each new geometry – this is impractical for our application. Surrogate models like FNO and U-Net learn a mapping across the entire configuration space, enabling real-time inference.

**Q: What does the variable-radius FNO solution imply for architecture choice?**
It shows that spectral truncation is not a fundamental architectural barrier – it can be addressed through training data diversity. For practical deployment I would therefore recommend including variable crystal sizes in the training distribution before switching to adaptive Fourier modes.
