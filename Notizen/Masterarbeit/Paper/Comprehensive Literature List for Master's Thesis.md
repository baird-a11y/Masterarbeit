
## Physics-Based Machine Learning for Predicting Flow Fields in Multi-Crystal Sedimentation Systems

This curated reading list provides balanced coverage from foundational theory through cutting-edge applications, specifically tailored for a UNet-based approach to multi-crystal sedimentation with stream function and residual learning methods.

---

## 1. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations

**Citation:** Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2019). Journal of Computational Physics, 378, 686-707.

**Why this paper matters:** This is THE foundational paper that every researcher in physics-informed ML must read, with over 15,000 citations. It introduces the PINN framework that encodes physical laws (described by PDEs) directly into neural network training through automatic differentiation. Critically for your thesis, it demonstrates the methodology on classic fluid dynamics problems including Navier-Stokes equations and incompressible flow, showing how PINNs can infer velocity and pressure fields from sparse data. This establishes the theoretical foundation for physics-based constraints in your flow field predictions.

**Specific aspect:** Foundational PINN methodology – embedding physical laws as loss function constraints

---

## 2. Physics-informed machine learning

**Citation:** Karniadakis, G.E., Kevrekidis, I.G., Lu, L., Perdikaris, P., Wang, S., & Yang, L. (2021). Nature Reviews Physics, 3(6), 422-440.

**Why this paper matters:** This comprehensive review (4,288 citations) provides the broader landscape of physics-informed ML, covering neural operators, domain decomposition, and diverse applications including fluid mechanics. Exceptionally well-written and accessible for Master's students, it discusses current capabilities, limitations, and future directions. Reading this after the foundational Raissi paper will help you understand where your thesis fits in the field and what methodological innovations are emerging. It bridges theory and practical implementation across multiple physics domains.

**Specific aspect:** Comprehensive review of physics-informed ML – theoretical foundations, implementations, and applications

---

## 3. DeepCFD: Efficient Steady-State Laminar Flow Approximation with Deep Convolutional Neural Networks

**Citation:** Ribeiro, M.D., Rehman, A., Ahmed, S., & Dengel, A. (2020). arXiv:2004.08826.

**Why this paper matters:** DeepCFD presents a U-Net-based CNN model that directly learns complete solutions to Navier-Stokes equations for both velocity and pressure fields, achieving three orders of magnitude speedup over traditional CFD. The architecture employs multiple decoders (one per output variable) and demonstrates practical applicability to aerodynamic optimization. This paper provides concrete architectural guidance for implementing your UNet-based approach for flow field prediction, including input representations using signed distance functions and multi-class flow region labeling.

**Specific aspect:** Multi-decoder U-Net architecture for simultaneous velocity and pressure field prediction

---

## 4. NSFnets (Navier-Stokes Flow nets): Physics-informed neural networks for the incompressible Navier-Stokes equations

**Citation:** Jin, X., Cai, S., Li, H., & Karniadakis, G.E. (2021). Journal of Computational Physics, 426, 109951.

**Why this paper matters:** With 767 citations, this paper is directly relevant to your stream function approach. It develops PINNs specifically for incompressible flows using velocity-pressure formulations that automatically ensure divergence-free velocity fields by enforcing the continuity equation (∇·u = 0) directly in the network architecture. This eliminates the need for extra computational cost or penalty terms. The paper investigates dynamic weighting schemes to balance data and physics components during training, which will be valuable for your residual learning approach combining analytical Stokes solutions with learned corrections.

**Specific aspect:** Divergence-free velocity field representation through incompressibility constraints in neural architectures

---

## 5. Neural Conservation Laws: A Divergence-Free Perspective

**Citation:** Richter-Powell, J., Lipman, Y., & Chen, R.T.Q. (2022). Advances in Neural Information Processing Systems (NeurIPS).

**Why this paper matters:** This paper proposes a novel approach to parameterizing neural networks that inherently satisfy conservation laws by design using differential forms and automatic differentiation. Rather than using soft penalty methods, this constructs vector fields that are provably divergence-free by exploiting the mathematical identity d²=0 from differential geometry. This provides a principled alternative to your stream function approach – networks are divergence-free by construction without post-processing. The universal approximation guarantees and GitHub implementations make this immediately applicable to ensuring mass conservation in your multi-crystal sedimentation predictions.

**Specific aspect:** Stream function-like representation through differential forms for divergence-free flows by design

---

## 6. Sedimentation in a dilute dispersion of spheres

**Citation:** Batchelor, G.K. (1972). Journal of Fluid Mechanics, 52(2), 245-268.

**Why this paper matters:** This is THE foundational theoretical work (1,142 citations) on multi-particle hydrodynamic interactions in Stokes flow – the regime relevant to your crystal sedimentation problem. Batchelor rigorously solved how particles interact through the viscous fluid, creating velocity disturbances that decay as 1/r and affect all neighboring particles. He showed that mean settling velocity U = U₀(1 - 6.55c) for dilute suspensions. This provides the theoretical physics your ML model should learn to reproduce. Understanding these classical results will help you validate whether your neural network predictions capture the correct multi-particle interaction physics.

**Specific aspect:** Multi-particle hydrodynamic interactions in dilute suspensions – theoretical foundation for hindered settling

---

## 7. Crystal settling in a vigorously convecting magma chamber

**Citation:** Martin, D. & Nokes, R. (1988). Nature, 332, 534-536.

**Why this paper matters:** This paper directly addresses your application domain – crystal sedimentation in geophysical systems. Through laboratory experiments and theory, the authors demonstrated that crystal settling remains efficient in magma chambers despite vigorous convection, with particles preferentially accumulating in downwelling regions. This is essential for understanding the geophysical context of your work, including fractional crystallization and cumulate formation in layered igneous intrusions. It bridges the gap between fundamental sedimentation physics (Batchelor) and real geophysical processes in multi-crystal systems, providing physical intuition for the phenomena your ML model must capture.

**Specific aspect:** Crystal settling in geophysical magma chamber systems with thermal convection

---

## 8. Learning two-phase microstructure evolution using neural operators and autoencoder architectures

**Citation:** Oommen, V., Shukla, K., Goswami, S., Dingreville, R., & Karniadakis, G.E. (2022). npj Computational Materials, 8, 190.

**Why this paper matters:** This state-of-the-art paper from Nature's computational materials journal demonstrates a breakthrough framework combining convolutional autoencoders with DeepONet (Deep Operator Networks) for learning two-phase dynamics. The innovation lies in learning phase-field evolution in a compressed latent space, achieving 29% speedup while maintaining accuracy with robustness to 10% noise. Most importantly for your thesis, it shows how to integrate ML surrogates with traditional numerical solvers in a hybrid approach – directly applicable to your residual learning method combining analytical Stokes solutions with learned corrections. The multi-scale feature capture is particularly relevant for your 1-15 crystal systems.

**Specific aspect:** Neural operator learning for coupled multi-phase system dynamics with hybrid ML-numerical solver integration

---

## 9. Inverse Problems in Geodynamics Using Machine Learning Algorithms

**Citation:** Shahnas, M.H., Yuen, D.A., & Pysklywec, R.N. (2018). Journal of Geophysical Research: Solid Earth, 123, 296-310.

**Why this paper matters:** This paper demonstrates applying supervised machine learning to solve inverse problems in mantle dynamics, using snapshots from numerical convection models as training data. The authors successfully predict mantle density anomalies and characterize flow patterns, with discussion of extending to deep learning for constraining viscosity and thermal/chemical properties. This provides a concrete example of ML applied to geodynamic flow problems similar to your crystal sedimentation context. It shows how to structure training data from geophysical simulations and what types of inverse problems (parameter estimation, flow characterization) are tractable with ML approaches.

**Specific aspect:** Machine learning for geodynamic inverse problems and mantle flow pattern characterization

---

## Recommended Reading Order

**Phase 1 – Foundations (Start here):**

1. Raissi et al. (2019) – Core PINN methodology
2. Batchelor (1972) – Multi-particle sedimentation physics
3. Martin & Nokes (1988) – Geophysical crystal settling context

**Phase 2 – Architectures & Methods:** 4. Karniadakis et al. (2021) – Comprehensive physics-informed ML review 5. Ribeiro et al. (2020) – UNet implementation for flow fields 6. Jin et al. (2021) – Divergence-free formulations (stream function approach) 7. Richter-Powell et al. (2022) – Conservation laws by design

**Phase 3 – Advanced Applications:** 8. Oommen et al. (2022) – State-of-the-art two-phase systems 9. Shahnas et al. (2018) – Geodynamics applications

---

## Key Takeaways for Your Thesis

This reading list positions you to:

**Understand fundamentals:** Papers 1, 2, and 6 provide the theoretical foundation for physics-informed ML and multi-particle sedimentation physics.

**Implement your three methods:** Papers 3-5 give concrete guidance on UNet architectures (direct learning), stream function approaches (ensuring divergence-free fields), and residual/hybrid methods (combining physics-based solutions with learned corrections).

**Validate and contextualize:** Papers 7 and 9 provide geophysical context for crystal sedimentation, while paper 8 shows cutting-edge approaches to multi-component systems that suggest future research directions.

**Next steps identified:** The literature points to promising directions including neural operators (Paper 8), conservation-law-preserving architectures (Paper 5), and hybrid physics-ML solvers (Papers 4, 8) – all potentially applicable to extending your work beyond the Master's thesis.