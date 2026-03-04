---
type: course
status: active
tags: [teaching, NQS, research]
---
# Rydberg model as a long-range transverse-field Ising model

## 1. Introduction

One of the central challenges of quantum many-body physics is the **exponential growth of the Hilbert space**. Consider a chain of $N$ spin-$1/2$ particles, where each site can be either spin-up $|\uparrow\rangle$ or spin-down $|\downarrow\rangle$. For a single spin, there are $2$ basis states. For two spins, there are $2^2 = 4$ states: $|\uparrow\uparrow\rangle$, $|\uparrow\downarrow\rangle$, $|\downarrow\uparrow\rangle$, $|\downarrow\downarrow\rangle$. In general, the full quantum state of $N$ spins requires specifying $2^N$ complex coefficients — one for each basis configuration. For $N = 30$, this is already $\sim 10^9$ numbers; for $N = 100$, the number exceeds the count of atoms in the observable universe. Storing and diagonalizing such a state directly, a method known as **exact diagonalization (ED)**, is therefore limited in practice to $N \lesssim 20$–$30$ sites. To go beyond this, one must find smarter ways to represent and optimize quantum states.

The key insight that makes this possible is that physically relevant ground states — in particular, gapped ground states far from a phase transition — are far from generic. They occupy only a tiny, structured corner of the full Hilbert space. This structure is captured by the **entanglement entropy**. To define it, consider partitioning the system into two subsystems $L$ (left) and $R$ (right). The reduced density matrix of $L$ is obtained by tracing out $R$:
$$\rho_L = \mathrm{Tr}_R\, |\psi\rangle\langle\psi|,$$
and the von Neumann entanglement entropy is
$$S(\rho_L) = -\mathrm{Tr}(\rho_L \log \rho_L).$$

Let us work this out for the simplest case: two spins, $L = $ spin 1 and $R = $ spin 2. Consider first a **product state**
$$|\psi_{\mathrm{prod}}\rangle = |\uparrow\rangle_1 \otimes |\downarrow\rangle_2.$$
The full density matrix is $|\psi\rangle\langle\psi|$, and tracing out spin 2 gives $\rho_L = |\uparrow\rangle\langle\uparrow|$, a pure state with a single eigenvalue $\lambda = 1$. Therefore
$$S = -(1)\log(1) = 0.$$
There is no entanglement: the two spins are in definite, independent states.

Now consider the **Bell state**
$$|\psi_{\mathrm{Bell}}\rangle = \frac{1}{\sqrt{2}}\left(|\uparrow\downarrow\rangle - |\downarrow\uparrow\rangle\right).$$
Tracing out spin 2 gives $\rho_L = \frac{1}{2}|\uparrow\rangle\langle\uparrow| + \frac{1}{2}|\downarrow\rangle\langle\downarrow|$, a maximally mixed state with two equal eigenvalues $\lambda_{1,2} = 1/2$. Therefore
$$S = -2 \times \frac{1}{2}\log\frac{1}{2} = \log 2.$$
This is the maximum possible entanglement for a two-level system: spin 1 is in a completely mixed state with no definite value until measured, and its outcome is perfectly correlated with that of spin 2.

For a system of $N$ spins, $S$ can range from $0$ (product state) up to $\sim N \log 2$ (volume law), where the latter means entanglement grows with the total number of spins. However, gapped ground states of local Hamiltonians generically satisfy an **area law**: $S$ scales not with the volume of $L$ but with the size of the boundary between $L$ and $R$ [1]. In one dimension, the boundary is just two points regardless of where we cut, so the area law simply means $S \leq \mathrm{const}$, independent of system size — much closer to the product-state limit than to the Bell-state limit. At a quantum critical point, this breaks down and the entropy grows logarithmically as $S \sim \frac{c}{3} \log L$, where $c$ is the central charge of the underlying conformal field theory. These different scalings are the key diagnostic that determines which numerical method is best suited for a given problem.

This distinction in entanglement structure has direct consequences for numerical methods. **Matrix product states (MPS)** represent a quantum state as a contraction of $N$ matrices, one per site, each of dimension $D \times D$. The bond dimension $D$ controls the amount of entanglement the ansatz can capture: the entanglement entropy of an MPS is bounded by $S \leq \log D$. For area-law states, a modest $D$ is sufficient for an accurate representation. The **density matrix renormalization group (DMRG)** algorithm finds the optimal MPS variationally and has become the method of choice for 1D quantum systems [2]. However, MPS faces fundamental difficulties when the area law breaks down: at critical points, $D$ must grow as a power law in $L$; in two dimensions, mapping the lattice to a 1D snake forces $D \sim e^{L_y}$, an exponential cost in the transverse width $L_y$.

A complementary approach is offered by **neural quantum states (NQS)**, introduced by Carleo and Troyer [3]. Here, the wave function amplitudes $\psi(\sigma_1, \ldots, \sigma_N)$ are parametrized directly by a neural network, where $\sigma_i \in \{\uparrow, \downarrow\}$ labels the spin configuration. The network is then optimized variationally using **variational Monte Carlo (VMC)**: one samples spin configurations according to the Born probability $|\psi(\boldsymbol{\sigma})|^2$ and minimizes the energy expectation value $\langle H \rangle$ over the network parameters. Unlike MPS, NQS place no explicit constraint on the entanglement structure — the expressibility of the ansatz is not tied to the area law, and the number of parameters grows only polynomially with $N$ regardless of the interaction range. A recent review of NQS architectures and their applications is given in Ref. [4]; its Fig. 1 gives a useful picture of how NQS compares to MPS and other tensor network ansätze in terms of representational power.

In this project, we use a one-dimensional spin-$1/2$ chain with **power-law interactions** $V_{ij} \sim 1/r^\alpha$ as a tunable testbed to explore when NQS prevails and when MPS/DMRG is a sufficient — or superior — tool. The model is physically motivated by Rydberg atom arrays and other experimental platforms, as described in Sec. 2. By varying the exponent $\alpha$, we can systematically move between regimes where the area law is guaranteed and regimes where it breaks down, providing a concrete and controlled setting in which to benchmark the two approaches.

---

## 2. Model

### 2.1 The Rydberg Hamiltonian

Rydberg atom arrays consist of neutral atoms trapped in optical tweezers and arranged in a lattice. Each atom is treated as a **two-level system** with a ground state $|g\rangle$ and a highly excited Rydberg state $|e\rangle$. In the local basis $\{|e\rangle, |g\rangle\}$, the relevant single-site operators are
$$n_i = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}, \qquad \sigma^x_i = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix},$$
where $n_i = |e\rangle\langle e|_i$ is the occupation operator ($n_i = 1$ if atom $i$ is in the excited state, $n_i = 0$ otherwise) and $\sigma^x_i = |g\rangle\langle e|_i + |e\rangle\langle g|_i$ drives transitions between the two levels. Three physical processes compete:

- **Coherent driving** at Rabi frequency $\Omega/2$: the laser drives coherent transitions between $|g\rangle$ and $|e\rangle$ at each site independently, introducing quantum fluctuations that compete with the interaction-induced order.
- **Detuning** $\delta$: the laser frequency is offset from atomic resonance by $\delta$, acting as a chemical potential — positive $\delta$ favors excitation, negative $\delta$ favors the ground state.
- **Van der Waals interactions**: two atoms simultaneously in the Rydberg state repel each other with energy $C_6/r_{ij}^6$, where $r_{ij}$ is their separation. At short distances this strongly suppresses neighboring excitations — the **Rydberg blockade**.

The Hamiltonian as written in Ref. [5] is
$$H = \frac{\Omega}{2} \sum_i \sigma^x_i - \delta \sum_i n_i + \sum_{i < j} \frac{C_6}{r_{ij}^6}\, n_i n_j.$$

### 2.2 Mapping to the transverse-field Ising model

The Hamiltonian above takes a more transparent form in the spin-$1/2$ language. We identify
$$|e\rangle \equiv |\uparrow\rangle, \qquad |g\rangle \equiv |\downarrow\rangle,$$
so that $\sigma^z_i = |{\uparrow}\rangle\langle{\uparrow}|_i - |{\downarrow}\rangle\langle{\downarrow}|_i$ has eigenvalues $+1$ (excited) and $-1$ (ground). The occupation operator then reads
$$n_i = \frac{\mathbf{1} + \sigma^z_i}{2},$$
which one can verify directly: $n_i|\uparrow\rangle = |\uparrow\rangle$ and $n_i|\downarrow\rangle = 0$. Substituting into each term of $H$, the detuning term gives
$$-\delta \sum_i n_i = -\frac{\delta}{2}\sum_i \mathbf{1} - \frac{\delta}{2}\sum_i \sigma^z_i,$$
where the first part is a global constant that we drop. The interaction term gives
$$\frac{C_6}{r_{ij}^6}\,n_i n_j = \frac{C_6}{r_{ij}^6} \cdot \frac{(\mathbf{1}+\sigma^z_i)(\mathbf{1}+\sigma^z_j)}{4} = \frac{C_6}{4 r_{ij}^6}\left(1 + \sigma^z_i + \sigma^z_j + \sigma^z_i\sigma^z_j\right).$$
The constant and single-site $\sigma^z$ terms renormalize $\delta$; collecting everything, the Hamiltonian becomes
$$\boxed{H = \frac{\Omega}{2}\sum_i \sigma^x_i - \tilde{\delta}\sum_i \sigma^z_i + \sum_{i<j} \frac{J}{r_{ij}^6}\,\sigma^z_i \sigma^z_j,}$$
where $J = C_6/4$ and $\tilde{\delta} = \delta/2 + \sum_{j\neq i} C_6/(2r_{ij}^6)$ absorbs all single-site corrections. This is a **transverse-field Ising model** with long-range $1/r^6$ interactions: $\sigma^x$ drives quantum fluctuations, $\sigma^z$ biases the magnetization, and $\sigma^z_i\sigma^z_j$ is the Ising coupling.

### 2.3 Generalization: tunable power law

In Rydberg systems the exponent $\alpha = 6$ is fixed by atomic physics [7]. However, the same Ising Hamiltonian structure
$$H = \frac{\Omega}{2}\sum_i \sigma^x_i - \tilde{\delta}\sum_i \sigma^z_i + \sum_{i<j} \frac{J}{r_{ij}^\alpha}\,\sigma^z_i \sigma^z_j$$
with tunable $\alpha$ is realized across a range of quantum simulation platforms:

| Platform | Mechanism | $\alpha$ | References |
|---|---|---|---|
| Rydberg atoms | Van der Waals | $6$ | [5,7] |
| Polar molecules, magnetic atoms | Dipole-dipole | $3$ | [8,9] |
| Trapped ions | Phonon-mediated | $0$ to $3$ | [10,11] |

In trapped-ion experiments, $\alpha$ is continuously tunable by adjusting the laser detuning relative to the motional sidebands of the ion crystal [10,11], making this the platform of choice for systematically exploring the role of interaction range.

Treating $\alpha$ as a free parameter, we can ask: for which values of $\alpha$ is the ground state efficiently representable by MPS, and where does NQS offer a genuine advantage? The answer is provided by a rigorous result of Kuwahara and Saito [6]: for any **gapped** 1D ground state with power-law interactions, the area law $S \leq \mathrm{const}$ holds whenever $\alpha > 2$. In this regime MPS with modest bond dimension $D$ is guaranteed to be efficient. For $\alpha \leq 2$, the total interaction strength across any bipartition diverges in the thermodynamic limit, the area law is no longer guaranteed, and MPS may require exponentially large $D$. This is precisely the regime where NQS, carrying no entanglement constraint, is expected to have an advantage. The goal of this project is to test this expectation numerically.

---

## 3. Tasks

This project is organized as a mini research investigation. The tasks below are intentionally open-ended — the goal is not to follow a fixed recipe but to develop physical intuition for when and why different numerical methods succeed or fail.

### 3.1 Exact diagonalization as ground truth

Before running any variational calculation, establish a reliable reference. For small system sizes ($L \lesssim 16$), the full Hamiltonian can be diagonalized exactly — this is computationally feasible on a laptop for spin-$1/2$ chains, since the Hilbert space dimension is only $2^{16} = 65536$. Exact diagonalization (ED) can be performed directly within NetKet using its built-in exact solver, or using an external package such as QuSpin [12]. The ED ground state energy and wave function serve as the benchmark against which all subsequent variational results are measured.

### 3.2 Benchmarking NQS and DMRG

With ED results in hand, run both NQS (using the provided code) and DMRG (using e.g. ITensor or TeNPy) on the same small systems and verify that both methods recover the ED ground state energy to acceptable accuracy. Pay attention to convergence: how many optimization steps does NQS require, and what bond dimension does DMRG need? This step builds confidence in both methods and calibrates expectations before moving to larger systems where ED is no longer available.

A useful special case is the limit $\tilde{\delta} = 0$ with large $\alpha$ (effectively nearest-neighbor interactions), which reduces to the exactly solvable 1D transverse-field Ising chain [13]. The critical point occurs at $J = \Omega/2$, and the exact spectrum is known analytically via Jordan-Wigner fermionization. This provides a sharp, parameter-free benchmark.

### 3.3 Role of interaction range: varying $\alpha$

This is the central comparison of the project. Fix a representative point in the $( \tilde{\delta}, J)$ parameter space and vary the interaction exponent $\alpha$. For each $\alpha$, compute the ground state energy and entanglement entropy $S$ using both NQS and DMRG, and compare against ED where available. The theoretical expectation from Ref. [6] is that DMRG should be efficient for $\alpha > 2$ (area law guaranteed) and begin to struggle as $\alpha \to 2$ and below, while NQS accuracy should be less sensitive to $\alpha$.

Scale up the system size $L$ progressively to see how the two methods diverge. Track not just the energy but also convergence indicators: the bond dimension required by DMRG to achieve a given accuracy, and the variance of the NQS energy estimator. These diagnostics tell you *why* a method struggles, not just *that* it does.

Based on these results, identify a value of $\alpha$ — likely somewhere below $2$, for instance $\alpha \approx 1.5$ — where NQS offers a clear advantage. The choice should be motivated by your numerical evidence.

### 3.4 Phase diagram at strong long-range interactions

Having identified a regime where NQS is the superior method, use it to map out the phase diagram in the $(\tilde{\delta}, J)$ plane at your chosen $\alpha$. Before running calculations, reason from the limiting cases:

- **$J \to 0$**: interactions vanish, spins respond independently to the competing $\sigma^x$ and $\sigma^z$ fields — the system is in a disordered, non-interacting state.
- **$J \to \infty$, $\alpha \leq 2$**: long-range interactions dominate; in this regime mean-field theory becomes increasingly accurate and ferromagnetic order is expected.
- **$\tilde{\delta} \to \pm\infty$**: the longitudinal field polarizes all spins along $\pm z$.

From these limits one can argue that at least one phase boundary must exist in the $(\tilde{\delta}, J)$ plane, separating a disordered phase from an ordered one. The nature of the transition — its universality class, whether it is continuous or first-order — and whether additional phases exist between the two limits are questions to be answered numerically.

Choose an appropriate order parameter (e.g. the staggered or uniform magnetization $\langle \sigma^z \rangle$), compute it across the parameter plane, and identify the phase boundary. Discuss how the phase diagram changes with $\alpha$, and what this implies about the role of interaction range in stabilizing or destabilizing ordered phases.

---

## References

[1] J. Eisert, M. Cramer, and M. B. Plenio, "Colloquium: Area laws for the entanglement entropy," *Rev. Mod. Phys.* **82**, 277 (2010).

[2] U. Schollwöck, "The density-matrix renormalization group in the age of matrix product states," *Ann. Phys.* **326**, 96 (2011).

[3] G. Carleo and M. Troyer, "Solving the quantum many-body problem with artificial neural networks," *Science* **355**, 602 (2017).

[4] H. Lange, A. Van de Walle, A. Abedinnia, and A. Bohrdt, "From architectures to applications: A review of neural quantum states," arXiv:2402.09402 (2024).

[5] R. Samajdar, W. W. Ho, H. Pichler, M. D. Lukin, and S. Sachdev, "Complex density wave orders and quantum phase transitions in a model of square-lattice Rydberg atom arrays," *Phys. Rev. Lett.* **124**, 103601 (2020).

[6] T. Kuwahara and K. Saito, "Area law of noncritical ground states in 1D long-range interacting systems," *Nat. Commun.* **11**, 4478 (2020).

[7] L. Béguin, A. Vernier, R. Chicireanu, T. Lahaye, and A. Browaeys, "Direct measurement of the van der Waals interaction between two Rydberg atoms," *Phys. Rev. Lett.* **110**, 263201 (2013).

[8] B. Yan, S. A. Moses, B. Gadway, J. P. Covey, K. R. A. Hazzard, A. M. Rey, D. S. Jin, and J. Ye, "Observation of dipolar spin-exchange interactions with lattice-confined polar molecules," *Nature* **501**, 521 (2013).

[9] M. Lu, N. Q. Burdick, S. H. Youn, and B. L. Lev, "Strongly dipolar Bose-Einstein condensate of dysprosium," *Phys. Rev. Lett.* **107**, 190401 (2011).

[10] D. Porras and J. I. Cirac, "Effective quantum spin systems with trapped ions," *Phys. Rev. Lett.* **92**, 207901 (2004).

[11] R. Islam, C. Senko, W. C. Campbell, S. Korenblit, J. Smith, A. Lee, E. E. Edwards, C.-C. J. Wang, J. K. Freericks, and C. Monroe, "Emergence and frustration of magnetism with variable-range interactions in a quantum simulator," *Science* **340**, 583 (2013).

[12] P. Weinberg and M. Bukov, "QuSpin: a Python package for dynamics and exact diagonalization of quantum many body systems," *SciPost Phys.* **2**, 003 (2017).

[13] P. Pfeuty, "The one-dimensional Ising model with a transverse field," *Ann. Phys.* **57**, 79 (1970).
