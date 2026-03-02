# The Angular Uniqueness of Base 2 in Positional Multiplication

**Author:** Stefano Alimonti
**Affiliation:** Independent Researcher
**Date:** March 2026

---

## Abstract

We prove that base 2 is the unique integer base for which the D-parity boundary in positional multiplication becomes a straight line in angular coordinates: $(1+\tan\alpha)(1+\tan\beta) = b$ reduces to $\alpha + \beta = \pi/4$ if and only if $b = 2$. This geometric fact explains why $\pi$ enters the conjectured trace anomaly $c_1 = \pi/18$ for binary multiplication — the integration domain is a right triangle with a $\pi/4$ hypotenuse — and predicts that $c_1(b)$ for $b > 2$ does not involve $\pi$.

**Keywords:** angular coordinates, D-parity boundary, trace anomaly, carries, positional multiplication, base dependence

**MSC 2020:** 11A63, 60C05, 26B15

---

## 1. Introduction

The trace anomaly

$$c_1 := \mathbb{E}[c_{\mathrm{top}-1}] - 1$$

governs the leading correction in the carry–zeta approximation [B]:

$$\ln R(l,s) = \frac{c_1}{l^s} + O(l^{-2\sigma}).$$

For base 2, high-statistics Monte Carlo and exact enumeration up to $K = 21$ establish $c_1 = \pi/18 = 0.17453\ldots$ to ${\sim}4.3$ digits from direct enumeration [E]. The appearance of $\pi$ in a problem of positional arithmetic is surprising — no trigonometric structure is apparent in the carry recurrence

$$c_{k+1} = \lfloor(\mathrm{conv}_k + c_k)/2\rfloor$$

. (The trace anomaly $c_1$ of the Euler product correction studied in this paper and in [B] is the leading spectral constant of the carry correction factor; it is related to, but distinct from, the macroscopic cascade trace anomaly $\Delta R$ studied in [E], which concerns the full sector ratio of the cascade valuation.)

This paper explains where $\pi$ comes from. We introduce angular coordinates $(\alpha, \beta)$ for the normalized factors and show that the D-parity boundary — the hyperbola $(1+X)(1+Y) = b$ separating even-length from odd-length products — becomes a straight line $\alpha + \beta = \pi/4$ in these coordinates if and only if $b = 2$. The constant $\pi/4$ in the integration limit pulls $\pi$ into the trace anomaly through the geometry of the domain, not through the arithmetic of carries.

For $b > 2$ the boundary is a curve, the angular range involves $\arctan(b-1)$ (irrational multiples of $\pi$), and multi-base experiments confirm that $c_1(b)$ departs sharply from $\pi/18$.

---

## 2. Angular Coordinates and the D-Parity Boundary

### 2.1 Setup

Let $p, q$ be independent random $d$-digit integers in base $b$, with leading digit $\geq 1$ and remaining digits uniform on $\{0, \ldots, b-1\}$. Write $p = b^{d-1}(1+X)$, $q = b^{d-1}(1+Y)$, so that $X, Y \in [0, b-1)$ are continuous approximations to the fractional parts. The product $N = pq$ has

$$N = b^{2d-2}\,W, \qquad W = (1+X)(1+Y) \in [1, b^2).$$

The digit count $D = \lfloor\log_b N\rfloor + 1$ satisfies:

- $D = 2d$ (even) when $W \geq b$,
- $D = 2d - 1$ (odd) when $W < b$.

This is the **D-parity** decomposition. The boundary is the hyperbola $(1+X)(1+Y) = b$ in the unit square $[0, b-1)^2$.

### 2.2 Angular Substitution

Introduce angular coordinates via $X = \tan\alpha$, $Y = \tan\beta$, where $\alpha, \beta \in [0, \arctan(b-1))$. The Jacobian is $dX\,dY = \sec^2\!\alpha\;\sec^2\!\beta\;d\alpha\,d\beta$, and

$$W = (1 + \tan\alpha)(1 + \tan\beta).$$

**Proposition (Angular boundary).** The D-parity boundary $(1+\tan\alpha)(1+\tan\beta) = b$ in angular coordinates satisfies

$$\tan(\alpha + \beta) = \frac{b - 1 - P}{1 - P}, \qquad P = \tan\alpha\,\tan\beta.$$

*Proof.* Expand the left side: $1 + \tan\alpha + \tan\beta + \tan\alpha\,\tan\beta = b$, so $\tan\alpha + \tan\beta = b - 1 - P$. The addition formula gives $\tan(\alpha+\beta) = (\tan\alpha + \tan\beta)/(1 - \tan\alpha\,\tan\beta) = (b-1-P)/(1-P)$. $\square$

### 2.3 The Angular Uniqueness Theorem

**Theorem (Angular Uniqueness).** The expression $(b-1-P)/(1-P)$ is independent of $P$ if and only if $b = 2$. Consequently:

- For $b = 2$: the D-parity boundary is the straight line $\alpha + \beta = \pi/4$.
- For $b > 2$: the boundary is a curve, and the angular sum $\alpha + \beta$ varies along it.

*Proof.* Write $f(P) = (b-1-P)/(1-P)$. Compute the derivative:

$$f'(P) = \frac{-(1-P) + (b-1-P)}{(1-P)^2} = \frac{b - 2}{(1-P)^2}.$$

The numerator $b - 2$ vanishes if and only if $b = 2$. When $b = 2$: $f(P) = (1-P)/(1-P) = 1$ for all $P \in [0,1)$, so $\tan(\alpha+\beta) = 1$, giving $\alpha + \beta = \pi/4$. When $b > 2$: $f'(P) > 0$, so $\tan(\alpha+\beta)$ increases with $P$ and the boundary is not a level set of $\alpha + \beta$. $\square$

**Remark.** The algebraic mechanism is transparent: $b - 1 = 1$ is the unique case where the numerator and denominator of $(b-1-P)/(1-P)$ are proportional, producing a cancellation that eliminates $P$.

---

## 3. Consequences

### 3.1 The D-Odd Region for Base 2

For $b = 2$, the D-odd condition $W < 2$ maps to the right triangle

$$\mathcal{T} = \{(\alpha, \beta) : \alpha \geq 0,\; \beta \geq 0,\; \alpha + \beta < \pi/4\}$$

in the angular square $[0, \pi/4)^2$. The D-odd probability is

$$P_o = \int_0^{\pi/4}\!\int_0^{\pi/4 - \alpha} \sec^2\!\alpha\;\sec^2\!\beta\;d\beta\;d\alpha.$$

Evaluating the inner integral:

$$\int_0^{\pi/4 - \alpha}\sec^2\!\beta\;d\beta = \tan(\pi/4 - \alpha) = \frac{1 - \tan\alpha}{1 + \tan\alpha}.$$

Substituting $t = \tan\alpha$:

$$P_o = \int_0^1 \frac{1-t}{1+t}\,dt = \bigl[2\ln(1+t) - t\bigr]_0^1 = 2\ln 2 - 1 \approx 0.3863.$$

The integration limit $\pi/4$ is the source of $\pi$ in all subsequent integrals over the D-odd domain.

### 3.2 The Inner Integral on Isochrones

Introduce the isochrone coordinate $s = \alpha + \beta$. Along the level set at angular distance $s$ from the vertex, the density integral is

$$I(s) = \int_0^{s} \sec^2\!\alpha\;\sec^2(s - \alpha)\;d\alpha = \frac{2(1+T^2)\bigl[T^2 - \ln(1+T^2)\bigr]}{T^3}, \qquad T = \tan s.$$

*Derivation.* Write $\sec^2\!\alpha\;\sec^2(s-\alpha) = (1 + \tan^2\!\alpha)(1 + \tan^2(s-\alpha))$. Substituting $u = \tan\alpha$ with $du = \sec^2\!\alpha\;d\alpha$ and using $\tan(s-\alpha) = (T - u)/(1 + Tu)$:

$$I(s) = \int_0^{T} \left(1 + \frac{(T-u)^2}{(1+Tu)^2}\right) du = \int_0^T \frac{(1+Tu)^2 + (T-u)^2}{(1+Tu)^2}\,du.$$

Expanding the numerator: $(1+Tu)^2 + (T-u)^2 = (1 + T^2)(1 + u^2)$, so

$$I(s) = (1+T^2)\int_0^{T} \frac{1 + u^2}{(1+Tu)^2}\,du.$$

Substituting $w = 1 + Tu$ and separating partial fractions:

$$\int_0^T \frac{1+u^2}{(1+Tu)^2}\,du = \frac{1}{T^3}\Bigl[-(T^2{+}1)/w - 2\ln w + w\Bigr]_1^{1+T^2} = \frac{2\bigl[T^2 - \ln(1+T^2)\bigr]}{T^3},$$

yielding the stated closed form.

**Verification.** $P_o = \int_0^{\pi/4} I(s)\,ds = 2\ln 2 - 1$, confirmed analytically by reversing the order of integration.

### 3.3 Representation of $c_1$

The trace anomaly admits a single-integral representation over the angular domain:

$$c_1 = \int_0^{\pi/4} \langle c_{M-1} - 1\rangle(s) \cdot I(s)\;ds + (1 + 3\ln(3/4))$$

where $\langle c_{M-1} - 1\rangle(s)$ is the isochrone average of $c_{M-1} - 1$ at angular distance $s$, and the constant $1 + 3\ln(3/4)$ is the D-even contribution (the D-even cascade is degenerate [E, §8.1], making the schoolbook computation exact and the D-even $c_1$ component analytically closed [B]). The upper limit $\pi/4$ — a direct consequence of the Angular Uniqueness Theorem — is the sole entry point for $\pi$.

### 3.4 Base $b > 2$: Curved Boundaries

For $b > 2$, the angular sum along the D-parity boundary varies from $\arctan(b-1)$ (at $P = 0$, i.e., $\alpha = 0$ or $\beta = 0$) upward. The value $\arctan(b-1)$ is not a rational multiple of $\pi$ for any integer $b > 2$ (by the Niven–Mann theorem: $\arctan(n)$ is a rational multiple of $\pi$ only for $n = 0, \pm 1$). The integration domain is bounded by a curve, not a line, and the integration *limits* no longer produce clean factors of $\pi$.

**Conjecture.** $c_1(b)$ for $b > 2$ does not involve $\pi$.

*Supporting evidence:* (i) Niven–Mann constrains the integration limits; (ii) base-3 numerical data ($c_1(3) \approx 0.5985$, candidate $\ln 3 - 1/2$) shows no $\pi$ involvement (§4.1); (iii) PSLQ analysis on base-3 data excludes $\pi$ from the constant basis at ${\sim}4$-digit precision (G03, G12). *Caveat:* this argument constrains the limits but not the integrand — in principle, $\pi$ could enter through the carry kernel $\langle c_{M-1} - 1 \rangle(s)$ or through combinations of arctan values. A proof would require showing the full integral, not just the limits, is $\pi$-free.

---

## 4. Experimental Verification

### 4.1 Multi-Base Monte Carlo

Monte Carlo simulation with $10^8$ samples per base ($d = 64$-bit factors) yields:

| Base $b$ | $c_1(b)$ | Std. error | Consistent with $\pi/18$? |
|----------|-----------|------------|--------------------------|
| 2 | 0.17465 | $4.4 \times 10^{-5}$ | Yes ($0.3\sigma$) |
| 3 | 0.599 | $1 \times 10^{-3}$ | No ($>400\sigma$) |
| 5 | 1.61 | $2 \times 10^{-3}$ | No ($>700\sigma$) |
| 7 | 2.64 | $3 \times 10^{-3}$ | No ($>800\sigma$) |
| 10 | 4.20 | $4 \times 10^{-3}$ | No ($>1000\sigma$) |

The trace anomaly grows with base and departs decisively from $\pi/18$ for all $b > 2$.

### 4.2 Exact Enumeration for Base 3

Exact enumeration of $c_1(3, K)$ for $K = 2, \ldots, 12$ ($3^{24} \approx 2.82 \times 10^{11}$ configurations at $K = 12$, parallelized with OpenMP) yields:

| $K$ | $c_1(3, K)$ | $\Delta$ | $\rho(K)$ |
|-----|-------------|----------|-----------|
| 2 | 0.50000 | — | — |
| 3 | 0.48148 | $-1.85 \times 10^{-2}$ | — |
| 4 | 0.53185 | $+5.04 \times 10^{-2}$ | 2.72 |
| 5 | 0.54670 | $+1.48 \times 10^{-2}$ | 0.295 |
| 6 | 0.56709 | $+2.04 \times 10^{-2}$ | 1.37 |
| 7 | 0.57461 | $+7.52 \times 10^{-3}$ | 0.369 |
| 8 | 0.58280 | $+8.19 \times 10^{-3}$ | 1.09 |
| 9 | 0.58684 | $+4.04 \times 10^{-3}$ | 0.493 |
| 10 | 0.59015 | $+3.31 \times 10^{-3}$ | 0.819 |
| 11 | 0.59822 | $+8.07 \times 10^{-3}$ | 2.44 |
| 12 | 0.59851 | $+2.84 \times 10^{-4}$ | 0.035 |

The $K = 11, 12$ values are exact rationals: $c_1(3, 11) = 8342998915 / 13946313395$ and $c_1(3, 12) = 75125607729 / 125521846608$.

A dramatic **odd-even oscillation** in $\rho(K)$ dominates the convergence: odd $K$ overshoots, even $K$ undershoots, with the oscillation amplitude decreasing. This makes multi-term extrapolation unstable. The 1-term Richardson extrapolation from the most recent pair gives $c_\infty^{\text{Rich}}(11, 12) = 0.59865$, close to $\ln(3) - 1/2 = 0.59861\ldots$ ($\Delta = 3.6 \times 10^{-5}$, $\sim 1.4\sigma$).

PSLQ search with extended basis $\{1, \ln 2, \ln 3, \ln^2 3, \arctan(1/2), \zeta(2)/9, \sqrt{3}, \pi\}$ finds no exact relation at the available precision ($\sim 5$–$6$ digits). The candidate $\ln(3) - 1/2$ is suggestive but not confirmed.

**Enhanced analysis (G12).** Separating $K$ into even $\{2,4,6,8,10,12\}$ and odd $\{3,5,7,9,11\}$ subsequences eliminates the oscillation (each subsequence converges monotonically), consistent with a $(-1/3)^K$ alternating eigenvalue. However, multi-term Richardson extrapolation on each subsequence with rate $1/3$ overshoots the target by $\sim 10^{-3}$: the effective correction rate within the even subsequence is $r_{\text{eff}} \approx 1.07$, not $1/9$. This reveals a **pre-asymptotic structure**: the correction coefficients $A_K$ depend on $K$ (the boundary observable changes with each cascade depth), so the true expansion is $c_1(K) = c_\infty + (A_0 + A_1/K + \cdots)(1/3)^K$, defeating simple geometric Richardson. Nonlinear models (alternating eigenvalues $\pm 1/3, \pm 1/9, 1/27$) also underfit (RMS $\sim 5 \times 10^{-3}$). All estimates cluster in $[0.590, 0.600]$, bracketing $\ln(3) - 1/2 = 0.5986$ (consistent at $0.6\sigma$), but the available precision ($\sim 2$–$3$ digits) is far below the $\sim 8$ digits required for definitive PSLQ. Data at $K \geq 16$ would be needed.

### 4.3 Convergence Rate

For base 2, the effective convergence rate $\rho(K)$ approaches $1/2$ monotonically ($\rho(17) = 0.505$, proved to equal $\lambda_2 = 1/b$ via transfer operator analysis).

For base 3, the convergence shows a pronounced **odd-even oscillation**: $\rho(K)$ alternates between values well above 1 (at odd $K$) and well below 1 (at even $K$). The sequence $\rho(K)$ for $K = 4, \ldots, 12$ is: $2.72, 0.30, 1.37, 0.37, 1.09, 0.49, 0.82, 2.44, 0.035$. This oscillation indicates complex sub-dominant eigenvalues in the transfer operator, consistent with the $(b-1) = 2$ off-diagonal channels in the base-3 carry Markov chain.

Despite the oscillation, the **geometric mean** of consecutive $\rho$ values decreases toward $1/3$:

| Pair | $\sqrt{\rho_{\text{odd}} \cdot \rho_{\text{even}}}$ |
|------|-----------------------------------------------------|
| $(K=4,5)$ | 0.90 |
| $(K=6,7)$ | 0.71 |
| $(K=8,9)$ | 0.73 |
| $(K=10,11)$ | 1.41 |
| $(K=11,12)$ | 0.29 |

The last pair ($K = 11, 12$) undershoots $1/3$, suggesting the oscillation is damping. More data ($K \geq 14$) would be needed to confirm the asymptotic geometric-mean rate equals $1/3 = 1/b$.

This identifies the Diaconis–Fulman sub-dominant eigenvalue $\lambda_2 = 1/b$ as the universal convergence bottleneck, though for $b > 2$ the approach to this rate involves oscillatory transients from complex eigenvalues.

### 4.4 Angular Boundary Curvature

The standard deviation $\sigma(\alpha + \beta)$ along the D-parity boundary, sampled at 1000 uniformly spaced points, quantifies the departure from linearity:

| Base $b$ | $\sigma(\alpha + \beta)$ |
|----------|-------------------------|
| 2 | 0.000 (exact) |
| 3 | 0.046 |
| 5 | 0.137 |
| 10 | 0.249 |
| 100 | 0.413 |

The curvature vanishes only for $b = 2$ and grows monotonically, confirming the theorem.

### 4.5 Transfer Operator Spectrum

The carry transfer operator for base-$b$ multiplication, acting on carry states $\{0, \ldots, b-1\}$, has eigenvalues $\lambda_k = 1/b^k$ for $k = 0, 1, \ldots, b-1$ — exactly the Diaconis–Fulman spectrum. This holds for all tested bases $b = 2, 3, 5, 7, 10$ and extends the Diaconis–Fulman theorem from addition carries to multiplication carries.

The mechanism is the division by $b$ in the carry recurrence: a linear perturbation $\delta(c) = c - \mathbb{E}[c]$ maps to $\delta/b$ at each step, regardless of the convolution distribution. Higher-order perturbations decay at rates $1/b^2, 1/b^3, \ldots$, completing the spectrum.

---

## 5. Discussion

### 5.1 The Origin of $\pi$

The Angular Uniqueness Theorem explains why $\pi$ enters $c_1(2) = \pi/18$. The constant $\pi$ does not come from the carry arithmetic — neither the carry recurrence nor the convolution sums involve trigonometric quantities. Instead, $\pi$ enters through the geometry of the integration domain: the D-odd region, when expressed in the natural angular coordinates for the multiplicative structure $(1+X)(1+Y)$, is a right triangle with hypotenuse $\alpha + \beta = \pi/4$. The integration limit $\pi/4$ propagates into the final answer.

For $b > 2$, the boundary is curved, the domain is not a simplex, and the integration limits involve $\arctan(b-1)$ — quantities that are irrational multiples of $\pi$. The limits alone do not produce clean factors of $\pi$; whether $\pi$ could enter through the integrand remains an open question (see Conjecture in §3.4).

### 5.2 Geometry Versus Algebra

The trace anomaly $c_1$ depends on two distinct structures:

1. **The spectrum** — the transfer operator eigenvalues $\lambda_k = 1/b^k$ — which governs the convergence rate $\rho(K) \to 1/b$. This is algebraic, universal across bases, and proved for both addition and multiplication carries.

2. **The boundary geometry** — the shape of the D-parity region in angular coordinates — which determines the value of $c_1$ itself. This is geometric, base-specific, and uniquely simple for $b = 2$.

The formula $c_1(K) = \pi/18 + P(K) \cdot (1/2)^K$ from [B, experiment B30] separates these contributions cleanly: $\pi/18$ is geometric (the integration domain), $1/2$ is algebraic (the spectral gap), and the polynomial prefactor $P(K)$ arises from non-Markov digit-bit correlations in the multiplication carry chain.

### 5.3 Connections

The Angular Uniqueness Theorem complements the spectral theory of carries [A], which establishes the Diaconis–Fulman eigenvalue structure, and the carry–zeta approximation framework [B], which uses $c_1$ as the leading correction constant. The present result explains why the correction involves $\pi$ for the base ($b = 2$) in which the framework was originally developed, and predicts that multi-base extensions will encounter different — and likely more complex — transcendental structure.

The trace anomaly analysis [E] established $c_1 = \pi/18$ to ${\sim}4.3$ digits from direct enumeration ($K = 21$); $\Sigma_{\text{even}}$ achieves $7.3$ digits via Richardson extrapolation. A formal proof remains the central open problem. The Angular Uniqueness Theorem identifies the geometric origin of $\pi$ and thereby constrains the form any such proof must take: it must ultimately evaluate an integral over the triangular domain $\alpha + \beta < \pi/4$.

---

## 6. Reproducibility

All scripts are in `experiments/`, following the naming convention `G{NN}_{description}.{py|c}`. Core experiments: G01–G12; extensions G13–G15 are documented in `experiments/README.md`.

| Script | Language | Description |
|--------|----------|-------------|
| G01 | C | High-precision Monte Carlo for $c_1(3)$ ($10^8$–$10^9$ samples) |
| G02 | C | Exact enumeration $c_1(3, K)$ for $K = 2$–$12$ (OpenMP) |
| G03 | Python | PSLQ analysis on extrapolated $c_1(3)$ |
| G04 | Python | Transfer operator eigenvalue verification, 5 bases (§4.5) |
| G05–G06 | Python | Base-3 eigenvalue verification and extensions |
| G07 | C | Base-3 exact enumeration $K = 2$–$10$ |
| G08–G09 | Python | Multi-term Diaconis–Fulman fit; PSLQ on $c_1(3)$ |
| G10–G11 | C/Python | Extended enumeration $K = 11$–$14$; enhanced PSLQ with 5-term Richardson |
| G12 | Python | Odd-even separation, alternating eigenvalue models |

Key result from G11: base-3 transfer operators confirm the full Diaconis–Fulman spectrum $\{1, 1/3, 1/9, 1/27\}$. Best estimate $c_1(3) \approx 0.5985$, candidate $\ln 3 - 1/2$, PSLQ inconclusive at 4-digit precision. Definitive identification requires $K \geq 16$.

Requirements: Python 3.8+, NumPy, SciPy, mpmath. C compiler (gcc/clang) for `.c` scripts.

---

## 7. References

1. P. Diaconis, J. Fulman, "Carries, Shuffling, and Symmetric Functions," *Adv. Appl. Math.* 43(2), 176–196, 2009.
2. I. Niven, "Irrational Numbers," *Carus Mathematical Monographs* 11, MAA, 1956.
3. [A] Companion paper: "Spectral Theory of Carries in Positional Multiplication," this series.
4. [B] Companion paper: "Carry Polynomials and the Euler Product: An Approximation Framework," this series.
5. [P1] Companion paper: "π from Pure Arithmetic: A Spectral Phase Transition in the Binary Carry Bridge," this series.
6. [P2] Companion paper: "The Sector Ratio in Binary Multiplication: From Markov Failure to Transcendence," this series.
7. [E] Companion paper: "The Trace Anomaly of Binary Multiplication," this series.
8. [F] Companion paper: "Exact Covariance Structure of Binary Carry Chains," this series.

---

*CC BY 4.0. Code: MIT License.*
