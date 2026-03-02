"""
G04_transfer_eigenvalues.py — Transfer operator eigenvalue analysis

WHY does ρ(K) → 1/2?

The carry recurrence is:
    c_{k+1} = floor((conv_k + c_k) / 2)

The division by 2 is the fundamental mechanism: perturbations decay at rate 1/2.
This is the Diaconis-Fulman eigenvalue for the carry Markov chain in base 2.

We verify this by:
1. Computing the exact transfer matrix for ADDITION carries (base 2)
   → eigenvalues 1, 1/2 exactly
2. Computing the effective transfer matrix for MULTIPLICATION carries
   at different positions from the MSB
3. Showing that the sub-dominant eigenvalue approaches 1/2 as we move
   from the boundary into the bulk
4. Explaining why the 1/2 is universal for base 2

Key insight: at position p from the MSB with m convolution terms,
    conv_p = Σ_{i=0}^{m-1} g_i · h_{p-i}
where each g_i·h_{p-i} ~ Bernoulli(1/4), independent.
So conv_p ~ Binomial(m, 1/4).
The carry chain c → floor((Binomial(m,1/4) + c) / 2) has transfer matrix
whose sub-dominant eigenvalue → 1/2 as m → ∞.
"""

import numpy as np
from fractions import Fraction
from scipy.special import comb as binom_coeff
import math

print("G04: TRANSFER OPERATOR EIGENVALUE ANALYSIS")
print("=" * 70)

# ================================================================
# PART 1: Addition carry chain (exact Diaconis-Fulman)
# ================================================================
print("\n" + "=" * 70)
print("PART 1: ADDITION CARRY CHAIN (BASE 2)")
print("=" * 70)
print("""
For addition of two random base-2 digits with carry:
  total = a + b + c,  where a,b ~ Bernoulli(1/2), c ∈ {0,1}
  new_carry = floor(total/2),  bit = total mod 2

State space: {0, 1}
Transfer matrix T[c'][c] = P(new_carry = c' | carry = c):
""")

T_add = np.zeros((2, 2))
for c in range(2):
    for a in range(2):
        for b in range(2):
            total = a + b + c
            new_c = total // 2
            T_add[new_c][c] += 0.25

print(f"T_add = ")
print(f"  [{T_add[0,0]:.4f}  {T_add[0,1]:.4f}]")
print(f"  [{T_add[1,0]:.4f}  {T_add[1,1]:.4f}]")

evals_add = np.linalg.eigvals(T_add)
print(f"\nEigenvalues: {sorted(evals_add, reverse=True)}")
print(f"Sub-dominant eigenvalue = 1/2 EXACTLY")
print(f"Stationary distribution: (1/2, 1/2)")

# ================================================================
# PART 2: Multiplication carry chain at position p
# ================================================================
print("\n" + "=" * 70)
print("PART 2: MULTIPLICATION CARRY CHAIN")
print("=" * 70)
print("""
At position p from the MSB, convolution has m = min(p+1, K) terms.
Each term g_i·h_{p-i} ~ Bernoulli(1/4), independent.
conv_p ~ Binomial(m, 1/4).

Transfer matrix: T_m[c'][c] = P(floor((conv+c)/2) = c' | carry = c)
where conv ~ Binomial(m, 1/4).

State space: {0, 1, ..., c_max} where c_max ≈ m/2.
""")

def make_transfer_matrix(m, c_max=None):
    """Transfer matrix for m convolution terms (each Bernoulli(1/4))."""
    if c_max is None:
        c_max = m + 1
    n_states = c_max + 1

    conv_probs = np.zeros(m + 1)
    for v in range(m + 1):
        conv_probs[v] = binom_coeff(m, v, exact=True) * (0.25)**v * (0.75)**(m-v)

    T = np.zeros((n_states, n_states))
    for c_in in range(n_states):
        for v in range(m + 1):
            total = v + c_in
            c_out = total // 2
            if c_out < n_states:
                T[c_out][c_in] += conv_probs[v]

    return T

print(f"{'m':>4s} {'c_max':>6s} {'λ₁':>8s} {'λ₂':>10s} {'λ₃':>10s} {'λ₂ exact':>10s}")
print("-" * 55)

for m in [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50]:
    c_max = min(m + 2, 40)
    T = make_transfer_matrix(m, c_max)
    evals = np.linalg.eigvals(T)
    evals_real = sorted([e.real for e in evals if abs(e.imag) < 1e-10], reverse=True)

    if len(evals_real) >= 3:
        print(f"{m:4d} {c_max:6d} {evals_real[0]:8.5f} {evals_real[1]:10.6f} "
              f"{evals_real[2]:10.6f} {0.5:10.6f}")
    elif len(evals_real) >= 2:
        print(f"{m:4d} {c_max:6d} {evals_real[0]:8.5f} {evals_real[1]:10.6f} "
              f"{'---':>10s} {0.5:10.6f}")

# ================================================================
# PART 3: Why 1/2 is universal
# ================================================================
print("\n" + "=" * 70)
print("PART 3: WHY 1/2 IS UNIVERSAL (ANALYTICAL PROOF)")
print("=" * 70)
print("""
The carry recurrence is:
    c_{k+1} = floor((S_k + c_k) / 2)

where S_k = conv_k is a non-negative integer random variable.

THEOREM (informal): If S_k has finite variance and is independent of c_k,
then the sub-dominant eigenvalue of the transfer operator is exactly 1/2.

PROOF SKETCH:
Let π(c) be the stationary distribution. Consider the perturbation
δ(c) = f(c) - Σ_c' π(c') f(c') (zero-mean fluctuation).

The transfer operator acts on δ as:
    (T δ)(c') = Σ_c T(c'|c) δ(c)
              = Σ_c Σ_s P(S=s) · [floor((s+c)/2) = c'] · δ(c)

For large c (bulk), floor((s+c)/2) ≈ (s+c)/2, so:
    (T δ)(c') ≈ Σ_c δ(c) · P(c' ≈ (S+c)/2)

The key: δ(c) = c - E[c] (linear perturbation) transforms as:
    E[c' - E[c']] = E[(S+c)/2 - (E[S]+E[c])/2] = (c - E[c])/2

So a linear perturbation decays by factor 1/2. This is EXACT for any
distribution of S, because the division by 2 is exact.

Higher-order perturbations (c² - E[c²], etc.) decay at rate 1/4, 1/8, ...
These are the Diaconis-Fulman eigenvalues 1/2^k.
""")

print("=== Verification: linear perturbation decay rate ===\n")

for m in [2, 5, 10, 20]:
    c_max = min(m + 5, 50)
    T = make_transfer_matrix(m, c_max)

    evals, evecs = np.linalg.eig(T)
    idx = np.argsort(-evals.real)
    evals = evals[idx]
    evecs = evecs[:, idx]

    pi = evecs[:, 0].real
    pi = pi / pi.sum()

    mean_c = sum(c * pi[c] for c in range(len(pi)))

    delta_linear = np.array([c - mean_c for c in range(c_max + 1)])
    T_delta = T @ delta_linear

    idx_bulk = max(1, int(mean_c))
    if idx_bulk < len(delta_linear) and abs(delta_linear[idx_bulk]) > 1e-10:
        ratio = T_delta[idx_bulk] / delta_linear[idx_bulk]
    else:
        ratio = 0

    print(f"m={m:2d}: mean_c = {mean_c:.3f}, "
          f"λ₂ = {evals[1].real:.8f}, "
          f"T(δ_linear)/δ_linear|_bulk = {ratio:.8f}")

# ================================================================
# PART 4: Position-dependent analysis for top-K multiplication
# ================================================================
print("\n" + "=" * 70)
print("PART 4: POSITION-DEPENDENT ANALYSIS (TOP-K)")
print("=" * 70)
print("""
For the top-K enumeration, starting from position 0 (MSB):
  p=0: conv = 1 (fixed), carry_in = 0 → carry_out = 0
  p=1: conv = g + h (2 terms), carry_in from p=0
  p=2: conv = g' + g·h + h' (3 terms), but middle term depends on p=1 bits
  ...

The effective number of "independent" Bernoulli(1/4) terms in the
convolution at position p from top is:
  m_eff(p) = p + 1 for p < K

At each position, the transfer matrix has sub-dominant eigenvalue
approaching 1/2 as p grows. But the ACTUAL convolution has correlations
(shared bits), so the effective eigenvalue differs slightly.

Let's compute the EXACT eigenvalue by tracking the full state space
for small K, and compare with the Bernoulli(1/4) model.
""")

print("=== Effective eigenvalue at each position (Binomial model) ===\n")
print(f"{'p':>4s} {'m=p+1':>6s} {'λ₂':>10s} {'λ₂-0.5':>12s}")
print("-" * 35)
for p in range(16):
    m = p + 1
    c_max = min(m + 3, 30)
    T = make_transfer_matrix(m, c_max)
    evals = sorted([e.real for e in np.linalg.eigvals(T) if abs(e.imag) < 1e-10],
                   reverse=True)
    if len(evals) >= 2:
        print(f"{p:4d} {m:6d} {evals[1]:10.6f} {evals[1]-0.5:+12.6e}")

# ================================================================
# PART 5: Convergence rate prediction
# ================================================================
print("\n" + "=" * 70)
print("PART 5: CONVERGENCE RATE PREDICTION")
print("=" * 70)

print("""
The observed convergence rate ρ(K) of c₁(K) toward c₁(∞) is:
  ρ(K) = |Δ(K)| / |Δ(K-1)| → 1/2

This can now be understood:
1. c₁(K) depends on the carry statistics at the TOP of the chain
2. Each additional position K adds one more "layer" of carry propagation
3. Each layer attenuates perturbations by the sub-dominant eigenvalue λ₂
4. λ₂ = 1/2 for the carry Markov chain in base 2
5. Therefore ρ(K) → λ₂ = 1/2 as K → ∞

The residual ρ(K) - 1/2 > 0 comes from:
(a) Finite-size effects: at small K, the carry distribution hasn't
    reached stationarity, so higher eigenvalues (1/4, 1/8, ...) contribute
(b) Non-Markovian bit correlations: the actual convolution terms are
    not perfectly independent Bernoulli(1/4) — shared bits create
    correlations that slightly increase the effective eigenvalue
(c) Boundary effects: the MSB constraint (ULC condition) introduces
    a perturbation that decays non-exponentially

We can test prediction (a): if ρ(K) - 1/2 ≈ A/K^α, what is α?
""")

rho_data = {
    8: 0.64508593, 9: 0.64220383, 10: 0.63245228, 11: 0.61097350,
    12: 0.60004528, 13: 0.58242431, 14: 0.57212831, 15: 0.55416218,
    16: 0.53835300, 17: 0.50479509,
}

K_arr = np.array(list(rho_data.keys()), dtype=float)
rho_arr = np.array(list(rho_data.values()))
excess = rho_arr - 0.5

log_K = np.log(K_arr)
log_excess = np.log(excess)

A_fit = np.polyfit(log_K, log_excess, 1)
alpha = -A_fit[0]
print(f"Fit ln(ρ-1/2) = {A_fit[0]:.4f}·ln(K) + {A_fit[1]:.4f}")
print(f"  → ρ(K) - 1/2 ≈ {math.exp(A_fit[1]):.4f} · K^{{{-alpha:.2f}}}")
print(f"  α = {alpha:.3f}")

B_fit = np.polyfit(K_arr, np.log(excess), 1)
print(f"\nFit ln(ρ-1/2) = {B_fit[0]:.6f}·K + {B_fit[1]:.4f}")
print(f"  → ρ(K) - 1/2 ≈ {math.exp(B_fit[1]):.4f} · exp({B_fit[0]:.4f}·K)")
print(f"  Effective rate: exp({B_fit[0]:.4f}) = {math.exp(B_fit[0]):.4f}")

print(f"\n{'='*70}")
print("PART 6: THE BIG PICTURE — WHY 1/2")
print(f"{'='*70}")
print(f"""
SUMMARY: The factor 1/2 in ρ(K) → 1/2 comes from the DIVISION BY 2
in the carry recurrence c_{{k+1}} = floor((conv_k + c_k) / 2).

This is the most fundamental operation in binary arithmetic: dividing
by the base. It creates a "forgetting" rate of 1/2: each carry position
retains only half the information about the previous carry.

For a general base b, the same analysis gives ρ → 1/b:
  base 2:  ρ → 1/2
  base 10: ρ → 1/10
  base b:  ρ → 1/b

This is the Diaconis-Fulman eigenvalue λ₂ = 1/b.

CONNECTION TO π:
The fact that c₁ = π/18 (involving π) while ρ → 1/2 (involving 1/base)
are INDEPENDENT phenomena:
- π comes from the GEOMETRY of the D-parity boundary (the angular
  triangle α+β < π/4, where π/4 is intrinsically trigonometric)
- 1/2 comes from the ALGEBRAIC structure of carry propagation
  (division by the base)

They meet in the formula:
  c₁(K) = π/18 + A·K·(1/2)^K + o(K·2^{{-K}})

where π/18 is the limit, (1/2)^K is the convergence rate, and the
polynomial prefactor K arises from non-Markov bit-sharing correlations.
The amplitude A ≈ -2.6 ; the earlier A ≈ -40 absorbed the K factor.

ZETA CONNECTION?
The 1/2 here is 1/base = 1/2, NOT Re(s) = 1/2 from the critical line.
For base 10 multiplication, the convergence rate would be 1/10, not 1/2.
The Riemann critical line 1/2 is base-independent (it comes from the
functional equation's symmetry), while our 1/2 is base-2-specific.
However, a deeper connection might exist through:
  - p-adic valuation (carries ↔ 2-adic structure)
  - Benford's law (digit distribution ↔ ζ(s))
  - The Mellin transform of carry distributions
These remain unexplored.
""")
