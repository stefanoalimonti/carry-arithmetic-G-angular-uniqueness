"""
G08_amplitude_multiterm.py — Multi-eigenvalue amplitude extraction

The carry transfer operator has eigenvalues 1/2^n (Diaconis-Fulman).
So the exact expansion is:
    c₁(K) = π/18 + A₁·(1/2)^K + A₂·(1/4)^K + A₃·(1/8)^K + ...

Using exact c₁(K) data for K=4..17, we can extract coefficients A_n
by solving an overdetermined linear system.

We also test the hypothesis that logarithmic corrections exist:
    c₁(K) = π/18 + (A₁ + B₁·K)·(1/2)^K + ...
"""

import numpy as np
from fractions import Fraction
import mpmath
mpmath.mp.dps = 50

print("G08: MULTI-EIGENVALUE AMPLITUDE EXTRACTION")
print("=" * 70)

raw_data = {
    4:  (41, 41),
    5:  (205, 195),
    6:  (956, 881),
    7:  (4174, 3759),
    8:  (17717, 15635),
    9:  (73301, 63853),
    10: (299404, 258613),
    11: (1212075, 1041063),
    12: (4882521, 4178907),
    13: (19607966, 16745419),
    14: (78609132, 67045167),
    15: (314829727, 268306599),
    16: (1260189799, 1073488167),
    17: (5042657326, 4294460189),
}

exact_data = {}
for K, (s, n) in raw_data.items():
    exact_data[K] = Fraction(s - n, n)

pi_18 = mpmath.pi / 18

print("\n--- Exact c₁(K) data ---")
print(f"{'K':>3s} {'c₁(K)':>22s} {'δ = c₁-π/18':>18s}")
print("-" * 50)
for K in sorted(exact_data.keys()):
    c1_mp = mpmath.mpf(exact_data[K].numerator) / mpmath.mpf(exact_data[K].denominator)
    delta = c1_mp - pi_18
    print(f"{K:3d} {float(c1_mp):22.15f} {float(delta):+18.12e}")

# ================================================================
# PART 1: Pure geometric model
# ================================================================
print(f"\n{'='*70}")
print("PART 1: PURE GEOMETRIC MODEL")
print("δ(K) = Σ_{n=1}^{N} A_n · (1/2^n)^K")
print("=" * 70)

K_vals = sorted(exact_data.keys())
for n_terms in [2, 3, 4, 5, 6]:
    K_use = K_vals[-n_terms*2:] if n_terms*2 <= len(K_vals) else K_vals
    n_data = len(K_use)

    M = mpmath.matrix(n_data, n_terms)
    b_vec = mpmath.matrix(n_data, 1)

    for i, K in enumerate(K_use):
        c1_mp = mpmath.mpf(exact_data[K].numerator) / mpmath.mpf(exact_data[K].denominator)
        b_vec[i] = c1_mp - pi_18
        for j in range(n_terms):
            rate = mpmath.mpf(1) / mpmath.power(2, j + 1)
            M[i, j] = mpmath.power(rate, K)

    try:
        Mt = M.T
        MtM = Mt * M
        Mtb = Mt * b_vec
        A_coeffs = mpmath.lu_solve(MtM, Mtb)

        residuals = M * A_coeffs - b_vec
        rms = float(mpmath.sqrt(sum(residuals[i]**2 for i in range(n_data)) / n_data))

        print(f"\n--- {n_terms} terms (using K={K_use[0]}..{K_use[-1]}, {n_data} points) ---")
        for j in range(n_terms):
            print(f"  A_{j+1} (rate 1/{2**(j+1)}) = {float(A_coeffs[j]):+.8e}")
        print(f"  RMS residual = {rms:.4e}")

        print("  Reconstruction check:")
        for K in [K_vals[-1], K_vals[-2]]:
            c1_mp = mpmath.mpf(exact_data[K].numerator) / mpmath.mpf(exact_data[K].denominator)
            recon = pi_18
            for j in range(n_terms):
                rate = mpmath.mpf(1) / mpmath.power(2, j + 1)
                recon += A_coeffs[j] * mpmath.power(rate, K)
            err = float(c1_mp - recon)
            print(f"    K={K}: c₁ = {float(c1_mp):.15f}, recon = {float(recon):.15f}, err = {err:+.4e}")
    except Exception as e:
        print(f"\n--- {n_terms} terms: FAILED ({e}) ---")

# ================================================================
# PART 2: Geometric + logarithmic correction
# ================================================================
print(f"\n{'='*70}")
print("PART 2: GEOMETRIC + LOGARITHMIC CORRECTION")
print("δ(K) = (A₁ + B₁·K)·(1/2)^K + (A₂ + B₂·K)·(1/4)^K + ...")
print("=" * 70)

for n_rates in [1, 2, 3]:
    n_params = 2 * n_rates
    K_use = K_vals
    n_data = len(K_use)
    if n_data < n_params:
        continue

    M = mpmath.matrix(n_data, n_params)
    b_vec = mpmath.matrix(n_data, 1)

    for i, K in enumerate(K_use):
        c1_mp = mpmath.mpf(exact_data[K].numerator) / mpmath.mpf(exact_data[K].denominator)
        b_vec[i] = c1_mp - pi_18
        for j in range(n_rates):
            rate = mpmath.mpf(1) / mpmath.power(2, j + 1)
            rK = mpmath.power(rate, K)
            M[i, 2*j] = rK
            M[i, 2*j + 1] = K * rK

    try:
        Mt = M.T
        MtM = Mt * M
        Mtb = Mt * b_vec
        coeffs = mpmath.lu_solve(MtM, Mtb)

        residuals = M * coeffs - b_vec
        rms = float(mpmath.sqrt(sum(residuals[i]**2 for i in range(n_data)) / n_data))

        print(f"\n--- {n_rates} rates × (A + B·K), {n_params} params ---")
        for j in range(n_rates):
            print(f"  Rate 1/{2**(j+1)}: A = {float(coeffs[2*j]):+.8e}, B = {float(coeffs[2*j+1]):+.8e}")
        print(f"  RMS residual = {rms:.4e}")
    except Exception as e:
        print(f"\n--- {n_rates} rates × (A+BK): FAILED ({e}) ---")

# ================================================================
# PART 3: PSLQ on A₁ (dominant amplitude)
# ================================================================
print(f"\n{'='*70}")
print("PART 3: PSLQ ON DOMINANT AMPLITUDE A₁")
print("=" * 70)

K_use = K_vals[-6:]
n_data = len(K_use)
M = mpmath.matrix(n_data, 3)
b_vec = mpmath.matrix(n_data, 1)
for i, K in enumerate(K_use):
    c1_mp = mpmath.mpf(exact_data[K].numerator) / mpmath.mpf(exact_data[K].denominator)
    b_vec[i] = c1_mp - pi_18
    M[i, 0] = mpmath.power(mpmath.mpf(1)/2, K)
    M[i, 1] = mpmath.power(mpmath.mpf(1)/4, K)
    M[i, 2] = mpmath.power(mpmath.mpf(1)/8, K)
coeffs3 = mpmath.lu_solve(M.T * M, M.T * b_vec)
A1_best = coeffs3[0]
print(f"Best A₁ from 3-term fit (K={K_use[0]}..{K_use[-1]}): {float(A1_best):.10f}")

print(f"\nPSLQ search for A₁ with various bases:")
A1_f = float(A1_best)
bases_pslq = [
    ("A₁, 1, π, ln2", [A1_f, 1.0, float(mpmath.pi), float(mpmath.log(2))]),
    ("A₁, 1, π, ln2, π²", [A1_f, 1.0, float(mpmath.pi), float(mpmath.log(2)), float(mpmath.pi**2)]),
    ("A₁, 1, π, ln2, π/18", [A1_f, 1.0, float(mpmath.pi), float(mpmath.log(2)), float(mpmath.pi/18)]),
    ("A₁, 1, π², ln²2", [A1_f, 1.0, float(mpmath.pi**2), float(mpmath.log(2)**2)]),
]

for label, vec in bases_pslq:
    try:
        vec_mp = [mpmath.mpf(v) for v in vec]
        rel = mpmath.pslq(vec_mp, maxcoeff=1000, maxsteps=5000)
        if rel is not None:
            terms = []
            for i, c in enumerate(rel):
                if c != 0:
                    terms.append(f"{c}·x{i}")
            print(f"  {label}: {' + '.join(terms)} = 0")
            if rel[0] != 0:
                val = sum(c * v for c, v in zip(rel[1:], vec[1:])) / (-rel[0])
                print(f"    → A₁ = {val:.10f} (from relation)")
        else:
            print(f"  {label}: no relation found")
    except Exception as e:
        print(f"  {label}: error ({e})")

# ================================================================
# PART 4: Per-K A estimates with Richardson extrapolation
# ================================================================
print(f"\n{'='*70}")
print("PART 4: PER-K AMPLITUDE ESTIMATES")
print("=" * 70)

print(f"\nA_K^(0) = δ(K) · 2^K  (0th order)")
print(f"A_K^(1) = (2·A_{K}^(0) - A_{K-1}^(0))  (Richardson, eliminating (1/4)^K)")
print(f"A_K^(2) = (4/3·A_{K}^(1) - 1/3·A_{K-1}^(1))  (eliminating (1/8)^K)")

A0 = {}
A1_rich = {}
A2_rich = {}

print(f"\n{'K':>3s} {'δ(K)':>16s} {'A_K^(0)':>14s} {'A_K^(1)':>14s} {'A_K^(2)':>14s}")
print("-" * 65)
for K in K_vals:
    c1_mp = mpmath.mpf(exact_data[K].numerator) / mpmath.mpf(exact_data[K].denominator)
    delta = c1_mp - pi_18
    a0 = delta * mpmath.power(2, K)
    A0[K] = a0

    a1_str = "---"
    if K - 1 in A0:
        a1 = 2 * a0 - A0[K-1]
        A1_rich[K] = a1
        a1_str = f"{float(a1):14.6f}"

    a2_str = "---"
    if K - 1 in A1_rich:
        a2 = mpmath.mpf(4)/3 * A1_rich[K] - mpmath.mpf(1)/3 * A1_rich[K-1]
        A2_rich[K] = a2
        a2_str = f"{float(a2):14.6f}"

    print(f"{K:3d} {float(delta):+16.10e} {float(a0):14.6f} {a1_str:>14s} {a2_str:>14s}")

if A2_rich:
    last_K = max(A2_rich.keys())
    print(f"\nBest A₁ estimate (A^(2) at K={last_K}): {float(A2_rich[last_K]):.8f}")

print(f"\n{'='*70}")
print("SUMMARY")
print("=" * 70)
print("""
The multi-term fit decomposes δ(K) = c₁(K) - π/18 into eigenvalue
contributions A_n · (1/2^n)^K.

Key: the amplitude A₁ of the dominant (1/2)^K term carries most of
the information. If A₁ has a closed form involving π, ln2, etc.,
it would complete the formula for c₁(K) at all K.
""")
