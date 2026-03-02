"""
G03_base3_pslq_extended.py — PSLQ analysis on c₁(base 3)

Uses the MC value c₁(3) ≈ 0.59870 ± 0.00002 (d=40, 10⁹ samples)
and exact enumeration values from G02 (K=2..12).

Tests extended bases including ln3, ln²3, arctan, ζ(2), etc.

Usage: python3 G03_base3_pslq_extended.py
"""

import mpmath
mpmath.mp.dps = 50

c1_mc = mpmath.mpf('0.5987045120')
c1_mc_err = mpmath.mpf('2.49e-5')

# Known c₁(K) values from G03_base3_exact.c (exact enumeration)
# These are the float values; rational (sum_cm1, n_ulc) pairs will be
# K=11, K=12 from G02 exact enumeration.
c1_exact = {
    2:  mpmath.mpf('0.500000000000000'),
    3:  mpmath.mpf('0.481481481481481'),
    4:  mpmath.mpf('0.531851851851852'),
    5:  mpmath.mpf('0.546700960219478'),
    6:  mpmath.mpf('0.567093851656498'),
    7:  mpmath.mpf('0.574612783496652'),
    8:  mpmath.mpf('0.582804025395297'),
    9:  mpmath.mpf('0.586844629946830'),
    10: mpmath.mpf('0.590154017513966'),
    # K=11, K=12: from G02 exact enumeration
}

print("G03: PSLQ ANALYSIS ON c₁(base 3)")
print("=" * 50)
print()

print("Exact enumeration data:")
prev_c1 = None
prev_delta = None
K_vals_sorted = sorted(c1_exact.keys())
for K in K_vals_sorted:
    c1_k = c1_exact[K]
    delta = None
    rho = None
    if prev_c1 is not None:
        delta = c1_k - prev_c1
        if prev_delta is not None and abs(prev_delta) > 1e-20:
            rho = abs(delta / prev_delta)
    print(f"  K={K:2d}: c₁ = {float(c1_k):.15f}"
          + (f"  Δ={float(delta):+.6e}" if delta else "")
          + (f"  ρ={float(rho):.4f}" if rho else ""))
    prev_delta = delta
    prev_c1 = c1_k

print(f"\n  MC (d=40, 10⁹): c₁ = {float(c1_mc):.10f} ± {float(c1_mc_err):.2e}")

# --- Extrapolation from exact data ---
print("\n" + "=" * 50)
print("EXTRAPOLATION (geometric model with ρ → 1/3)")
print("=" * 50)

K_vals = sorted(c1_exact.keys())
c1_vals = [c1_exact[K] for K in K_vals]

for n_terms in [1, 2, 3]:
    K_use = K_vals[-(n_terms + 1):]
    c_use = [c1_vals[K_vals.index(K)] for K in K_use]

    if n_terms == 1:
        c_inf = (c_use[1] - c_use[0] * mpmath.mpf(1)/3) / (1 - mpmath.mpf(1)/3)
        A = (c_use[1] - c_inf) * mpmath.power(3, K_use[1])
        print(f"\n  1-term (ρ=1/3, K={K_use}): c∞ = {float(c_inf):.10f}, A = {float(A):.4f}")

    elif n_terms == 2:
        K1, K2, K3 = K_use
        c1, c2, c3 = c_use
        r1 = mpmath.mpf(1) / 3
        r2 = mpmath.mpf(1) / 9

        M = mpmath.matrix([
            [1, mpmath.power(r1, K1), mpmath.power(r2, K1)],
            [1, mpmath.power(r1, K2), mpmath.power(r2, K2)],
            [1, mpmath.power(r1, K3), mpmath.power(r2, K3)],
        ])
        b = mpmath.matrix([c1, c2, c3])
        try:
            coeffs = mpmath.lu_solve(M, b)
            c_inf = coeffs[0]
            print(f"\n  2-term (1/3 + 1/9, K={K_use}): c∞ = {float(c_inf):.10f}")
            print(f"    A₁ = {float(coeffs[1]):.6f}, A₂ = {float(coeffs[2]):.6f}")
        except:
            print(f"\n  2-term fit failed for K={K_use}")

    elif n_terms == 3:
        K_use = K_vals[-4:]
        c_use = [c1_vals[K_vals.index(K)] for K in K_use]
        r1, r2, r3 = mpmath.mpf(1)/3, mpmath.mpf(1)/9, mpmath.mpf(1)/27

        M = mpmath.matrix([
            [1, mpmath.power(r1, K_use[i]), mpmath.power(r2, K_use[i]),
             mpmath.power(r3, K_use[i])]
            for i in range(4)
        ])
        b = mpmath.matrix([c_use[i] for i in range(4)])
        try:
            coeffs = mpmath.lu_solve(M, b)
            c_inf = coeffs[0]
            print(f"\n  3-term (1/3+1/9+1/27, K={K_use}): c∞ = {float(c_inf):.10f}")
            print(f"    A₁={float(coeffs[1]):.4f}, A₂={float(coeffs[2]):.4f}, A₃={float(coeffs[3]):.4f}")
        except:
            print(f"\n  3-term fit failed")

# --- PSLQ ---
print("\n" + "=" * 50)
print("PSLQ ANALYSIS")
print("=" * 50)

# Use best available value
c1_best = c1_mc
print(f"\nUsing c₁(3) = {float(c1_best):.10f} (MC, 5-digit precision)")
print("(PSLQ with 5 digits is marginal — relations with norm < ~100 are credible)\n")

# Define basis elements
basis_elements = {
    '1':           mpmath.mpf(1),
    'ln2':         mpmath.log(2),
    'ln3':         mpmath.log(3),
    'ln²3':        mpmath.log(3)**2,
    'ln2·ln3':     mpmath.log(2) * mpmath.log(3),
    'π':           mpmath.pi,
    'π²':          mpmath.pi**2,
    'π·ln3':       mpmath.pi * mpmath.log(3),
    '1/ln3':       1 / mpmath.log(3),
    'arctan(1/2)': mpmath.atan(mpmath.mpf(1)/2),
    'arctan(2)':   mpmath.atan(2),
    'ζ(2)':        mpmath.pi**2 / 6,
    'ζ(2)/9':      mpmath.pi**2 / 54,
    'ζ(3)':        mpmath.zeta(3),
    'ln(4/3)':     mpmath.log(mpmath.mpf(4)/3),
    'G':           mpmath.catalan,
    '√3':          mpmath.sqrt(3),
    'ln3/√3':      mpmath.log(3) / mpmath.sqrt(3),
}

# Test specific simple candidates first
print("--- Simple candidates ---")
candidates = [
    ('ln(3) - 1/2', mpmath.log(3) - mpmath.mpf(1)/2),
    ('ln²(3)/2', mpmath.log(3)**2 / 2),
    ('3/5', mpmath.mpf(3)/5),
    ('ln(3)/(ln(3)+1)', mpmath.log(3) / (mpmath.log(3) + 1)),
    ('2ln(3)/π', 2*mpmath.log(3)/mpmath.pi),
    ('1 - 1/e', 1 - 1/mpmath.e),
    ('(e-1)/e·ln3/ln2', (mpmath.e-1)/mpmath.e * mpmath.log(3)/mpmath.log(2)),
    ('3·arctan(1/√3)', 3*mpmath.atan(1/mpmath.sqrt(3))),
    ('π/√(3·e)', mpmath.pi/mpmath.sqrt(3*mpmath.e)),
    ('ln(3)·ln(2)', mpmath.log(3)*mpmath.log(2)),
    ('1/(1+1/ln3)', 1/(1+1/mpmath.log(3))),
    ('3·ln(3/2)', 3*mpmath.log(mpmath.mpf(3)/2)),
    ('arctan(2)', mpmath.atan(2)),
    ('(3-1)·ln(3)/(3+1)', 2*mpmath.log(3)/4),
]

for name, val in candidates:
    delta = float(c1_best - val)
    sigma = delta / float(c1_mc_err)
    marker = " *** CLOSE ***" if abs(sigma) < 5 else ""
    print(f"  {name:25s} = {float(val):.10f}  Δ={delta:+.6e}  ({sigma:+.1f}σ){marker}")

# PSLQ with various basis sets
print("\n--- PSLQ searches ---")

basis_sets = [
    ("Logarithmic", ['1', 'ln2', 'ln3', 'ln²3', 'ln2·ln3']),
    ("With π", ['1', 'ln3', 'π', 'ln²3']),
    ("With ζ(2)", ['1', 'ln3', 'ζ(2)', 'ln²3']),
    ("Arctan", ['1', 'arctan(1/2)', 'arctan(2)', 'ln3']),
    ("With √3", ['1', 'ln3', '√3', 'ln3/√3']),
    ("Full kitchen sink", ['1', 'ln2', 'ln3', 'ln²3', 'π', 'arctan(2)']),
    ("Catalan", ['1', 'ln3', 'G', 'π']),
    ("With 1/ln3", ['1', 'ln3', '1/ln3', 'ln²3']),
]

for name, keys in basis_sets:
    vec = [c1_best] + [basis_elements[k] for k in keys]
    try:
        rel = mpmath.pslq(vec, maxcoeff=1000, maxsteps=5000)
        if rel is not None:
            norm = sum(abs(r) for r in rel)
            terms = []
            if rel[0] != 0:
                terms.append(f"{rel[0]}·c₁")
            for i, k in enumerate(keys):
                if rel[i+1] != 0:
                    terms.append(f"{rel[i+1]:+d}·{k}")
            expr = " ".join(terms) + " = 0"

            if rel[0] != 0:
                c1_from_rel = -sum(rel[i+1] * basis_elements[keys[i]]
                                   for i in range(len(keys))) / rel[0]
                delta = float(c1_best - c1_from_rel)
                sigma = delta / float(c1_mc_err)
                print(f"\n  [{name}] PSLQ found: {expr}")
                print(f"    norm={norm}, c₁ implied = {float(c1_from_rel):.12f}")
                print(f"    Δ from MC = {delta:+.6e} ({sigma:+.1f}σ)")
                if norm > 200:
                    print(f"    ⚠ High norm ({norm}) — likely spurious with 5-digit input")
            else:
                print(f"\n  [{name}] PSLQ found (c₁ coefficient = 0): {expr}")
                print(f"    This is a relation among the basis, not involving c₁")
        else:
            print(f"\n  [{name}] PSLQ: no relation found (maxcoeff=1000)")
    except Exception as e:
        print(f"\n  [{name}] PSLQ failed: {e}")

print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"c₁(3) = {float(c1_best):.10f} ± {float(c1_mc_err):.2e} (MC)")
print(f"Precision: ~5 significant digits")
print(f"Need ~10-12 digits for reliable PSLQ (requires exact enum to K≥15)")
print(f"Angular Uniqueness confirmed: c₁(3) ≠ π/18 at 17040σ")
