#!/usr/bin/env python3
"""
G15 — Mellin transform with exact cascade data.

E57/E59/E61 computed the Mellin transform M[f](s) of the carry density from
MC data, finding NO peaks at zeta zeros (resolution-limited by MC noise).

NOW we have exact closed-form cascade contributions for J=1..7 (from E65-E67).
This allows us to build a much more precise "cascade Mellin transform":

  M_cascade(s) = Σ_{J=1}^7 M[contrib_J](s)

where each contrib_J is known exactly as (r_J + Σ n_{J,p} ln(p)) / 4^J.

This script explores:
  A. The "cascade Dirichlet series" D(s) = Σ_{J,p} n_{J,p} p^{-s} / 4^J
  B. The "cascade L-function" L_c(s) = Σ_J contrib_J(∞) · J^{-s}
  C. Mellin of the cascade density: interpreting cascade depth J as a scale
  D. Poles/zeros of D(s) vs Riemann zeta zeros
  E. The η(s) connection: does D(1/2+it) peak at zeta zeros?
  F. Functional equation test

Requires: mpmath
"""
import sys
from math import log, pi, sqrt, atan2
from fractions import Fraction
from collections import defaultdict

try:
    import mpmath
    mpmath.mp.dps = 50
    HAS_MPMATH = True
except ImportError:
    HAS_MPMATH = False
    print("WARNING: mpmath not available. Install for full precision.")

DATA = {
    1: {0: 2, 2: 18, 3: -8, 5: -12, 7: 7},
    2: {0: 14, 2: 2, 3: -132, 5: 40, 7: -4, 11: 22, 13: 8},
    3: {0: 61, 2: 1512, 3: 464, 5: 24, 7: -152, 11: -8, 13: -8,
        17: -768, 19: 76, 23: 184, 29: 16},
    4: {0: 341, 2: 2808, 3: -3392, 5: 344, 7: 1464, 11: -6144,
        13: 672, 17: 1360, 19: 16, 23: -16, 29: -16, 31: 976,
        37: 160, 41: 128, 43: 688, 53: 32, 61: 32},
    5: {0: 1490, 2: 135648, 3: 37888, 5: -30688, 7: -12704,
        11: 10848, 13: -43616, 17: 3040, 19: -32, 23: 1152, 29: 96,
        31: -10848, 37: 1024, 41: 3936, 43: -32, 47: -32, 53: -32,
        59: -32, 61: 3872, 67: 1072, 71: 2272, 73: -8384, 79: 5056,
        83: 2656, 89: 192, 97: 192, 101: 4160, 109: 64, 113: 64,
        127: -4064},
    6: {0: 7095, 2: 371936, 3: -208192, 5: 89728, 7: 127680,
        11: 97472, 13: 101632, 17: 53184, 19: 38848, 23: 9344,
        29: 2560, 31: -15616, 37: 10496, 41: 64, 43: -393216,
        47: 1152, 53: -9984, 59: 576, 61: -19328, 67: -8512,
        71: 9152, 73: 23360, 79: 10176, 83: 64, 89: -64,
        97: -27200, 101: -64, 103: 13120, 107: -64, 109: 13888,
        127: 32448, 131: 4192, 137: 3456, 139: 8896, 149: 640,
        151: 19328, 157: -10112, 163: 10432, 173: -11136,
        179: 11456, 181: -11648, 191: -12224, 193: 640,
        197: 16128, 223: -14272, 229: 128, 233: 128, 241: 128},
    7: {0: 30581, 2: 8699776, 3: 1154944, 5: 659328, 7: 4928,
        11: 227840, 13: 111872, 17: 181888, 19: 153344, 23: 30208,
        29: 249600, 31: -3712, 37: -82496, 41: 246272, 43: 507264,
        47: 1280, 53: 180352, 59: 76032, 61: -21760, 67: 80000,
        71: 45568, 73: -26624, 79: -40576, 83: -128, 89: 34432,
        97: 331008, 101: 640, 103: -51072, 107: -27776, 109: -55424,
        113: 108928, 127: -308608, 131: -45824, 137: 59776,
        139: 54272, 149: -96000, 151: 39040, 163: 256, 167: 128,
        173: -128, 179: -128, 181: 46208, 191: 48768, 193: -216320,
        197: -128, 199: 50816, 211: -128, 223: 56960, 229: -128,
        233: -128, 239: 61056, 241: 61568, 251: 64128,
        257: -3145728, 263: 33664, 269: 8704, 271: 69376,
        277: -72192, 281: -72704, 283: 72448, 293: 48384,
        307: 39296, 311: 39808, 313: -39936, 317: -40704,
        331: 42368, 337: 66560, 347: 44416, 349: -44800, 353: 768,
        367: -46976, 373: -47872, 379: 48512, 383: 49024,
        389: 128768, 397: 57600, 401: 768, 409: 256, 421: 256,
        431: -55168, 433: 57600, 443: -56704, 449: 768, 457: 256,
        461: 256, 487: -62336, 509: 256},
}

J_MAX = 7

LN = {}
for J in DATA:
    for p in DATA[J]:
        if p >= 2:
            LN[p] = log(p)


def contrib_J_real(J):
    """Compute contrib_J(∞) = (r_J + Σ n_{J,p} ln(p)) / 4^J as a float."""
    r_J = DATA[J].get(0, 0)
    log_part = sum(n * LN.get(p, 0) for p, n in DATA[J].items() if p >= 2)
    return (r_J + log_part) / 4**J


# First 30 zeta zeros (imaginary parts)
ZETA_ZEROS = [
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
    67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
    79.337375, 82.910381, 84.735493, 87.425275, 88.809111,
    92.491899, 94.651344, 95.870634, 98.831194, 101.317851,
]


# ====================================================================
print("=" * 80)
print("PART A: CASCADE DIRICHLET SERIES D(s) = Σ n_{J,p} p^{-s} / 4^J")
print("=" * 80)
print()
sys.stdout.flush()

def D_cascade(s_complex):
    """D(s) = Σ_{J=1}^7 Σ_p n_{J,p} p^{-s} / 4^J  (complex s)."""
    if HAS_MPMATH:
        result = mpmath.mpf(0)
        for J in range(1, J_MAX + 1):
            for p, n in DATA[J].items():
                if p >= 2:
                    result += mpmath.mpf(n) * mpmath.power(p, -s_complex) / mpmath.mpf(4**J)
        return result
    else:
        result = 0.0
        for J in range(1, J_MAX + 1):
            for p, n in DATA[J].items():
                if p >= 2:
                    result += n * p**(-s_complex) / 4**J
        return result


print("D(s) for real s:\n")
print(f"{'s':>6s} {'D(s)':>16s}")

for s_val in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0]:
    d_val = float(D_cascade(s_val))
    print(f"{s_val:6.1f} {d_val:16.10f}")

# ====================================================================
print("\n\n" + "=" * 80)
print("PART B: CASCADE L-FUNCTION L_c(s) = Σ_J contrib_J(∞) · J^{-s}")
print("=" * 80)
print()
sys.stdout.flush()

contribs = {}
for J in range(1, J_MAX + 1):
    contribs[J] = contrib_J_real(J)
    print(f"  contrib_{J}(∞) = {contribs[J]:+.12f}")

SIGMA_ODD = pi/18 - (1 + 3*log(3/4))
cumul = sum(contribs.values())
print(f"\n  Σ_{'{1..7}'} contrib_J = {cumul:.12f}")
print(f"  Target Σ_odd           = {SIGMA_ODD:.12f}")
print(f"  Captured: {cumul/SIGMA_ODD*100:.2f}%")


def L_cascade(s_complex):
    """L_c(s) = Σ_{J=1}^7 contrib_J · J^{-s}."""
    if HAS_MPMATH:
        result = mpmath.mpf(0)
        for J in range(1, J_MAX + 1):
            result += mpmath.mpf(contribs[J]) * mpmath.power(J, -s_complex)
        return result
    else:
        result = 0.0
        for J in range(1, J_MAX + 1):
            result += contribs[J] * J**(-s_complex)
        return result


print(f"\nL_c(s) for real s:")
print(f"{'s':>6s} {'L_c(s)':>16s}")

for s_val in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
    lc_val = float(L_cascade(s_val))
    print(f"{s_val:6.1f} {lc_val:16.10f}")

# ====================================================================
print("\n\n" + "=" * 80)
print("PART C: DEPTH-SCALE MELLIN TRANSFORM")
print("=" * 80)
print()
sys.stdout.flush()

print("""Interpreting cascade depth J as a "scale" variable:
  M_depth(s) = Σ_J contrib_J · (1/4)^{J·s} = Σ_J contrib_J · 4^{-Js}

This is the Mellin transform of the "cascade measure" at scale 4^J.
If the cascade has self-similar structure, M_depth(s) should have
special values at particular s.
""")

def M_depth(s_complex):
    """M_depth(s) = Σ_J contrib_J · 4^{-Js}."""
    if HAS_MPMATH:
        result = mpmath.mpf(0)
        for J in range(1, J_MAX + 1):
            result += mpmath.mpf(contribs[J]) * mpmath.power(4, -J * s_complex)
        return result
    else:
        result = 0.0
        for J in range(1, J_MAX + 1):
            result += contribs[J] * 4**(-J * s_complex)
        return result

print(f"{'s':>6s} {'M_depth(s)':>16s} {'4^s · M':>16s}")

for s_val in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
    md = float(M_depth(s_val))
    scaled = md * 4**s_val
    print(f"{s_val:6.1f} {md:16.10f} {scaled:16.10f}")

print(f"\nAt s=0: M_depth(0) = Σ contrib_J = {float(M_depth(0)):.10f}")
print(f"At s=1: M_depth(1) = Σ contrib_J/4^J = {float(M_depth(1)):.10f}")

# ====================================================================
print("\n\n" + "=" * 80)
print("PART D: D(1/2 + it) VS RIEMANN ZETA ZEROS")
print("=" * 80)
print()
sys.stdout.flush()

print("Scanning |D(1/2 + it)| along the critical line:\n")

if HAS_MPMATH:
    sigma = mpmath.mpf('0.5')
    t_scan = [mpmath.mpf(t) for t in range(0, 105)]
    for t0 in ZETA_ZEROS[:15]:
        for dt in [-0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5]:
            t_scan.append(mpmath.mpf(t0 + dt))

    t_scan = sorted(set(t_scan))

    results = []
    for t in t_scan:
        s = sigma + 1j * t
        d_val = mpmath.mpf(0) + 1j * mpmath.mpf(0)
        for J in range(1, J_MAX + 1):
            for p, n in DATA[J].items():
                if p >= 2:
                    d_val += mpmath.mpf(n) * mpmath.power(p, -s) / mpmath.mpf(4**J)
        magnitude = abs(d_val)
        results.append((float(t), float(magnitude)))

    results.sort(key=lambda x: x[0])

    print(f"{'t':>10s} {'|D(1/2+it)|':>14s} {'Near zeta zero?':>16s}")
    for t_val, mag in results:
        near_zero = ""
        for z0 in ZETA_ZEROS:
            if abs(t_val - z0) < 0.5:
                near_zero = f" <-- ζ zero ~{z0:.1f}"
                break
        if near_zero or mag > 0.1 or t_val % 5 < 0.01:
            print(f"{t_val:10.3f} {mag:14.8f} {near_zero}")

    top_peaks = sorted(results, key=lambda x: -x[1])[:20]
    print(f"\nTop 20 peaks of |D(1/2+it)|:")
    for t_val, mag in top_peaks:
        near = ""
        for z0 in ZETA_ZEROS:
            if abs(t_val - z0) < 1.0:
                near = f" [near ζ zero {z0:.2f}]"
        print(f"  t = {t_val:8.3f}, |D| = {mag:.8f}{near}")

    zeta_zero_mags = []
    for z0 in ZETA_ZEROS[:15]:
        s = sigma + 1j * mpmath.mpf(z0)
        d_val = mpmath.mpf(0) + 1j * mpmath.mpf(0)
        for J in range(1, J_MAX + 1):
            for p, n in DATA[J].items():
                if p >= 2:
                    d_val += mpmath.mpf(n) * mpmath.power(p, -s) / mpmath.mpf(4**J)
        zeta_zero_mags.append((z0, float(abs(d_val))))

    print(f"\n|D(1/2+it)| at first 15 zeta zeros:")
    print(f"{'ζ zero t':>10s} {'|D|':>14s}")
    for z0, mag in zeta_zero_mags:
        print(f"{z0:10.3f} {mag:14.8f}")

    avg_at_zeros = sum(m for _, m in zeta_zero_mags) / len(zeta_zero_mags)
    all_mags = [m for _, m in results]
    avg_overall = sum(all_mags) / len(all_mags)

    print(f"\n  Mean |D| at zeta zeros: {avg_at_zeros:.8f}")
    print(f"  Mean |D| overall:       {avg_overall:.8f}")
    print(f"  Ratio: {avg_at_zeros / avg_overall:.4f}")

    if abs(avg_at_zeros / avg_overall - 1) < 0.3:
        print(f"  -> No significant enhancement at zeta zeros.")
    else:
        print(f"  -> {'ENHANCEMENT' if avg_at_zeros > avg_overall else 'SUPPRESSION'} at zeta zeros!")

else:
    print("  (mpmath required for complex arithmetic — skipping)")

# ====================================================================
print("\n\n" + "=" * 80)
print("PART E: THE η(s) CONNECTION — PARITY PROJECTION")
print("=" * 80)
print()
sys.stdout.flush()

print("""The Dirichlet eta function η(s) = (1 - 2^{1-s})ζ(s) is related to
the parity eigenvector φ(c) = (-1)^c of the carry operator.

Test: is D(s) / η(s) smooth (no zeros/poles) for Re(s) > 0?
If so, D(s) "factors through" η(s), meaning the cascade encodes
the same alternating structure as the eta function.
""")

if HAS_MPMATH:
    print(f"{'s':>8s} {'D(s)':>16s} {'η(s)':>14s} {'D/η':>14s} {'ζ(s)':>14s} {'D/ζ':>14s}")

    for s_val in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 8.0]:
        d_val = float(D_cascade(s_val))
        z_val = float(mpmath.zeta(s_val)) if s_val != 1.0 else float('inf')
        eta_val = float((1 - 2**(1 - s_val)) * mpmath.zeta(s_val)) if s_val != 1.0 else float(mpmath.log(2))

        d_over_eta = d_val / eta_val if abs(eta_val) > 1e-15 else float('nan')
        d_over_zeta = d_val / z_val if z_val != float('inf') and abs(z_val) > 1e-15 else float('nan')

        print(f"{s_val:8.1f} {d_val:16.10f} {eta_val:14.10f} "
              f"{d_over_eta:14.10f} {z_val:14.10f} {d_over_zeta:14.10f}")

    print(f"\nIf D/η or D/ζ is approximately constant, that's a signal.")
    d_over_eta_vals = []
    for s_val in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        d_val = float(D_cascade(s_val))
        eta_val = float((1 - 2**(1 - s_val)) * mpmath.zeta(s_val))
        d_over_eta_vals.append(d_val / eta_val)

    spread = max(d_over_eta_vals) - min(d_over_eta_vals)
    mean_de = sum(d_over_eta_vals) / len(d_over_eta_vals)
    print(f"  D/η values for s = 1.5..5: {[f'{v:.6f}' for v in d_over_eta_vals]}")
    print(f"  Mean: {mean_de:.6f}, spread: {spread:.6f}, CV: {spread/abs(mean_de):.4f}")

    if spread / abs(mean_de) < 0.1:
        print(f"  -> D(s) ≈ {mean_de:.4f} · η(s)  (approximate proportionality!)")
    else:
        print(f"  -> D/η is NOT constant. No simple proportionality.")

# ====================================================================
print("\n\n" + "=" * 80)
print("PART F: FUNCTIONAL EQUATION TEST")
print("=" * 80)
print()
sys.stdout.flush()

print("""For ζ(s): the functional equation is ξ(s) = ξ(1-s) where
  ξ(s) = π^{-s/2} Γ(s/2) ζ(s)

Test: does D(s) satisfy any approximate functional equation?
We check D(s) vs D(1-s), D(s) vs D(2-s), etc.
""")

if HAS_MPMATH:
    print(f"{'s':>8s} {'D(s)':>14s} {'D(1-s)':>14s} {'D(2-s)':>14s} {'D(s)/D(1-s)':>14s}")

    for s_val in [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
        ds = float(D_cascade(s_val))
        d1ms = float(D_cascade(1 - s_val))
        d2ms = float(D_cascade(2 - s_val))
        ratio = ds / d1ms if abs(d1ms) > 1e-15 else float('nan')
        print(f"{s_val:8.2f} {ds:14.8f} {d1ms:14.8f} {d2ms:14.8f} {ratio:14.8f}")

    print(f"\nWith Gamma correction (ξ-type):")
    print(f"{'s':>8s} {'ξ_D(s)':>14s} {'ξ_D(1-s)':>14s} {'Ratio':>14s}")

    for s_val in [0.25, 0.5, 0.75, 1.5, 2.0]:
        ds = D_cascade(s_val)
        d1ms = D_cascade(1 - s_val)
        gamma_s = mpmath.gamma(s_val / 2)
        gamma_1ms = mpmath.gamma((1 - s_val) / 2)

        xi_s = float(mpmath.pi**(-s_val/2) * gamma_s * ds)
        xi_1ms = float(mpmath.pi**(-(1-s_val)/2) * gamma_1ms * d1ms)

        ratio = xi_s / xi_1ms if abs(xi_1ms) > 1e-15 else float('nan')
        print(f"{s_val:8.2f} {xi_s:14.8f} {xi_1ms:14.8f} {ratio:14.8f}")

# ====================================================================
print("\n\n" + "=" * 80)
print("PART G: DECOMPOSITION BY PRIME — INDIVIDUAL p^{-s} COEFFICIENTS")
print("=" * 80)
print()
sys.stdout.flush()

print("For each prime p, its total contribution to D(s) is:")
print("  a_p(s) = Σ_J n_{J,p} / 4^J  ×  p^{-s}\n")

prime_totals = defaultdict(Fraction)
for J in range(1, J_MAX + 1):
    for p, n in DATA[J].items():
        if p >= 2:
            prime_totals[p] += Fraction(n, 4**J)

print(f"{'p':>5s} {'a_p = Σ n/4^J':>16s} {'|a_p|':>14s} {'a_p × ln(p)':>14s}")

for p in sorted(prime_totals.keys()):
    a_p = float(prime_totals[p])
    abs_a = abs(a_p)
    a_ln = a_p * log(p)
    if abs_a > 1e-10:
        print(f"{p:5d} {a_p:+16.10f} {abs_a:14.10f} {a_ln:+14.10f}")

print(f"\nThe log-weighted sum Σ a_p · ln(p) should give the logarithmic part of Σ_odd.")
total_log = sum(float(prime_totals[p]) * log(p) for p in prime_totals)
print(f"  Σ a_p · ln(p) = {total_log:.10f}")

SIGMA_ODD = pi/18 - (1 + 3*log(3/4))
rational_sum = sum(DATA[J].get(0, 0) / 4**J for J in range(1, J_MAX + 1))
print(f"  Rational part  = {rational_sum:.10f}")
print(f"  Total          = {rational_sum + total_log:.10f}")
print(f"  Target Σ_odd   = {SIGMA_ODD:.10f}")

# Check which primes have the largest |a_p|
sorted_primes = sorted(prime_totals.keys(), key=lambda p: -abs(float(prime_totals[p])))
print(f"\nTop 15 primes by |a_p|:")
for p in sorted_primes[:15]:
    a_p = float(prime_totals[p])
    print(f"  p = {p:5d}: a_p = {a_p:+.10f}")

# ====================================================================
print("\n\n" + "=" * 80)
print("PART H: COMPARISON WITH E57 MC-BASED MELLIN")
print("=" * 80)
print()
sys.stdout.flush()

print("""E57 computed M[f](s) from Monte Carlo data for f(T).
Our D(s) is structurally different: it's a Dirichlet series over primes,
not a Mellin integral over [0,1].

The BRIDGE is: Σ_odd = (π/4) ∫₀¹ f(T) K(T) dT
                     = Σ_J contrib_J(∞)
                     = rational + Σ_p a_p · ln(p)

So D(s) is the "prime decomposition" of the same quantity that
M[f](s) measures in the angular/Mellin domain.

Connection to ζ(s):
  D(s) = Σ_p a_p p^{-s}
  -ζ'(s)/ζ(s) = Σ_p ln(p) p^{-s} / (1 - p^{-s})

If a_p ≈ c · ln(p) for some constant c, then D(s) ≈ c · (-ζ'(s)/ζ(s)).
""")

if HAS_MPMATH:
    print(f"Testing: a_p vs c · ln(p):")
    c_vals = []
    for p in sorted(prime_totals.keys()):
        a_p = float(prime_totals[p])
        if abs(a_p) > 1e-8 and log(p) > 0:
            c_vals.append(a_p / log(p))

    if c_vals:
        mean_c = sum(c_vals) / len(c_vals)
        std_c = sqrt(sum((c - mean_c)**2 for c in c_vals) / len(c_vals))
        print(f"  a_p / ln(p) values: mean = {mean_c:.6f}, std = {std_c:.6f}")
        print(f"  CV = {std_c / abs(mean_c):.4f}")

        if std_c / abs(mean_c) < 0.2:
            print(f"  -> a_p ≈ {mean_c:.4f} · ln(p)  (approximate!)")
            print(f"     This would mean D(s) ≈ {mean_c:.4f} · (-ζ'(s)/ζ(s))")
        else:
            print(f"  -> a_p / ln(p) is NOT constant. The prime structure is richer.")

        print(f"\n  Detailed a_p / ln(p) for small primes:")
        for p in sorted(prime_totals.keys())[:20]:
            a_p = float(prime_totals[p])
            if abs(a_p) > 1e-8:
                print(f"    p = {p:5d}: a_p/ln(p) = {a_p/log(p):+.8f}")


# ====================================================================
print("\n\n" + "=" * 80)
print("SYNTHESIS")
print("=" * 80)
print(f"""
KEY FINDINGS:

1. CASCADE DIRICHLET SERIES D(s):
   D(s) = Σ_{{J,p}} n_{{J,p}} p^{{-s}} / 4^J is well-defined for Re(s) > 0.
   It encodes the prime structure of the cascade in Dirichlet form.

2. CRITICAL LINE SCAN:
   |D(1/2 + it)| tested at the first 15 Riemann zeta zeros.
   Enhancement/suppression ratio reported above.

3. η(s) CONNECTION:
   D(s)/η(s) tested for approximate constancy.
   If constant, D(s) inherits the zeta zeros through η(s) = (1-2^{{1-s}})ζ(s).

4. FUNCTIONAL EQUATION:
   D(s) vs D(1-s) tested directly and with Gamma correction.
   No exact functional equation expected (D is a finite sum), but
   approximate symmetry would signal a deeper structure.

5. PRIME DECOMPOSITION:
   The per-prime coefficients a_p = Σ_J n_{{J,p}}/4^J show whether
   the cascade "prefers" certain primes. The 7-depth truncation limits
   accuracy for large primes.

6. COMPARISON WITH MC MELLIN:
   D(s) is the Dirichlet-series face of the same information that
   M[f](s) encodes in the angular domain. The bridge is the
   decomposition Σ_odd = rational + Σ a_p ln(p).
""")

print("=" * 80)
print("  END OF G15")
print("=" * 80)
