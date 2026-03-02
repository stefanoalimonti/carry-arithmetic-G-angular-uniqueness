"""
G05_base3_verification.py — Verify ρ → 1/b for different bases

CRITICAL TEST: If our ρ(K) → 1/2 is truly 1/base (not 1/2 from the critical line),
then for BASE 3 multiplication, the transfer operator eigenvalue should be 1/3.

We compute:
1. Addition carry chain in base b → eigenvalue 1/b (exact)
2. Multiplication carry chain with Multinomial(m, 1/b²) convolution → eigenvalue → 1/b
3. Direct c₁(K) computation for base 3 and ratio check
"""

import numpy as np
from scipy.special import comb as binom_coeff
from itertools import product as cart_product
import math

print("G05: BASE DEPENDENCE OF ρ → 1/b")
print("=" * 70)

# ================================================================
# PART 1: Addition carry chain in base b
# ================================================================
print("\nPART 1: ADDITION CARRY CHAIN FOR DIFFERENT BASES\n")

for b in [2, 3, 5, 10]:
    T = np.zeros((b, b))
    for c in range(b):
        for a in range(b):
            for d in range(b):
                total = a + d + c
                new_c = total // b
                if new_c < b:
                    T[new_c][c] += 1.0 / (b * b)

    evals = sorted(np.linalg.eigvals(T).real, reverse=True)
    print(f"Base {b:2d}: eigenvalues = [{', '.join(f'{e:.6f}' for e in evals[:min(4, len(evals))])}]")
    print(f"         λ₂ = {evals[1]:.6f}, 1/b = {1/b:.6f}, "
          f"match = {'YES' if abs(evals[1] - 1/b) < 1e-10 else 'NO'}")

# ================================================================
# PART 2: Multiplication carry transfer matrix for base b
# ================================================================
print(f"\n{'='*70}")
print("PART 2: MULTIPLICATION CARRY TRANSFER MATRIX FOR BASE b")
print("=" * 70)

def mult_transfer_base_b(b, m, c_max=None):
    """
    Transfer matrix for multiplication carry in base b.
    conv_p = sum of m terms, each = product of two base-b digits.
    Product of two uniform [0,b-1] digits: possible values 0..(b-1)^2
    with known distribution.
    """
    digit_prod_probs = np.zeros((b-1)**2 + 1)
    for a in range(b):
        for d in range(b):
            digit_prod_probs[a * d] += 1.0 / (b * b)

    max_conv = m * (b - 1)**2
    if c_max is None:
        c_max = max_conv // (b - 1) + 2

    conv_probs = np.zeros(max_conv + 1)
    conv_probs[0] = 1.0
    for _ in range(m):
        new_probs = np.zeros(len(conv_probs) + len(digit_prod_probs) - 1)
        for v, pv in enumerate(conv_probs):
            if pv > 0:
                for w, pw in enumerate(digit_prod_probs):
                    if pw > 0:
                        new_probs[v + w] += pv * pw
        conv_probs = new_probs

    n_states = c_max + 1
    T = np.zeros((n_states, n_states))
    for c_in in range(n_states):
        for v in range(len(conv_probs)):
            if conv_probs[v] > 0:
                total = v + c_in
                c_out = total // b
                if c_out < n_states:
                    T[c_out][c_in] += conv_probs[v]

    return T

for b in [2, 3, 5]:
    print(f"\n--- Base {b} ---")
    print(f"{'m':>4s} {'λ₂':>10s} {'1/b':>10s} {'λ₂-1/b':>12s}")
    print("-" * 40)
    for m in [1, 2, 3, 5, 8, 12, 20]:
        c_max = min(m * (b-1)**2 // (b-1) + 5, 60)
        T = mult_transfer_base_b(b, m, c_max)
        evals = sorted([e.real for e in np.linalg.eigvals(T) if abs(e.imag) < 1e-8],
                        reverse=True)
        if len(evals) >= 2:
            print(f"{m:4d} {evals[1]:10.6f} {1/b:10.6f} {evals[1]-1/b:+12.2e}")

# ================================================================
# PART 3: Direct c₁(K) for base 3
# ================================================================
print(f"\n{'='*70}")
print("PART 3: EXACT c₁(K) FOR BASE 3")
print("=" * 70)

def c1_base_b(b, K):
    """
    Exact c₁(K) for base-b multiplication of K-digit numbers.
    Enumerates all pairs (X, Y) of K-digit base-b numbers with leading digit ≥ 1,
    checks if the product has 2K-1 digits (D-odd analogue),
    and computes E[c_{M-1} - 1].
    """
    min_val = b**(K-1)
    max_val = b**K - 1
    sum_cm1 = 0
    count_ulc = 0

    for x in range(min_val, max_val + 1):
        for y in range(min_val, max_val + 1):
            digits_x = []
            digits_y = []
            xx, yy = x, y
            for _ in range(K):
                digits_x.append(xx % b)
                digits_y.append(yy % b)
                xx //= b
                yy //= b

            M = 2 * K
            carries = [0] * (M + 1)
            for j in range(M):
                conv_j = 0
                for i in range(K):
                    if 0 <= j - i < K:
                        conv_j += digits_x[i] * digits_y[j - i]
                total = conv_j + carries[j]
                carries[j + 1] = total // b

            product = x * y
            D = len(str(product)) if product > 0 else 1
            actual_D = 0
            p = product
            while p > 0:
                actual_D += 1
                p //= b

            if carries[M - 1] > 0 and carries[M] == 0:
                sum_cm1 += carries[M - 1]
                count_ulc += 1

    if count_ulc == 0:
        return None, 0
    return sum_cm1 / count_ulc - 1, count_ulc

print(f"\nBase 3 multiplication: c₁(K) and convergence ratio ρ(K)")
print(f"{'K':>3s} {'c₁(K)':>14s} {'Δ':>14s} {'ρ':>10s} {'n_ulc':>10s}")
print("-" * 55)

c1_vals_b3 = {}
prev_delta = None
for K in range(2, 7):
    c1_val, n_ulc = c1_base_b(3, K)
    c1_vals_b3[K] = c1_val
    if K >= 3 and c1_vals_b3.get(K-1) is not None:
        delta = c1_val - c1_vals_b3[K-1]
        if prev_delta is not None and abs(prev_delta) > 1e-15:
            rho = abs(delta / prev_delta)
            print(f"{K:3d} {c1_val:14.10f} {delta:+14.6e} {rho:10.6f} {n_ulc:10d}")
        else:
            print(f"{K:3d} {c1_val:14.10f} {delta:+14.6e} {'---':>10s} {n_ulc:10d}")
        prev_delta = delta
    else:
        print(f"{K:3d} {c1_val:14.10f} {'---':>14s} {'---':>10s} {n_ulc:10d}")

print(f"\nExpected convergence rate for base 3: ρ → 1/3 = {1/3:.6f}")

# ================================================================
# PART 4: Direct c₁(K) for base 10
# ================================================================
print(f"\n{'='*70}")
print("PART 4: c₁(K) FOR BASE 10 (K=2,3 only — gets expensive)")
print("=" * 70)

for b in [10]:
    print(f"\nBase {b}:")
    print(f"{'K':>3s} {'c₁(K)':>14s} {'n_ulc':>10s}")
    print("-" * 30)
    for K in [2, 3]:
        c1_val, n_ulc = c1_base_b(b, K)
        if c1_val is not None:
            print(f"{K:3d} {c1_val:14.10f} {n_ulc:10d}")
        else:
            print(f"{K:3d} {'N/A':>14s} {n_ulc:10d}")

print(f"\n{'='*70}")
print("CONCLUSION")
print(f"{'='*70}")
print("""
If ρ(K) → 1/3 for base 3 and ρ(K) → 1/10 for base 10,
this PROVES that our 1/2 is 1/base, not Re(s)=1/2.

The eigenvalue spectrum of the carry transfer operator is:
  λ_k = 1/b^k,  k = 0, 1, 2, ...
  
giving eigenvalues 1, 1/b, 1/b², ...

For base 2: 1, 1/2, 1/4, 1/8, ...
For base 3: 1, 1/3, 1/9, 1/27, ...
For base b: 1, 1/b, 1/b², ...

The sub-dominant eigenvalue 1/b controls the convergence rate of c₁(K).
""")
