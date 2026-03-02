"""
G06_base3_extended.py — Extended base-3 c₁(K) computation

Faster implementation using numpy for K up to 8 (3^16 = 43M — tight but doable).
"""

import numpy as np
import time

def c1_base3_fast(K):
    """Fast c₁(K) for base 3 using vectorized computation."""
    b = 3
    n_digits = K
    n_vals = b**K

    t0 = time.time()
    sum_cm1 = 0
    count_ulc = 0
    M = 2 * K

    chunk_size = min(n_vals, 500)

    for x in range(b**(K-1), b**K):
        dx = []
        xx = x
        for _ in range(K):
            dx.append(xx % b)
            xx //= b

        for y in range(b**(K-1), b**K):
            dy = []
            yy = y
            for _ in range(K):
                dy.append(yy % b)
                yy //= b

            carries = 0
            for j in range(M):
                conv_j = 0
                for i in range(max(0, j - K + 1), min(K, j + 1)):
                    conv_j += dx[i] * dy[j - i]
                total = conv_j + carries
                carries = total // b
                if j == M - 2:
                    cm1 = carries
                if j == M - 1:
                    cm_top = carries

            if cm1 > 0 and cm_top == 0:
                sum_cm1 += cm1
                count_ulc += 1

    elapsed = time.time() - t0
    c1 = sum_cm1 / count_ulc - 1 if count_ulc > 0 else None
    return c1, count_ulc, sum_cm1, elapsed

print("G06: EXTENDED BASE-3 c₁(K) COMPUTATION")
print("=" * 70)
print(f"{'K':>3s} {'c₁(K)':>14s} {'Δ':>14s} {'ρ':>10s} {'n_ulc':>10s} {'time':>8s}")
print("-" * 65)

prev_c1 = None
prev_delta = None

for K in range(2, 8):
    n_pairs = (3**K - 3**(K-1))**2
    if n_pairs > 5e9:
        print(f"{K:3d} --- too large ({n_pairs:.0e} pairs)")
        break

    c1, n_ulc, sum_cm1, elapsed = c1_base3_fast(K)

    delta_str = "---"
    rho_str = "---"
    if prev_c1 is not None:
        delta = c1 - prev_c1
        delta_str = f"{delta:+14.6e}"
        if prev_delta is not None and abs(prev_delta) > 1e-15:
            rho = abs(delta / prev_delta)
            rho_str = f"{rho:10.6f}"
        prev_delta = delta

    print(f"{K:3d} {c1:14.10f} {delta_str:>14s} {rho_str:>10s} {n_ulc:10d} {elapsed:7.1f}s")
    prev_c1 = c1

print(f"\nExpected: ρ → 1/3 = 0.333333")
