# Experiments — Paper G

| Script | Lang | Description | Referenced in |
|--------|------|-------------|---------------|
| `G01_base3_highprec_mc.c` | C | High-precision Monte Carlo for c₁(3) | §4.1 |
| `G02_base3_exact_k12.c` | C | Exact enumeration c₁(3, K) for K = 2–12 | §4.2 |
| `G03_pslq_extended.py` | Python | PSLQ analysis on extrapolated c₁(3) | §3.4, §4.2 |
| `G04_transfer_eigenvalues.py` | Python | Transfer operator eigenvalue verification (5 bases) | §4.5 |
| `G05_base3_verification.py` | Python | Base-3 eigenvalue verification | §4.5 |
| `G06_base3_extended.py` | Python | Base-3 extended analysis | §4.5 |
| `G07_base3_exact.c` | C | Base-3 exact enumeration K = 2–10 | §4.2 |
| `G08_amplitude_multiterm.py` | Python | Multi-term Diaconis–Fulman amplitude fit | §4.3 |
| `G09_base3_pslq.py` | Python | Base-3 PSLQ on c₁(3) | §3.4, §4.2 |
| `G10_base3_exact_k13.c` | C | Extended enumeration K = 11–14 | §4.2 |
| `G11_enhanced_pslq.py` | Python | Enhanced PSLQ with 5-term Richardson | §4.2 |
| `G12_enhanced_pslq_k12.py` | Python | Odd-even separation, alternating eigenvalue models | §4.2, §4.3 |
| `G13_padic_multiplication.py` | Python | p-adic multiplication analysis | §3.4 |
| `G14_benford_cascade.py` | Python | Benford cascade analysis | §4.3 |
| `G15_mellin_exact_cascade.py` | Python | Mellin transform exact cascade | §3.2 |

## Requirements

Python >= 3.8, NumPy, SciPy, SymPy, mpmath. C compiler (gcc/clang) with optional OpenMP.
