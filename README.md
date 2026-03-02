# carry-arithmetic-G-angular-uniqueness

**The Angular Uniqueness of Base 2 in Positional Multiplication**

*Author: Stefano Alimonti* · [ORCID 0009-0009-1183-1698](https://orcid.org/0009-0009-1183-1698)

## Main Result

Base 2 is the unique integer base for which the D-parity boundary in the multiplication unit square is a straight line ($\alpha + \beta = \pi/4$). This geometric rigidity explains why $\pi$ enters the trace anomaly $c_1 = \pi/18$ exclusively in base 2. For bases $b > 2$, the boundary is curved and the constant $c_1(b)$ involves only logarithms.

## Status

Complete. 15 experiments (Python + C).

## Repository Structure

```
paper/angular_uniqueness.md           The paper
experiments/
  G01_base3_highprec_mc.c             High-precision Monte Carlo (C)
  G02_base3_exact_k12.c               Exact enumeration K=2-12 (C)
  G03_pslq_extended.py                PSLQ analysis
  G04_transfer_eigenvalues.py         Transfer operator eigenvalues
  G05_base3_verification.py           Base-3 eigenvalue verification
  G06_base3_extended.py               Base-3 extended analysis
  G07_base3_exact.c                   Base-3 exact enumeration (C)
  G08_amplitude_multiterm.py          Multi-term amplitude fit
  G09_base3_pslq.py                   Base-3 PSLQ
  G10_base3_exact_k13.c              Extended enumeration K=13 (C)
  G11_enhanced_pslq.py                Enhanced PSLQ with Richardson
  G12_enhanced_pslq_k12.py            Enhanced PSLQ at K=12
  G13_padic_multiplication.py         p-adic multiplication analysis
  G14_benford_cascade.py              Benford cascade analysis
  G15_mellin_exact_cascade.py         Mellin transform exact cascade
```

## Reproduction

```bash
# Python experiments
pip install numpy scipy sympy mpmath
python experiments/G03_pslq_extended.py
# ... through G15

# C experiments (require gcc/clang, optionally OpenMP)
cc -O3 -fopenmp -o g01 experiments/G01_base3_highprec_mc.c -lm
cc -O3 -fopenmp -o g02 experiments/G02_base3_exact_k12.c -lm
```

## Dependencies

- Python >= 3.8, NumPy, SciPy, SymPy, mpmath
- C compiler (gcc/clang) with optional OpenMP support

## Companion Papers

| Label | Title | Repository |
|-------|-------|------------|
| [A] | Spectral Theory of Carries | [`carry-arithmetic-A-spectral-theory`](https://github.com/stefanoalimonti/carry-arithmetic-A-spectral-theory) |
| [B] | Carry Polynomials and the Euler Product | [`carry-arithmetic-B-zeta-approximation`](https://github.com/stefanoalimonti/carry-arithmetic-B-zeta-approximation) |
| [E] | The Trace Anomaly of Binary Multiplication | [`carry-arithmetic-E-trace-anomaly`](https://github.com/stefanoalimonti/carry-arithmetic-E-trace-anomaly) |
| [F] | Exact Covariance Structure | [`carry-arithmetic-F-covariance-structure`](https://github.com/stefanoalimonti/carry-arithmetic-F-covariance-structure) |
| [P1] | Pi from Pure Arithmetic | [`carry-arithmetic-P1-pi-spectral`](https://github.com/stefanoalimonti/carry-arithmetic-P1-pi-spectral) |
| [P2] | The Sector Ratio in Binary Multiplication | [`carry-arithmetic-P2-sector-ratio`](https://github.com/stefanoalimonti/carry-arithmetic-P2-sector-ratio) |

### Citation

```bibtex
@article{alimonti2026angular_uniqueness,
  author  = {Alimonti, Stefano},
  title   = {The Angular Uniqueness of Base 2 in Positional Multiplication},
  year    = {2026},
  note    = {Preprint},
  url     = {https://github.com/stefanoalimonti/carry-arithmetic-G-angular-uniqueness}
}
```

## License

Paper: CC BY 4.0. Code: MIT License.
