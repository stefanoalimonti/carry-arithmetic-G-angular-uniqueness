/*
 * G02_base3_exact_k12.c — Exact enumeration of c₁(K) for base 3, K=2..12
 *
 * OpenMP parallelized over x-loop for multi-core speedup.
 *
 * K=10:  ~1.55×10⁹  pairs →  ~15 min (1 core), ~2 min (8 cores)
 * K=11:  ~1.39×10¹⁰ pairs →  ~2.2 hours (1 core), ~17 min (8 cores)
 * K=12:  ~1.26×10¹¹ pairs →  ~20 hours (1 core), ~2.5 hours (8 cores)
 *
 * Compile (with OpenMP):
 *   clang -O3 -Xpreprocessor -fopenmp -lomp -o G02_base3_exact_k12 G02_base3_exact_k12.c -lm
 *
 * If OpenMP not available (still works, single-threaded):
 *   cc -O3 -o G02_base3_exact_k12 G02_base3_exact_k12.c -lm
 *
 * Run:
 *   ./G02_base3_exact_k12
 *
 * To run ONLY K=11 and K=12 (skip previously computed K≤10):
 *   ./G02_base3_exact_k12 11
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define BASE 3
#define MAX_K 14
#define MAX_POS (2 * MAX_K + 5)

static void run_K(int K) {
    int n_free = K - 1;
    long long n_per_num = 2;
    for (int i = 0; i < n_free; i++) n_per_num *= BASE;
    long long n_pairs = n_per_num * n_per_num;

    long long min_val = 1;
    for (int i = 0; i < n_free; i++) min_val *= BASE;
    long long max_val = min_val * BASE - 1;

    int D_conv = 2 * K - 1;

    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif

    printf("K=%d: %lld pairs (%.2e), %d thread(s)\n",
           K, n_pairs, (double)n_pairs, n_threads);
    fflush(stdout);

    double t_start;
#ifdef _OPENMP
    t_start = omp_get_wtime();
#else
    t_start = (double)clock() / CLOCKS_PER_SEC;
#endif

    long long total_n_ulc = 0;
    long long total_sum_cm1 = 0;
    long long progress_done = 0;
    long long progress_total = max_val - min_val + 1;

#pragma omp parallel reduction(+:total_n_ulc, total_sum_cm1)
    {
        int local_conv[MAX_POS];
        int local_carries[MAX_POS];
        int dx[MAX_K], dy[MAX_K];

#pragma omp for schedule(dynamic, 64)
        for (long long x = min_val; x <= max_val; x++) {
            /* Extract digits of x */
            {
                long long xx = x;
                for (int i = 0; i < K; i++) {
                    dx[i] = (int)(xx % BASE);
                    xx /= BASE;
                }
            }

            long long local_ulc = 0, local_sum = 0;

            for (long long y = min_val; y <= max_val; y++) {
                /* Extract digits of y */
                {
                    long long yy = y;
                    for (int i = 0; i < K; i++) {
                        dy[i] = (int)(yy % BASE);
                        yy /= BASE;
                    }
                }

                /* Convolution */
                int max_pos = D_conv + 4;
                for (int p = 0; p <= max_pos; p++) local_conv[p] = 0;

                for (int i = 0; i < K; i++)
                    for (int j = 0; j < K; j++)
                        local_conv[i + j] += dx[i] * dy[j];

                /* Carry propagation */
                for (int p = 0; p <= max_pos; p++) local_carries[p] = 0;

                for (int pos = 0; pos < D_conv; pos++) {
                    int total = local_conv[pos] + local_carries[pos];
                    local_carries[pos + 1] = total / BASE;
                }
                for (int pos = D_conv; pos <= max_pos - 1; pos++) {
                    if (local_carries[pos] == 0) break;
                    local_carries[pos + 1] = local_carries[pos] / BASE;
                }

                /* Find M_top */
                int M_top = 0;
                for (int pos = max_pos - 1; pos >= 1; pos--) {
                    if (local_carries[pos] > 0) { M_top = pos; break; }
                }
                if (M_top <= 0) continue;

                int c_M = local_carries[M_top];
                int cm1 = local_carries[M_top - 1];

                if (c_M > 0 && local_carries[M_top + 1] == 0) {
                    local_ulc++;
                    local_sum += cm1;
                }
            }

            total_n_ulc += local_ulc;
            total_sum_cm1 += local_sum;

            /* Progress reporting (from thread 0 only) */
#ifdef _OPENMP
            if (omp_get_thread_num() == 0) {
                long long done;
#pragma omp atomic read
                done = total_n_ulc;
                /* Report every ~5% */
                static int last_pct = -1;
                long long my_x_count = x - min_val + 1;
                int pct = (int)(my_x_count * 100 * n_threads / progress_total);
                if (pct > 100) pct = 100;
                if (pct != last_pct && pct % 5 == 0) {
                    last_pct = pct;
                    double now = omp_get_wtime();
                    double elapsed = now - t_start;
                    double eta = (pct > 0) ? elapsed * (100.0 / pct - 1.0) : 0;
                    printf("  [~%d%%] n_ulc=%lld, elapsed=%.0fs, ETA=%.0fs\n",
                           pct, done, elapsed, eta);
                    fflush(stdout);
                }
            }
#else
            {
                long long my_x_count = x - min_val + 1;
                int pct = (int)(my_x_count * 100 / progress_total);
                static int last_pct_s = -1;
                if (pct != last_pct_s && pct % 10 == 0) {
                    last_pct_s = pct;
                    double now = (double)clock() / CLOCKS_PER_SEC;
                    double elapsed = now - t_start;
                    double eta = (pct > 0) ? elapsed * (100.0 / pct - 1.0) : 0;
                    printf("  [%d%%] n_ulc=%lld, elapsed=%.0fs, ETA=%.0fs\n",
                           pct, total_n_ulc, elapsed, eta);
                    fflush(stdout);
                }
            }
#endif
        }
    } /* end parallel */

    double t_end;
#ifdef _OPENMP
    t_end = omp_get_wtime();
#else
    t_end = (double)clock() / CLOCKS_PER_SEC;
#endif
    double elapsed = t_end - t_start;

    double c1 = (double)total_sum_cm1 / total_n_ulc - 1.0;

    printf("\n  K=%d RESULT:\n", K);
    printf("  n_pairs   = %lld\n", n_pairs);
    printf("  n_ulc     = %lld\n", total_n_ulc);
    printf("  sum_cm1   = %lld\n", total_sum_cm1);
    printf("  c1(K)     = %.15f\n", c1);
    printf("  sum-n_ulc = %lld  (for rational: c1 = %lld/%lld)\n",
           total_sum_cm1 - total_n_ulc,
           total_sum_cm1 - total_n_ulc, total_n_ulc);
    printf("  time: %.1f sec (%.1f min)\n\n", elapsed, elapsed / 60.0);
    fflush(stdout);
}

int main(int argc, char **argv) {
    printf("G02: EXACT BASE-3 c₁(K) ENUMERATION (OpenMP)\n");
    printf("=============================================\n\n");

#ifdef _OPENMP
    printf("OpenMP enabled: %d threads available\n\n", omp_get_max_threads());
#else
    printf("OpenMP NOT available — running single-threaded\n");
    printf("For multi-core: clang -O3 -Xpreprocessor -fopenmp -lomp ...\n\n");
#endif

    int K_start = 2;
    if (argc > 1) {
        K_start = atoi(argv[1]);
        printf("Starting from K=%d (skipping lower values)\n\n", K_start);
    }

    /* Known exact values from f96 (K=2..10) for reference */
    printf("Previously computed exact values (f96/E50):\n");
    printf("  K=2:  c₁ =  0.500000000000000\n");
    printf("  K=3:  c₁ =  0.481481481481481\n");
    printf("  K=4:  c₁ =  0.531851851851852\n");
    printf("  K=5:  c₁ =  0.546700960219478\n");
    printf("  K=6:  c₁ =  0.567093851656498\n");
    printf("  K=7:  c₁ =  0.574612783496652\n");
    printf("  K=8:  c₁ =  0.582804025395297\n");
    printf("  K=9:  c₁ =  0.586844629946830\n");
    printf("  K=10: c₁ =  0.590154017513966\n");
    printf("  (MC:  c₁(∞) ≈ 0.59870 ± 0.00002)\n\n");

    double prev_c1 = 0;
    double prev_delta = 0;
    int has_prev = 0, has_prev_delta = 0;

    /* If starting from K>2, set prev values from K_start-1 */
    if (K_start == 11) {
        prev_c1 = 0.590154017513966; /* K=10 */
        has_prev = 1;
        prev_delta = 0.590154017513966 - 0.586844629946830; /* K10-K9 */
        has_prev_delta = 1;
    } else if (K_start == 12) {
        /* Will be set after K=11 runs */
    }

    for (int K = K_start; K <= 12; K++) {
        run_K(K);

        /* Read back for delta/rho computation */
        /* We recompute c1 from stdout-parsed values; simpler: just track */
    }

    printf("=== DONE ===\n");
    printf("Feed sum_cm1 and n_ulc values to PSLQ script.\n");
    printf("c₁(K) = (sum_cm1 - n_ulc) / n_ulc\n");

    return 0;
}
