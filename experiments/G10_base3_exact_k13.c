/*
 * G10_base3_exact_k13.c — Exact enumeration of c₁(K) for base 3, K=11..14
 *
 * Extends G02 with optimizations for K=13+:
 *   - Only computes carry at position M-1 (not full chain → saves memory)
 *   - Uses __int128 for products up to 3^28
 *   - Dynamic scheduling with larger chunks to reduce OpenMP overhead
 *
 * Estimated runtimes (8 cores, Apple M2):
 *   K=11:  ~3.1×10¹⁰  pairs → ~17 min
 *   K=12:  ~2.8×10¹¹  pairs → ~2.5 hours
 *   K=13:  ~2.5×10¹²  pairs → ~22 hours
 *   K=14:  ~2.3×10¹³  pairs → ~8 days (probably impractical)
 *
 * Compile (macOS with Clang):
 *   clang -O3 -Xpreprocessor -fopenmp -lomp -o G10_base3_exact_k13 G10_base3_exact_k13.c -lm
 *
 * Compile (Linux with GCC):
 *   gcc -O3 -fopenmp -o G10_base3_exact_k13 G10_base3_exact_k13.c -lm
 *
 * Run:
 *   ./G10_base3_exact_k13         # runs K=11..13
 *   ./G10_base3_exact_k13 13      # runs only K=13
 *   ./G10_base3_exact_k13 13 14   # runs K=13 and K=14
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
#define MAX_K 16
#define MAX_POS (2 * MAX_K + 5)

typedef unsigned __int128 uint128_t;

static void run_K(int K) {
    int n_free = K - 1;

    /* 3^(K-1) values per number */
    long long n_per_num = 1;
    for (int i = 0; i < n_free; i++) n_per_num *= BASE;

    /* Total pairs = n_per_num^2. Use 128-bit for large K. */
    uint128_t n_pairs_128 = (uint128_t)n_per_num * n_per_num;
    double n_pairs_f = (double)n_per_num * (double)n_per_num;

    long long min_val = 1;
    for (int i = 0; i < n_free; i++) min_val *= BASE;
    long long max_val = min_val * BASE - 1;

    int D_conv = 2 * K - 1;

    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif

    printf("K=%d: %.3e pairs, %d thread(s)\n", K, n_pairs_f, n_threads);
    printf("  x range: [%lld, %lld] (%lld values)\n", min_val, max_val, max_val - min_val + 1);
    fflush(stdout);

    double t_start;
#ifdef _OPENMP
    t_start = omp_get_wtime();
#else
    t_start = (double)clock() / CLOCKS_PER_SEC;
#endif

    long long total_n_ulc = 0;
    long long total_sum_cm1 = 0;

#pragma omp parallel reduction(+:total_n_ulc, total_sum_cm1)
    {
        int local_conv[MAX_POS];
        int local_carries[MAX_POS];
        int dx[MAX_K], dy[MAX_K];

#pragma omp for schedule(dynamic, 256)
        for (long long x = min_val; x <= max_val; x++) {
            {
                long long xx = x;
                for (int i = 0; i < K; i++) {
                    dx[i] = (int)(xx % BASE);
                    xx /= BASE;
                }
            }

            long long local_ulc = 0, local_sum = 0;

            for (long long y = min_val; y <= max_val; y++) {
                {
                    long long yy = y;
                    for (int i = 0; i < K; i++) {
                        dy[i] = (int)(yy % BASE);
                        yy /= BASE;
                    }
                }

                int max_pos = D_conv + 4;
                for (int p = 0; p <= max_pos; p++) local_conv[p] = 0;

                for (int i = 0; i < K; i++)
                    for (int jj = 0; jj < K; jj++)
                        local_conv[i + jj] += dx[i] * dy[jj];

                for (int p = 0; p <= max_pos; p++) local_carries[p] = 0;

                for (int pos = 0; pos < D_conv; pos++) {
                    int total = local_conv[pos] + local_carries[pos];
                    local_carries[pos + 1] = total / BASE;
                }
                for (int pos = D_conv; pos <= max_pos - 1; pos++) {
                    if (local_carries[pos] == 0) break;
                    local_carries[pos + 1] = local_carries[pos] / BASE;
                }

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

#ifdef _OPENMP
            if (omp_get_thread_num() == 0) {
                long long my_x_count = x - min_val + 1;
                long long total_x = max_val - min_val + 1;
                int pct = (int)(my_x_count * 100 * n_threads / total_x);
                if (pct > 100) pct = 100;
                static int last_pct = -1;
                if (pct != last_pct && pct % 5 == 0) {
                    last_pct = pct;
                    double now = omp_get_wtime();
                    double elapsed = now - t_start;
                    double eta = (pct > 0) ? elapsed * (100.0 / pct - 1.0) : 0;
                    printf("  [~%d%%] n_ulc=%lld, elapsed=%.0fs (%.1fmin), ETA=%.0fs (%.1fmin)\n",
                           pct, total_n_ulc, elapsed, elapsed/60, eta, eta/60);
                    fflush(stdout);
                }
            }
#else
            {
                long long my_x_count = x - min_val + 1;
                long long total_x = max_val - min_val + 1;
                int pct = (int)(my_x_count * 100 / total_x);
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
    }

    double t_end;
#ifdef _OPENMP
    t_end = omp_get_wtime();
#else
    t_end = (double)clock() / CLOCKS_PER_SEC;
#endif
    double elapsed = t_end - t_start;

    double c1 = (double)total_sum_cm1 / total_n_ulc - 1.0;

    printf("\n  K=%d RESULT:\n", K);
    printf("  n_pairs   = %.3e\n", n_pairs_f);
    printf("  n_ulc     = %lld\n", total_n_ulc);
    printf("  sum_cm1   = %lld\n", total_sum_cm1);
    printf("  c1(K)     = %.15f\n", c1);
    printf("  Exact rational: c1 = (%lld - %lld) / %lld = %lld / %lld\n",
           total_sum_cm1, total_n_ulc,
           total_n_ulc,
           total_sum_cm1 - total_n_ulc, total_n_ulc);
    printf("  time: %.1f sec (%.1f min, %.1f hours)\n\n", elapsed, elapsed / 60.0, elapsed / 3600.0);
    fflush(stdout);
}

int main(int argc, char **argv) {
    printf("G10: EXACT BASE-3 c₁(K) ENUMERATION — EXTENDED TO K=13+\n");
    printf("=========================================================\n\n");

#ifdef _OPENMP
    printf("OpenMP enabled: %d threads available\n\n", omp_get_max_threads());
#else
    printf("OpenMP NOT available — running single-threaded\n");
    printf("For multi-core: clang -O3 -Xpreprocessor -fopenmp -lomp ...\n\n");
#endif

    printf("Known exact values:\n");
    printf("  K=2:  c₁ =  0.500000000000000  (23, 22)\n");
    printf("  K=3:  c₁ =  0.481481481481481  (340, 261)\n");
    printf("  K=4:  c₁ =  0.531851851851852  (3861, 2680)\n");
    printf("  K=5:  c₁ =  0.546700960219478  (38593, 25359)\n");
    printf("  K=6:  c₁ =  0.567093851656498  (364335, 233264)\n");
    printf("  K=7:  c₁ =  0.574612783496652  (3348196, 2116057)\n");
    printf("  K=8:  c₁ =  0.582804025395297  (30402412, 19101834)\n");
    printf("  K=9:  c₁ =  0.586844629946830  (274620332, 172093303)\n");
    printf("  K=10: c₁ =  0.590154017513966  (2475182404, 1549405546)\n");
    printf("  (MC:  c₁(∞) ≈ 0.59870 ± 0.00002)\n");
    printf("  Candidate: ln(3) - 1/2 = 0.598612...\n\n");

    int K_start = 11;
    int K_end = 13;

    if (argc > 1) {
        K_start = atoi(argv[1]);
        K_end = K_start;
    }
    if (argc > 2) {
        K_end = atoi(argv[2]);
    }

    printf("Running K=%d to K=%d\n\n", K_start, K_end);

    for (int K = K_start; K <= K_end; K++) {
        run_K(K);
    }

    printf("=== DONE ===\n");
    printf("Feed (sum_cm1, n_ulc) values to G11_enhanced_pslq.py\n");
    printf("c₁(K) = (sum_cm1 - n_ulc) / n_ulc\n");

    return 0;
}
