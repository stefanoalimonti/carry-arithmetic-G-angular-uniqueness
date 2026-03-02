/*
 * f96_base3_exact.c — Exact enumeration of c₁(K) for base 3
 *
 * For K-digit base-3 numbers: digits d_{K-1}...d_0
 *   d_{K-1} ∈ {1, 2}, d_i ∈ {0, 1, 2} for i < K-1
 * Total configs per number: 2 * 3^{K-1}
 * Total pairs: (2·3^{K-1})^2
 *
 * K   pairs          feasibility
 * 2   16             instant
 * 3   144            instant
 * 4   1296           instant
 * 5   11664          <1s
 * 6   104976         <1s
 * 7   944784         ~1s
 * 8   8503056        ~10s
 * 9   76527504       ~90s
 * 10  688747536      ~15min
 *
 * Compile: cc -O3 -o f96_base3_exact f96_base3_exact.c -lm
 */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

#define BASE 3
#define MAX_K 12
#define MAX_POS (2 * MAX_K + 5)

int main(void) {
    printf("f96: EXACT BASE-3 c₁(K) ENUMERATION\n");
    printf("====================================\n\n");

    double prev_c1 = 0;
    double prev_delta = 0;
    int has_prev = 0, has_prev_delta = 0;

    for (int K = 2; K <= 10; K++) {
        int n_free = K - 1;
        long long n_per_num = 2;
        for (int i = 0; i < n_free; i++) n_per_num *= BASE;
        long long n_pairs = n_per_num * n_per_num;

        if (n_pairs > 2000000000LL) {
            printf("K=%d: %lld pairs — skipping (too large)\n\n", K, n_pairs);
            break;
        }

        clock_t t0 = clock();

        long long n_ulc = 0;
        long long sum_cm1 = 0;

        int D_conv = 2 * K - 1;

        long long min_val = 1;
        for (int i = 0; i < n_free; i++) min_val *= BASE;
        long long max_val = min_val * BASE - 1;

        for (long long x = min_val; x <= max_val; x++) {
            int dx[MAX_K];
            {
                long long xx = x;
                for (int i = 0; i < K; i++) {
                    dx[i] = (int)(xx % BASE);
                    xx /= BASE;
                }
            }

            for (long long y = min_val; y <= max_val; y++) {
                int dy[MAX_K];
                {
                    long long yy = y;
                    for (int i = 0; i < K; i++) {
                        dy[i] = (int)(yy % BASE);
                        yy /= BASE;
                    }
                }

                int conv[MAX_POS] = {0};
                for (int i = 0; i < K; i++)
                    for (int j = 0; j < K; j++)
                        conv[i + j] += dx[i] * dy[j];

                int carries[MAX_POS] = {0};
                for (int pos = 0; pos < D_conv; pos++) {
                    int total = conv[pos] + carries[pos];
                    carries[pos + 1] = total / BASE;
                }
                for (int pos = D_conv; pos < MAX_POS - 1; pos++) {
                    if (carries[pos] == 0) break;
                    carries[pos + 1] = carries[pos] / BASE;
                }

                int M_top = 0;
                for (int pos = MAX_POS - 2; pos >= 1; pos--) {
                    if (carries[pos] > 0) { M_top = pos; break; }
                }
                if (M_top <= 0) continue;

                int c_M = carries[M_top];
                int cm1 = carries[M_top - 1];

                /* D-odd: product has exactly 2K-1 digits in base 3
                 * This means c_{2K-2} > 0 and c_{2K-1} = 0
                 * Equivalently, M_top = 2K-2 (for D-odd with M = 2K-1)
                 * But more generally: c_M > 0 and c_{M+1} = 0.
                 * For the trace anomaly: c₁ = E[c_{M-1} | valid] - 1
                 * where "valid" = c_M > 0 (generalization of ULC to base b).
                 */
                if (c_M > 0 && carries[M_top + 1] == 0) {
                    n_ulc++;
                    sum_cm1 += cm1;
                }
            }
        }

        double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;

        double c1 = (double)sum_cm1 / n_ulc - 1.0;

        printf("K=%d:\n", K);
        printf("  n_pairs   = %lld\n", n_pairs);
        printf("  n_ulc     = %lld\n", n_ulc);
        printf("  sum_cm1   = %lld\n", sum_cm1);
        printf("  c1(K)     = %.15f\n", c1);
        printf("  sum-n_ulc = %lld\n", sum_cm1 - n_ulc);

        if (has_prev) {
            double delta = c1 - prev_c1;
            printf("  Δ         = %+.10e\n", delta);
            if (has_prev_delta && fabs(prev_delta) > 1e-18) {
                double rho = fabs(delta / prev_delta);
                printf("  ρ         = %.8f  (target: 1/3 = %.8f)\n", rho, 1.0/3.0);
            }
            prev_delta = delta;
            has_prev_delta = 1;
        }
        printf("  time: %.2f sec\n\n", elapsed);

        prev_c1 = c1;
        has_prev = 1;
    }

    printf("=== EXACT RATIONAL VALUES ===\n");
    printf("(Run f96c_base3_pslq.py with these sum_cm1 / n_ulc values)\n");

    return 0;
}
