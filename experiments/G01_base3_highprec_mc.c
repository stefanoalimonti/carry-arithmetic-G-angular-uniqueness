/*
 * G01_base3_highprec_mc.c — High-precision Monte Carlo for c₁(base 3)
 *
 * Computes c₁(3) with 10^9 samples at d=40 for ~6-digit precision.
 * Also runs d=20, d=60 to check d-independence.
 *
 * Compile:  cc -O3 -o G01_base3_highprec_mc G01_base3_highprec_mc.c -lm
 * Run:      ./G01_base3_highprec_mc
 *
 * Expected runtime: ~30-60 minutes for 10^9 samples at d=40 on M-series Mac.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

#define BASE 3
#define MAX_D   70
#define MAX_POS 160

static uint64_t rng_s[4];
static uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}
static uint64_t next_rand(void) {
    const uint64_t result = rotl(rng_s[1] * 5, 7) * 9;
    const uint64_t t = rng_s[1] << 17;
    rng_s[2] ^= rng_s[0]; rng_s[3] ^= rng_s[1];
    rng_s[1] ^= rng_s[2]; rng_s[0] ^= rng_s[3];
    rng_s[2] ^= t; rng_s[3] = rotl(rng_s[3], 45);
    return result;
}
static void seed_rng(uint64_t seed) {
    rng_s[0] = seed;
    rng_s[1] = seed ^ 0x9E3779B97F4A7C15ULL;
    rng_s[2] = seed ^ 0x6C62272E07BB0142ULL;
    rng_s[3] = seed ^ 0xBF58476D1CE4E5B9ULL;
    for (int i = 0; i < 20; i++) next_rand();
}

static int rand_digit_base3(void) {
    return (int)(next_rand() % 3);
}

static void mc_run(int d, long long n_samples) {
    int D_conv = 2 * d - 1;

    printf("BASE 3, d=%d, %lld samples (%.0eM):\n",
           d, n_samples, (double)n_samples / 1e6);
    fflush(stdout);

    long long n_valid = 0;
    double sum_cm1 = 0.0, sum_cm1_sq = 0.0;
    long long n_even = 0, n_odd = 0;
    double sum_even = 0.0, sum_odd = 0.0;

    long long cm_counts[BASE];
    double cm_sums[BASE], cm_sq[BASE];
    memset(cm_counts, 0, sizeof(cm_counts));
    memset(cm_sums, 0, sizeof(cm_sums));
    memset(cm_sq, 0, sizeof(cm_sq));

    int dp[MAX_D], dq[MAX_D];
    int conv[MAX_POS], carries[MAX_POS];

    clock_t t0 = clock();
    long long report_interval = n_samples / 10;
    if (report_interval < 1) report_interval = 1;

    for (long long iter = 0; iter < n_samples; iter++) {
        if (iter > 0 && iter % report_interval == 0) {
            double c1_now = sum_cm1 / n_valid - 1.0;
            double se_now = sqrt((sum_cm1_sq / n_valid -
                            (sum_cm1 / n_valid) * (sum_cm1 / n_valid)) /
                            n_valid);
            double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;
            printf("  [%3lld%%] c₁ = %.8f ± %.2e  (%.0fs)\n",
                   iter * 100 / n_samples, c1_now, se_now, elapsed);
            fflush(stdout);
        }

        dp[0] = 1 + (int)(next_rand() % 2);
        dq[0] = 1 + (int)(next_rand() % 2);
        for (int i = 1; i < d; i++) {
            dp[i] = rand_digit_base3();
            dq[i] = rand_digit_base3();
        }

        memset(conv, 0, sizeof(int) * (D_conv + 5));
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++) {
                int pos = (d - 1 - i) + (d - 1 - j);
                conv[pos] += dp[i] * dq[j];
            }

        memset(carries, 0, sizeof(int) * MAX_POS);
        int last_nonzero_digit = 0;
        for (int pos = 0; pos <= D_conv; pos++) {
            int total = conv[pos] + carries[pos];
            int digit = total % BASE;
            carries[pos + 1] += total / BASE;
            if (digit != 0 || pos == 0) last_nonzero_digit = pos;
        }
        for (int pos = D_conv + 1; pos < MAX_POS - 1; pos++) {
            if (carries[pos] == 0) break;
            int total = carries[pos];
            int digit = total % BASE;
            carries[pos + 1] += total / BASE;
            if (digit != 0) last_nonzero_digit = pos;
        }

        int M_top = 0;
        for (int pos = MAX_POS - 2; pos >= 1; pos--) {
            if (carries[pos] > 0) { M_top = pos; break; }
        }
        if (M_top <= 0) continue;

        int c_M = carries[M_top];
        int cm1 = carries[M_top - 1];

        if (c_M <= 0 || c_M >= BASE) continue;

        n_valid++;
        sum_cm1 += cm1;
        sum_cm1_sq += (double)cm1 * cm1;

        cm_counts[c_M]++;
        cm_sums[c_M] += cm1;
        cm_sq[c_M] += (double)cm1 * cm1;

        int D = last_nonzero_digit + 1;
        if (D % 2 == 0) { n_even++; sum_even += cm1; }
        else             { n_odd++;  sum_odd  += cm1; }
    }

    clock_t t1 = clock();
    double elapsed = (double)(t1 - t0) / CLOCKS_PER_SEC;

    double c1 = sum_cm1 / n_valid - 1.0;
    double var = sum_cm1_sq / n_valid - (sum_cm1 / n_valid) * (sum_cm1 / n_valid);
    double se = sqrt(var / n_valid);

    printf("\n  RESULT: c₁(3) = %.10f ± %.2e  (n_valid=%lld, %.1fs)\n",
           c1, se, n_valid, elapsed);
    printf("  Valid fraction: %.6f\n", (double)n_valid / n_samples);

    printf("\n  By c_M:\n");
    for (int c = 1; c < BASE; c++) {
        if (cm_counts[c] > 0) {
            double c1_c = cm_sums[c] / cm_counts[c] - 1.0;
            double var_c = cm_sq[c] / cm_counts[c] -
                           (cm_sums[c] / cm_counts[c]) * (cm_sums[c] / cm_counts[c]);
            double se_c = sqrt(var_c / cm_counts[c]);
            printf("    c_M=%d: c₁ = %.10f ± %.2e  (frac=%.4f)\n",
                   c, c1_c, se_c, (double)cm_counts[c] / n_valid);
        }
    }

    double p_e = (double)n_even / n_valid;
    double c1_e = (n_even > 0) ? sum_even / n_even - 1.0 : 0;
    double c1_o = (n_odd > 0) ? sum_odd / n_odd - 1.0 : 0;
    printf("\n  D-parity: P_even=%.6f, c₁^even=%.10f, c₁^odd=%.10f\n",
           p_e, c1_e, c1_o);
    printf("  Check: P_e·c₁^e + P_o·c₁^o = %.10f (should = %.10f)\n",
           p_e * c1_e + (1 - p_e) * c1_o, c1);

    printf("\n  Reference constants:\n");
    printf("    ln(3)/9     = %.15f\n", log(3.0) / 9.0);
    printf("    ln(3)^2/2   = %.15f\n", log(3.0) * log(3.0) / 2.0);
    printf("    (3-1)ln3/9  = %.15f\n", 2.0 * log(3.0) / 9.0);
    printf("    π/18        = %.15f  (base-2 value, should NOT match)\n",
           M_PI / 18.0);
    printf("    Δ from π/18 = %+.6e (%+.1fσ)\n",
           c1 - M_PI / 18.0, (c1 - M_PI / 18.0) / se);
    printf("\n");
    fflush(stdout);
}

int main(void) {
    printf("G01: HIGH-PRECISION c₁(base 3) MONTE CARLO\n");
    printf("============================================\n\n");

    seed_rng(0x314159265358979BULL);

    printf("=== Phase 1: d=20 warmup (10^8 samples) ===\n\n");
    mc_run(20, 100000000LL);

    printf("=== Phase 2: d=40 main run (10^9 samples) ===\n\n");
    mc_run(40, 1000000000LL);

    printf("=== Phase 3: d=60 verification (10^8 samples) ===\n\n");
    mc_run(60, 100000000LL);

    printf("=== DONE ===\n");
    printf("Feed the c₁(3) value to E52_base3_pslq.py for PSLQ analysis.\n");

    return 0;
}
