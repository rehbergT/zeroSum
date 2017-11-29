#include "RegressionDataScheme.h"

void RegressionDataScheme::optimizeParameterAmbiguity(int iterations) {
#ifdef AVX_VERSION
    double* tmp = (double*)aligned_alloc(ALIGNMENT, K * sizeof(double));
#else
    double* tmp = (double*)malloc(K * sizeof(double));
#endif

    double v1, v2, cost1, cost2, between;

    v1 = sum_a(offset, K) / K;
    for (int l = 0; l < K; l++)
        offset[l] -= v1;

    for (int j = 0; j < P; j++) {
        for (int l = 0; l < K; l++)
            tmp[l] = beta[INDEX(j, l, memory_P)];

        v1 = sum_a(tmp, K) / K;
        if (v[j] == 0.0) {
            for (int l = 0; l < K; l++)
                beta[INDEX(j, l, memory_P)] -= v1;
        } else {
            v2 = median(tmp, K);
            if (v1 == v2)
                continue;

            for (int it = 0; it < iterations; it++) {
                cost1 = penaltyCost(tmp, v1);
                between = (v1 + v2) / 2.0;
                cost2 = penaltyCost(tmp, between);

                if (cost1 < cost2)
                    v2 = between;
                else
                    v1 = between;
            }

            cost1 = penaltyCost(tmp, v1);
            cost2 = penaltyCost(tmp, v2);

            if (cost2 < cost1)
                v1 = v2;

            for (int l = 0; l < K; l++) {
                beta[INDEX(j, l, memory_P)] -= v1;
            }
        }
    }

    cSum = sum_a(beta, P);

    free(tmp);
}
