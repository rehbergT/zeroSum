#include "zeroSum.h"

double zeroSum::penaltyCost(double* coefs, double t) {
    double cost = 0.0;
    double tmp;

    for (uint32_t l = 0; l < K; l++) {
        tmp = coefs[l] - t;
        cost += 0.5 * (1.0 - alpha) * tmp * tmp + alpha * fabs(tmp);
    }

    return cost;
}

int cmpfunc(const void* a, const void* b) {
    if (*(double*)a < *(double*)b)
        return -1;
    if (*(double*)a == *(double*)b)
        return 0;
    if (*(double*)a > *(double*)b)
        return 1;
    return 0;
}

double median(double* x, uint32_t N) {
    qsort(x, N, sizeof(double), cmpfunc);

    if (N % 2 == 0)
        return ((x[N / 2] + x[N / 2 - 1]) / 2.0);
    else
        return x[N / 2];
}

void zeroSum::optimizeParameterAmbiguity(uint32_t fold, uint32_t iterations) {
    double* betaf = &beta[INDEX_TENSOR_COL(0, fold, memory_P, K)];
    double* interceptf = &intercept[INDEX_COL(fold, K)];
    double* vf = &v[INDEX_COL(fold, memory_P)];
    double* tmp = &tmp_array1[INDEX_COL(fold, memory_N)];

    double v1, v2, cost1, cost2, between;

    v1 = arraySum(interceptf, K) / K;
    for (uint32_t l = 0; l < K; l++)
        interceptf[l] -= v1;

    for (uint32_t j = 0; j < P; j++) {
        for (uint32_t l = 0; l < K; l++)
            tmp[l] = betaf[INDEX(j, l, memory_P)];

        v1 = arraySum(tmp, K) / K;
        if (vf[j] == 0.0) {
            for (uint32_t l = 0; l < K; l++)
                betaf[INDEX(j, l, memory_P)] -= v1;
        } else {
            v2 = median(tmp, K);
            if (v1 == v2)
                continue;

            for (uint32_t it = 0; it < iterations; it++) {
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

            for (uint32_t l = 0; l < K; l++) {
                betaf[INDEX(j, l, memory_P)] -= v1;
            }
        }
    }

    cSum = arraySumAvx(betaf, memory_P);
}
