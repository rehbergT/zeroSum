#include "RegressionDataScheme.h"

void RegressionDataScheme::predict() {
    memset(xTimesBeta, 0.0, memory_N * K * sizeof(double));

    if (type <= 8) {
        a_add_scalar_b(xTimesBeta, offset[0], xTimesBeta, N);
        for (int j = 0; j < P; j++)
            add_a_add_scalar_b(&x[INDEX(0, j, memory_N)], beta[j], xTimesBeta,
                               N);

        if (type > 4) {
            for (int i = 0; i < N; i++)
                xTimesBeta[i] = 1.0 / (1.0 + exp(-xTimesBeta[i]));
        }
    } else if (type <= 12) {
        double* xbTmp;
        double* ptr;
        for (int l = 0; l < K; ++l) {
            ptr = &beta[INDEX(0, l, memory_P)];

            xbTmp = &xTimesBeta[INDEX(0, l, memory_N)];

            a_add_scalar_b(xbTmp, offset[l], xbTmp, N);

            for (int j = 0; j < P; ++j)
                add_a_add_scalar_b(&x[INDEX(0, j, memory_N)], ptr[j], xbTmp, N);
        }

        for (int i = 0; i < N; i++) {
            double tmp = 0.0;

            for (int l = 0; l < K; ++l)
                tmp += exp(xTimesBeta[INDEX(i, l, memory_N)]);

            for (int l = 0; l < K; ++l)
                xTimesBeta[INDEX(i, l, memory_N)] =
                    exp(xTimesBeta[INDEX(i, l, memory_N)]) / tmp;
        }
    } else {
        for (int j = 0; j < P; j++)
            add_a_add_scalar_b(&x[INDEX(0, j, memory_N)], beta[j], xTimesBeta,
                               N);
    }
}
