#include "RegressionDataScheme.h"

int RegressionDataScheme::checkYsubXtimesBeta() {
    double* xb;
    double* ptr;
    double tmp;
    long double diffs = 0.0;
    long double diffs2 = 0.0;

    for (int l = 0; l < K; l++) {
        xb = &xTimesBeta[INDEX(0, l, memory_N)];
        ptr = &beta[INDEX(0, l, memory_P)];

        diffs = 0.0;
        for (int i = 0; i < N; i++) {
            tmp = y[INDEX(i, l, memory_N)] - offset[l];

            for (int j = 0; j < P; j++)
                tmp -= ptr[j] * x[INDEX(i, j, memory_N)];

            diffs += fabs(tmp - xb[i]);
        }

        diffs2 += diffs;
        diffs /= N;
    }

    if (diffs2 > 1e-10)
        return FALSE;
    else
        return TRUE;
}

int RegressionDataScheme::checkXtimesBeta() {
    double* xb;
    double* ptr;
    double tmp;
    long double diffs = 0.0;
    long double diffs2 = 0.0;

    for (int l = 0; l < K; l++) {
        xb = &xTimesBeta[INDEX(0, l, memory_N)];
        ptr = &beta[INDEX(0, l, memory_P)];

        diffs = 0.0;
        for (int i = 0; i < N; i++) {
            tmp = offset[l];

            for (int j = 0; j < P; j++)
                tmp += ptr[j] * x[INDEX(i, j, memory_N)];

            diffs += fabs(tmp - xb[i]);
        }

        diffs2 += diffs;
        diffs /= N;
    }

    if (diffs2 > 1e-10)
        return FALSE;
    else
        return TRUE;
}
