#include "zeroSum.h"

void zeroSum::refreshApproximation(uint32_t fold, uint32_t l, uint32_t _updateCost) {
    uint32_t ii = INDEX_TENSOR_COL(l, fold, memory_N, K);
    double* yOrgl = &yOrg[INDEX_COL(l, memory_N)];
    double* yl = &y[ii];
    double* xb = &xTimesBeta[ii];
    double* wl = &w[ii];

    double* wOrgf = &wOrg[INDEX_COL(fold, memory_N)];
    double p;

    if (type == binomial || type == multinomial) {
        for (uint32_t i = 0; i < N; i++) {
            if (type == binomial) {
                if (xb[i] >= 0.0) {
                    p = 1.0 / (exp(-xb[i]) + 1.0);
                } else {
                    p = exp(xb[i]);
                    p = p / (p + 1.0);
                }

            } else {
                p = 0.0;
                for (uint32_t k = 0; k < K; k++)
                    p += exp(xTimesBeta[INDEX_TENSOR(i, k, fold, memory_N, K)]);

                p = exp(xTimesBeta[INDEX_TENSOR(i, l, fold, memory_N, K)]) / p;
            }

            wl[i] = p * (1.0 - p);

            if (wl[i] < DBL_EPSILON * 100) {
                yl[i] = 0.0;
                wl[i] = 0.0;
            } else {
                yl[i] = xb[i] + (yOrgl[i] - p) / wl[i];
                wl[i] *= wOrgf[i];
            }
        }
    } else {
        double* tmpArray = &tmp_array1[INDEX_COL(fold, memory_N)];
        for (uint32_t i = 0; i < N; ++i)
            wl[i] = wOrgf[i] * exp(xb[i]);

        tmpArray[0] = arraySumAvx(wl, N);
        for (uint32_t i = 1; i < N; ++i) {
            tmpArray[i] = tmpArray[i - 1] - wl[i - 1];
        }

        double tmp1, tmp_wl, tmp_yl;
        for (uint32_t i = 0; i < N; ++i) {
            tmp_wl = 0.0;
            tmp_yl = 0.0;
            if (wOrgf[i] != 0.0) {
                for (uint32_t k = 0; k <= i; k++) {
                    if (d[INDEX(k, fold, memory_N)] != 0.0) {
                        tmp1 =
                            d[INDEX(k, fold, memory_N)] * wl[i] / tmpArray[k];

                        tmp_wl += tmp1 * (tmpArray[k] - wl[i]) / tmpArray[k];
                        tmp_yl += tmp1;
                    }
                }

                tmp_yl = xb[i] + (wOrgf[i] * status[i] - tmp_yl) / tmp_wl;
            }
            wl[i] = tmp_wl;
            yl[i] = tmp_yl;
        }
    }

    approxFailed[fold] = false;
    uint32_t lowW = 0;
    for (uint32_t i = 0; i < N; ++i) {
        if (wl[i] < 1e-6)
            lowW++;
        if (std::isnan(wl[i])) {
            approxFailed[fold] = true;
            break;
        }
    }
    if ((double)lowW / (double)N > 0.7) {
        approxFailed[fold] = true;
    }

#ifdef DEBUG
    if (approxFailed[fold])
        PRINT("approximation failed! lowW: %f\n", (double)lowW / (double)N);

#endif

    if (_updateCost)
        updateCost(fold, l);
}
