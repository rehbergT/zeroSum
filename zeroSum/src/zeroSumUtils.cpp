#include "zeroSum.h"

void zeroSum::meanCentering() {
    // get the weights of the full-sample-fit
    double* wf = &wOrg[INDEX_COL(nFold, memory_N)];
    double sumWeights = 1.0 / arraySumAvx(wf, memory_N);
    dgemv_(&BLAS_T, &memory_N, &memory_P, &sumWeights, x, &memory_N, wf,
           &BLAS_I_ONE, &BLAS_D_ZERO, featureMean, &BLAS_I_ONE);

    for (uint32_t j = 0; j < P; j++) {
        double* xptr = &x[INDEX_COL(j, memory_N)];
        for (uint32_t i = 0; i < N; i++) {
            xptr[i] -= featureMean[j];
        }
    }
}

void zeroSum::standardizeResponse() {
    for (uint32_t f = 0; f < nFold1; f++) {
        double* wf = &wOrg[INDEX_COL(f, memory_N)];

        // calculate the sum of all weights and calculate the inverse
        double sumWeights = 1.0 / arraySumAvx(wf, memory_N);

        // calculate the weighted mean of y
        double yMean =
            ddot_(&memory_N, wf, &BLAS_I_ONE, y, &BLAS_I_ONE) * sumWeights;

        // calculate the weighted standard deviation y
        ySD[f] = 0.0;
        for (uint32_t i = 0; i < N; i++) {
            ySD[f] += wf[i] * pow(y[i] - yMean, 2);
        }
        ySD[f] = sqrt(ySD[f]);

        double* vf = &v[INDEX_COL(f, memory_P)];
        for (uint32_t j = 0; j < P; j++)
            vf[j] /= ySD[f];
    }
}

void zeroSum::standardizeData() {
    for (uint32_t f = 0; f < nFold1; f++) {
        double* wf = &wOrg[INDEX_COL(f, memory_N)];
        double* vf = &v[INDEX_COL(f, memory_P)];

        // calculate the sum of all weights and calculate the inverse
        double sumWeights = 1.0 / arraySumAvx(wf, memory_N);

        // calculates the weighted mean of each feature and stores it in
        // beta (beta is only used as intermediate memory)
        dgemv_(&BLAS_T, &memory_N, &memory_P, &sumWeights, x, &memory_N, wf,
               &BLAS_I_ONE, &BLAS_D_ZERO, beta, &BLAS_I_ONE);

        for (uint32_t j = 0; j < P; j++) {
            double* xptr = &x[INDEX_COL(j, memory_N)];
            double tmp = 0.0;
            for (uint32_t i = 0; i < N; i++) {
                tmp += wf[i] * pow(xptr[i] - beta[j], 2);
            }
            vf[j] *= sqrt(tmp * sumWeights);
        }
    }

    memset(beta, 0.0, memory_P * sizeof(double));
}

void zeroSum::activeSetRemoveZeros(uint32_t fold) {
    double* betaf = &beta[INDEX_TENSOR_COL(0, fold, memory_P, K)];

    for (auto i = activeSet[fold].begin(); i != activeSet[fold].end();) {
        uint32_t k = *i;
        bool isZero = true;
        for (uint32_t l = 0; l < K; l++)
            if (betaf[INDEX(k, l, memory_P)] != 0.0)
                isZero = false;

        if (isZero) {
            i = activeSet[fold].erase(i);
        } else {
            ++i;
        }
    }
}

bool zeroSum::activeSetInsert(uint32_t fold, uint32_t k) {
    auto it = activeSet[fold].insert(k);
    bool inserted = it.second;
    return inserted;
}

bool zeroSum::activeSetContains(uint32_t fold, uint32_t k) {
    auto it = activeSet[fold].find(k);
    bool found = it != activeSet[fold].end();
    return found;
}

uint32_t zeroSum::activeSetGetElement(uint32_t fold, uint32_t k) {
    uint32_t ii = 0;
    for (auto i = activeSet[fold].begin(); i != activeSet[fold].end(); i++) {
        if (ii == k) {
            return *i;
        }
        ii++;
    }
    return -1;
}

void zeroSum::adjustWeights() {
    for (uint32_t f = 0; f < nFold; f++) {
        uint32_t foldSize = 0;

        for (uint32_t i = 0; i < N; i++) {
            if ((foldid[i] - 1) == f || foldid[i] == 0)
                foldSize++;
        }

        double scaler1 = (double)N / (double)(N - foldSize);
        double scaler2 = (double)N / (double)foldSize;

        /** we have to adjust the weights of the samples: 1. rescaling of the
         * contribution to the loglikelihood by replace 1/N with 1/(N-foldsize)
         * by multiplying with scaler1. 2. the loglikelihood should be evaluated
         * on all samples of the fold -> replace 1/N with 1/foldsize by
         * multiplying with scaler2. 3. If the sample is part of the fold which
         * should be dropped then set the weight to zero.  */

        for (uint32_t n = 0; n < N; n++) {
            bool sampleIsInLeftOutFold = (foldid[n] - 1) == f;

            if (sampleIsInLeftOutFold) {
                for (uint32_t k = 0; k < K; k++)
                    w[INDEX_TENSOR(n, k, f, memory_N, K)] = 0.0;
                wOrg[INDEX(n, f, memory_N)] = 0.0;
                wCV[INDEX(n, f, memory_N)] *= scaler2;
            } else {
                for (uint32_t k = 0; k < K; k++)
                    w[INDEX_TENSOR(n, k, f, memory_N, K)] *= scaler1;
                wOrg[INDEX(n, f, memory_N)] *= scaler1;
                wCV[INDEX(n, f, memory_N)] *= 0.0;
            }
        }
    }
}

void zeroSum::calcCoxRegressionD() {
    memset(d, 0.0, memory_N * nFold1 * sizeof(double));

    for (uint32_t f = 0; f < nFold1; f++) {
        uint32_t i = 0;
        while (i < N) {
            uint32_t ii = INDEX(i, f, memory_N);
            if (status[i] == 0 || wOrg[ii] == 0.0) {
                i++;
                continue;
            }

            d[ii] = wOrg[ii];
            uint32_t k;
            for (k = i + 1; k < N && yOrg[k - 1] == yOrg[k] &&
                            status[k - 1] == 1 && status[k] == 1;
                 k++) {
                if (status[k] == 1)
                    d[ii] += wOrg[INDEX(k, f, memory_N)];
            }
            i = k;
        }
    }
}

void zeroSum::predict() {
    memset(xTimesBeta, 0.0, memory_N * K * nFold1 * sizeof(double));

    for (uint32_t f = 0; f < nFold; f++) {
        double* xb = &xTimesBeta[INDEX_TENSOR_COL(0, f, memory_N, K)];
        double* betaf = &beta[INDEX_TENSOR_COL(0, f, memory_P, K)];

        if (type == gaussian || type == binomial) {
            a_add_scalar_b(xb, intercept[f], memory_N);
            for (uint32_t j = 0; j < P; j++)
                daxpy_(&memory_N, &betaf[j], &x[INDEX_COL(j, memory_N)],
                       &BLAS_I_ONE, xb, &BLAS_I_ONE);

            if (type == binomial) {
                for (uint32_t i = 0; i < N; i++)
                    xb[i] = 1.0 / (1.0 + exp(-xb[i]));
            }
        } else if (type == multinomial) {
            double* xbTmp;
            double* ptr;
            for (uint32_t l = 0; l < K; ++l) {
                ptr = &betaf[INDEX_COL(l, memory_P)];

                xbTmp = &xb[INDEX_COL(l, memory_N)];

                a_add_scalar_b(xbTmp, intercept[INDEX(l, f, K)], memory_N);

                for (uint32_t j = 0; j < P; ++j)
                    daxpy_(&memory_N, &ptr[j], &x[INDEX_COL(j, memory_N)],
                           &BLAS_I_ONE, xbTmp, &BLAS_I_ONE);
            }

            for (uint32_t i = 0; i < N; i++) {
                double tmp = 0.0;

                for (uint32_t l = 0; l < K; ++l)
                    tmp += exp(xb[INDEX(i, l, memory_N)]);

                for (uint32_t l = 0; l < K; ++l)
                    xb[INDEX(i, l, memory_N)] =
                        exp(xb[INDEX(i, l, memory_N)]) / tmp;
            }
        } else {
            for (uint32_t j = 0; j < P; j++)
                daxpy_(&memory_N, &betaf[j], &x[INDEX_COL(j, memory_N)],
                       &BLAS_I_ONE, xb, &BLAS_I_ONE);
        }
    }
}

double zeroSum::arraySumAvx(double* a, uint32_t n) {
    double sum;

#if !defined(__APPLE__) || !defined(__arm64__)
    if (avxType == avx2) {
        arraySumKernelAVX2(a, n, &sum);
    } else if (avxType == avx512) {
        arraySumKernelAVX512(a, n, &sum);
    } else {
#endif
        sum = 0.0;
        for (uint32_t i = 0; i < n; ++i)
            sum += a[i];
#if !defined(__APPLE__) || !defined(__arm64__)
    }
#endif

    return sum;
}

double zeroSum::arraySum(double* a, uint32_t n) {
    double sum = 0.0;
    for (uint32_t i = 0; i < n; ++i)
        sum += a[i];
    return sum;
}

double zeroSum::weightedAbsSum(double* a, double* b, uint32_t n) {
    double sum = 0;
    for (uint32_t i = 0; i < n; ++i)
        sum += b[i] * fabs(a[i]);

    return sum;
}

double zeroSum::weightedSquareSum(double* a, double* b, uint32_t n) {
    double sum;

#if !defined(__APPLE__) || !defined(__arm64__)
    if (avxType == avx2) {
        weightedSquareSumKernelAVX2(a, b, n, &sum);
    } else if (avxType == avx512) {
        weightedSquareSumKernelAVX512(a, b, n, &sum);
    } else {
#endif
        sum = 0.0;
        for (uint32_t i = 0; i < n; ++i)
            sum += a[i] * a[i] * b[i];
#if !defined(__APPLE__) || !defined(__arm64__)
    }
#endif
    return sum;
}

double zeroSum::weightedResidualSquareSum(double* a,
                                          double* b,
                                          double* c,
                                          uint32_t n) {
    double sum;

#if !defined(__APPLE__) || !defined(__arm64__)
    if (avxType == avx2) {
        weightedResidualSquareSumKernelAVX2(a, b, c, n, &sum);
    } else if (avxType == avx512) {
        weightedResidualSquareSumKernelAVX512(a, b, c, n, &sum);
    } else {
#endif
        sum = 0.0;
        for (uint32_t i = 0; i < n; ++i)
            sum += a[i] * pow(b[i] - c[i], 2.0);
#if !defined(__APPLE__) || !defined(__arm64__)
    }
#endif
    return sum;
}

double zeroSum::squareWeightedSum(double* a, double* b, uint32_t n) {
    double sum;

#if !defined(__APPLE__) || !defined(__arm64__)
    if (avxType == avx2) {
        squareWeightedSumKernelAVX2(a, b, n, &sum);
    } else if (avxType == avx512) {
        squareWeightedSumKernelAVX512(a, b, n, &sum);
    } else {
#endif
        sum = 0.0;
        for (uint32_t i = 0; i < n; i++) {
            double tmp = a[i] * b[i];
            sum += pow(tmp, 2);
        }
#if !defined(__APPLE__) || !defined(__arm64__)
    }
#endif

    return sum;
}

void zeroSum::a_sub_b(double* a, double* b, double* c, uint32_t n) {
#if !defined(__APPLE__) || !defined(__arm64__)
    if (avxType == avx2) {
        a_sub_bKernelAVX2(a, b, c, N);
    } else if (avxType == avx512) {
        a_sub_bKernelAVX512(a, b, c, N);
    } else {
#endif
        for (uint32_t i = 0; i < n; ++i)
            c[i] = a[i] - b[i];
#if !defined(__APPLE__) || !defined(__arm64__)
    }
#endif
}

void zeroSum::a_add_scalar_b(double* a, double b, uint32_t n) {
#if !defined(__APPLE__) || !defined(__arm64__)
    if (avxType == avx2) {
        a_add_scalar_bKernelAVX2(a, &b, n);
    } else if (avxType == avx512) {
        a_add_scalar_bKernelAVX2(a, &b, n);
    } else {
#endif
        for (uint32_t i = 0; i < n; ++i)
            a[i] += b;
#if !defined(__APPLE__) || !defined(__arm64__)
    }
#endif
}
