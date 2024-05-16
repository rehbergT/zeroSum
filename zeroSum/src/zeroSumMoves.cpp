#include "zeroSum.h"

void zeroSum::interceptMove(uint32_t fold, uint32_t l) {
    if (approxFailed[fold])
        return;

    uint32_t ii = INDEX_TENSOR_COL(l, fold, memory_N, K);

    double* yf = &y[ii];
    double* xb = &xTimesBeta[ii];
    double* ww = &w[ii];
    double bk;

#if !defined(__APPLE__) || !defined(__arm64__)
    if (avxType == avx2) {
        interceptMoveKernelAVX2(yf, xb, ww, memory_N, &bk);
    } else if (avxType == avx512) {
        interceptMoveKernelAVX512(yf, xb, ww, memory_N, &bk);
    } else {
#endif
        bk = 0.0;
        for (uint32_t i = 0; i < memory_N; i++)
            bk += (yf[i] - xb[i]) * ww[i];
#if !defined(__APPLE__) || !defined(__arm64__)
    }
#endif

    double sumW = arraySumAvx(ww, memory_N);

    if (fabs(sumW) < 1000 * DBL_EPSILON)
        return;

    uint32_t ind = INDEX(l, fold, K);
    double oldbeta = intercept[ind];
    intercept[ind] = bk / sumW + oldbeta;

    double diff = intercept[ind] - oldbeta;

    a_add_scalar_b(xb, diff, memory_N);

    if (type != gaussian && !useApprox)
        refreshApproximation(fold, l);
}

uint32_t zeroSum::cdMove(uint32_t fold, uint32_t k, uint32_t l) {
    if (approxFailed[fold])
        return 0;

    uint32_t ii = INDEX_TENSOR_COL(l, fold, memory_P, K);

    double* betak = &beta[ii + k];
    double* xk = &x[INDEX_COL(k, memory_N)];

    ii = INDEX_TENSOR_COL(l, fold, memory_N, K);

    double* yf = &y[ii];
    double* xb = &xTimesBeta[ii];
    double* ww = &w[ii];

    double ak, bk;
#if !defined(__APPLE__) || !defined(__arm64__)
    if (avxType == avx2) {
        cvMoveKernelAVX2(yf, xb, ww, xk, betak, memory_N, &ak, &bk);
    } else if (avxType == avx512) {
        cvMoveKernelAVX512(yf, xb, ww, xk, betak, memory_N, &ak, &bk);
    } else {
#endif
        ak = 0.0;
        bk = 0.0;
        for (uint32_t i = 0; i < N; i++) {
            double tmp = ww[i] * xk[i];
            ak += tmp * xk[i];
            bk += tmp * (yf[i] - xb[i] + *betak * xk[i]);
        }
#if !defined(__APPLE__) || !defined(__arm64__)
    }
#endif

    double vk = v[INDEX(k, fold, memory_P)];
    ak += lambda[fold] * (1.0 - alpha) * pow(vk, 2);

    double tmp = lambda[fold] * alpha * vk;

    double bk1 = bk + tmp;
    double bk2 = bk - tmp;

    if (bk1 < 0.0) {
        tmp = bk1 / ak;
    } else if (bk2 > 0.0) {
        tmp = bk2 / ak;
    } else {
        tmp = 0.0;
    }

    double diff = tmp - *betak;
    *betak = tmp;

    if (fabs(diff) < BETA_CHANGE_PRECISION)
        return 0;
    else {
        daxpy_(&memory_N, &diff, xk, &BLAS_I_ONE, xb, &BLAS_I_ONE);
        if (useApprox)
            refreshApproximation(fold, l);

        if (useIntercept)
            interceptMove(fold, l);

        return 1;
    }
}

uint32_t zeroSum::cdMoveZS(uint32_t fold, uint32_t k, uint32_t s, uint32_t l) {
    if (approxFailed[fold])
        return 0;

    uint32_t ii = INDEX_TENSOR_COL(l, fold, memory_P, K);

    double* betak = &beta[ii + k];
    double* betas = &beta[ii + s];

    ii = INDEX_TENSOR_COL(l, fold, memory_N, K);
    double* yf = &y[ii];
    double* xb = &xTimesBeta[ii];
    double* ww = &w[ii];

    double* xk = &x[INDEX_COL(k, memory_N)];
    double* xs = &x[INDEX_COL(s, memory_N)];

    double ukus = u[k] / u[s];
    double l1 = lambda[fold] * (1.0 - alpha);

    ii = INDEX_COL(fold, memory_P);
    double vk = v[ii + k];
    double vs = v[ii + s];

    double ak = l1 * (pow(vk, 2) + pow(vs, 2) * pow(ukus, 2));
    double bk = l1 * pow(vs, 2) * ukus * (*betas + ukus * *betak);

#if !defined(__APPLE__) || !defined(__arm64__)
    if (avxType == avx2) {
        cvMoveZSKernelAVX2(yf, xb, ww, xk, xs, betak, &ukus, memory_N, &ak,
                           &bk);
    } else if (avxType == avx512) {
        cvMoveZSKernelAVX512(yf, xb, ww, xk, xs, betak, &ukus, memory_N, &ak,
                             &bk);
    } else {
#endif
        double tmp;
        for (uint32_t i = 0; i < N; i++) {
            tmp = xs[i] * ukus - xk[i];
            ak += ww[i] * pow(tmp, 2);
            bk -= ww[i] * tmp * (yf[i] - xb[i] - *betak * tmp);
        }
#if !defined(__APPLE__) || !defined(__arm64__)
    }
#endif

    double tmp3 = lambda[fold] * alpha / ak;

    vs *= ukus;

    double case14 = tmp3 * (vk - vs);
    double case23 = tmp3 * (vk + vs);

    bk /= ak;
    double bk1 = bk - case14;
    double bk4 = bk + case14;
    double bk2 = bk - case23;
    double bk3 = bk + case23;

    tmp3 = u[s] * *betas + u[k] * *betak;

    double bs1 = tmp3 - u[k] * bk1;
    double bs2 = tmp3 - u[k] * bk2;
    double bs3 = tmp3 - u[k] * bk3;
    double bs4 = tmp3 - u[k] * bk4;

    double diffk = *betak;
    double diffs = *betas;

    uint32_t defined = true;
    if (bk1 > 0 && bs1 > 0) {
        *betak = bk1;
        *betas = bs1 / u[s];
    } else if (bk2 > 0 && bs2 < 0) {
        *betak = bk2;
        *betas = bs2 / u[s];
    } else if (bk3 < 0 && bs3 > 0) {
        *betak = bk3;
        *betas = bs3 / u[s];
    } else if (bk4 < 0 && bs4 < 0) {
        *betak = bk4;
        *betas = bs4 / u[s];
    } else {
        defined = false;
    }

    if (defined == false)
        return 0;
    else {
        diffk -= *betak;
        diffs -= *betas;
#if !defined(__APPLE__) || !defined(__arm64__)
        if (avxType == avx2) {
            cvMoveZSKernel2AVX2(xb, &diffk, xk, &diffs, xs, memory_N);
        } else if (avxType == avx512) {
            cvMoveZSKernel2AVX512(xb, &diffk, xk, &diffs, xs, memory_N);
        } else {
#endif
            for (uint32_t i = 0; i < N; i++)
                xb[i] -= xk[i] * diffk + xs[i] * diffs;
#if !defined(__APPLE__) || !defined(__arm64__)
        }
#endif

        if (useApprox)
            refreshApproximation(fold, l);

        if (useIntercept)
            interceptMove(fold, l);

        return 1;
    }
}

void zeroSum::cdMove_parallel(uint32_t* improving, uint32_t steps) {
    std::mutex mutex;
    parallel.doParallelChunked(P, [&](size_t s) {
        for (uint64_t fold = 0; fold < (uint64_t)nFold1; fold++) {
            // cycle over all multinomial classes (if type !=
            // multinomial K=1)
            for (uint64_t l = 0; l < (uint64_t)K; l++) {
                // if the activeset search of the previous search has not
                // changed the active set then the optimus is reached and
                // improving is still zero and we can skip this fold
                if (improving[fold] == 0 && steps > 0)
                    continue;

                if (approxFailed[fold])
                    continue;

                uint32_t ii = INDEX_TENSOR_COL(l, fold, memory_P, K);

                double* betak = &beta[ii + s];
                double* xk = &x[INDEX_COL(s, memory_N)];

                ii = INDEX_TENSOR_COL(l, fold, memory_N, K);

                double* yf = &y[ii];
                double* xb = &xTimesBeta[ii];
                double* ww = &w[ii];

                double ak, bk;
#if !defined(__APPLE__) || !defined(__arm64__)
                if (avxType == avx2) {
                    cvMoveKernelAVX2(yf, xb, ww, xk, betak, memory_N, &ak, &bk);
                } else if (avxType == avx512) {
                    cvMoveKernelAVX512(yf, xb, ww, xk, betak, memory_N, &ak,
                                       &bk);
                } else {
#endif
                    ak = 0.0;
                    bk = 0.0;
                    for (uint32_t i = 0; i < N; i++) {
                        double tmp = ww[i] * xk[i];
                        ak += tmp * xk[i];
                        bk += tmp * (yf[i] - xb[i] + *betak * xk[i]);
                    }
#if !defined(__APPLE__) || !defined(__arm64__)
                }
#endif

                double vs = v[INDEX(s, fold, memory_P)];
                ak += lambda[fold] * (1.0 - alpha) * pow(vs, 2);

                double tmp = lambda[fold] * alpha * vs;
                double bk1 = bk + tmp;
                double bk2 = bk - tmp;

                if (bk1 < 0.0) {
                    tmp = bk1 / ak;
                } else if (bk2 > 0.0) {
                    tmp = bk2 / ak;
                } else {
                    tmp = 0.0;
                }

                double diff = tmp - *betak;

                if (fabs(diff) >= BETA_CHANGE_PRECISION) {
                    std::lock_guard<std::mutex> lock(mutex);
                    parallelActiveSet[fold].push_back(
                        {(uint32_t)l, (uint32_t)s, 0});
                }
            }
        }
    });
}

void zeroSum::cdMoveZS_parallel(uint32_t* improving, uint32_t steps) {
    uint64_t comb = (uint64_t)(P * (P - 1) / 2.0);

    // subtract y -> xTimesBeta now stores -y + \beta_0 + x^T \beta
    // has to be reverted at the end of this function!
    for (uint32_t fold = 0; fold < nFold1; fold++) {
        for (uint64_t l = 0; l < (uint64_t)K; l++) {
            uint32_t ii = INDEX_TENSOR_COL(l, fold, memory_N, K);
            daxpy_(&memory_N, &BLAS_D_MINUS_ONE, &y[ii], &BLAS_I_ONE,
                   &xTimesBeta[ii], &BLAS_I_ONE);
        }
    }

    std::mutex mutex;

    parallel.doParallelChunked(comb, [&](size_t c) {
        uint32_t k = P - 2 - (uint32_t)(sqrt(2 * (comb - c) - 1.75) - 0.5);
        uint32_t s = (uint32_t)(c + k + 1 - comb + (P - k) * ((P - k) - 1) / 2);

        for (uint64_t fold = 0; fold < (uint64_t)nFold1; fold++) {
            // cycle over all multinomial classes (if type !=
            // multinomial K=1)
            for (uint64_t l = 0; l < (uint64_t)K; l++) {
                // if the activeset search of the previous search
                // has not changed the active set then the optimus
                // is reached and improving is still zero and we can
                // skip this fold
                // if step>0 the loop with uint32_t s should
                // only go over the non active set coefficients
                // and the loop over k should go over the active
                // set
                // -> skip if s is in the active set or k is not
                // in the active set
                if (steps > 0 &&
                    (improving[fold] == 0 || !activeSetContains(fold, k) ||
                     !activeSetContains(fold, s)))
                    continue;

                // zero-sum weight u_k = 0 means that this
                // coefficient has to be updated with cdMove.
                // But this is handled in the convergence phase.
                // Normally just a few values in u are zero ->
                // just add it to active set
                if (u[k] == 0 || u[s] == 0 || approxFailed[fold])
                    continue;

                double ukus = u[k] / u[s];

                uint32_t ii = INDEX_TENSOR_COL(l, fold, memory_P, K);

                double* betak = &beta[ii + k];
                double* betas = &beta[ii + s];

                ii = INDEX_TENSOR_COL(l, fold, memory_N, K);
                double* xb = &xTimesBeta[ii];
                double* ww = &w[ii];

                double* xk = &x[INDEX_COL(k, memory_N)];
                double* xs = &x[INDEX_COL(s, memory_N)];

                double l1 = lambda[fold] * (1.0 - alpha);

                ii = INDEX_COL(fold, memory_P);
                double vk = v[ii + k];
                double vs = v[ii + s];

                double ak = l1 * (pow(vk, 2) + pow(vs, 2) * pow(ukus, 2));
                double bk = l1 * pow(vs, 2) * ukus * (*betas + ukus * *betak);

#if !defined(__APPLE__) || !defined(__arm64__)
                if (avxType == avx2) {
                    cvMoveZSParallelKernelAVX2(xb, ww, xk, xs, betak, &ukus,
                                               memory_N, &ak, &bk);
                } else if (avxType == avx512) {
                    cvMoveZSParallelKernelAVX512(xb, ww, xk, xs, betak, &ukus,
                                                 memory_N, &ak, &bk);
                } else {
#endif
                    double tmp;
                    for (uint32_t i = 0; i < N; i++) {
                        tmp = xs[i] * ukus - xk[i];
                        ak += ww[i] * pow(tmp, 2);
                        bk += ww[i] * tmp * (xb[i] + *betak * tmp);
                    }
#if !defined(__APPLE__) || !defined(__arm64__)
                }
#endif

                double tmp3 = lambda[fold] * alpha / ak;

                vs *= ukus;

                double case14 = tmp3 * (vk - vs);
                double case23 = tmp3 * (vk + vs);

                bk /= ak;
                double bk1 = bk - case14;
                double bk4 = bk + case14;
                double bk2 = bk - case23;
                double bk3 = bk + case23;

                tmp3 = u[s] * *betas + u[k] * *betak;

                double bs1 = tmp3 - u[k] * bk1;
                double bs2 = tmp3 - u[k] * bk2;
                double bs3 = tmp3 - u[k] * bk3;
                double bs4 = tmp3 - u[k] * bk4;

                double diffk = *betak;
                double diffs = *betas;

                uint32_t defined = true;
                if (bk1 > 0 && bs1 > 0) {
                    diffk -= bk1;
                    diffs -= bs1 / u[s];
                } else if (bk2 > 0 && bs2 < 0) {
                    diffk -= bk2;
                    diffs -= bs2 / u[s];
                } else if (bk3 < 0 && bs3 > 0) {
                    diffk -= bk3;
                    diffs -= bs3 / u[s];
                } else if (bk4 < 0 && bs4 < 0) {
                    diffk -= bk4;
                    diffs -= bs4 / u[s];
                } else {
                    defined = false;
                }

                if (!(defined == false ||
                      (fabs(diffk) < BETA_CHANGE_PRECISION &&
                       fabs(diffs) < BETA_CHANGE_PRECISION))) {
                    std::lock_guard<std::mutex> lock(mutex);
                    parallelActiveSet[fold].push_back({(uint32_t)l, s, k});
                }
            }
        }
    });

    // revert y subtraction -> xTimesBeta now stores again \beta_0 + x^T \beta
    for (uint32_t fold = 0; fold < nFold1; fold++) {
        for (uint64_t l = 0; l < (uint64_t)K; l++) {
            uint32_t ii = INDEX_TENSOR_COL(l, fold, memory_N, K);
            daxpy_(&memory_N, &BLAS_D_ONE, &y[ii], &BLAS_I_ONE, &xTimesBeta[ii],
                   &BLAS_I_ONE);
        }
    }
}

uint32_t zeroSum::cdMoveZSRotated(uint32_t fold,
                                  uint32_t n,
                                  uint32_t m,
                                  uint32_t s,
                                  uint32_t l,
                                  double theta) {
    if (approxFailed[fold])
        return 0;

    uint32_t ii = INDEX_TENSOR_COL(l, fold, memory_P, K);

    double* betan = &beta[ii + n];
    double* betam = &beta[ii + m];
    double* betas = &beta[ii + s];

    ii = INDEX_TENSOR_COL(l, fold, memory_N, K);
    double* yf = &y[ii];
    double* xb = &xTimesBeta[ii];
    double* ww = &w[ii];

    double* xn = &x[INDEX_COL(n, memory_N)];
    double* xm = &x[INDEX_COL(m, memory_N)];
    double* xs = &x[INDEX_COL(s, memory_N)];

    ii = INDEX_COL(fold, memory_N);
    double* tmp_array1f = &tmp_array1[ii];

    double cosT = cos(theta);
    double sinT = sin(theta);

    double unum1 = (-u[n] * cosT + u[m] * sinT) / u[s];
    double unum2 = (u[n] * cosT - u[m] * sinT) / u[s];

    double l1 = lambda[fold] * alpha;
    double l2 = lambda[fold] - l1;

#if !defined(__APPLE__) || !defined(__arm64__)
    if (avxType == avx2) {
        cdMoveZSRotatedKernelAVX2(xm, xn, xs, &sinT, &cosT, &unum2, memory_N,
                                  tmp_array1f);

    } else if (avxType == avx512) {
        cdMoveZSRotatedKernelAVX512(xm, xn, xs, &sinT, &cosT, &unum2, memory_N,
                                    tmp_array1f);

    } else {
#endif
        for (uint32_t i = 0; i < memory_N; ++i)
            tmp_array1f[i] = xm[i] * sinT + xn[i] * (-cosT) + xs[i] * unum2;
#if !defined(__APPLE__) || !defined(__arm64__)
    }
#endif

    double vm = v[INDEX(m, fold, memory_P)];
    double vn = v[INDEX(n, fold, memory_P)];
    double vs = v[INDEX(s, fold, memory_P)];

    double a =
        weightedSquareSum(tmp_array1f, ww, N) +
        l2 * (pow(vn * cosT, 2) + pow(vm * sinT, 2) + pow(vs * unum1, 2));

    double b;
#if !defined(__APPLE__) || !defined(__arm64__)
    if (avxType == avx2) {
        cdMoveZSRotatedKernel2AVX2(yf, xb, ww, tmp_array1f, memory_N, &b);
    } else if (avxType == avx512) {
        cdMoveZSRotatedKernel2AVX512(yf, xb, ww, tmp_array1f, memory_N, &b);
    } else {
#endif
        b = 0.0;
        for (uint32_t i = 0; i < memory_N; ++i)
            b += ww[i] * (yf[i] - xb[i]) * tmp_array1f[i];
#if !defined(__APPLE__) || !defined(__arm64__)
    }
#endif

    b = -b - l2 * (pow(vn, 2) * *betan * cosT - pow(vm, 2) * *betam * sinT +
                   pow(vs, 2) * *betas * unum1);

    double t1 = vn * cosT - vm * sinT;
    double t2 = vn * cosT + vm * sinT;

    unum1 *= vs;
    unum2 *= vs;

    double c1 = t1 + unum1;
    double c2 = t1 + unum2;
    double c3 = t2 + unum1;
    double c4 = t2 + unum2;
    double c5 = -t2 + unum1;
    double c6 = -t2 + unum2;
    double c7 = -t1 + unum1;
    double c8 = -t1 + unum2;

    double tmp2 = u[n] * *betan + u[m] * *betam + u[s] * *betas;
    double tmp3;
    double diffn = *betan;
    double diffm = *betam;
    double diffs = *betas;

    double betanNew = NAN;
    double betamNew = NAN;
    double betasNew = NAN;

    if (std::isnan(betanNew)) {
        tmp3 = (b - l1 * c1) / a;

        betanNew = tmp3 * cosT + *betan;
        betamNew = (-tmp3) * sinT + *betam;
        betasNew = (tmp2 - u[n] * betanNew - u[m] * betamNew) / u[s];

        if (betanNew > 0.0 && betamNew > 0.0 && betasNew > 0.0) {
            *betan = betanNew;
            *betam = betamNew;
            *betas = betasNew;
        } else {
            betanNew = NAN;
        }
    }

    if (std::isnan(betanNew)) {
        tmp3 = (b - l1 * c2) / a;

        betanNew = tmp3 * cosT + *betan;
        betamNew = (-tmp3) * sinT + *betam;
        betasNew = (tmp2 - u[n] * betanNew - u[m] * betamNew) / u[s];

        if (betanNew > 0.0 && betamNew > 0.0 && betasNew < 0.0) {
            *betan = betanNew;
            *betam = betamNew;
            *betas = betasNew;
        } else {
            betanNew = NAN;
        }
    }

    if (std::isnan(betanNew)) {
        tmp3 = (b - l1 * c3) / a;

        betanNew = tmp3 * cosT + *betan;
        betamNew = (-tmp3) * sinT + *betam;
        betasNew = (tmp2 - u[n] * betanNew - u[m] * betamNew) / u[s];

        if (betanNew > 0.0 && betamNew < 0.0 && betasNew > 0.0) {
            *betan = betanNew;
            *betam = betamNew;
            *betas = betasNew;
        } else {
            betanNew = NAN;
        }
    }

    if (std::isnan(betanNew)) {
        tmp3 = (b - l1 * c4) / a;

        betanNew = tmp3 * cosT + *betan;
        betamNew = (-tmp3) * sinT + *betam;
        betasNew = (tmp2 - u[n] * betanNew - u[m] * betamNew) / u[s];

        if (betanNew > 0.0 && betamNew < 0.0 && betasNew < 0.0) {
            *betan = betanNew;
            *betam = betamNew;
            *betas = betasNew;
        } else {
            betanNew = NAN;
        }
    }

    if (std::isnan(betanNew)) {
        tmp3 = (b - l1 * c5) / a;

        betanNew = tmp3 * cosT + *betan;
        betamNew = (-tmp3) * sinT + *betam;
        betasNew = (tmp2 - u[n] * betanNew - u[m] * betamNew) / u[s];

        if (betanNew < 0.0 && betamNew > 0.0 && betasNew > 0.0) {
            *betan = betanNew;
            *betam = betamNew;
            *betas = betasNew;
        } else {
            betanNew = NAN;
        }
    }

    if (std::isnan(betanNew)) {
        tmp3 = (b - l1 * c6) / a;

        betanNew = tmp3 * cosT + *betan;
        betamNew = (-tmp3) * sinT + *betam;
        betasNew = (tmp2 - u[n] * betanNew - u[m] * betamNew) / u[s];

        if (betanNew < 0.0 && betamNew > 0.0 && betasNew < 0.0) {
            *betan = betanNew;
            *betam = betamNew;
            *betas = betasNew;
        } else {
            betanNew = NAN;
        }
    }

    if (std::isnan(betanNew)) {
        tmp3 = (b - l1 * c7) / a;

        betanNew = tmp3 * cosT + *betan;
        betamNew = (-tmp3) * sinT + *betam;
        betasNew = (tmp2 - u[n] * betanNew - u[m] * betamNew) / u[s];

        if (betanNew < 0.0 && betamNew < 0.0 && betasNew > 0.0) {
            *betan = betanNew;
            *betam = betamNew;
            *betas = betasNew;
        } else {
            betanNew = NAN;
        }
    }

    if (std::isnan(betanNew)) {
        tmp3 = (b - l1 * c8) / a;

        betanNew = tmp3 * cosT + *betan;
        betamNew = (-tmp3) * sinT + *betam;
        betasNew = (tmp2 - u[n] * betanNew - u[m] * betamNew) / u[s];

        if (betanNew < 0.0 && betamNew < 0.0 && betasNew < 0.0) {
            *betan = betanNew;
            *betam = betamNew;
            *betas = betasNew;
        } else {
            betanNew = NAN;
        }
    }

    if (std::isnan(betanNew))
        return 0;

    diffn -= *betan;
    diffm -= *betam;
    diffs -= *betas;

#if !defined(__APPLE__) || !defined(__arm64__)
    if (avxType == avx2) {
        cvMoveZSRotatedKernel3AVX2(xb, &diffn, xn, &diffm, xm, &diffs, xs,
                                   memory_N);
    } else if (avxType == avx512) {
        cvMoveZSRotatedKernel3AVX512(xb, &diffn, xn, &diffm, xm, &diffs, xs,
                                     memory_N);
    } else {
#endif
        for (uint32_t i = 0; i < memory_N; ++i)
            xb[i] -= xn[i] * diffn + xm[i] * diffm + xs[i] * diffs;
#if !defined(__APPLE__) || !defined(__arm64__)
    }
#endif

    if (fabs(diffn) < BETA_CHANGE_PRECISION)
        return 0;
    else {
        if (useApprox)
            refreshApproximation(fold, l);

        if (useIntercept)
            interceptMove(fold, l);

        return 1;
    }
}

void zeroSum::lsSaOffsetMove(uint32_t fold, uint32_t l) {
    // for updating the intercept one has to use the local approximation ->
    // refresh the approximation (only necessary in the useApprox=false
    // case, else its allready refreshed) -> update the intercept -> update the
    // loglikihood/cost
    if (type != gaussian && !useApprox) {
        refreshApproximation(fold, l);
    }
    interceptMove(fold, l);
    updateCost(fold, l);
}

uint32_t zeroSum::lsSaMove(uint32_t fold,
                           uint32_t k,
                           uint32_t s,
                           uint32_t l,
                           double delta_k,
                           double* rng,
                           double temperature) {
    if (u[s] == 0)
        return 0;

    uint32_t col_k = INDEX_COL(k, memory_N);

    uint32_t ii = INDEX_TENSOR_COL(0, fold, memory_N, K);
    double* xb = &xTimesBeta[ii];
    double* xbTmp = &tmp_array2[ii];
    memcpy(xbTmp, xb, sizeof(double) * memory_N * K);

    double tmp_loglikelihood = 0.0;
    double tmp_ridge = 0.0;
    double tmp_lasso = 0.0;
    double tmp_fusion = 0.0;

    double betak = 0.0;
    double betas = 0.0;
    double betakNew = 0.0;
    double betasNew = 0.0;
    double delta_s = 0.0;

    ii = INDEX_TENSOR_COL(l, fold, memory_P, K);
    betak = beta[ii + k];
    betakNew = betak + delta_k;

    double vk = v[INDEX(k, fold, memory_P)];
    double vs = v[INDEX(s, fold, memory_P)];

    if (useZeroSum && u[k] != 0) {
        delta_s = -delta_k * u[k] / u[s];
        betas = beta[ii + s];
        betasNew = betas + delta_s;

        uint32_t col_s = INDEX_COL(s, memory_N);
        double delta_k_tmp = delta_k;
        double delta_s_tmp = delta_s;

        daxpy_(&memory_N, &delta_k_tmp, &x[col_k], &BLAS_I_ONE,
               &xbTmp[INDEX_COL(l, memory_N)], &BLAS_I_ONE);
        daxpy_(&memory_N, &delta_s_tmp, &x[col_s], &BLAS_I_ONE,
               &xbTmp[INDEX_COL(l, memory_N)], &BLAS_I_ONE);

        tmp_lasso = lasso[fold] + vk * (fabs(betakNew) - fabs(betak)) +
                    vs * (fabs(betasNew) - fabs(betas));

        tmp_ridge = ridge[fold] +
                    vk * vk * (betakNew * betakNew - betak * betak) +
                    vs * vs * (betasNew * betasNew - betas * betas);
    } else {
        double delta_k_tmp = delta_k;

        // calculate xb = delta_k_tmp * x_k + xb
        daxpy_(&memory_N, &delta_k_tmp, &x[col_k], &BLAS_I_ONE,
               &xbTmp[INDEX_COL(l, memory_N)], &BLAS_I_ONE);

        tmp_lasso = lasso[fold] + vk * (fabs(betakNew) - fabs(betak));
        tmp_ridge =
            ridge[fold] + vk * vk * (betakNew * betakNew - betak * betak);
    }

    if (useApprox) {
        uint32_t ii = INDEX_TENSOR_COL(l, fold, memory_N, K);
        tmp_loglikelihood =
            -0.5 * weightedResidualSquareSum(&w[ii], &y[ii],
                                             &xbTmp[INDEX_COL(l, memory_N)],
                                             memory_N);
    } else {
        tmp_loglikelihood = calcLogLikelihood(fold, xbTmp);
    }
    double fusionSum = 0.0;
    if (useFusion) {
        double* sumTmp = &fusionPartialSumsTmp[INDEX_COL(fold, memory_nc)];

        memcpy(sumTmp,
               &fusionPartialSums[INDEX_TENSOR_COL(l, fold, memory_nc, K)],
               nc * sizeof(double));

        struct fusionKernel* currEl = fusionKernel[k];

        while (currEl != NULL) {
            sumTmp[currEl->i] += currEl->value * delta_k;
            currEl = currEl->next;
        }

        if (useZeroSum && u[k] != 0) {
            currEl = fusionKernel[s];
            while (currEl != NULL) {
                sumTmp[currEl->i] += currEl->value * delta_s;
                currEl = currEl->next;
            }
        }
        // calc sum of absolute values
        fusionSum = dasum_(&nc, sumTmp, &BLAS_I_ONE);
        tmp_fusion = fusion[fold] - fusionSums[INDEX(l, fold, K)] + fusionSum;
    }

    double tmp_cost =
        -tmp_loglikelihood +
        lambda[fold] * ((1.0 - alpha) * tmp_ridge / 2.0 + alpha * tmp_lasso) +
        gamma * tmp_fusion;

    double deltaE = tmp_cost - cost[fold];

    if (deltaE < 0.0 || (rng != NULL && *rng < exp(-deltaE / temperature))) {
        memcpy(xb, xbTmp, sizeof(double) * memory_N * K);

        ridge[fold] = tmp_ridge;
        lasso[fold] = tmp_lasso;
        loglikelihood[fold] = tmp_loglikelihood;
        cost[fold] = tmp_cost;

        if (useFusion) {
            fusion[fold] = tmp_fusion;
            memcpy(&fusionPartialSums[INDEX_TENSOR_COL(l, fold, memory_nc, K)],
                   &fusionPartialSumsTmp[INDEX_COL(fold, memory_nc)],
                   nc * sizeof(double));
            fusionSums[INDEX(l, fold, K)] = fusionSum;
        }

        beta[ii + k] = betakNew;
        if (std::isnan(betakNew))
            PRINT("ERROR2!!!\n");

        if (useZeroSum && u[k] != 0)
            beta[ii + s] = betasNew;

        if (useApprox)
            refreshApproximation(fold, l, true);

        return 1;
    } else {
        return 0;
    }
}
