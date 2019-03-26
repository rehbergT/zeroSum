#include "zeroSum.h"

void zeroSum::costFunctionAllFolds() {
    for (uint32_t f = 0; f < nFold1; f++)
        costFunction(f);
}

double zeroSum::calcLogLikelihood(uint32_t fold, double* xb) {
    double ll = 0.0;

    double* wOrgf = &wOrg[INDEX_COL(fold, memory_N)];

    if (type == gaussian) {
        ll = -weightedResidualSquareSum(wOrgf, yOrg, xb, memory_N) * 0.5;
    } else if (type == binomial) {
        double tmp1 = 0.0;
        for (uint32_t i = 0; i < N; ++i)
            tmp1 = fma(wOrgf[i], fma(yOrg[i], xb[i], -log1p(exp(xb[i]))), tmp1);

        ll = tmp1;
    } else if (type == multinomial) {
        double tmp1 = 0.0;
        double tmp2 = 0.0;
        double tmp3 = 0.0;
        double tmp4 = 0.0;
        double a = *std::max_element(xb, xb + memory_N * K);

        for (uint32_t i = 0; i < N; ++i) {
            tmp1 = 0.0;
            tmp2 = 0.0;
            for (uint32_t l = 0; l < K; ++l) {
                uint32_t m = INDEX(i, l, memory_N);
                tmp3 = xb[m];
                tmp1 = fma(yOrg[m], tmp3, tmp1);
                tmp2 += exp(tmp3 - a);
            }
            tmp4 = fma(wOrgf[i], (tmp1 - log(tmp2) - a), tmp4);
        }
        ll = tmp4;
    } else {
        double* tmp_arrayf = &tmp_array1[INDEX_COL(fold, memory_N)];
        double* dFold = &d[INDEX_COL(fold, memory_N)];

        double a = *std::max_element(xb, xb + N);
        for (uint32_t i = 0; i < N; ++i) {
            tmp_arrayf[i] = wOrgf[i] * exp(xb[i] - a);

            if (status[i] != 0.0)
                ll = fma(wOrgf[i], xb[i], ll);
        }

        long double tmp1 = arraySumAvx(tmp_arrayf, memory_N);
        long double tmp2 = (log(tmp1) + a) * dFold[0];

        for (uint32_t i = 1; i < N; ++i) {
            tmp1 -= tmp_arrayf[i - 1];
            if (tmp1 < COX_MIN_PRECISION)
                tmp1 = arraySum(&tmp_arrayf[i], N - i);
            if (dFold[i] == 0.0)
                continue;
            tmp2 = fma((log(tmp1) + a), dFold[i], tmp2);
        }

        ll -= tmp2;
    }
    return ll;
}

void zeroSum::costFunction(uint32_t f) {
    double* betaf = &beta[INDEX_TENSOR_COL(0, f, memory_P, K)];
    double* xb = &xTimesBeta[INDEX_TENSOR_COL(0, f, memory_N, K)];

    // calculate the weighted lasso and ridge component, however only if
    // necessary alpha=1 -> lasso only, alpha=0 -> ridge only
    ridge[f] = 0.0;
    lasso[f] = 0.0;
    for (uint32_t l = 0; l < K; l++) {
        double* betafl = &betaf[INDEX_COL(l, memory_P)];
        if (alpha != 1.0)
            ridge[f] +=
                squareWeightedSum(betafl, &v[INDEX_COL(f, memory_P)], P);

        if (alpha != 0.0)
            lasso[f] += weightedAbsSum(betafl, &v[INDEX_COL(f, memory_P)], P);
    }

    // we need to calculate the linear model \beta_0 + x %*% \beta

    double* tmp;  // tmp double pointer
    for (uint32_t l = 0, k = 0; l < K; l++) {
        tmp = &intercept[INDEX(l, f,
                               K)];  // get the intercept of class l and fold f

        // init xb with the intercept
        for (uint32_t i = 0; i < memory_N; i++, k++)
            xb[k] = *tmp;
    }

    // calc the matrix times matrix multiplication
    dgemm_(&BLAS_NO, &BLAS_NO, &memory_N, &K, &memory_P, &BLAS_D_ONE, x,
           &memory_N, betaf, &memory_P, &BLAS_D_ONE, xb, &memory_N);

    loglikelihood[f] = calcLogLikelihood(f, xb);

    cost[f] = -loglikelihood[f] +
              lambda[f] * ((1.0 - alpha) * ridge[f] / 2.0 + alpha * lasso[f]);

    if (useFusion) {
        double* fusionPartialSumsf =
            &fusionPartialSums[INDEX_TENSOR_COL(0, f, memory_nc, K)];
        double* fusionSumsf = &fusionSums[INDEX_COL(f, K)];

        memset(fusionPartialSumsf, 0.0, memory_nc * K * sizeof(double));

        double* ps;
        double* ptr;
        struct fusionKernel* currEl;

        for (uint32_t l = 0; l < K; l++) {
            ps = &fusionPartialSumsf[INDEX_COL(l, memory_nc)];
            ptr = &betaf[INDEX_COL(l, memory_P)];

            for (uint32_t j = 0; j < P; j++) {
                currEl = fusionKernel[j];
                while (currEl != NULL) {
                    ps[currEl->i] += currEl->value * ptr[j];
                    currEl = currEl->next;
                }
            }
            // calc sum of absolute values
            fusionSumsf[l] = dasum_(&nc, ps, &BLAS_I_ONE);
        }

        fusion[f] = arraySum(fusionSumsf, K);
        cost[f] += gamma * fusion[f];
    }

    return;
}

void zeroSum::updateCost(uint32_t fold, uint32_t l) {
    cost[fold] += loglikelihood[fold];

    if (useApprox) {
        uint32_t ii = INDEX_TENSOR_COL(l, fold, memory_N, K);
        loglikelihood[fold] =
            -0.5 * weightedResidualSquareSum(&w[ii], &y[ii], &xTimesBeta[ii],
                                             memory_N);
    } else {
        uint32_t ii = INDEX_TENSOR_COL(0, fold, memory_N, K);
        double* xb = &xTimesBeta[ii];
        loglikelihood[fold] = calcLogLikelihood(fold, xb);
    }
    cost[fold] -= loglikelihood[fold];
}
