#include "RegressionDataScheme.h"

void RegressionDataScheme::costFunction(void) {
    if (alpha != 1.0)
        ridge = sum_square_a_times_b(beta, v, P);

    if (alpha != 0.0)
        lasso = sum_abs_a_times_b(beta, v, P);

    double* xb = xTimesBeta;

    if (type <= 4) {
        memcpy(xb, y, memory_N * sizeof(double));
        a_add_scalar_b(xb, -offset[0], xb, N);

        for (int j = 0; j < P; ++j)
            add_a_add_scalar_b(&x[INDEX(0, j, memory_N)], -beta[j], xb, N);

        loglikelihood = -sum_square_a_times_b(xb, wOrg, N) * 0.5;
    } else if (type <= 8) {
        memset(xb, 0.0, memory_N * sizeof(double));
        memcpy(y, yOrg, memory_N * sizeof(double));
        a_add_scalar_b(xb, offset[0], xb, N);

        for (int j = 0; j < P; ++j)
            add_a_add_scalar_b(&x[INDEX(0, j, memory_N)], beta[j], xb, N);

        double tmp1 = 0.0;

        for (int i = 0; i < N; ++i)
            tmp1 += wOrg[i] * (yOrg[i] * xb[i] - log1p(exp(xb[i])));

        if (useApprox)
            a_sub_b(yOrg, xb, xb, N);

        loglikelihood = tmp1;
    } else if (type <= 12) {
        double* ptr;
        for (int l = 1; l < K; ++l) {
            ptr = &beta[INDEX(0, l, memory_P)];

            if (alpha != 1.0)
                ridge += sum_square_a_times_b(ptr, v, P);

            if (alpha != 0.0)
                lasso += sum_abs_a_times_b(ptr, v, P);
        }

        memset(xb, 0.0, memory_N * K * sizeof(double));
        memcpy(y, yOrg, memory_N * K * sizeof(double));

        double* xbTmp;
        for (int l = 0; l < K; ++l) {
            ptr = &beta[INDEX(0, l, memory_P)];

            xbTmp = &xb[INDEX(0, l, memory_N)];

            a_add_scalar_b(xbTmp, offset[l], xbTmp, N);

            for (int j = 0; j < P; ++j)
                add_a_add_scalar_b(&x[INDEX(0, j, memory_N)], ptr[j], xbTmp, N);
        }

        double tmp1 = 0.0;
        double tmp2 = 0.0;
        double tmp3 = 0.0;
        double tmp4 = 0.0;

        double a = *std::max_element(xb, xb + N);
        for (int l = 0; l < K; ++l) {
            xbTmp = &xb[INDEX(0, l, memory_N)];
            double a1 = *std::max_element(xbTmp, xbTmp + N);
            if (a1 > a)
                a = a1;
        }

        for (int i = 0; i < N; ++i) {
            tmp1 = 0.0;
            tmp2 = 0.0;
            for (int l = 0; l < K; ++l) {
                tmp3 = xb[INDEX(i, l, memory_N)];
                tmp1 += yOrg[INDEX(i, l, memory_N)] * tmp3;
                tmp2 += exp(tmp3 - a);
            }
            tmp4 += wOrg[i] * (tmp1 - log(tmp2) - a);
        }
        loglikelihood = tmp4;

        if (useApprox) {
            for (int l = 0; l < K; ++l) {
                xbTmp = &xb[INDEX(0, l, memory_N)];
                a_sub_b(&yOrg[INDEX(0, l, memory_N)], xbTmp, xbTmp, N);
            }
        }
    } else {
        memset(xb, 0.0, memory_N * sizeof(double));
        memcpy(y, yOrg, memory_N * sizeof(double));
        for (int j = 0; j < P; ++j)
            add_a_add_scalar_b(&x[INDEX(0, j, memory_N)], beta[j], xb, N);

        loglikelihood = 0.0;
        double a = *std::max_element(xb, xb + N);
        for (int i = 0; i < N; ++i) {
            tmp_array1[i] = wOrg[i] * exp(xb[i] - a);

            if (status[i] != 0.0)
                loglikelihood += wOrg[i] * xb[i];
        }

        double tmp1 = sum_a(tmp_array1, N);
        double tmp2 = (log(tmp1) + a) * d[0];

        for (int i = 1; i < N; ++i) {
            tmp1 -= tmp_array1[i - 1];
            if (tmp1 < COX_MIN_PRECISION)
                tmp1 = sum_a(&tmp_array1[i], N - i);

            if (d[i] == 0.0)
                continue;
            tmp2 += (log(tmp1) + a) * d[i];
        }

        loglikelihood -= tmp2;
        if (useApprox)
            a_sub_b(yOrg, xb, xb, N);
    }

    cost =
        -loglikelihood + lambda * ((1.0 - alpha) * ridge / 2.0 + alpha * lasso);

    if (isFusion) {
        fusion = 0.0;
        memset(fusionPartialSums, 0.0, memory_nc * K * sizeof(double));

        double* ps;
        double* ptr;
        struct fusionKernel* currEl;

        for (int l = 0; l < K; l++) {
            ps = &fusionPartialSums[INDEX(0, l, memory_nc)];
            ptr = &beta[INDEX(0, l, memory_P)];

            for (int j = 0; j < P; j++) {
                currEl = fusionKernel[j];
                while (currEl != NULL) {
                    ps[currEl->i] += currEl->value * ptr[j];
                    currEl = currEl->next;
                }
            }
            fusionSums[l] = sum_abs_a(ps, nc);
        }

        fusion = sum_a(fusionSums, K);
        cost += gamma * fusion;
    }
    return;
}

void RegressionDataScheme::updateCost(int l) {
    double* xb = &xTimesBeta[INDEX(0, l, memory_N)];

    cost += loglikelihood;

    if (type <= 4 || useApprox) {
        loglikelihood = -sum_square_a_times_b(xb, w, N) * 0.5;
    } else if (type <= 8) {
        double tmp = 0.0;

        for (int i = 0; i < N; ++i) {
            tmp += wOrg[i] * (yOrg[i] * xb[i] - log1p(exp(xb[i])));
        }

        loglikelihood = tmp;
    } else if (type <= 12) {
        double tmp1 = 0.0;
        double tmp2 = 0.0;
        double tmp3 = 0.0;
        double tmp4 = 0.0;

        double a = *std::max_element(xb, xb + N);
        double* xbTmp;
        for (int l = 0; l < K; ++l) {
            xbTmp = &xTimesBeta[INDEX(0, l, memory_N)];
            double a1 = *std::max_element(xbTmp, xbTmp + N);
            if (a1 > a)
                a = a1;
        }

        for (int i = 0; i < N; ++i) {
            tmp1 = 0.0;
            tmp2 = 0.0;
            for (int ll = 0; ll < K; ++ll) {
                tmp3 = xTimesBeta[INDEX(i, ll, memory_N)];
                tmp1 += yOrg[INDEX(i, ll, memory_N)] * tmp3;
                tmp2 += exp(tmp3 - a);
            }
            tmp4 += wOrg[i] * (tmp1 - log(tmp2) - a);
        }

        loglikelihood = tmp4;
    } else {
        loglikelihood = 0.0;
        double a = *std::max_element(xb, xb + N);

        for (int i = 0; i < N; ++i) {
            tmp_array1[i] = wOrg[i] * exp(xTimesBeta[i] - a);
            if (status[i] != 0.0)
                loglikelihood += wOrg[i] * xTimesBeta[i];
        }

        double tmp1 = sum_a(tmp_array1, N);
        double tmp2 = (log(tmp1) + a) * d[0];

        for (int i = 1; i < N; ++i) {
            tmp1 -= tmp_array1[i - 1];
            if (tmp1 < COX_MIN_PRECISION)
                tmp1 = sum_a(&tmp_array1[i], N - i);

            if (d[i] == 0.0)
                continue;
            tmp2 += (log(tmp1) + a) * d[i];
        }
        loglikelihood -= tmp2;
    }

    cost -= loglikelihood;
}

double RegressionDataScheme::penaltyCost(double* coefs, double t) {
    double cost = 0.0;
    double tmp;

    for (int l = 0; l < K; l++) {
        tmp = coefs[l] - t;
        cost += 0.5 * (1.0 - alpha) * tmp * tmp + alpha * fabs(tmp);
    }

    return cost;
}
