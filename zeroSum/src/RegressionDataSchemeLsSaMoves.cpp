#include "RegressionDataScheme.h"

void RegressionDataScheme::lsSaOffsetMove(int l) {
    if (type > 4 && !useApprox)
        refreshApproximation(l);
    offsetMove(l, TRUE);
}

int RegressionDataScheme::lsSaMove(int k,
                                   int s,
                                   int l,
                                   double delta_k,
                                   double* rng,
                                   double temperature) {
    int col_k = INDEX(0, k, memory_N);

    double* xbTmp = tmp_array1;
    double* xb = &xTimesBeta[INDEX(0, l, memory_N)];
    memcpy(xbTmp, xb, sizeof(double) * N);

    double tmp_loglikelihood = 0.0;
    double tmp_ridge = 0.0;
    double tmp_lasso = 0.0;
    double tmp_fusion = 0.0;

    double betak = 0.0, betas = 0.0, betakNew = 0.0, betasNew = 0.0,
           delta_s = 0.0;

    betak = beta[INDEX(k, l, memory_P)];
    betakNew = betak + delta_k;

    if (isZeroSum) {
        delta_s = -delta_k * u[k] / u[s];
        betas = beta[INDEX(s, l, memory_P)];
        betasNew = betas + delta_s;

        int col_s = INDEX(0, s, memory_N);

        if (type <= 4 || useApprox)
            sub_a_times_scalar_b_sub_c_times_scalar_d(
                &x[col_k], delta_k, &x[col_s], delta_s, xbTmp, N);
        else
            sub_a_times_scalar_b_sub_c_times_scalar_d(
                &x[col_k], -delta_k, &x[col_s], -delta_s, xbTmp, N);

        tmp_lasso =
            lasso - fabs(betak) - fabs(betas) + fabs(betakNew) + fabs(betasNew);

        tmp_ridge = ridge - betak * betak - betas * betas +
                    betakNew * betakNew + betasNew * betasNew;
    } else {
        if (type <= 4 || useApprox)
            add_a_add_scalar_b(&x[col_k], -delta_k, xbTmp, N);
        else
            add_a_add_scalar_b(&x[col_k], delta_k, xbTmp, N);

        tmp_lasso = lasso - fabs(betak) + fabs(betakNew);
        tmp_ridge = ridge - betak * betak + betakNew * betakNew;
    }

    if (type <= 4) {
        tmp_loglikelihood = -sum_square_a_times_b(xbTmp, w, N) * 0.5;
    } else if (type <= 8) {
        if (useApprox) {
            tmp_loglikelihood = -sum_square_a_times_b(xbTmp, w, N) * 0.5;
        } else {
            for (int i = 0; i < N; ++i)
                tmp_loglikelihood +=
                    wOrg[i] * (yOrg[i] * xbTmp[i] - log(1.0 + exp(xbTmp[i])));
        }
    } else if (type <= 12) {
        if (useApprox) {
            tmp_loglikelihood = -sum_square_a_times_b(xbTmp, w, N) * 0.5;
        } else {
            double tmp1 = 0.0;
            double tmp2 = 0.0;
            double tmp3 = 0.0;
            double tmp4 = 0.0;

            for (int i = 0; i < N; ++i) {
                tmp1 = 0.0;
                tmp2 = 0.0;
                for (int ll = 0; ll < K; ++ll) {
                    if (ll != l)
                        tmp3 = xTimesBeta[INDEX(i, ll, memory_N)];
                    else
                        tmp3 = xbTmp[i];

                    tmp1 += yOrg[INDEX(i, ll, memory_N)] * tmp3;
                    tmp2 += exp(tmp3);
                }
                tmp4 += wOrg[i] * (tmp1 - log(tmp2));
            }

            tmp_loglikelihood = tmp4;
        }
    } else if (type <= 16) {
        if (useApprox) {
            tmp_loglikelihood = -sum_square_a_times_b(xbTmp, w, N) * 0.5;
        } else {
            double tmp1 = 0.0;
            for (int i = 0; i < N; ++i) {
                tmp_array2[i] = exp(xbTmp[i]);
                tmp1 += wOrg[i] * tmp_array2[i];

                if (status[i] != 0.0)
                    tmp_loglikelihood += wOrg[i] * xbTmp[i];
            }

            double tmp2 = log(tmp1) * d[0];
            for (int i = 1; i < N; ++i) {
                tmp1 = tmp1 - wOrg[i - 1] * tmp_array2[i - 1];
                tmp2 += log(tmp1) * d[i];
            }
            tmp_loglikelihood -= tmp2;
        }
    }

    double fusionSum = 0.0;
    if (isFusion) {
        memcpy(fusionPartialSumsTmp, &fusionPartialSums[INDEX(0, l, memory_nc)],
               nc * sizeof(double));

        struct fusionKernel* currEl = fusionKernel[k];

        while (currEl != NULL) {
            fusionPartialSumsTmp[currEl->i] += currEl->value * delta_k;
            currEl = currEl->next;
        }

        if (isZeroSum) {
            currEl = fusionKernel[s];
            while (currEl != NULL) {
                fusionPartialSumsTmp[currEl->i] += currEl->value * delta_s;
                currEl = currEl->next;
            }
        }

        fusionSum = sum_abs_a(fusionPartialSumsTmp, nc);
        tmp_fusion = fusion - fusionSums[l] + fusionSum;
    }

    double tmp_cost =
        -tmp_loglikelihood +
        lambda * ((1.0 - alpha) * tmp_ridge / 2.0 + alpha * tmp_lasso) +
        gamma * tmp_fusion;

    double deltaE = tmp_cost - cost;

    if (deltaE < 0.0 || (rng != NULL && *rng < exp(-deltaE / temperature))) {
        memcpy(xb, xbTmp, sizeof(double) * N);

        ridge = tmp_ridge;
        lasso = tmp_lasso;
        fusion = tmp_fusion;
        loglikelihood = tmp_loglikelihood;
        cost = tmp_cost;

        if (isFusion) {
            memcpy(&fusionPartialSums[INDEX(0, l, memory_nc)],
                   fusionPartialSumsTmp, nc * sizeof(double));
            fusionSums[l] = fusionSum;
        }

        beta[INDEX(k, l, memory_P)] = betakNew;

        if (isZeroSum)
            beta[INDEX(s, l, memory_P)] = betasNew;

        if (useApprox)
            refreshApproximation(l, TRUE);

        if (useOffset)
            lsSaOffsetMove(l);

        return 1;
    } else {
        return 0;
    }
}
