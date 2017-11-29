#include "RegressionDataScheme.h"

void RegressionDataScheme::offsetMove(int l, int _updateCost) {
    int ii = INDEX(0, l, memory_N);

    double* xb = &xTimesBeta[ii];
    double bk;

    if (type <= 4 || useApprox)
        bk = sum_a_times_b(xb, w, N);
    else
        bk = sum_a_sub_b_times_c(&y[ii], xb, w, N);

    double sumW = sum_a(w, N);

    if (fabs(sumW) < 1000 * DBL_EPSILON)
        return;

    double oldbeta = offset[l];
    offset[l] = bk / sumW + oldbeta;

    double diff = oldbeta - offset[l];

    if (type <= 4 || useApprox)
        a_add_scalar_b(xb, diff, xb, N);
    else
        a_add_scalar_b(xb, -diff, xb, N);

    if (type > 4 && !useApprox)
        refreshApproximation(l);

    if (_updateCost)
        updateCost(l);
}

int RegressionDataScheme::cdMove(int k, int l) {
    double* betak = &beta[INDEX(k, l, memory_P)];
    double* xk = &x[INDEX(0, k, memory_N)];

    a_times_b(w, xk, tmp_array1, N);

    double ak =
        sum_a_times_b(xk, tmp_array1, N) + lambda * (1.0 - alpha) * v[k];

    a_times_scalar_b(xk, *betak, tmp_array2, N);

    double* xb = &(xTimesBeta[INDEX(0, l, memory_N)]);

    double bk = sum_a_add_b_times_c(xb, tmp_array2, tmp_array1, N);
    double tmp = lambda * alpha * v[k];

    double bk1 = bk + tmp;
    double bk2 = bk - tmp;

    if (bk1 < 0.0) {
        tmp = bk1 / ak;
    } else if (bk2 > 0.0) {
        tmp = bk2 / ak;
    } else {
        tmp = 0.0;
    }

    double diff = *betak - tmp;
    *betak = tmp;

    add_a_add_scalar_b(xk, diff, xb, N);
    if (fabs(diff) < BETA_CHANGE_PRECISION)
        return 0;
    else {
        if (useApprox)
            refreshApproximation(l);

        if (useOffset)
            offsetMove(l);

        return 1;
    }
}

int RegressionDataScheme::cdMoveZS(int k, int s, int l) {
    double* betak = &beta[INDEX(k, l, memory_P)];
    double* betas = &beta[INDEX(s, l, memory_P)];

    double* xb = &xTimesBeta[INDEX(0, l, memory_N)];
    double* xk = &x[INDEX(0, k, memory_N)];
    double* xs = &x[INDEX(0, s, memory_N)];

    double ukus = u[k] / u[s];
    double l1 = lambda * (1.0 - alpha);

    a_times_scalar_b_sub_c(xs, ukus, xk, tmp_array1, N);

    double ak = sum_square_a_times_b(tmp_array1, w, N) +
                l1 * (v[k] + v[s] * ukus * ukus);

    a_add_scalar_b_times_c_sub_d_times_e(xb, *betak, xk, xs, ukus, tmp_array2,
                                         N);

    double bk = -sum_a_times_b_times_c(w, tmp_array1, tmp_array2, N) +
                l1 * v[s] * ukus * (*betas + ukus * *betak);
    double tmp3 = lambda * alpha;

    double case14 = tmp3 * (v[k] - v[s] * ukus);
    double case23 = tmp3 * (v[k] + v[s] * ukus);

    double bk1 = (bk - case14) / ak;
    double bk2 = (bk - case23) / ak;
    double bk3 = (bk + case23) / ak;
    double bk4 = (bk + case14) / ak;

    tmp3 = u[s] * *betas + u[k] * *betak;

    double bs1 = tmp3 - u[k] * bk1;
    double bs2 = tmp3 - u[k] * bk2;
    double bs3 = tmp3 - u[k] * bk3;
    double bs4 = tmp3 - u[k] * bk4;

    double diffk = -*betak;
    double diffs = -*betas;

    int defined = true;
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
        diffk += *betak;
        diffs += *betas;
        sub_a_times_scalar_b_sub_c_times_scalar_d(xk, diffk, xs, diffs, xb, N);
        if (useApprox)
            refreshApproximation(l);

        if (useOffset)
            offsetMove(l);

        return 1;
    }
}

int RegressionDataScheme::cdMoveZSRotated(int n,
                                          int m,
                                          int s,
                                          int l,
                                          double theta) {
    double* betan = &beta[INDEX(n, l, memory_P)];
    double* betam = &beta[INDEX(m, l, memory_P)];
    double* betas = &beta[INDEX(s, l, memory_P)];

    double* xb = &xTimesBeta[INDEX(0, l, memory_N)];

    double* xn = &x[INDEX(0, n, memory_N)];
    double* xm = &x[INDEX(0, m, memory_N)];
    double* xs = &x[INDEX(0, s, memory_N)];

    double cosT = cos(theta);
    double sinT = sin(theta);

    double unum1 = (-u[n] * cosT + u[m] * sinT) / u[s];
    double unum2 = (u[n] * cosT - u[m] * sinT) / u[s];

    double l1 = lambda * alpha;
    double l2 = lambda - l1;

    a_times_scalar_b_add_c_times_scalar_d_add_d_times_scalar_f(
        xm, sinT, xn, -cosT, xs, unum2, tmp_array1, N);

    double a =
        sum_square_a_times_b(tmp_array1, w, N) +
        l2 * (v[n] * cosT * cosT + v[m] * sinT * sinT + v[s] * unum1 * unum1);

    double b = -sum_a_times_b_times_c(w, xb, tmp_array1, N) -
               l2 * (v[n] * *betan * cosT - v[m] * *betam * sinT +
                     v[s] * *betas * unum1);

    double t1 = v[n] * cosT - v[m] * sinT;
    double t2 = v[n] * cosT + v[m] * sinT;

    unum1 *= v[s];
    unum2 *= v[s];

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
    double diffn = -*betan;
    double diffm = -*betam;
    double diffs = -*betas;

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

    diffn += *betan;
    diffm += *betam;
    diffs += *betas;

    sub_a_times_scalar_b_sub_c_times_scalar_d_sub_d_times_scalar_f(
        xn, diffn, xm, diffm, xs, diffs, xb, N);

    if (fabs(diffn) < BETA_CHANGE_PRECISION)
        return 0;
    else {
        if (useApprox)
            refreshApproximation(l);

        if (useOffset)
            offsetMove(l);

        return 1;
    }
}
