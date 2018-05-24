#include "RegressionCV.h"

// #define DEBUG

void RegressionCV::localSearch(int seed) {
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int g = lengthGamma - 1; g >= 0; g--) {
        for (int f = 0; f < nFold + 1; f++) {
#ifdef DEBUG
            double timet, costStart, costEnd;
            struct timespec ts0, ts1;
            clock_gettime(CLOCK_REALTIME, &ts0);
            PRINT("Starting LS\n");
            cv_data[g][f].costFunction();
            costStart = cv_data[g][f].cost;

            PRINT("0: Loglikelihood: %e lasso: %e ridge: %e cost: %e sum=%e\n",
                  cv_data[g][f].loglikelihood, cv_data[g][f].lasso,
                  cv_data[g][f].ridge, cv_data[g][f].cost,
                  sum_a_times_b(cv_data[g][f].beta, cv_data[g][f].u, P));
#endif

            int k = 0, s = 0, t;
            double delta_k, e1 = 0.0, e2 = 0.0;

            seed = f + seed +
                   (int)((cv_data[g][f].lambda + cv_data[g][f].gamma) * 1e3);
            std::mt19937_64 mt(seed);
            std::uniform_real_distribution<double> rng(0.0, 1.0);

            for (int j = 0; j < P; j++)
                cv_data[g][f].checkActiveSet(j);

            double intervalSize = INTERVAL_SIZE;

            int sweep = cv_data[g][f].isZeroSum ? P * (P - 1) : P;
            sweep = (int)ceil((double)sweep * cv_data[g][f].downScaler);

            for (int step = 0; step < MAX_STEPS; step++) {
                long counter = 0;
                long attempts = 0;

                cv_data[g][f].costFunction();
                e1 = cv_data[g][f].cost;

                // random sweeps -> looking for new coefficients
                for (int sr = 0; sr < SWEEPS_RANDOM; sr++) {
                    for (int l = 0; l < K; l++) {
                        if (cv_data[g][f].useApprox)
                            cv_data[g][f].refreshApproximation(l, TRUE);

                        if (cv_data[g][f].useOffset)
                            cv_data[g][f].lsSaOffsetMove(l);

                        for (int i = 0; i < sweep; i++) {
                            k = floor(rng(mt) * P);

                            if (cv_data[g][f].isZeroSum) {
                                s = floor(rng(mt) * P);
                                if (s == k)
                                    continue;
                            }

                            attempts++;

                            // choose a random amount
                            delta_k =
                                rng(mt) * intervalSize - intervalSize * 0.5;

                            t = cv_data[g][f].lsSaMove(k, s, l, delta_k);

                            if (t != 0) {
                                counter++;
                                cv_data[g][f].checkActiveSet(k);
                                if (cv_data[g][f].isZeroSum)
                                    cv_data[g][f].checkActiveSet(s);

                                if (cv_data[g][f].useOffset)
                                    cv_data[g][f].lsSaOffsetMove(l);
                            }
                        }
                    }

                    if (type == MULTINOMIAL || type == MULTINOMIAL_ZS) {
                        cv_data[g][f].optimizeParameterAmbiguity(100);
                        cv_data[g][f].costFunction();
                    }
                }

                int P2 = (cv_data[g][f].isZeroSum)
                             ? cv_data[g][f].activeSet.size()
                             : 1;
                P2 = (int)ceil((double)P2 * cv_data[g][f].downScaler);

                // active set sweeps -> adjust coeffiecnts
                for (int sw = 0; sw < SWEEPS_ACTIVESET + SWEEPS_NULL; sw++) {
                    for (int l = 0; l < K; l++) {
                        double* betaPtr =
                            &cv_data[g][f].beta[INDEX(0, l, memory_P)];

                        if (cv_data[g][f].useApprox)
                            cv_data[g][f].refreshApproximation(l, TRUE);

                        if (cv_data[g][f].useOffset)
                            cv_data[g][f].lsSaOffsetMove(l);

                        for (const int& k : cv_data[g][f].activeSet) {
                            for (int n = 0; n < P2; n++) {
                                if (cv_data[g][f].isZeroSum) {
                                    s = cv_data[g][f].activeSet[n];
                                    if (s == k ||
                                        rng(mt) > cv_data[g][f].downScaler)
                                        continue;
                                }

                                if (sw < SWEEPS_ACTIVESET) {
                                    // choose a random amount
                                    delta_k = rng(mt) * intervalSize -
                                              intervalSize * 0.5;
                                } else {
                                    // try to get beta_k to zero
                                    delta_k = -betaPtr[k];
                                }

                                t = cv_data[g][f].lsSaMove(k, s, l, delta_k);

                                attempts++;
                                if (t != 0) {
                                    counter++;
                                    if (cv_data[g][f].useOffset)
                                        cv_data[g][f].lsSaOffsetMove(l);
                                }
                            }
                        }

                        if (type == MULTINOMIAL || type == MULTINOMIAL_ZS) {
                            cv_data[g][f].optimizeParameterAmbiguity(100);
                            cv_data[g][f].costFunction();
                        }

                        if (cv_data[g][f].isFusion) {
                            for (const int& k : cv_data[g][f].activeSet) {
                                s = k + 1;
                                if (s == P)
                                    s = k - 1;

                                if (cv_data[g][f].isZeroSum)
                                    delta_k = (betaPtr[s] - betaPtr[k]) * 0.5;
                                else
                                    delta_k = (betaPtr[s] - betaPtr[k]);

                                t = cv_data[g][f].lsSaMove(k, s, l, delta_k);

                                attempts++;
                                if (t != 0) {
                                    counter++;
                                    if (cv_data[g][f].useOffset)
                                        cv_data[g][f].lsSaOffsetMove(l);
                                }
                            }
                        }
                    }
                }

                cv_data[g][f].removeCoefsWithZeroFromActiveSet();

#ifdef DEBUG
                PRINT(
                    "1: Loglikelihood: %e lasso: %e ridge: %e fusion: %e cost: "
                    "%e "
                    "sum=%e sum=%e t1=%d t2=%d i_size: %e\n",
                    cv_data[g][f].loglikelihood, cv_data[g][f].lasso,
                    cv_data[g][f].ridge, cv_data[g][f].fusion,
                    cv_data[g][f].cost, sum_a(cv_data[g][f].beta, P),
                    sum_a_times_b(cv_data[g][f].beta, cv_data[g][f].u, P),
                    cv_data[g][f].checkXtimesBeta(),
                    cv_data[g][f].checkYsubXtimesBeta(), intervalSize);
#endif

                cv_data[g][f].costFunction();
                e2 = cv_data[g][f].cost;

#ifdef DEBUG
                if (cv_data[g][f].useApprox)
                    cv_data[g][f].refreshApproximation(K - 1, TRUE);

                PRINT(
                    "2: Loglikelihood: %e lasso: %e ridge: %e fusion: %e cost: "
                    "%e "
                    "sum=%e sum=%e t1=%d t2=%d\n",
                    cv_data[g][f].loglikelihood, cv_data[g][f].lasso,
                    cv_data[g][f].ridge, cv_data[g][f].fusion,
                    cv_data[g][f].cost, sum_a(cv_data[g][f].beta, P),
                    sum_a_times_b(cv_data[g][f].beta, cv_data[g][f].u, P),
                    cv_data[g][f].checkXtimesBeta(),
                    cv_data[g][f].checkYsubXtimesBeta());
                PRINT("e2-e1: %e %e\n", e2 - e1, -cv_data[g][f].precision);

                PRINT("step:%d size=%e accreptrate: %e  deltaE: %e\n", step,
                      intervalSize, (double)counter / (double)(attempts),
                      e2 - e1);
#endif

                intervalSize *= INTERVAL_SHRINK;
                if (e2 - e1 > -cv_data[g][f].precision)
                    break;
            }

            for (int j = 0; j < K * memory_P; j++)
                if (fabs(cv_data[g][f].beta[j]) < 100 * DBL_EPSILON)
                    cv_data[g][f].beta[j] = 0.0;

#ifdef DEBUG
            cv_data[g][f].costFunction();
            costEnd = cv_data[g][f].cost;

            PRINT("Cost start: %e cost end: %e diff %e\n", costStart, costEnd,
                  costEnd - costStart);

            clock_gettime(CLOCK_REALTIME, &ts1);
            timet =
                (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;
            PRINT("time taken = %e s\n", timet);
#endif
        }
    }
}
