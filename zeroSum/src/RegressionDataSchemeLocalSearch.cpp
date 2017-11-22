#include "RegressionDataScheme.h"

// #define DEBUG

void RegressionDataScheme::localSearch(int seed) {
#ifdef DEBUG
    double timet, costStart, costEnd;
    struct timespec ts0, ts1;
    clock_gettime(CLOCK_REALTIME, &ts0);
    PRINT("Starting LS\n");
    costFunction();
    costStart = cost;

    PRINT("Loglikelihood: %e lasso: %e ridge: %e cost: %e sum=%e\n",
          loglikelihood, lasso, ridge, cost, sum_a_times_b(beta, u, P));
#endif

    int k = 0, s = 0, t;
    double delta_k, e1 = 0.0, e2 = 0.0;

    std::mt19937_64 mt(seed);
    std::uniform_real_distribution<double> rng(0.0, 1.0);

    for (int j = 0; j < P; j++)
        checkActiveSet(j);

    double intervalSize = INTERVAL_SIZE;

    int sweep = isZeroSum ? P * (P - 1) : P;

    for (int step = 0; step < MAX_STEPS; step++) {
        long counter = 0;
        long attempts = 0;

        costFunction();
        e1 = cost;

        // random sweeps -> looking for new coefficients
        for (int sr = 0; sr < SWEEPS_RANDOM; sr++) {
            for (int l = 0; l < K; l++) {
                if (useApprox)
                    refreshApproximation(l, TRUE);

                if (useOffset)
                    lsSaOffsetMove(l);

                for (int i = 0; i < sweep * downScaler; i++) {
                    k = floor(rng(mt) * P);

                    if (isZeroSum) {
                        s = floor(rng(mt) * P);
                        if (s == k)
                            continue;
                    }

                    attempts++;

                    // choose a random amount
                    delta_k = rng(mt) * intervalSize - intervalSize * 0.5;

                    t = lsSaMove(k, s, l, delta_k);

                    if (t != 0) {
                        counter++;
                        checkActiveSet(k);
                        if (isZeroSum)
                            checkActiveSet(s);
                    }
                }
            }
        }

        int P2 = (isZeroSum) ? activeSet.size() : 1;

        // active set sweeps -> adjust coeffiecnts
        for (int sw = 0; sw < SWEEPS_ACTIVESET + SWEEPS_NULL; sw++) {
            for (int l = 0; l < K; l++) {
                double* betaPtr = &beta[INDEX(0, l, memory_P)];

                if (useApprox)
                    refreshApproximation(l, TRUE);

                if (useOffset)
                    lsSaOffsetMove(l);

                for (const int& k : activeSet) {
                    for (int n = 0; n < P2; n++) {
                        if (isZeroSum) {
                            s = activeSet[n];
                            if (s == k || rng(mt) > downScaler)
                                continue;
                        }

                        if (sw < SWEEPS_ACTIVESET) {
                            // choose a random amount
                            delta_k =
                                rng(mt) * intervalSize - intervalSize * 0.5;
                        } else {
                            // try to get beta_k to zero
                            delta_k = -betaPtr[k];
                        }

                        t = lsSaMove(k, s, l, delta_k);

                        attempts++;
                        if (t != 0)
                            counter++;
                    }
                }
            }
        }

        checkWholeActiveSet();

#ifdef DEBUG
        PRINT(
            "1: Loglikelihood: %e lasso: %e ridge: %e fusion: %e cost: %e "
            "sum=%e sum=%e t1=%d t2=%d i_size: %e\n",
            loglikelihood, lasso, ridge, fusion, cost, sum_a(beta, P),
            sum_a_times_b(beta, u, P), checkXtimesBeta(), checkYsubXtimesBeta(),
            intervalSize);
#endif

        costFunction();
        e2 = cost;

#ifdef DEBUG
        if (useApprox)
            refreshApproximation(K - 1, TRUE);

        PRINT(
            "2: Loglikelihood: %e lasso: %e ridge: %e fusion: %e cost: %e "
            "sum=%e sum=%e t1=%d t2=%d\n",
            loglikelihood, lasso, ridge, fusion, cost, sum_a(beta, P),
            sum_a_times_b(beta, u, P), checkXtimesBeta(),
            checkYsubXtimesBeta());
        PRINT("e2-e1: %e %e\n", e2 - e1, -precision);

        PRINT("step:%d size=%e accreptrate: %e  deltaE: %e\n", step,
              intervalSize, (double)counter / (double)(attempts), e2 - e1);
#endif

        intervalSize *= INTERVAL_SHRINK;

        if (e2 - e1 > -precision)
            break;
    }

#ifdef DEBUG
    costFunction();
    costEnd = cost;

    PRINT("Cost start: %e cost end: %e diff %e\n", costStart, costEnd,
          costEnd - costStart);

    clock_gettime(CLOCK_REALTIME, &ts1);
    timet = (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;
    PRINT("time taken = %e s\n", timet);
#endif
}
