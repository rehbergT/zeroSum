#include "zeroSum.h"

// localSearch setttings
#define MAX_STEPS 500
#define INTERVAL_SIZE 0.2
#define INTERVAL_SHRINK 0.9
#define LS_MIN_REPEATS 100

#define SWEEPS_RANDOM 15
#define SWEEPS_ACTIVESET 8
#define SWEEPS_NULL 1
#define SWEEPS_FUSED 10

void zeroSum::doFitUsingLocalSearch(uint32_t seed) {
    std::vector<std::mt19937_64> mt;
    std::uniform_real_distribution<double> rng(0.0, 1.0);
    for (uint32_t f = 0; f < nFold1; f++) {
        uint32_t fold_seed = f + seed + (uint32_t)((lambda[f] + gamma) * 1e3);
        mt.push_back(std::mt19937_64(fold_seed));
    }

    threadPool.doParallelChunked(nFold1, [&](size_t f) {
        double* betaf = &beta[INDEX_TENSOR_COL(0, f, memory_P, K)];
#ifdef DEBUG
        double timet, costStart, costEnd;
        struct timespec ts0, ts1;
        clock_gettime(CLOCK_REALTIME, &ts0);
        PRINT("Starting LS\n");
        costFunction(f);
        costStart = cost[f];

        PRINT(
            "0: Loglikelihood: %e lasso: %e ridge: %e cost: %e "
            "sum=%e\n",
            loglikelihood[f], lasso[f], ridge[f], cost[f],
            ddot_(&memory_P, betaf, &BLAS_I_ONE, u, &BLAS_I_ONE));
#endif

        uint32_t k = 0, s = 0, t;
        double delta_k, e1 = 0.0, e2 = 0.0;

        for (uint32_t j = 0; j < P; j++)
            activeSetInsert(f, j);

        double intervalSize = INTERVAL_SIZE;

        uint32_t sweep = P * (P - 1);
        sweep = (uint32_t)ceil((double)sweep * downScaler);

        for (uint32_t step = 0; step < MAX_STEPS; step++) {
            long counter = 0;
            long attempts = 0;

            costFunction(f);
            e1 = cost[f];

            // random sweeps -> looking for new coefficients
            for (uint32_t sr = 0; sr < SWEEPS_RANDOM; sr++) {
                for (uint32_t l = 0; l < K; l++) {
                    if (useApprox)
                        refreshApproximation(f, l, true);

                    if (useIntercept)
                        lsSaOffsetMove(f, l);

                    for (uint32_t i = 0; i < sweep; i++) {
                        k = floor(rng(mt[f]) * P);

                        if (useZeroSum) {
                            s = floor(rng(mt[f]) * P);
                            if (s == k)
                                continue;
                        }

                        attempts++;

                        // choose a random amount
                        delta_k =
                            rng(mt[f]) * intervalSize - intervalSize * 0.5;

                        t = lsSaMove(f, k, s, l, delta_k);

                        if (t != 0) {
                            counter++;
                            activeSetInsert(f, k);
                            if (useZeroSum)
                                activeSetInsert(f, s);

                            if (useIntercept)
                                lsSaOffsetMove(f, l);
                        }
                    }
                }

                if (type == multinomial && !useFusion) {
                    optimizeParameterAmbiguity(f, 100);
                    costFunction(f);
                }
            }

            uint32_t P2 = useZeroSum ? activeSet[f].size() : 1;
            P2 = (uint32_t)ceil((double)P2 * downScaler);

            // active set sweeps -> adjust coeffiecnts
            for (uint32_t sw = 0; sw < SWEEPS_ACTIVESET + SWEEPS_NULL; sw++) {
                for (uint32_t l = 0; l < K; l++) {
                    double* betaPtr = &betaf[INDEX_COL(l, memory_P)];

                    if (useApprox)
                        refreshApproximation(f, l, true);

                    if (useIntercept)
                        lsSaOffsetMove(f, l);

                    for (const uint32_t& k : activeSet[f]) {
                        if (useZeroSum) {
                            for (const uint32_t& s : activeSet[f]) {
                                if (!activeSetContains(f, s))
                                    continue;
                                if (s == k || rng(mt[f]) > downScaler)
                                    continue;

                                if (sw < SWEEPS_ACTIVESET) {
                                    // choose a random amount
                                    delta_k = rng(mt[f]) * intervalSize -
                                              intervalSize * 0.5;
                                } else {
                                    // try to get beta_k to zero
                                    delta_k = -betaPtr[k];
                                }

                                t = lsSaMove(f, k, s, l, delta_k);

                                attempts++;
                                if (t != 0) {
                                    counter++;
                                    if (useIntercept)
                                        lsSaOffsetMove(f, l);
                                }
                            }
                        } else {
                            if (sw < SWEEPS_ACTIVESET) {
                                // choose a random amount
                                delta_k = rng(mt[f]) * intervalSize -
                                          intervalSize * 0.5;
                            } else {
                                // try to get beta_k to zero
                                delta_k = -betaPtr[k];
                            }

                            t = lsSaMove(f, k, s, l, delta_k);

                            attempts++;
                            if (t != 0) {
                                counter++;
                                if (useIntercept)
                                    lsSaOffsetMove(f, l);
                            }
                        }
                    }

                    if (type == multinomial && !useFusion) {
                        optimizeParameterAmbiguity(f, 100);
                        costFunction(f);
                    }

                    if (useFusion) {
                        for (uint32_t k = 0; k < P; k++) {
                            if (!activeSetContains(f, k))
                                continue;
                            s = k + 1;
                            if (s == P)
                                s = k - 1;

                            if (useZeroSum)
                                delta_k = (betaPtr[s] - betaPtr[k]) * 0.5;
                            else
                                delta_k = (betaPtr[s] - betaPtr[k]);

                            t = lsSaMove(f, k, s, l, delta_k);

                            attempts++;
                            if (t != 0) {
                                counter++;
                                if (useIntercept)
                                    lsSaOffsetMove(f, l);
                            }
                        }
                    }
                }
            }

            activeSetRemoveZeros(f);

#ifdef DEBUG
            PRINT(
                "1: Loglikelihood: %e lasso: %e ridge: %e fusion: %e "
                "cost: %e "
                "sum=%e sum=%e i_size: %e\n",
                loglikelihood[f], lasso[f], ridge[f], fusion[f], cost[f],
                arraySumAvx(betaf, memory_P),
                ddot_(&memory_P, betaf, &BLAS_I_ONE, u, &BLAS_I_ONE),
                intervalSize);
#endif

            costFunction(f);
            e2 = cost[f];

#ifdef DEBUG
            if (useApprox)
                refreshApproximation(f, K - 1, true);

            PRINT(
                "2: Loglikelihood: %e lasso: %e ridge: %e fusion: %e "
                "cost: %e "
                "sum=%e sum=%e\n",
                loglikelihood[f], lasso[f], ridge[f], fusion[f], cost[f],
                arraySumAvx(betaf, memory_P),
                ddot_(&memory_P, betaf, &BLAS_I_ONE, u, &BLAS_I_ONE));
            PRINT("e2-e1: %e %e\n", e2 - e1, -precision);

            PRINT("step:%d size=%e accreptrate: %e  deltaE: %e\n", step,
                  intervalSize, (double)counter / (double)(attempts), e2 - e1);
#endif

            intervalSize *= INTERVAL_SHRINK;
            if (e2 - e1 > -precision)
                break;
        }

        for (uint32_t j = 0; j < K * memory_P; j++)
            if (fabs(betaf[j]) < 100 * DBL_EPSILON)
                betaf[j] = 0.0;

#ifdef DEBUG
        costFunction(f);
        costEnd = cost[f];

        PRINT("Cost start: %e cost end: %e diff %e\n", costStart, costEnd,
              costEnd - costStart);

        clock_gettime(CLOCK_REALTIME, &ts1);
        timet = (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;
        PRINT("time taken = %e s\n", timet);
#endif
    });
}
