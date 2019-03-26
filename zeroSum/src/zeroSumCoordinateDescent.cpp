#include "zeroSum.h"

#define MIN_SUCCESS_RATE 0.90

void zeroSum::doFitUsingCoordinateDescent(uint32_t seed) {
    // coordinate descent is only working for gamma=0!
    costFunctionAllFolds();

#ifdef DEBUG
    double* costStart = (double*)malloc(nFold1 * sizeof(double));
    double timet;
    struct timespec ts0, ts1;
    clock_gettime(CLOCK_REALTIME, &ts0);
#endif

    std::vector<std::mt19937_64> mt;
    std::uniform_real_distribution<double> rng(0.0, 1.0);
    for (uint32_t f = 0; f < nFold1; f++) {
        uint32_t fold_seed = f + seed + (uint32_t)(lambda[f] * 1e3);
        mt.push_back(std::mt19937_64(fold_seed));
    }

    threadPool.doParallelChunked(nFold1, [&](size_t f) {
        double* betaf = &beta[INDEX_TENSOR_COL(0, f, memory_P, K)];
        double* last_betaf = &last_beta[INDEX_TENSOR_COL(0, f, memory_P, K)];
        double* interceptf = &intercept[INDEX_COL(f, K)];
        double* last_interceptf = &last_intercept[INDEX_COL(f, K)];

#ifdef DEBUG
        costStart[f] = cost[f];
        PRINT(
            "Loglikelihood: %e lasso: %e ridge: %e cost: %e sum=%e "
            "sum=%e\n",
            loglikelihood[f], lasso[f], ridge[f], cost[f],
            arraySumAvx(beta, memory_P),
            ddot_(&memory_P, betaf, &BLAS_I_ONE, u, &BLAS_I_ONE));
#endif
        activeSet[f].clear();

        uint32_t success, attempts;
        uint32_t gettingWorse = false;
        for (uint32_t steps = 0; steps < 100 && !gettingWorse; steps++) {
#ifdef DEBUG
            PRINT("Step: %d\nFind active set\n", steps);
#endif
            uint32_t activeSetChange = 0;

            double costBefore = cost[f];
            memcpy(last_betaf, betaf, memory_P * K * sizeof(double));
            memcpy(last_interceptf, interceptf, K * sizeof(double));

            // cycle over all multinomial classes (if type !=
            // multinomial K=1)
            for (uint32_t l = 0; l < K; l++) {
                // every move checks if the approximation has failed.
                // Has to be reset for each class, since it could have
                // failed for class l=0 but could be valid for l=1
                approxFailed[f] = false;
                if (type != gaussian)
                    refreshApproximation(f, l);

                // update the intercept
                if (useIntercept)
                    interceptMove(f, l);

                // Cycle over the coefficients using the update schemes.
                // In the zerosum case there are (P^2 - P) / 2 unique
                // combinations possible. If the active set has already
                // determined (steps>0), only the combinations between
                // the coefficients of the activeset with the
                // coefficients which are not in active set have to be
                // tested. In the non zero-sum case the update scheme
                // only has to be iterated over the P coefficients.
                if (useZeroSum) {
                    for (uint32_t s = 0; s < P; s++) {
                        // if the coefficients have a zero-sum weight
                        // (u) of 0 we have to apply the cdMove. We can
                        // then skip the quadratic update search for
                        // beta_s
                        if (u[s] == 0) {
                            uint32_t change = cdMove(f, s, l);
                            if (change)
                                activeSetChange += activeSetInsert(f, s);
                            continue;
                        }

                        if (steps > 0) {
                            // if step>0 the loop with uint32_t s should
                            // only go over the non active set
                            // coefficients
                            // -> skip if s is in the active set
                            if (activeSetContains(f, s))
                                continue;

                            // second coefficient should be of the
                            // activeset
                            for (const uint32_t& k : activeSet[f]) {
                                // zero-sum weight u_k = 0 means that
                                // this coef has to be updated with
                                // cdMove. But this is done by the
                                // uint32_t s loop -> just skipt here
                                if (u[k] == 0)
                                    continue;

                                uint32_t change = cdMoveZS(f, s, k, l);

                                if (change) {
                                    activeSetChange += activeSetInsert(f, k);
                                    activeSetChange += activeSetInsert(f, s);
                                }
                            }
                        } else {
                            // cycle over all (P^2 - P) / 2 unique
                            // combinations
                            for (uint32_t k = s + 1; k < P; k++) {
                                // zero-sum weight u_k = 0 means that
                                // this coef has to be updated with
                                // cdMove. But this is done by the
                                // uint32_t s loop -> just skipt here
                                if (u[k] == 0)
                                    continue;

                                uint32_t change = cdMoveZS(f, s, k, l);

                                if (change) {
                                    activeSetChange += activeSetInsert(f, k);
                                    activeSetChange += activeSetInsert(f, s);
                                }
                            }
                        }
                    }
                } else {
                    for (uint32_t s = 0; s < P; s++) {
                        uint32_t change = cdMove(f, s, l);
                        if (change)
                            activeSetChange += activeSetInsert(f, s);
                    }
                }

                if (type == multinomial) {
                    optimizeParameterAmbiguity(f, 200);
                }
            }
            // polish (local search) to get small coefs where the cd is
            // undefined to zero
            if (usePolish) {
                // cost function has to be executed to init all ls
                // variables
                costFunction(f);
                for (uint32_t l = 0; l < K; l++) {
                    // get a pointer to the current coefficients for
                    // easier access
                    double* betaPtr =
                        &beta[INDEX_TENSOR_COL(l, f, memory_P, K)];
                    if (useApprox)
                        refreshApproximation(f, l, true);

                    for (const uint32_t& s : activeSet[f]) {
                        // if beta_s is zero skip -> we dont want a zero
                        // coefficient as target. The local search move
                        // maintains the zerosum contraint
                        // -> dont apply the move on coefficients which
                        // are not in the zerosum constraint
                        if (fabs(betaPtr[s]) < DBL_EPSILON * 100 || u[s] == 0)
                            continue;

                        for (const uint32_t& k : activeSet[f]) {
                            // skip if s==k since the move is not
                            // defined for this case. Also, skip if
                            // beta_k is 0
                            if (s == k ||
                                fabs(betaPtr[k]) < DBL_EPSILON * 100 ||
                                u[k] == 0)
                                continue;
                            // try to move the amount of beta_k to
                            // beta_s. If it does not harm the cost
                            // function the move is performed.
                            lsSaMove(f, k, s, l, -betaPtr[k]);
                        }
                    }
                }
                activeSetRemoveZeros(f);
            }

            // this cost function call is necessary to init all
            // approximation variables
            costFunction(f);
            double costAfter = cost[f];

#ifdef DEBUG
            activeSetPrint(f);
            PRINT(
                "Loglikelihood: %e lasso: %e ridge: %e cost: %e sum=%e "
                "sum=%e\n",
                loglikelihood[f], lasso[f], ridge[f], cost[f],
                arraySumAvx(beta, memory_P),
                ddot_(&memory_P, betaf, &BLAS_I_ONE, u, &BLAS_I_ONE));
#endif
            // if due to numerical instabilities the costfunction gets
            // nan or the quality of the model decreases, restore the
            // previous configuration and stop the optimization
            if (std::isnan(costAfter) || std::isinf(costAfter) ||
                costBefore < costAfter) {
                memcpy(betaf, last_betaf, memory_P * K * sizeof(double));
                memcpy(interceptf, last_interceptf, K * sizeof(double));
                costFunction(f);
                break;
            }

            // if the activeset search has not changed the active set or
            // if the active set is empty then stop
            if (activeSetChange == 0 || activeSet[f].size() == 0) {
                break;
            }

#ifdef DEBUG
            PRINT("converge\n");
#endif

            while (true) {
                costBefore = cost[f];
                memcpy(last_betaf, betaf, memory_P * K * sizeof(double));
                memcpy(last_interceptf, interceptf, K * sizeof(double));

                success = 0;

                for (uint32_t l = 0; l < K; l++) {
                    approxFailed[f] = false;
                    if (type != gaussian)
                        refreshApproximation(f, l);
                    if (useIntercept)
                        interceptMove(f, l);

                    if (useZeroSum) {
                        for (const uint32_t& s : activeSet[f]) {
                            if (u[s] == 0) {
                                success += cdMove(f, s, l);
                                continue;
                            }
                            for (const uint32_t& k : activeSet[f]) {
                                if (s == k || u[k] == 0)
                                    continue;
                                success += cdMoveZS(f, s, k, l);
                            }
                        }
                        attempts =
                            activeSet[f].size() * (activeSet[f].size() - 1);

                        if (rotatedUpdates &&
                            (double)success / (double)attempts <=
                                MIN_SUCCESS_RATE) {
                            for (const uint32_t& s : activeSet[f]) {
                                if (u[s] == 0)
                                    continue;

                                for (const uint32_t& k : activeSet[f]) {
                                    if (s == k || u[k] == 0)
                                        continue;

                                    uint32_t h =
                                        floor(rng(mt[f]) * activeSet[f].size());

                                    // select a random element of the
                                    // set
                                    h = activeSetGetElement(f, h);

                                    if (h == s || h == k || u[h] == 0)
                                        continue;

                                    cdMoveZSRotated(f, s, k, h, l,
                                                    rng(mt[f]) * M_PI);
                                }
                            }
                        }

                    } else {
                        for (const uint32_t& s : activeSet[f]) {
                            success += cdMove(f, s, l);
                        }
                    };
                }

                if (type == multinomial) {
                    optimizeParameterAmbiguity(f, 200);
                }

                costFunction(f);
                costAfter = cost[f];
#ifdef DEBUG
                PRINT(
                    "Loglikelihood: %e lasso: %e ridge: %e cost: %e "
                    "sum=%e "
                    "sum=%e\tChange: costBefore=%e costAfter=%e %e %e "
                    "(success:%d)\n",
                    loglikelihood[f], lasso[f], ridge[f], cost[f],
                    arraySumAvx(betaf, memory_P),
                    ddot_(&memory_P, betaf, &BLAS_I_ONE, u, &BLAS_I_ONE),
                    costBefore, costAfter, fabs(costAfter - costBefore),
                    precision, success);
#endif
                if (std::isnan(costAfter) || costBefore < costAfter) {
                    gettingWorse = true;
                    memcpy(betaf, last_betaf, memory_P * K * sizeof(double));
                    memcpy(interceptf, last_interceptf, K * sizeof(double));
                    costFunction(f);
                    break;
                }

                if (success == 0 || fabs(costAfter - costBefore) < precision)
                    break;
            }
        }
    });

    if (type == multinomial) {
        threadPool.doParallelChunked(
            nFold1, [&](size_t f) { optimizeParameterAmbiguity(f, 200); });
    }

    threadPool.doParallelChunked(memory_P * K * nFold1, [&](size_t j) {
        if (fabs(beta[j]) < 100 * DBL_EPSILON)
            beta[j] = 0.0;
    });
#ifdef DEBUG
    costFunctionAllFolds();

    for (uint32_t f = 0; f < nFold1; f++) {
        double costEnd = cost[f];
        PRINT("Cost start: %e cost end: %e diff %e\n", costStart[f], costEnd,
              costEnd - costStart[f]);
    }

    clock_gettime(CLOCK_REALTIME, &ts1);
    timet = (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;
    PRINT("time taken = %e s\n", timet);

    free(costStart);
#endif
}
