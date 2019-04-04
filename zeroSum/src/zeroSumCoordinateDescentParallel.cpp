#include "zeroSum.h"

#define MIN_SUCCESS_RATE 0.90

void zeroSum::doFitUsingCoordinateDescentParallel(uint32_t seed) {
    // coordinate descent is only working for gamma=0!
    costFunctionAllFolds();
    uint32_t* improving = (uint32_t*)malloc(nFold1 * sizeof(uint32_t));

#ifdef DEBUG
    double* costStart = (double*)malloc(nFold1 * sizeof(double));

    double timet;
    struct timespec ts0, ts1;
    clock_gettime(CLOCK_REALTIME, &ts0);
#endif

    std::vector<std::mt19937_64> mt;
    std::uniform_real_distribution<double> rng(0.0, 1.0);

    // initialize random numbers and clear activeset
    for (uint32_t f = 0; f < nFold1; f++) {
        activeSet[f].clear();
        improving[f] = 1;
        uint32_t fold_seed = f + seed + (uint32_t)(lambda[f] * 1e3);
        mt.push_back(std::mt19937_64(fold_seed));

#ifdef DEBUG
        costStart[f] = cost[f];
        PRINT(
            "fold: %d Loglikelihood: %e lasso: %e ridge: %e cost: %e sum=%e "
            "sum%e\n",
            f, loglikelihood[f], lasso[f], ridge[f], cost[f],
            arraySumAvx(&beta[INDEX_TENSOR_COL(0, f, memory_P, K)], memory_P),
            ddot_(&memory_P, &beta[INDEX_TENSOR_COL(0, f, memory_P, K)],
                  &BLAS_I_ONE, u, &BLAS_I_ONE));
#endif
    }

    for (uint32_t steps = 0; steps < 100; steps++) {
        std::atomic<uint32_t> stop;
        stop = 0;
#ifdef DEBUG
        PRINT("Step: %d\nFind active set\n", steps);
#endif

        parallel.doParallelChunked(nFold1, [&](size_t f) {
            // if the activeset search of the previous search has not
            // changed the active set then thle optimus is reached and
            // improving is still zero and we can skip this fold
            if (improving[f] == 0 && steps > 0) {
                stop++;
                return;
            }

            for (uint32_t l = 0; l < K; l++) {
                // every move checks if the approximation has failed. Has to
                // be reset for each class, since it could have failed for
                // class l=0 but could be valid for l=1
                approxFailed[f] = false;
                if (type != gaussian)
                    refreshApproximation(f, l);

                // update the intercept
                if (useIntercept)
                    interceptMove(f, l);
            }

            parallelActiveSet[f].clear();
        });
        if (stop == nFold1)
            break;

        // Cycle over the coefficients using the update schemes.
        // In the zerosum case there are (P^2 - P) / 2 unique
        // combinations possible. If the active set has already
        // determined (steps>0), only the combinations between the
        // coefficients of the activeset with the coefficients which
        // are not in active set have to be tested. In the non
        // zero-sum case the update scheme only has to be iterated
        // over the P coefficients.
        if (useZeroSum) {
            cdMoveZS_parallel(improving, steps);
        } else {
            cdMove_parallel(improving, steps);
        }

        parallel.doParallelChunked(nFold1, [&](size_t f) {
            // if the activeset search has not changed the active set then
            // improving is still zero and we can skip this fold
            if (improving[f] == 0) {
                return;
            }

            // reset improving
            improving[f] = 0;
            for (const auto& ind : parallelActiveSet[f]) {
                if (!useZeroSum || u[ind[1]] == 0) {
                    uint32_t change = cdMove(f, ind[1], ind[0]);
                    if (change)
                        improving[f] += activeSetInsert(f, ind[1]);
                } else {
                    if (u[ind[2]] == 0)
                        continue;
                    uint32_t change = cdMoveZS(f, ind[1], ind[2], ind[0]);

                    if (change) {
                        improving[f] += activeSetInsert(f, ind[1]);
                        improving[f] += activeSetInsert(f, ind[2]);
                    }
                }
            }

            double* betaf = &beta[INDEX_TENSOR_COL(0, f, memory_P, K)];
            double* last_betaf =
                &last_beta[INDEX_TENSOR_COL(0, f, memory_P, K)];
            double* interceptf = &intercept[INDEX_COL(f, K)];
            double* last_interceptf = &last_intercept[INDEX_COL(f, K)];

#ifdef DEBUG
            PRINT("converge (fold=%d)\n", f);
#endif
            // this cost function call is neccessary to init all
            // approximation variables
            costFunction(f);

            while (true) {
                double costBefore = cost[f];
                memcpy(last_betaf, betaf, memory_P * K * sizeof(double));
                memcpy(last_interceptf, interceptf, K * sizeof(double));

                uint32_t success = 0;
                uint32_t attempts;

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

                                    // select a random element of the set
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
                    }
                }

                if (type == multinomial) {
                    optimizeParameterAmbiguity(f, 200);
                }

                costFunction(f);
                double costAfter = cost[f];
#ifdef DEBUG
                PRINT(
                    "Fold: %d Loglikelihood: %e lasso: %e ridge: %e cost: "
                    "%e "
                    "sum=%e sum=%e\tChange: costBefore=%e costAfter=%e %e "
                    "%e "
                    "(success:%d)\n",
                    f, loglikelihood[f], lasso[f], ridge[f], cost[f],
                    arraySumAvx(betaf, memory_P),
                    ddot_(&memory_P, betaf, &BLAS_I_ONE, u, &BLAS_I_ONE),
                    costBefore, costAfter, fabs(costAfter - costBefore),
                    precision, success);
#endif

                // if due to numerical instabilities the costfunction gets
                // nan or the quality of the model decreases, restore the
                // previous configuration and stop the optimization
                if (std::isnan(costAfter) || costBefore < costAfter) {
                    memcpy(betaf, last_betaf, memory_P * K * sizeof(double));
                    memcpy(interceptf, last_interceptf, K * sizeof(double));
                    costFunction(f);
                    break;
                }

                if (success == 0 || fabs(costAfter - costBefore) < precision)
                    break;
            }

            // polish (local search) to get small coefs where the cd is
            // undefined to zero
            if (usePolish) {
                // cost function has to be executed to init all ls variables
                costFunction(f);

                for (uint32_t l = 0; l < K; l++) {
                    // get a pointer to the current coefficients for easier
                    // access
                    double* betaPtr =
                        &beta[INDEX_TENSOR_COL(l, f, memory_P, K)];
                    if (useApprox)
                        refreshApproximation(f, l, true);

                    for (const uint32_t& s : activeSet[f]) {
                        // if beta_s is zero skip -> we dont want a zero
                        // coefficient as target. The local search move
                        // maintains the zerosum contraint
                        // -> dont apply the move on coefficients which are
                        // not in the zerosum constraint
                        if (fabs(betaPtr[s]) < DBL_EPSILON * 100 || u[s] == 0)
                            continue;

                        for (const uint32_t& k : activeSet[f]) {
                            // skip if s==k since the move is not defined
                            // for this case. Also, skip if beta_k is 0
                            if (s == k ||
                                fabs(betaPtr[k]) < DBL_EPSILON * 100 ||
                                u[k] == 0)
                                continue;
                            // try to move the amount of beta_k to beta_s.
                            // If it does not harm the cost function the
                            // move is performed.
                            lsSaMove(f, k, s, l, -betaPtr[k]);
                        }
                    }
                }
                activeSetRemoveZeros(f);
            }
        });
    }

    if (type == multinomial) {
        parallel.doParallelChunked(
            nFold1, [&](size_t f) { optimizeParameterAmbiguity(f, 200); });
    }

    parallel.doParallelChunked(memory_P * K * nFold1, [&](size_t j) {
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
    free(improving);
}
