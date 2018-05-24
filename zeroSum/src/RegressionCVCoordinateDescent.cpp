#include "RegressionCV.h"

#define MIN_SUCCESS_RATE 0.90

void printActiveSet(std::vector<int> activeSet) {
    std::sort(activeSet.begin(), activeSet.end());
    for (auto i = 0u; i < activeSet.size(); ++i) {
        printf("%d ", activeSet[i]);
    }
    printf("\n");
}

void RegressionCV::coordinateDescent(int seed) {
    // coordinate descent is only working for gamma=0!
    // -> use a reference to the gamma=0 container
    std::vector<CvRegressionData>& data = cv_data[0];

#pragma omp parallel for schedule(dynamic)
    for (int f = 0; f < nFold + 1; f++) {
#ifdef DEBUG
        double timet, costStart, costEnd;
        struct timespec ts0, ts1;
        clock_gettime(CLOCK_REALTIME, &ts0);
#endif

        data[f].costFunction();

#ifdef DEBUG
        costStart = data[f].cost;
        PRINT("Loglikelihood: %e lasso: %e ridge: %e cost: %e sum=%e sum=%e\n",
              data[f].loglikelihood, data[f].lasso, data[f].ridge, data[f].cost,
              sum_a(data[f].beta, P),
              sum_a_times_b(data[f].beta, data[f].u, P));
#endif

        data[f].activeSet.clear();

        int success, attempts;
        int gettingWorse = FALSE;

        double* last_beta = (double*)malloc(memory_P * K * sizeof(double));
        double* last_offset = (double*)malloc(K * sizeof(double));

        seed = f + seed + (int)(data[f].lambda * 1e3);
        std::mt19937_64 mt(seed);
        std::uniform_real_distribution<double> rng(0.0, 1.0);

        for (int steps = 0; steps < 100 && !gettingWorse; steps++) {
#ifdef DEBUG
            PRINT("Step: %d\nFind active set\n", steps);
#endif
            data[f].removeCoefsWithZeroFromActiveSet();
            int activeSetChange = 0;

            double costBefore = data[f].cost;
            memcpy(last_beta, data[f].beta, memory_P * K * sizeof(double));
            memcpy(last_offset, data[f].offset, K * sizeof(double));

            // cycle over all multinomial classes (if type != multinomial K=1)
            for (int l = 0; l < K; l++) {
                // every move checks if the approximation has failed. Has to be
                // reset for each class, since it could have failed for class
                // l=0 but could be valid for l=1
                data[f].approxFailed = FALSE;
                if (type >= BINOMIAL)
                    data[f].refreshApproximation(l);

                // update the offset
                if (data[f].useOffset)
                    data[f].offsetMove(l);

                // Cycle over the coefficients using the update schemes.
                // In the zerosum case there are (P^2 - P) / 2 unique
                // combinations possible. If the active set has already
                // determined (steps>0), only the combinations between the
                // coefficients of the activeset with the coefficients which are
                // not in active set have to be tested. In the non zero-sum case
                // the update scheme only has to be iterated over the P
                // coefficients.
                if (data[f].isZeroSum) {
                    for (int s = 0; s < P; s++) {
                        // if the coefficients have a zero-sum weight (u) of 0
                        // we have to apply the cdMove. We can
                        // then skip the quadratic update search for beta_s
                        if (data[f].u[s] == 0) {
                            int change = data[f].cdMove(s, l);
                            if (change)
                                activeSetChange += data[f].checkActiveSet(s);
                            continue;
                        }

                        if (steps > 0) {
                            // if step>0 the loop with int s should
                            // only go over the non active set coefficients ->
                            // skip if s is in the active set
                            if (data[f].isInActiveSet(s))
                                continue;

                            // second coefficient should be of the activeset
                            for (auto h = 0u; h < data[f].activeSet.size();
                                 h++) {
                                int k = data[f].activeSet[h];
                                // zero-sum weight u_k = 0 means that this coef
                                // has to be updated with cdMove. But this is
                                // done by the int s loop -> just skipt here
                                if (data[f].u[k] == 0)
                                    continue;

                                int change = data[f].cdMoveZS(s, k, l);

                                if (change) {
                                    activeSetChange +=
                                        data[f].checkActiveSet(k);
                                    activeSetChange +=
                                        data[f].checkActiveSet(s);
                                }
                            }
                        } else {
                            // cycle over all (P^2 - P) / 2 unique combinations
                            for (int k = s + 1; k < P; k++) {
                                // zero-sum weight u_k = 0 means that this coef
                                // has to be updated with cdMove. But this is
                                // done by the int s loop -> just skipt here
                                if (data[f].u[k] == 0)
                                    continue;

                                int change = data[f].cdMoveZS(s, k, l);

                                if (change) {
                                    activeSetChange +=
                                        data[f].checkActiveSet(k);
                                    activeSetChange +=
                                        data[f].checkActiveSet(s);
                                }
                            }
                        }
                    }
                } else {
                    for (int s = 0; s < P; s++) {
                        int change = data[f].cdMove(s, l);
                        if (change)
                            activeSetChange += data[f].checkActiveSet(s);
                    }
                }

                if (type == MULTINOMIAL || type == MULTINOMIAL_ZS) {
                    data[f].optimizeParameterAmbiguity(200);
                }
            }
            // polish (local search) to get small coefs where the cd is
            // undefined to zero
            if (data[f].polish) {
                // the polish can operate on the actual loglikelihood if the
                // approximation has failed.
                int useApproxBak = data[f].useApprox;
                if (data[f].approxFailed)
                    data[f].useApprox = FALSE;

                // cost function has to be executed to init all ls variables
                data[f].costFunction();

                for (int l = 0; l < K; l++) {
                    // get a pointer to the current coefficients for easier
                    // access
                    double* betaPtr = &data[f].beta[INDEX(0, l, memory_P)];
                    if (data[f].useApprox)
                        data[f].refreshApproximation(l, TRUE);

                    for (const int& s : data[f].activeSet) {
                        // if beta_s is zero skip -> we dont want a zero
                        // coefficient as target. The local search move
                        // maintains the zerosum contraint
                        // -> dont apply the move on coefficients which are not
                        // in the zerosum constraint
                        if (fabs(betaPtr[s]) < DBL_EPSILON * 100 ||
                            data[f].u[s] == 0)
                            continue;
                        for (const int& k : data[f].activeSet) {
                            // skip if s==k since the move is not defined for
                            // this case. Also, skip if beta_k is 0
                            if (s == k ||
                                fabs(betaPtr[k]) < DBL_EPSILON * 100 ||
                                data[f].u[k] == 0)
                                continue;
                            // try to move the amount of beta_k to beta_s.
                            // If it does not harm the cost function the move is
                            // performed
                            data[f].lsSaMove(k, s, l, -betaPtr[k]);
                        }
                    }
                }

                // restore the original setting
                data[f].useApprox = useApproxBak;
            }

#ifdef DEBUG
            printActiveSet(data[f].activeSet);
#endif
            // this cost function call is neccessary to init all approximation
            // variables
            data[f].costFunction();
            double costAfter = data[f].cost;

            // if due to numercial instabilities the costfunction gets nan or
            // the quality of the model decreases, restore the previous
            // configuration and stop the optimization
            if (std::isnan(costAfter) || costBefore < costAfter) {
                memcpy(data[f].beta, last_beta, memory_P * K * sizeof(double));
                memcpy(data[f].offset, last_offset, K * sizeof(double));
                data[f].costFunction();
                break;
            }

            // if the activeset search has not changed the active set or if the
            // active set is empty then stop
            if (activeSetChange == 0 || data[f].activeSet.empty()) {
                break;
            }

#ifdef DEBUG
            PRINT("converge\n");
#endif

            while (TRUE) {
                data[f].costFunction();
                costBefore = data[f].cost;
                memcpy(last_beta, data[f].beta, memory_P * K * sizeof(double));
                memcpy(last_offset, data[f].offset, K * sizeof(double));

                success = 0;

                for (int l = 0; l < K; l++) {
                    data[f].approxFailed = FALSE;
                    if (type >= BINOMIAL)
                        data[f].refreshApproximation(l);
                    if (data[f].useOffset)
                        data[f].offsetMove(l);

                    if (data[f].isZeroSum) {
                        for (const int& s : data[f].activeSet) {
                            if (data[f].u[s] == 0) {
                                success += data[f].cdMove(s, l);
                                continue;
                            }
                            for (const int& k : data[f].activeSet) {
                                if (s == k || data[f].u[k] == 0)
                                    continue;
                                success += data[f].cdMoveZS(s, k, l);
                            }
                        }
                        attempts = data[f].activeSet.size() *
                                   (data[f].activeSet.size() - 1);

                        if (data[f].diagonalMoves &&
                            (double)success / (double)attempts <=
                                MIN_SUCCESS_RATE) {
                            for (const int& s : data[f].activeSet) {
                                if (data[f].u[s] == 0)
                                    continue;

                                for (const int& k : data[f].activeSet) {
                                    if (s == k || data[f].u[k] == 0)
                                        continue;

                                    int h = floor(rng(mt) *
                                                  data[f].activeSet.size());
                                    h = data[f].activeSet[h];

                                    if (h == s || h == k || data[f].u[h] == 0)
                                        continue;

                                    data[f].cdMoveZSRotated(s, k, h, l,
                                                            rng(mt) * M_PI);
                                }
                            }
                        }

                    } else {
                        for (const int& s : data[f].activeSet) {
                            success += data[f].cdMove(s, l);
                        }
                    };

                    if (data[f].isZeroSum) {
                    }
                }

                if (type == MULTINOMIAL || type == MULTINOMIAL_ZS) {
                    data[f].optimizeParameterAmbiguity(200);
                }

                data[f].costFunction();
                costAfter = data[f].cost;
#ifdef DEBUG
                PRINT(
                    "Loglikelihood: %e lasso: %e ridge: %e cost: %e sum=%e "
                    "sum=%e\tChange: costBefore=%e costAfter=%e %e %e "
                    "(success:%d)\n",
                    data[f].loglikelihood, data[f].lasso, data[f].ridge,
                    data[f].cost, sum_a(data[f].beta, P),
                    sum_a_times_b(data[f].beta, data[f].u, P), costBefore,
                    costAfter, fabs(costAfter - costBefore), data[f].precision,
                    success);
#endif
                if (std::isnan(costAfter) || costBefore < costAfter) {
                    gettingWorse = TRUE;
                    memcpy(data[f].beta, last_beta,
                           memory_P * K * sizeof(double));
                    memcpy(data[f].offset, last_offset, K * sizeof(double));
                    data[f].costFunction();
                    break;
                }

                if (success == 0 ||
                    fabs(costAfter - costBefore) < data[f].precision)
                    break;
            }
        }

        if (type == MULTINOMIAL || type == MULTINOMIAL_ZS) {
            data[f].optimizeParameterAmbiguity(200);
        }

        for (int j = 0; j < K * memory_P; j++)
            if (fabs(data[f].beta[j]) < 100 * DBL_EPSILON)
                data[f].beta[j] = 0.0;

        free(last_beta);
        free(last_offset);
#ifdef DEBUG
        data[f].costFunction();
        costEnd = data[f].cost;

        PRINT("Cost start: %e cost end: %e diff %e\n", costStart, costEnd,
              costEnd - costStart);

        clock_gettime(CLOCK_REALTIME, &ts1);
        timet = (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;
        PRINT("time taken = %e s\n", timet);
#endif
    }
}
