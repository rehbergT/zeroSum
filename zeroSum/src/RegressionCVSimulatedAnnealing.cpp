#include "RegressionCV.h"

void RegressionCV::simulatedAnnealing(int seed) {
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int g = lengthGamma - 1; g >= 0; g--) {
        for (int f = 0; f < nFold + 1; f++) {
            cv_data[g][f].costFunction();

#ifdef DEBUG
            double timet, costStart, costEnd;
            struct timespec ts0, ts1;
            clock_gettime(CLOCK_REALTIME, &ts0);
            PRINT("Starting SA\n");
            costStart = cv_data[g][f].cost;

            PRINT("Loglikelihood: %e lasso: %e ridge: %e cost: %e\n",
                  cv_data[g][f].loglikelihood, cv_data[g][f].lasso,
                  cv_data[g][f].ridge, cv_data[g][f].cost);
#endif

            int k = 0, s = 0, t;
            double delta_k = 0.0, costOld = 0.0;

            seed = f + seed +
                   (int)((cv_data[g][f].lambda + cv_data[g][f].gamma) * 1e3);
            std::mt19937_64 mt(seed);
            std::uniform_real_distribution<double> rng(0.0, 1.0);

            double temperature = 0.0;
            double* betaPtr;
            double rnd = 0.0;
            int attempts = 0, success = 0;
            int sweep = cv_data[g][f].isZeroSum ? P * (P - 1) : P;

            // array for storing best solution
            double best_cost = DBL_MAX;

#ifdef AVX_VERSION
            double* best_beta = (double*)aligned_alloc(
                ALIGNMENT, memory_P * K * sizeof(double));
            double* best_offset =
                (double*)aligned_alloc(ALIGNMENT, K * sizeof(double));
#else
            double* best_beta = (double*)malloc(memory_P * K * sizeof(double));
            double* best_offset = (double*)malloc(K * sizeof(double));
#endif

            for (int i = 0; i < T_STEPS; ++i) {
                for (int l = 0; l < K; l++) {
                    betaPtr = &cv_data[g][f].beta[INDEX(0, l, memory_P)];
                    if (cv_data[g][f].useApprox)
                        cv_data[g][f].refreshApproximation(l, TRUE);
                    if (cv_data[g][f].useOffset)
                        cv_data[g][f].lsSaOffsetMove(l);

                    costOld = cv_data[g][f].cost;
                    k = floor(rng(mt) * P);

                    if (cv_data[g][f].isZeroSum) {
                        s = floor(rng(mt) * P);
                        if (s == k)
                            continue;
                    }

                    attempts++;

                    if (i % 20 == 0) {
                        // try to get beta_k to zero
                        delta_k = -betaPtr[k];
                    } else {
                        // choose a random amount
                        delta_k = rng(mt) * INTERVAL_SIZE - INTERVAL_SIZE * 0.5;
                    }
                    t = cv_data[g][f].lsSaMove(k, s, l, delta_k, &rnd, DBL_MAX);

                    if (t != 0) {
                        if (cv_data[g][f].useOffset)
                            cv_data[g][f].lsSaOffsetMove(l);
                    }

                    if (cv_data[g][f].cost > costOld) {
                        temperature += cv_data[g][f].cost - costOld;
                        success++;
                    }

                    if (cv_data[g][f].cost < best_cost) {
                        best_cost = cv_data[g][f].cost;
                        memcpy(best_beta, cv_data[g][f].beta,
                               memory_P * K * sizeof(double));
                        memcpy(best_offset, cv_data[g][f].offset,
                               K * sizeof(double));
                    }
                }
            }
            temperature /= (double)(success);
            temperature = -temperature / (log(T_START));

#ifdef DEBUG
            PRINT("Estimated Temperature: %e\n", temperature);
            PRINT(
                "Initial Energy: %e (loglikelihood: %e ridge term: %e, lasso: "
                "%e)\n",
                cv_data[g][f].cost, cv_data[g][f].loglikelihood,
                cv_data[g][f].ridge, cv_data[g][f].lasso);
            double average_Eng[MEASURE];
            int ii = 0;
#endif

            // reset beta (better for sparse solution)
            memset(cv_data[g][f].beta, 0, memory_P * K * sizeof(double));
            cv_data[g][f].costFunction();

            for (int step = 0; step < MAX_STEPS; step++) {
                costOld = cv_data[g][f].cost;

                attempts = 0;
                success = 0;
                for (int i = 0; i < THERMALIZE + MEASURE; ++i) {
                    for (int l = 0; l < K; l++) {
                        betaPtr = &cv_data[g][f].beta[INDEX(0, l, memory_P)];
                        if (cv_data[g][f].useApprox)
                            cv_data[g][f].refreshApproximation(l, TRUE);

                        if (cv_data[g][f].useOffset)
                            cv_data[g][f].lsSaOffsetMove(l);
                        for (int i = 0; i < sweep * cv_data[g][f].downScaler;
                             i++) {
                            k = floor(rng(mt) * P);

                            if (cv_data[g][f].isZeroSum) {
                                s = floor(rng(mt) * P);
                                if (s == k)
                                    continue;
                            }

                            attempts++;

                            if (i % 20 == 0) {
                                // try to get beta_k to zero
                                delta_k = -betaPtr[k];
                            } else {
                                // choose a random amount
                                delta_k = rng(mt) * INTERVAL_SIZE -
                                          INTERVAL_SIZE * 0.5;
                            }

                            rnd = rng(mt);
                            t = cv_data[g][f].lsSaMove(k, s, l, delta_k, &rnd,
                                                       temperature);

                            if (t != 0)
                                success++;

                            if (cv_data[g][f].cost < best_cost) {
                                best_cost = cv_data[g][f].cost;
                                memcpy(best_beta, cv_data[g][f].beta,
                                       memory_P * K * sizeof(double));
                                memcpy(best_offset, cv_data[g][f].offset,
                                       K * sizeof(double));
                            }
                        }
                    }
#ifdef DEBUG
                    if (i >= THERMALIZE) {
                        average_Eng[ii] = cv_data[g][f].cost;
                        ++ii;
                    }
#endif
                }

#ifdef DEBUG
                double acceptrate = (double)success / (double)attempts;
                double mean_E = mean(average_Eng, MEASURE);
                double sd_E = sd(average_Eng, MEASURE, &mean_E);
                PRINT(
                    "temperatur %e energy: %e var eng %e  accept: %f best E: "
                    "%e, "
                    "crit=%e   test=%d accept: %f\n",
                    temperature, mean_E, sd_E, acceptrate, best_cost,
                    acceptrate, acceptrate > cv_data[g][f].precision,
                    acceptrate);
#endif

                // call cost function to prevent numerical uncertainties
                if (type == MULTINOMIAL || type == MULTINOMIAL_ZS) {
                    cv_data[g][f].optimizeParameterAmbiguity(100);
                }
                cv_data[g][f].costFunction();

                temperature *= COOLING_FAKTOR;

                if (cv_data[g][f].cost - costOld > -cv_data[g][f].precision &&
                    step > 4)
                    break;
            }

            memcpy(cv_data[g][f].beta, best_beta,
                   memory_P * K * sizeof(double));
            memcpy(cv_data[g][f].offset, best_offset, K * sizeof(double));

            cv_data[g][f].costFunction();

            for (int j = 0; j < K * memory_P; j++)
                if (fabs(cv_data[g][f].beta[j]) < 100 * DBL_EPSILON)
                    cv_data[g][f].beta[j] = 0.0;

            free(best_beta);
            free(best_offset);

#ifdef DEBUG
            cv_data[g][f].costFunction();
            costEnd = cv_data[g][f].cost;

            PRINT("Cost start: %e cost end: %e diff %e\n", costStart, costEnd,
                  costStart - costEnd);

            clock_gettime(CLOCK_REALTIME, &ts1);
            timet =
                (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;
            PRINT("time taken = %e s\n", timet);
#endif
        }
    }
}
