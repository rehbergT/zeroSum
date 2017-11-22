#include "RegressionDataScheme.h"

void RegressionDataScheme::simulatedAnnealing(int seed) {
    costFunction();

#ifdef DEBUG
    double timet, costStart, costEnd;
    struct timespec ts0, ts1;
    clock_gettime(CLOCK_REALTIME, &ts0);
    PRINT("Starting SA\n");
    costStart = cost;

    PRINT("Loglikelihood: %e lasso: %e ridge: %e cost: %e\n", loglikelihood,
          lasso, ridge, cost);
#endif

    int k = 0, s = 0, t;
    double delta_k = 0.0, costOld = 0.0;

    std::mt19937_64 mt(seed);
    std::uniform_real_distribution<double> rng(0.0, 1.0);

    double temperature = 0.0;
    double* betaPtr;
    double rnd = 0.0;
    int attempts = 0, success = 0;
    int sweep = isZeroSum ? P * (P - 1) : P;

    // array for storing best solution
    double best_cost = DBL_MAX;

#ifdef AVX_VERSION
    double* best_beta =
        (double*)aligned_alloc(ALIGNMENT, memory_P * K * sizeof(double));
    double* best_offset = (double*)aligned_alloc(ALIGNMENT, K * sizeof(double));
#else
    double* best_beta = (double*)malloc(memory_P * K * sizeof(double));
    double* best_offset = (double*)malloc(K * sizeof(double));
#endif

    for (int i = 0; i < T_STEPS; ++i) {
        for (int l = 0; l < K; l++) {
            betaPtr = &beta[INDEX(0, l, memory_P)];
            if (useApprox)
                refreshApproximation(l, TRUE);
            if (useOffset)
                lsSaOffsetMove(l);

            costOld = cost;
            k = floor(rng(mt) * P);

            if (isZeroSum) {
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
            t = lsSaMove(k, s, l, delta_k, &rnd, DBL_MAX);

            if (cost > costOld) {
                temperature += cost - costOld;
                success++;
            }

            if (cost < best_cost) {
                best_cost = cost;
                memcpy(best_beta, beta, memory_P * K * sizeof(double));
                memcpy(best_offset, offset, K * sizeof(double));
            }
        }
    }
    temperature /= (double)(success);
    temperature = -temperature / (log(T_START));

#ifdef DEBUG
    PRINT("Estimated Temperature: %e\n", temperature);
    PRINT("Initial Energy: %e (loglikelihood: %e ridge term: %e, lasso: %e)\n",
          cost, loglikelihood, ridge, lasso);
    double average_Eng[MEASURE];
    int ii = 0;
#endif

    // reset beta (better for sparse solution)
    memset(beta, 0, memory_P * K * sizeof(double));
    costFunction();

    for (int step = 0; step < MAX_STEPS; step++) {
        costOld = cost;

        attempts = 0;
        success = 0;
        for (int i = 0; i < THERMALIZE + MEASURE; ++i) {
            for (int l = 0; l < K; l++) {
                betaPtr = &beta[INDEX(0, l, memory_P)];
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

                    if (i % 20 == 0) {
                        // try to get beta_k to zero
                        delta_k = -betaPtr[k];
                    } else {
                        // choose a random amount
                        delta_k = rng(mt) * INTERVAL_SIZE - INTERVAL_SIZE * 0.5;
                    }

                    rnd = rng(mt);
                    t = lsSaMove(k, s, l, delta_k, &rnd, temperature);

                    if (t != 0)
                        success++;

                    if (cost < best_cost) {
                        best_cost = cost;
                        memcpy(best_beta, beta, memory_P * K * sizeof(double));
                        memcpy(best_offset, offset, K * sizeof(double));
                    }
                }
            }
#ifdef DEBUG
            if (i >= THERMALIZE) {
                average_Eng[ii] = cost;
                ++ii;
            }
#endif
        }

#ifdef DEBUG
        double acceptrate = (double)success / (double)attempts;
        double mean_E = mean(average_Eng, MEASURE);
        double sd_E = sd(average_Eng, MEASURE, &mean_E);
        PRINT(
            "temperatur %e energy: %e var eng %e  accept: %f best E: %e, "
            "crit=%e   test=%d accept: %d\n",
            temperature, mean_E, sd_E, acceptrate, best_cost, acceptrate,
            acceptrate > precision, acceptrate);
#endif

        // call cost function to prevent numerical uncertainties
        costFunction();

        temperature *= COOLING_FAKTOR;
        if (cost - costOld > -precision && step > 4)
            break;
    }

    memcpy(beta, best_beta, memory_P * K * sizeof(double));
    memcpy(offset, best_offset, K * sizeof(double));

    costFunction();

    free(best_beta);

#ifdef DEBUG
    costFunction();
    costEnd = cost;

    PRINT("Cost start: %e cost end: %e diff %e\n", costStart, costEnd,
          costStart - costEnd);

    clock_gettime(CLOCK_REALTIME, &ts1);
    timet = (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;
    PRINT("time taken = %e s\n", timet);
#endif
}
