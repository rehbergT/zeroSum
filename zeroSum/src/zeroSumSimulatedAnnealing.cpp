#include "zeroSum.h"

// simulatedAnnealing settings
#define MAX_STEPS 500
#define INTERVAL_SIZE 0.2
#define INTERVAL_SHRINK 0.9

#define T_STEPS 10000
#define T_START 0.2
#define ACCEPTRATE 0.5

#define MEASURE 500
#define THERMALIZE 300
#define COOLING_FAKTOR 0.8

double mean(double* a, uint32_t n) {
    double sum = 0.0;
    for (uint32_t i = 0; i < n; ++i)
        sum += a[i];
    sum /= (double)n;

    return sum;
}

double sd(double* a, uint32_t n, double* mean_ptr) {
    double m = mean_ptr == nullptr ? *mean_ptr : mean(a, n);
    double squareA = 0.0;
    for (uint32_t i = 0; i < n; ++i)
        squareA += a[i] * a[i];
    double s = squareA / (double)n;
    s = s - m * m;
    return sqrt(s);
}

void zeroSum::doFitUsingSimulatedAnnealing(uint32_t seed) {
    std::vector<std::mt19937_64> mt;
    std::uniform_real_distribution<double> rng(0.0, 1.0);
    for (uint32_t f = 0; f < nFold1; f++) {
        uint32_t fold_seed = f + seed + (uint32_t)((lambda[f] + gamma) * 1e3);
        mt.push_back(std::mt19937_64(fold_seed));
    }

    threadPool.doParallelChunked(nFold1, [&](size_t f) {
        costFunction(f);
        double* betaf = &beta[INDEX_TENSOR_COL(0, f, memory_P, K)];
        double* interceptf = &intercept[INDEX_COL(f, K)];

#ifdef DEBUG
        double timet, costStart, costEnd;
        struct timespec ts0, ts1;
        clock_gettime(CLOCK_REALTIME, &ts0);
        PRINT("Starting SA\n");
        costStart = cost[f];

        PRINT("Loglikelihood: %e lasso: %e ridge: %e cost: %e\n",
              loglikelihood[f], lasso[f], ridge[f], cost[f]);
#endif

        uint32_t k = 0, s = 0, t;
        double delta_k = 0.0, costOld = 0.0;

        double temperature = 0.0;
        double* betaPtr;
        double rnd = 0.0;
        uint32_t attempts = 0, success = 0;
        uint32_t sweep = useZeroSum ? P * (P - 1) : P;

        // array for storing best solution
        double best_cost = DBL_MAX;

        double* best_beta = &last_beta[INDEX_TENSOR_COL(0, f, memory_P, K)];
        double* best_intercept = &last_intercept[INDEX_COL(f, K)];

        for (uint32_t i = 0; i < T_STEPS; ++i) {
            for (uint32_t l = 0; l < K; l++) {
                betaPtr = &betaf[INDEX_COL(l, memory_P)];
                if (useApprox)
                    refreshApproximation(f, l, true);
                if (useIntercept)
                    lsSaOffsetMove(f, l);

                costOld = cost[f];
                k = floor(rng(mt[f]) * P);

                if (useZeroSum) {
                    s = floor(rng(mt[f]) * P);
                    if (s == k)
                        continue;
                }

                attempts++;

                if (i % 20 == 0) {
                    // try to get beta_k to zero
                    delta_k = -betaPtr[k];
                } else {
                    // choose a random amount
                    delta_k = rng(mt[f]) * INTERVAL_SIZE - INTERVAL_SIZE * 0.5;
                }
                t = lsSaMove(f, k, s, l, delta_k, &rnd, DBL_MAX);

                if (t != 0 && useIntercept)
                    lsSaOffsetMove(f, l);

                if (cost[f] > costOld) {
                    temperature += cost[f] - costOld;
                    success++;
                }

                if (cost[f] < best_cost) {
                    best_cost = cost[f];
                    memcpy(best_beta, betaf, memory_P * K * sizeof(double));
                    memcpy(best_intercept, interceptf, K * sizeof(double));
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
            cost[f], loglikelihood[f], ridge[f], lasso[f]);
        double average_Eng[MEASURE];
        uint32_t ii = 0;
#endif

        // reset beta (better for sparse solution)
        memset(betaf, 0, memory_P * K * sizeof(double));
        costFunction(f);

        for (uint32_t step = 0; step < MAX_STEPS; step++) {
            costOld = cost[f];

            attempts = 0;
            success = 0;
            for (uint32_t i = 0; i < THERMALIZE + MEASURE; ++i) {
                for (uint32_t l = 0; l < K; l++) {
                    betaPtr = &betaf[INDEX_COL(l, memory_P)];
                    if (useApprox)
                        refreshApproximation(f, l, true);

                    if (useIntercept)
                        lsSaOffsetMove(f, l);
                    for (uint32_t i = 0; i < sweep * downScaler; i++) {
                        k = floor(rng(mt[f]) * P);

                        if (useZeroSum) {
                            s = floor(rng(mt[f]) * P);
                            if (s == k)
                                continue;
                        }

                        attempts++;

                        if (i % 20 == 0) {
                            // try to get beta_k to zero
                            delta_k = -betaPtr[k];
                        } else {
                            // choose a random amount
                            delta_k = rng(mt[f]) * INTERVAL_SIZE -
                                      INTERVAL_SIZE * 0.5;
                        }

                        rnd = rng(mt[f]);
                        t = lsSaMove(f, k, s, l, delta_k, &rnd, temperature);

                        if (t != 0)
                            success++;

                        if (cost[f] < best_cost) {
                            best_cost = cost[f];
                            memcpy(best_beta, betaf,
                                   memory_P * K * sizeof(double));
                            memcpy(best_intercept, interceptf,
                                   K * sizeof(double));
                        }
                    }
                }
#ifdef DEBUG
                if (i >= THERMALIZE) {
                    average_Eng[ii] = cost[f];
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
                temperature, mean_E, sd_E, acceptrate, best_cost, acceptrate,
                acceptrate > precision, acceptrate);
#endif

            // call cost function to prevent numerical uncertainties
            if (type == multinomial) {
                optimizeParameterAmbiguity(f, 100);
            }
            costFunction(f);

            temperature *= COOLING_FAKTOR;

            if (cost[f] - costOld > -precision && step > 4)
                break;
        }

        memcpy(betaf, best_beta, memory_P * K * sizeof(double));
        memcpy(interceptf, best_intercept, K * sizeof(double));

        costFunction(f);

        for (uint32_t j = 0; j < K * memory_P; j++)
            if (fabs(betaf[j]) < 100 * DBL_EPSILON)
                betaf[j] = 0.0;

#ifdef DEBUG
        costFunction(f);
        costEnd = cost[f];

        PRINT("Cost start: %e cost end: %e diff %e\n", costStart, costEnd,
              costStart - costEnd);

        clock_gettime(CLOCK_REALTIME, &ts1);
        timet = (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;
        PRINT("time taken = %e s\n", timet);
#endif
    });
}
