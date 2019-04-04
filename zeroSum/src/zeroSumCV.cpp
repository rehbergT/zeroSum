#include "zeroSum.h"

#ifdef R_PACKAGE
inline void checkInterrupt(void* dummy) {
    (void)dummy;
    R_CheckUserInterrupt();
}

bool checkInterrupt() {
    if (R_ToplevelExec(checkInterrupt, NULL) == false)
        return true;
    else
        return false;
}
#endif

std::vector<double> zeroSum::doCVRegression(char* path,
                                            char* name,
                                            uint32_t mpi_rank) {
    if (verbose) {
        if (avxType == avx2) {
            PRINT("Using AVX2\n");
        } else if (avxType == avx512) {
            PRINT("Using AVX512\n");
        }

        uint32_t threads_max = std::thread::hardware_concurrency();
        PRINT("Using %d threads (max. parallel executable: %d)\n", threads,
              threads_max);
    }

#ifdef DEBUG
    PRINT("x:\n");
    printMatrix(x, memory_N, P);
    PRINT("y:\n");
    printMatrix(y, N, K);
    PRINT("w:\n");
    printMatrix(w, N, 1);
    PRINT("u:\n");
    printMatrix(u, P, 1);
    PRINT("v:\n");
    printMatrix(v, P, 1);
    PRINT("lambda:\n");
    printMatrix(lambdaSeq.data(), lambdaSeq.size(), 1);
    PRINT("gamma size: %lu\n", gammaSeq.size());
    printMatrix(gammaSeq.data(), gammaSeq.size(), 1);
    PRINT("fold id:\n");
    printIntMatrix(foldid.data(), foldid.size(), 1);
    if (useFusion)
        printSparseMatrix(fusionKernel, P);
#endif

    adjustWeights();
    if (type == cox)
        calcCoxRegressionD();
    std::vector<double> cv_stats;
    std::vector<double> cv_tmp;
    cv_tmp.resize(nFold1);
    std::vector<double> cv_predict;
    cv_predict.resize(K * N);
    double lastCV = DBL_MAX;
    uint32_t worseLambdaSteps = 0;

    if (useCentering)
        meanCentering();

    // always standardize the response in the gaussian case
    if (type == 1)
        standardizeResponse();

    if (useStandardization)
        standardizeData();

    for (auto g = 0u; g < gammaSeq.size(); g++) {
        if (g > 0)
            memset(beta, 0.0, memory_P * K * nFold1 * sizeof(double));

        gamma = gammaSeq[g];

        for (auto j = 0u; j < lambdaSeq.size(); j++) {
#ifdef R_PACKAGE
            if (checkInterrupt()) {
                return cv_stats;
            }
#endif
            for (uint32_t f = 0; f < nFold1; f++) {
                lambda[f] = lambdaSeq[j];
                lambda[f] *= ySD[f];
            }

            if (algorithm == coodinateDescent) {
                // this case does not support fusion -> assumes gamma=0
                doFitUsingCoordinateDescent(seed);
            } else if (algorithm == simulatedAnnealing) {
                doFitUsingSimulatedAnnealing(seed);
                doFitUsingLocalSearch(seed);
            } else if (algorithm == localSearch) {
                doFitUsingLocalSearch(seed);
            } else if (algorithm == coodinateDescentParallel) {
                doFitUsingCoordinateDescentParallel(seed);
            }

            double* tmp = wOrg;
            wOrg = wCV;
            if (type == cox)
                calcCoxRegressionD();

            costFunctionAllFolds();

            wOrg = tmp;
            if (type == cox)
                calcCoxRegressionD();

            // calc mean-squared-error or deviance
            for (uint32_t f = 0; f < nFold1; f++)
                cv_tmp[f] = -2.0 * loglikelihood[f];

            predict();

            for (uint32_t f = 0; f < nFold; f++) {
                for (uint32_t i = 0; i < N; i++) {
                    if (wCV[INDEX(i, f, memory_N)] != 0.0) {
                        for (uint32_t l = 0; l < K; ++l)
                            cv_predict[INDEX(i, l, N)] =
                                xTimesBeta[INDEX_TENSOR(i, l, f, memory_N, K)];
                    }
                }
            }

            uint32_t cvImproving = 0;
            double cvError, cvErrorSD;

            double trainingError = cv_tmp[nFold];

            cvError = 0.0;
            for (uint32_t f = 0; f < nFold; f++)
                cvError += cv_tmp[f];

            if (std::isnan(cvError) || std::isinf(cvError) ||
                std::isnan(trainingError) || std::isinf(trainingError))
                break;

            if (nFold != 0)
                cvError /= nFold;

            cvErrorSD = 0.0;
            for (uint32_t f = 0; f < nFold; f++)
                cvErrorSD += cv_tmp[f] * cv_tmp[f];

            cvErrorSD /= (double)nFold;

            cvErrorSD -= cvError * cvError;
            cvErrorSD /= (double)nFold - 1.0;
            cvErrorSD = sqrt(cvErrorSD);

            if (cvError - lastCV > DBL_EPSILON)
                cvImproving++;

            lastCV = cvError;

            if (std::isnan(cvErrorSD))
                cvErrorSD = -1.0;

            if (path != NULL) {
                char outPath[MAX_LINE_LENGTH] = "";
                FILE* file;

                snprintf(outPath, sizeof(outPath), "%s%s_%d_stats.csv", path,
                         name, mpi_rank);

                file = fopen(outPath, "a");

                fprintf(file, "%a,%a,%a,%a,%a,", gamma,
                        lambda[nFold] / ySD[nFold], trainingError, cvError,
                        cvErrorSD);

                for (uint32_t k = 0; k < K; k++) {
                    double* betaf =
                        &beta[INDEX_TENSOR_COL(k, nFold, memory_P, K)];
                    double tmpOffset = intercept[INDEX(k, nFold, K)];

                    if (useIntercept && useCentering)
                        tmpOffset -= ddot_(&memory_P, betaf, &BLAS_I_ONE,
                                           featureMean, &BLAS_I_ONE);

                    fprintf(file, "%a,", tmpOffset);

                    for (uint32_t p = 0; p < P; p++)
                        fprintf(file, "%a,", betaf[p]);
                }

                for (uint32_t k = 0; k < N * K; k++)
                    fprintf(file, "%a,", cv_predict[k]);

                fprintf(file, "\n");
                fclose(file);
            } else {
                cv_stats.reserve(cv_stats.size() + K * (P + 1) + N * K);
                cv_stats.push_back(gamma);
                cv_stats.push_back(lambda[nFold] / ySD[nFold]);
                cv_stats.push_back(trainingError);
                cv_stats.push_back(cvError);
                cv_stats.push_back(cvErrorSD);

                for (uint32_t k = 0; k < K; k++) {
                    double* betaf =
                        &beta[INDEX_TENSOR_COL(k, nFold, memory_P, K)];
                    double tmpOffset = intercept[INDEX(k, nFold, K)];

                    if (useIntercept && useCentering)
                        tmpOffset -= ddot_(&memory_P, betaf, &BLAS_I_ONE,
                                           featureMean, &BLAS_I_ONE);

                    cv_stats.push_back(tmpOffset);

                    for (uint32_t p = 0; p < P; p++)
                        cv_stats.push_back(betaf[p]);
                }

                for (uint32_t k = 0; k < N * K; k++)
                    cv_stats.push_back(cv_predict[k]);
            }

            if (verbose) {
                PRINT("Step: %3d ", j);

                if (gammaSeq.size() != 1 || gammaSeq[0] != 0.0)
                    PRINT("G: %e ", gamma);

                PRINT("L: %e T-E %e CV-E %e CV-SD %e\n",
                      lambda[nFold] / pow(ySD[nFold], 2), trainingError,
                      cvError, cvErrorSD);
            }

            if (j > 1 && cvImproving != 0)
                worseLambdaSteps++;
            else
                worseLambdaSteps = 0;

            if (worseLambdaSteps > cvStop || std::isnan(cvError))
                break;
        }
    }
    return cv_stats;
}
