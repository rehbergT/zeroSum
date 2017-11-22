#include "RegressionCV.h"

RegressionCV::RegressionCV(RegressionData& data) {
    nFold = data.nFold;
    cvStop = data.cvStop;
    verbose = data.verbose;

    N = data.N;
    P = data.P;
    K = data.K;

    memory_N = data.memory_N;
    memory_P = data.memory_P;
    type = data.type;

    gammaSeq = data.gammaSeq;
    lambdaSeq = data.lambdaSeq;

    lengthGamma = data.lengthGamma;
    lengthLambda = data.lengthLambda;

    cv_data.reserve(lengthGamma);
    for (int i = 0; i < lengthGamma; i++) {
        std::vector<CvRegressionData> tmp;
        tmp.reserve(nFold + 1);
        for (int f = 0; f < nFold + 1; f++)
            tmp.push_back(CvRegressionData(data));

        cv_data.push_back(tmp);
    }
    cv_predict.resize(lengthGamma);
    cv_tmp.resize(lengthGamma);

    for (int i = 0; i < lengthGamma; i++) {
        cv_predict[i].resize(K * N);
        cv_tmp[i].resize(nFold + 1);
    }

    for (int f = 0; f < nFold; f++) {
        int foldSize = 0;
        for (int i = 0; i < N; i++) {
            if ((data.foldid[i] - 1) == f || data.foldid[i] == -1)
                foldSize++;
        }

        double scaler1 = (double)N / (double)(N - foldSize);
        double scaler2 = (double)N / (double)foldSize;

        for (int j = 0; j < lengthGamma; j++) {
            for (int n = 0; n < N; n++) {
                if ((data.foldid[n] - 1) == f) {
                    cv_data[j][f].w[n] = 0.0;
                    cv_data[j][f].wOrg[n] = 0.0;
                    cv_data[j][f].wCV[n] *= scaler2;
                } else {
                    cv_data[j][f].w[n] *= scaler1;
                    cv_data[j][f].wOrg[n] *= scaler1;
                    cv_data[j][f].wCV[n] = 0.0;
                }
            }

            if (type >= COX)
                cv_data[j][f].calcCoxRegressionD();
        }
    }
}

std::vector<double> RegressionCV::doCVRegression(int seed,
                                                 char* path,
                                                 char* name,
                                                 int mpi_rank) {
    std::vector<double> cv_stats;

#ifdef AVX_VERSION
    double* lastCV =
        (double*)aligned_alloc(ALIGNMENT, lengthGamma * sizeof(double));
#else
    double* lastCV = (double*)malloc(lengthGamma * sizeof(double));
#endif

    int worseLambdaSteps = 0;

    for (int j = 0; j < lengthLambda; j++) {
#ifdef R_PACKAGE
        R_CheckUserInterrupt();
#endif

#pragma omp parallel for collapse(2) schedule(dynamic)
        for (int f = 0; f < nFold + 1; f++)
            for (int i = lengthGamma - 1; i >= 0; i--) {
                cv_data[i][f].lambda = lambdaSeq[j];
                cv_data[i][f].gamma = gammaSeq[i];

                int cvSeed =
                    f + seed + (int)((lambdaSeq[j] + gammaSeq[i]) * 1e3);
                cv_data[i][f].doRegression(cvSeed);

                double* tmp = cv_data[i][f].wOrg;
                cv_data[i][f].wOrg = cv_data[i][f].wCV;

                if (type >= COX)
                    cv_data[i][f].calcCoxRegressionD();

                cv_data[i][f].costFunction();

                cv_data[i][f].wOrg = tmp;

                if (type >= COX)
                    cv_data[i][f].calcCoxRegressionD();

                cv_tmp[i][f] = cv_data[i][f].loglikelihood;

                if (f < nFold) {
                    cv_data[i][f].predict();

                    for (int ii = 0; ii < N; ii++) {
                        if (cv_data[i][f].wCV[ii] != 0.0) {
                            for (int l = 0; l < K; ++l)
                                cv_predict[i][INDEX(ii, l, memory_N)] =
                                    cv_data[i][f]
                                        .xTimesBeta[INDEX(ii, l, memory_N)];
                        }
                    }
                }
            }

        int cvImproving = 0;
        for (int i = 0; i < lengthGamma; i++) {
            double trainingError = cv_tmp[i][nFold];

            double cvError = 0.0;
            for (int ff = 0; ff < nFold; ff++)
                cvError += cv_tmp[i][ff];

            cvError /= nFold;

            double cvErrorSD = 0.0;
            for (int ff = 0; ff < nFold; ff++)
                cvErrorSD += cv_tmp[i][ff] * cv_tmp[i][ff];

            cvErrorSD /= (double)nFold;

            cvErrorSD -= cvError * cvError;
            cvErrorSD /= (double)nFold - 1.0;
            cvErrorSD = sqrt(cvErrorSD);

            if (std::isnan(cvErrorSD))
                cvErrorSD = -1.0;

            if (path != NULL) {
                char outPath[MAX_PATH_LENGTH] = "";
                FILE* file;

                snprintf(outPath, sizeof(outPath), "%s%s_%d_stats.csv", path,
                         name, mpi_rank);

                file = fopen(outPath, "a");

                fprintf(file, "%a,%a,%a,%a,%a,", gammaSeq[i], lambdaSeq[j],
                        trainingError, cvError, cvErrorSD);

                for (int k = 0; k < K; k++) {
                    fprintf(file, "%a,", cv_data[i][nFold].offset[k]);

                    for (int jj = 0; jj < P; jj++)
                        fprintf(file, "%a,",
                                cv_data[i][nFold].beta[INDEX(jj, k, memory_P)]);
                }

                for (int k = 0; k < N * K; k++)
                    fprintf(file, "%a,", cv_predict[i][k]);

                fprintf(file, "\n");
                fclose(file);
            } else {
                cv_stats.reserve(cv_stats.size() + K * (P + 1) + N * K);
                cv_stats.push_back(gammaSeq[i]);
                cv_stats.push_back(lambdaSeq[j]);
                cv_stats.push_back(trainingError);
                cv_stats.push_back(cvError);
                cv_stats.push_back(cvErrorSD);

                for (int k = 0; k < K; k++) {
                    cv_stats.push_back(cv_data[i][nFold].offset[k]);

                    for (int p = 0; p < P; p++)
                        cv_stats.push_back(
                            cv_data[i][nFold].beta[INDEX(p, k, memory_P)]);
                }

                for (int k = 0; k < N * K; k++)
                    cv_stats.push_back(cv_predict[i][k]);
            }

            if (verbose) {
                PRINT("Step: %3d ", j);

                if (lengthGamma != 1)
                    PRINT("G: %e ", gammaSeq[i]);

                PRINT("L: %e T-E %e CV-E %e CV-SD %e\n", lambdaSeq[j],
                      trainingError, cvError, cvErrorSD);
            }

            if (cvError - lastCV[i] < DBL_EPSILON)
                cvImproving++;

            lastCV[i] = cvError;
        }

        if (j > 1 && cvImproving != 0)
            worseLambdaSteps++;
        else
            worseLambdaSteps = 0;

        if (worseLambdaSteps > cvStop)
            break;
    }

    free(lastCV);

    return cv_stats;
}
