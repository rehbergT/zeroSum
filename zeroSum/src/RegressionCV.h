#ifndef REGRESSION_CD_H
#define REGRESSION_CD_H

#include <vector>
#include "CvRegressionData.h"
#include "RegressionData.h"

class RegressionCV {
   private:
    double* gammaSeq;
    int lengthGamma;

    double* lambdaSeq;
    int lengthLambda;

    int nFold;
    int N, memory_N, P, memory_P, K;
    int type;

    int cvStop;
    int verbose;

   public:
    std::vector<std::vector<CvRegressionData>> cv_data;

    std::vector<std::vector<double>> cv_tmp;
    std::vector<std::vector<double>> cv_predict;

    RegressionCV(RegressionData& data);

    std::vector<double> doCVRegression(int seed,
                                       char* path = nullptr,
                                       char* name = nullptr,
                                       int mpi_rank = 0);
};

#endif
