#ifndef REGRESSION_H
#define REGRESSION_H

#include <ctime>

#ifdef _OPENMP
#include <omp.h>
#endif


#include <vector>
#include "CvRegressionData.h"
#include "CvRegression.h"
#include "mathHelpers.h"
#include "RegressionData.h"

void doRegression(RegressionDataScheme* data, int seed);
void doCVRegression( RegressionData* data, double* gammaSeq,
        int gammaLength, double* lambdaSeq, int lambdaLength,
        double* cv_stats, int cv_cols, char* path, char* name, int mpi_rank,
        int seed);


#endif /* REGRESSION_H */
