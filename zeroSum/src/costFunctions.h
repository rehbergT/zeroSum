#ifndef COSTFUNCTION_H
#define COSTFUNCTION_H


#include <math.h>
#include <float.h>
#include <string.h>
#include <time.h>
#include "mathHelpers.h"
#include "regressionData.h"

void vectorElNetCostFunction(
                struct regressionData *data,                   
                double* res,
                double* energy,
                double* residum,
                double* ridge,
                double* lasso );


#endif /*COSTFUNCTION_H */
