#ifndef REGRESSION_H
#define REGRESSION_H

#include "costFunctions.h"
#include "mathHelpers.h"
#include "regressionData.h"

void calcOffsetElNetGradient(   struct regressionData *data,
                                double* restrict betasX);

void calcOffsetElNetGradientUpdate(   
                    struct regressionData *data,
                    double* restrict res,
                    double* restrict energy,
                    double* restrict residum, 
                    double* restrict ridge,
                    double* restrict lasso );

void zeroSumRegressionCD(  
            struct regressionData data,
            int verticalMoves );

void zeroSumRegressionSA(  
            struct regressionData data );

void zeroSumRegressionLS(  
            struct regressionData data,
            const int steps );

void elNetRegressionCD(  
            struct regressionData data );

void elNetRegressionSA(  
            struct regressionData data );

void elNetRegressionLS(  
            struct regressionData data,
            const int steps  );

#endif /* REGRESSION_H */
