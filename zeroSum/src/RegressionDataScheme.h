#ifndef REGRESSIONDATASCHEME_H
#define REGRESSIONDATASCHEME_H

#include <string.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include "fusionKernel.h"
#include "mathHelpers.h"
#include "settings.h"

class RegressionDataScheme {
   protected:
    void regressionDataSchemeAlloc();
    void regressionDataSchemeFree();
    void regressionDataSchemeShallowCopy(const RegressionDataScheme& source);
    void regressionDataSchemeDeepCopy(const RegressionDataScheme& source);
    void regressionDataSchemePointerMove(RegressionDataScheme& source);

    RegressionDataScheme();
    RegressionDataScheme(int _N, int _P, int _K, int _nc, int _type);
    RegressionDataScheme(const RegressionDataScheme& source);
    RegressionDataScheme(RegressionDataScheme&& source);
    ~RegressionDataScheme();

    RegressionDataScheme& operator=(const RegressionDataScheme& source);
    RegressionDataScheme& operator=(RegressionDataScheme&& source);

    std::vector<int> activeSet;

   public:
    double* x;
    double* yOrg;
    int* status;
    double* d;
    double* v;
    double* u;

    int* foldid;
    int nFold;

    // lambdaSeq and gammaSeq are managed from R
    // -> only pointer to memory
    double* lambdaSeq;
    double* gammaSeq;

    int lengthLambda;
    int lengthGamma;

    int type;
    int isFusion;
    int isZeroSum;

    int N;
    int P;
    int K;
    int nc;

    int memory_N;
    int memory_P;
    int memory_nc;

    double* y;
    double* w;
    double* wOrg;
    double* tmp_array1;
    double* tmp_array2;
    double* xTimesBeta;
    double* beta;
    double* offset;

    struct fusionKernel** fusionKernel;
    double* fusionPartialSums;
    double* fusionPartialSumsTmp;
    double* fusionSums;

    double cSum;
    double alpha;
    double lambda;
    double gamma;
    double downScaler;
    double precision;

    int diagonalMoves;
    int useOffset;
    int useApprox;
    int algorithm;
    int polish;
    int verbose;
    int cores;
    int cvStop;
    int approxFailed;

    double loglikelihood;
    double lasso;
    double ridge;
    double fusion;
    double cost;

   protected:
    // used to update costfunction after offset update
    void updateCost(int l);

    double penaltyCost(double* coefs, double t);
    int checkXtimesBeta();
    int checkYsubXtimesBeta();

    void optimizeParameterAmbiguity(int iterations = 10);

    // quadratic approximation of logistic loglikelihood
    void refreshApproximation(int l, int _updateCost = FALSE);

    bool checkActiveSet(int k);
    void checkWholeActiveSet();

    void coordinateDescent(int seed);
    void localSearch(int seed);
    void simulatedAnnealing(int seed);

   public:
    void costFunction(void);
    void calcCoxRegressionD();
    // make predictions and store in xb (used for cv error calculation)
    void predict();

    void offsetMove(int l, int _updateCost = FALSE);
    int cdMove(int k, int l);
    int cdMoveZS(int k, int s, int l);
    int cdMoveZSRotated(int n, int m, int s, int l, double theta);

    void lsSaOffsetMove(int l);
    int lsSaMove(int k,
                 int s,
                 int l,
                 double delta_k,
                 double* rng = NULL,
                 double temperature = 0);

    void doRegression(int seed);
};

#endif /* REGRESSIONDATASCHEME_H */
