#ifndef REGRESSIONDATASCHEME_H
#define REGRESSIONDATASCHEME_H

#include "fusionKernel.h"
#include "settings.h"
#include "mathHelpers.h"
#include <string.h>
#include <cstdio>
#include <random>
#include <set>

class RegressionDataScheme
{

protected:
    RegressionDataScheme( int _N, int _P, int _K, int _nc, int _type );
    RegressionDataScheme( const RegressionDataScheme& source);
    ~RegressionDataScheme();

    std::set<int> activeSet;

public:
    double* x;
    double* yOrg;
    double* v;
    double* u;

    int* foldid;
    int nFold;

    double* lambdaSeq;
    double* gammaSeq;

    int lengthLambda;
    int lengthGamma;

    int type;
    int isFusion;
    int isFused;
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

    double loglikelihood;
    double lasso;
    double ridge;
    double fusion;
    double cost;

protected:
    // used to update costfunction after offset update
    void updateCost( int l );

    double penaltyCost( double *coefs, double t );
    int checkXtimesBeta();
    int checkYsubXtimesBeta();

    void optimizeParameterAmbiguity( int iterations = 10 );

    // quadratic approximation of logistic loglikelihood
    void refreshApproximation( int l, int _updateCost = FALSE );

    bool checkActiveSet( int k );
    void checkWholeActiveSet();
    
public:

    void costFunction( void );

    // make predictions and store in xb (used for cv error calculation)
    void predict();

    void offsetMove( int l, int _updateCost = FALSE );
    int cdMove( int k, int l );
    int cdMoveFused( int k, int l );
    int cdMoveZS( int k, int s, int l );
    int cdMoveZSRotated( int n, int m, int s, int l, double theta );

    void lsSaOffsetMove( int l );
    int lsSaMove( int k, int s, int l, double delta_k,
                    double* rng = NULL, double temperature = 0 );

    void coordinateDescent( int seed );
    void localSearch( int seed );
    void simulatedAnnealing( int seed );

};

#endif /* REGRESSIONDATASCHEME_H */
