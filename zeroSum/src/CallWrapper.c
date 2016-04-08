#include "regressions.h"

SEXP CallWrapper(   SEXP _X, SEXP _Y, SEXP _beta, SEXP _lambda,
                    SEXP _alpha, SEXP _offset, SEXP _type,
                    SEXP _algorithm, SEXP _precision,
                    SEXP _verticalMoves, SEXP _polish
                )
{

    PROTECT( _X = AS_NUMERIC( _X ) );
    double* x = REAL( _X );

    int* dimX = INTEGER(GET_DIM( _X ));
    int N = dimX[0];
    int P = dimX[1];

    PROTECT( _Y = AS_NUMERIC( _Y ) );
    double* y = REAL( _Y );

    PROTECT( _lambda = AS_NUMERIC( _lambda ) );
    double lambda = REAL( _lambda )[0];

    PROTECT( _alpha = AS_NUMERIC( _alpha ) );
    double alpha = REAL( _alpha )[0];

    PROTECT( _offset = AS_INTEGER( _offset ) );
    int offset = INTEGER( _offset )[0];

    PROTECT( _type = AS_INTEGER( _type ) );
    int type = INTEGER( _type )[0];

    PROTECT( _algorithm = AS_INTEGER( _algorithm ) );
    int algorithm = INTEGER( _algorithm )[0];

    PROTECT( _precision = AS_NUMERIC( _precision ) );
    double precision = REAL( _precision )[0];

    PROTECT( _verticalMoves = AS_INTEGER( _verticalMoves ) );
    int verticalMoves = INTEGER( _verticalMoves )[0];

    PROTECT( _polish = AS_INTEGER( _polish ) );
    int polish = INTEGER( _polish )[0];
    
    PROTECT( _beta = AS_NUMERIC( _beta ) );
    double* beta = REAL( _beta );

    struct regressionData data = {
        x, y, N, P, beta, lambda, alpha, offset, precision
    };
    
    if( type == 0 )
    {
        if( algorithm == 0 )
        {
            elNetRegressionCD( data );
            if( polish != FALSE )
                elNetRegressionLS( data, polish);
        }
        else if( algorithm == 1 )
        {
            elNetRegressionSA( data );
            if( polish != FALSE )
                elNetRegressionLS( data, polish);            
        }
        else if( algorithm == 2 )
        {
            memset( beta, 0, P * sizeof(double) );
            elNetRegressionLS( data, 0);
        }       
        else if( algorithm == 3 )
        {
            elNetRegressionCD( data );
            elNetRegressionLS( data, 0);
        }
    }
    else if( type == 1 )
    {        
        if( algorithm == 0)
        {
            zeroSumRegressionCD( data, verticalMoves);
            if( polish != FALSE )
                zeroSumRegressionLS( data, polish);
        }
        else if( algorithm == 1 )
        {
            zeroSumRegressionSA( data );
            if( polish != FALSE )
                zeroSumRegressionLS( data, polish);
        }
        else if( algorithm == 2 )
        {
            memset( beta, 0, P * sizeof(double) );
            zeroSumRegressionLS( data, 0);
        }
        else if( algorithm == 3 )
        {
            verticalMoves = TRUE;
            zeroSumRegressionCD( data, verticalMoves);
            zeroSumRegressionLS( data, 0);
        }
    }   
    
    
    UNPROTECT(11);
    return R_NilValue;
}


static const 
R_CallMethodDef callMethods[] = {
    {"CallWrapper", (DL_FUNC) &CallWrapper, 11},
    {NULL, NULL, 0}
};


void R_init_zeroSum(DllInfo *info)
{
     R_registerRoutines(info, NULL, callMethods,  NULL, NULL);
}

