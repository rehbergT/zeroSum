#include "lambdaMax.h"

SEXP lambdaMax( SEXP _X, SEXP _beta0, SEXP _alpha)
{
    PROTECT( _X = AS_NUMERIC( _X ) );
    double* x = REAL( _X );

    int* dimX = INTEGER(GET_DIM( _X ));
    int N = dimX[0];
    int P = dimX[1];
    
    PROTECT( _beta0 = AS_NUMERIC( _beta0 ) );
    double* beta0 = REAL( _beta0 );
    
    PROTECT( _alpha = AS_NUMERIC( _alpha ) );
    double alpha = REAL( _alpha )[0];
        
    double MaxRes = DBL_MIN;
    double a;
    for(int k=2; k<P; ++k)
    {
        for(int s=1; s<k; ++s)
        {
            a = absSumDiffMult( &(x[ INDEX(0,s,N) ]), 
                                &(x[ INDEX(0,k,N) ]), 
                                beta0, N);
            if(a > MaxRes)
                MaxRes = a;            
        }        
    }
    
    double lambdaMax = MaxRes / ( 2.0 * (double)N * alpha );
    
    
    UNPROTECT(3);
    return ScalarReal(lambdaMax);
}
