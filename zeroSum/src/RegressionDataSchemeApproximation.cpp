#include "RegressionDataScheme.h"

void RegressionDataScheme::refreshApproximation(int l, int _updateCost )
{
    int ii = INDEX(0,l,memory_N);
    double* yl    = &y[ii];
    double* yOrgl = &yOrg[ii];
    double* xb    = &xTimesBeta[ii];
    double  p;

    // used for offset update when optimizing with localSearch on the real
    // loglikelihood
    if( useApprox )
    {
        for( int k=0; k<K; k++)
        {
            ii = INDEX(0,k,memory_N);
            a_sub_b( &y[ii], &xTimesBeta[ii], &xTimesBeta[ii], N);
        }
    }

    for( int i=0; i<N; i++ )
    {
        if( type <= 12 )
        {
            p = 1.0 / ( exp( -xb[i] ) + 1.0 );
        }
        else
        {
            p = 0.0;
            for( int k=0; k<K; k++ )
                p += exp( xTimesBeta[ INDEX(i,k,memory_N) ] );

            p = exp( xTimesBeta[ INDEX(i,l,memory_N) ] ) / p;
        }

        w[i]  = p * ( 1.0 - p );

        if( w[i] < DBL_EPSILON * 100 )
        {
            yl[i] = 0.0;
            w[i] = 0.0;
        }
        else
        {
            yl[i] = xb[i] + ( yOrgl[i] - p ) / w[i];
            w[i] *= wOrg[i];
        }
    }

    if( useApprox )
    {
        for( int k=0; k<K; k++)
        {
            ii = INDEX(0,k,memory_N);
            a_sub_b( &y[ii], &xTimesBeta[ii], &xTimesBeta[ii], N);
        }
    }

    if( _updateCost )
        this->updateCost(l);

}
