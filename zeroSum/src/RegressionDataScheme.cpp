#include "RegressionDataScheme.h"

bool RegressionDataScheme::checkActiveSet( int k )
{
    bool isZero = false;
    for( int l=0; l<K; l++ )
        if( beta[ INDEX(k,l,memory_P) ] == 0.0 ) isZero = true;

    if( isZero )
    {
        auto result = activeSet.erase(k);

        if( result != 0 )
            return true;
        else
            return false;
    }
    else
    {
        auto result = activeSet.insert(k);
        return result.second;
    }
}

void RegressionDataScheme::checkWholeActiveSet()
{
    for( auto it=activeSet.begin(); it!=activeSet.end(); )
    {
        if( fabs(beta[*it]) < DBL_EPSILON )
            activeSet.erase(it++);
        else
            ++it;
    }
}

void RegressionDataScheme::doRegression( int seed )
{
    if( algorithm == 1 )
    {
        coordinateDescent( seed );
        if( polish != FALSE )
        {
            localSearch( seed, polish );
        }
    }
    else if( algorithm == 2 )
    {
        simulatedAnnealing( seed );
    }
    else if( algorithm == 3 )
    {
        localSearch( seed, FALSE );
    }
    else if( algorithm == 4 )
    {
        coordinateDescent( seed );
        localSearch( seed, FALSE );
    }

}
