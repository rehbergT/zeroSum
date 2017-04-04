#include "RegressionData.h"

RegressionData::RegressionData( int _N, int _P, int _K, int _nc, int _type )
    : RegressionDataScheme( _N, _P, _K, _nc, _type)
{
    #ifdef AVX_VERSION
        x    = (double*)aligned_alloc( ALIGNMENT, memory_N * P * sizeof(double) );
        v    = (double*)aligned_alloc( ALIGNMENT, memory_P     * sizeof(double) );
        u    = (double*)aligned_alloc( ALIGNMENT, memory_P     * sizeof(double) );
    #else
        x    = (double*)malloc( memory_N * P * sizeof(double) );
        v    = (double*)malloc( memory_P     * sizeof(double) );
        u    = (double*)malloc( memory_P     * sizeof(double) );
    #endif

    memset( x, 0.0, memory_N * P * sizeof(double));
    memset( v, 0.0, memory_P * sizeof(double));
    memset( u, 0.0, memory_P * sizeof(double));

    if( type > 6 )
    {
        #ifdef AVX_VERSION
            yOrg = (double*)aligned_alloc( ALIGNMENT, memory_N * K * sizeof(double));
        #else
            yOrg = (double*)malloc( memory_N * K * sizeof(double));
        #endif

        memset( yOrg, 0.0, memory_N * K * sizeof(double));
    }
    if( isFusion )
    {
        fusionKernel = (struct fusionKernel**)malloc( P * sizeof(struct fusionKernel*));
        for( int j=0; j<P; j++)
            fusionKernel[j] = NULL;
    }
}

RegressionData::~RegressionData()
{
    free(x);
    free(v);
    free(u);

    if( type > 6 )
        free(yOrg);

    if( isFusion )
    {
        for( int j=0; j<P; ++j )
        {
            struct fusionKernel* currEl = fusionKernel[j];
            struct fusionKernel* nextEl;

            while( currEl != NULL)
            {
                nextEl = currEl->next;
                free(currEl);
                currEl = nextEl;
            }
        }
        free(fusionKernel);
    }
}

RegressionData::RegressionData( const RegressionData& source )
    : RegressionDataScheme( source )
{
    if( this != &source )
    {
        #ifdef AVX_VERSION
            x    = (double*)aligned_alloc( ALIGNMENT, memory_N * P * sizeof(double) );
            v    = (double*)aligned_alloc( ALIGNMENT, memory_P     * sizeof(double) );
            u    = (double*)aligned_alloc( ALIGNMENT, memory_P     * sizeof(double) );
        #else
            x    = (double*)malloc( memory_N * P * sizeof(double) );
            v    = (double*)malloc( memory_P     * sizeof(double) );
            u    = (double*)malloc( memory_P     * sizeof(double) );
        #endif

        memcpy( x, source.x, memory_N * P * sizeof(double));
        memcpy( v, source.y, memory_P * sizeof(double));
        memcpy( u, source.u, memory_P * sizeof(double));

        if( type > 6 )
        {
            #ifdef AVX_VERSION
                yOrg = (double*)aligned_alloc( ALIGNMENT, memory_N * K * sizeof(double));
            #else
                yOrg = (double*)malloc( memory_N * K * sizeof(double));
            #endif

            memcpy( yOrg, source.y, memory_N * K * sizeof(double));
        }
        if( isFusion )
        {
            fusionKernel = (struct fusionKernel**)malloc( P * sizeof(struct fusionKernel*));
            for( int j=0; j<P; j++)
            {
                struct fusionKernel* currEl = source.fusionKernel[j];

                while( currEl != NULL)
                {
                    fusionKernel[j] = appendElement(fusionKernel[j], currEl->i, currEl->value);
                    currEl = currEl->next;
                }
            }
        }
    }
}
