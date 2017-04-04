#include "CvRegressionData.h"

CvRegressionData::CvRegressionData( RegressionData* source )
    : RegressionDataScheme( source->N, source->P, source->K, source->nc, source->type )
{
    x      = source->x;
    yOrg   = source->yOrg;
    v      = source->v;
    u      = source->u;

    foldid = source->foldid;
    nFold  = source->nFold;

    lambdaSeq = source->lambdaSeq;
    gammaSeq  = source->gammaSeq;

    lengthLambda = source->lengthLambda;
    lengthGamma  = source->lengthGamma;

    memcpy( y,          source->y,          memory_N * K * sizeof(double));
    memcpy( w,          source->w,          memory_N     * sizeof(double));
    memcpy( wOrg,       source->wOrg,       memory_N     * sizeof(double));
    memcpy( tmp_array1, source->tmp_array1, memory_N     * sizeof(double));
    memcpy( tmp_array2, source->tmp_array2, memory_N      * sizeof(double));
    memcpy( xTimesBeta, source->xTimesBeta, memory_N * K * sizeof(double));
    memcpy( beta,       source->beta,       memory_P * K * sizeof(double));
    memcpy( offset,     source->offset,     K            * sizeof(double));

    fusionKernel =source->fusionKernel;
    if( isFusion )
    {
        memcpy(fusionPartialSums,   source->fusionPartialSums,    memory_nc * K * sizeof(double) );
        memcpy(fusionPartialSumsTmp,source->fusionPartialSumsTmp, nc            * sizeof(double) );
        memcpy(fusionSums,          source->fusionSums,           K             * sizeof(double) );
    }

    cSum       = source->cSum;
    alpha      = source->alpha;
    lambda     = source->lambda;
    gamma      = source->gamma;
    downScaler = source->downScaler;
    precision  = source->precision;

    diagonalMoves = source->diagonalMoves;
    useOffset     = source->useOffset;
    useApprox     = source->useApprox;
    algorithm     = source->algorithm;
    polish        = source->polish;
    verbose       = source->verbose;
    cores         = source->cores;
    cvStop        = source->cvStop;

    loglikelihood = source->loglikelihood;
    lasso         = source->lasso;
    ridge         = source->ridge;
    fusion        = source->fusion;
    cost          = source->cost;

    #ifdef AVX_VERSION
        wCV = (double*)aligned_alloc( ALIGNMENT, memory_N * sizeof(double) );
    #else
        wCV = (double*)malloc( memory_N * sizeof(double) );
    #endif

    memcpy( wCV, source->wOrg, memory_N * sizeof(double) );

}

CvRegressionData::~CvRegressionData()
{
    free(wCV);
}

CvRegressionData::CvRegressionData( const CvRegressionData& source )
    : RegressionDataScheme( source )
{
    if( this != &source )
    {
        #ifdef AVX_VERSION
            wCV = (double*)aligned_alloc( ALIGNMENT, memory_N * sizeof(double) );
        #else
            wCV = (double*)malloc( memory_N * sizeof(double) );
        #endif

        memcpy( wCV, source.wCV, memory_N * sizeof(double) );
    }
}
