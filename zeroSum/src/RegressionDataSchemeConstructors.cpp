#include "RegressionDataScheme.h"

RegressionDataScheme::RegressionDataScheme( int _N, int _P, int _K, int _nc, int _type )
{
    x      = NULL;
    yOrg   = NULL;
    v      = NULL;
    u      = NULL;

    foldid = NULL;
    nFold  = 0;

    lambdaSeq = NULL;
    gammaSeq  = NULL;

    lengthLambda = 0;
    lengthGamma  = 0;

    type = _type;
    if( type == FUSION_GAUSSIAN    || type == FUSION_GAUSSIAN_ZS ||
        type == FUSION_BINOMIAL    || type == FUSION_BINOMIAL_ZS ||
        type == FUSION_MULTINOMIAL || type == FUSION_MULTINOMIAL_ZS  )
    {
        isFusion = TRUE;
    }
    else
    {
        isFusion = FALSE;
    }

    if( type == FUSED_GAUSSIAN    || type == FUSED_GAUSSIAN_ZS ||
        type == FUSED_BINOMIAL    || type == FUSED_BINOMIAL_ZS ||
        type == FUSED_MULTINOMIAL || type == FUSED_MULTINOMIAL_ZS  )
    {
        isFused = TRUE;
    }
    else
    {
        isFused = FALSE;
    }

    if( type%2 == 0 )
        isZeroSum = TRUE;
    else
        isZeroSum = FALSE;

    N  = _N;
    P  = _P;
    K  = _K;
    nc = _nc;

    memory_N  = N;
    memory_P  = P;
    memory_nc = nc;

    #ifdef AVX_VERSION
        if( memory_N % ALIGNED_DOUBLES != 0)
            memory_N += ALIGNED_DOUBLES - memory_N % ALIGNED_DOUBLES;

        if( memory_P % ALIGNED_DOUBLES != 0)
            memory_P += ALIGNED_DOUBLES - memory_P % ALIGNED_DOUBLES;

        if( memory_nc % ALIGNED_DOUBLES != 0)
            memory_nc += ALIGNED_DOUBLES - memory_nc % ALIGNED_DOUBLES;

        y          = (double*)aligned_alloc( ALIGNMENT, memory_N * K * sizeof(double) );
        w          = (double*)aligned_alloc( ALIGNMENT, memory_N     * sizeof(double) );
        wOrg       = (double*)aligned_alloc( ALIGNMENT, memory_N     * sizeof(double) );
        tmp_array1 = (double*)aligned_alloc( ALIGNMENT, memory_N     * sizeof(double) );
        tmp_array2 = (double*)aligned_alloc( ALIGNMENT, memory_N     * sizeof(double) );
        xTimesBeta = (double*)aligned_alloc( ALIGNMENT, memory_N * K * sizeof(double) );
        beta       = (double*)aligned_alloc( ALIGNMENT, memory_P * K * sizeof(double) );
        offset     = (double*)aligned_alloc( ALIGNMENT, K            * sizeof(double) );
    #else
        y          = (double*)malloc( memory_N * K * sizeof(double) );
        w          = (double*)malloc( memory_N     * sizeof(double) );
        wOrg       = (double*)malloc( memory_N     * sizeof(double) );
        tmp_array1 = (double*)malloc( memory_N     * sizeof(double) );
        tmp_array2 = (double*)malloc( memory_N     * sizeof(double) );
        xTimesBeta = (double*)malloc( memory_N * K * sizeof(double) );
        beta       = (double*)malloc( memory_P * K * sizeof(double) );
        offset     = (double*)malloc( K            * sizeof(double) );
    #endif

    memset( y,          0.0, memory_N * K * sizeof(double));
    memset( w,          0.0, memory_N     * sizeof(double));
    memset( wOrg,       0.0, memory_N     * sizeof(double));
    memset( tmp_array1, 0.0, memory_N     * sizeof(double));
    memset( tmp_array2, 0.0, memory_N     * sizeof(double));
    memset( xTimesBeta, 0.0, memory_N * K * sizeof(double));
    memset( beta,       0.0, memory_P * K * sizeof(double));
    memset( offset,     0.0, K            * sizeof(double));

    fusionKernel = NULL;

    if( isFusion )
    {
        #ifdef AVX_VERSION
            fusionPartialSums    = (double*)aligned_alloc( ALIGNMENT, memory_nc * K * sizeof(double) );
            fusionPartialSumsTmp = (double*)aligned_alloc( ALIGNMENT, nc        * sizeof(double) );
            fusionSums           = (double*)aligned_alloc( ALIGNMENT, K             * sizeof(double) );
        #else
            fusionPartialSums    = (double*)malloc( memory_nc * K * sizeof(double) );
            fusionPartialSumsTmp = (double*)malloc( nc        * sizeof(double) );
            fusionSums           = (double*)malloc( K             * sizeof(double) );
        #endif

        memset( fusionPartialSums,    0.0, memory_nc * K * sizeof(double) );
        memset( fusionPartialSumsTmp, 0.0, nc            * sizeof(double) );
        memset( fusionSums,           0.0, K             * sizeof(double) );
    }

    cSum       = 0.0;
    alpha      = 0.0;
    lambda     = 0.0;
    gamma      = 0.0;
    downScaler = 0.0;
    precision  = 0.0;

    diagonalMoves = 0;
    useOffset     = 0;
    useApprox     = 0;
    algorithm     = 0;
    polish        = 0;
    verbose       = 0;
    cores         = 0;
    cvStop        = 0;

    loglikelihood = 0.0;
    lasso         = 0.0;
    ridge         = 0.0;
    fusion        = 0.0;
    cost          = 0.0;
}

RegressionDataScheme::~RegressionDataScheme()
{
    if( isFusion )
    {
        free(fusionSums);
        free(fusionPartialSumsTmp);
        free(fusionPartialSums);
    }

    free(offset);
    free(beta);
    free(xTimesBeta);
    free(tmp_array2);
    free(tmp_array1);
    free(wOrg);
    free(w);
    free(y);
}

RegressionDataScheme::RegressionDataScheme( const RegressionDataScheme& source)
{
    if( this != &source )
    {
        x      = source.x;
        yOrg   = source.yOrg;
        v      = source.v;
        u      = source.u;

        foldid = source.foldid;
        nFold  = source.nFold;

        lambdaSeq = source.lambdaSeq;
        gammaSeq  = source.gammaSeq;

        lengthLambda = source.lengthLambda;
        lengthGamma  = source.lengthGamma;

        type      = source.type;
        isFusion  = source.isFusion;
        isFused   = source.isFused;
        isZeroSum = source.isZeroSum;

        N  = source.N;
        P  = source.P;
        K  = source.K;
        nc = source.nc;

        memory_N  = source.memory_N;
        memory_P  = source.memory_P;
        memory_nc = source.memory_nc;

        #ifdef AVX_VERSION
            y          = (double*)aligned_alloc( ALIGNMENT, memory_N * K * sizeof(double) );
            w          = (double*)aligned_alloc( ALIGNMENT, memory_N     * sizeof(double) );
            wOrg       = (double*)aligned_alloc( ALIGNMENT, memory_N     * sizeof(double) );
            tmp_array1 = (double*)aligned_alloc( ALIGNMENT, memory_N     * sizeof(double) );
            tmp_array2 = (double*)aligned_alloc( ALIGNMENT, memory_N     * sizeof(double) );
            xTimesBeta = (double*)aligned_alloc( ALIGNMENT, memory_N * K * sizeof(double) );
            beta       = (double*)aligned_alloc( ALIGNMENT, memory_P * K * sizeof(double) );
            offset     = (double*)aligned_alloc( ALIGNMENT, K            * sizeof(double) );
        #else
            y          = (double*)malloc( memory_N * K * sizeof(double) );
            w          = (double*)malloc( memory_N     * sizeof(double) );
            wOrg       = (double*)malloc( memory_N     * sizeof(double) );
            tmp_array1 = (double*)malloc( memory_N     * sizeof(double) );
            tmp_array2 = (double*)malloc( memory_N     * sizeof(double) );
            xTimesBeta = (double*)malloc( memory_N * K * sizeof(double) );
            beta       = (double*)malloc( memory_P * K * sizeof(double) );
            offset     = (double*)malloc( K            * sizeof(double) );
        #endif

        memcpy( y,          source.y,          memory_N * K * sizeof(double));
        memcpy( w,          source.w,          memory_N     * sizeof(double));
        memcpy( wOrg,       source.wOrg,       memory_N     * sizeof(double));
        memcpy( tmp_array1, source.tmp_array1, memory_N     * sizeof(double));
        memcpy( tmp_array2, source.tmp_array2, memory_N      * sizeof(double));
        memcpy( xTimesBeta, source.xTimesBeta, memory_N * K * sizeof(double));
        memcpy( beta,       source.beta,       memory_P * K * sizeof(double));
        memcpy( offset,     source.offset,     K            * sizeof(double));

        fusionKernel = source.fusionKernel;

        if( isFusion )
        {
            #ifdef AVX_VERSION
                fusionPartialSums    = (double*)aligned_alloc( ALIGNMENT, memory_nc * K * sizeof(double) );
                fusionPartialSumsTmp = (double*)aligned_alloc( ALIGNMENT, nc        * sizeof(double) );
                fusionSums           = (double*)aligned_alloc( ALIGNMENT, K             * sizeof(double) );
            #else
                fusionPartialSums    = (double*)malloc( memory_nc * K * sizeof(double) );
                fusionPartialSumsTmp = (double*)malloc( nc        * sizeof(double) );
                fusionSums           = (double*)malloc( K             * sizeof(double) );
            #endif

            memcpy(fusionPartialSums,    source.fusionPartialSums,    memory_nc * K * sizeof(double) );
            memcpy(fusionPartialSumsTmp, source.fusionPartialSumsTmp, nc            * sizeof(double) );
            memcpy(fusionSums,           source.fusionSums,           K             * sizeof(double) );
        }

        cSum       = source.cSum;
        alpha      = source.alpha;
        lambda     = source.lambda;
        gamma      = source.gamma;
        downScaler = source.downScaler;
        precision  = source.precision;

        diagonalMoves = source.diagonalMoves;
        useOffset     = source.useOffset;
        useApprox     = source.useApprox;
        algorithm     = source.algorithm;
        polish        = source.polish;
        verbose       = source.verbose;
        cores         = source.cores;
        cvStop        = source.cvStop;

        loglikelihood = source.loglikelihood;
        lasso         = source.lasso;
        ridge         = source.ridge;
        fusion        = source.fusion;
        cost          = source.cost;
    }
}
