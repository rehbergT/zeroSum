#include "RegressionDataScheme.h"

void RegressionDataScheme::regressionDataSchemeAlloc() {
#ifdef AVX_VERSION
    y = (double*)aligned_alloc(ALIGNMENT, memory_N * K * sizeof(double));
    w = (double*)aligned_alloc(ALIGNMENT, memory_N * sizeof(double));
    wOrg = (double*)aligned_alloc(ALIGNMENT, memory_N * sizeof(double));
    tmp_array1 = (double*)aligned_alloc(ALIGNMENT, memory_N * sizeof(double));
    tmp_array2 = (double*)aligned_alloc(ALIGNMENT, memory_N * sizeof(double));
    xTimesBeta =
        (double*)aligned_alloc(ALIGNMENT, memory_N * K * sizeof(double));
    beta = (double*)aligned_alloc(ALIGNMENT, memory_P * K * sizeof(double));
    offset = (double*)aligned_alloc(ALIGNMENT, K * sizeof(double));
#else
    y = (double*)malloc(memory_N * K * sizeof(double));
    w = (double*)malloc(memory_N * sizeof(double));
    wOrg = (double*)malloc(memory_N * sizeof(double));
    tmp_array1 = (double*)malloc(memory_N * sizeof(double));
    tmp_array2 = (double*)malloc(memory_N * sizeof(double));
    xTimesBeta = (double*)malloc(memory_N * K * sizeof(double));
    beta = (double*)malloc(memory_P * K * sizeof(double));
    offset = (double*)malloc(K * sizeof(double));
#endif

    if (type >= COX) {
#ifdef AVX_VERSION
        status = (int*)aligned_alloc(ALIGNMENT, memory_N * 2 * sizeof(int));
        d = (double*)aligned_alloc(ALIGNMENT, memory_N * sizeof(double));
#else
        status = (int*)malloc(memory_N * 2 * sizeof(int));
        d = (double*)malloc(memory_N * sizeof(double));
#endif
    }

    if (isFusion) {
#ifdef AVX_VERSION
        fusionPartialSums =
            (double*)aligned_alloc(ALIGNMENT, memory_nc * K * sizeof(double));
        fusionPartialSumsTmp =
            (double*)aligned_alloc(ALIGNMENT, nc * sizeof(double));
        fusionSums = (double*)aligned_alloc(ALIGNMENT, K * sizeof(double));
#else
        fusionPartialSums = (double*)malloc(memory_nc * K * sizeof(double));
        fusionPartialSumsTmp = (double*)malloc(nc * sizeof(double));
        fusionSums = (double*)malloc(K * sizeof(double));
#endif
    }
}

void RegressionDataScheme::regressionDataSchemeFree() {
    if (isFusion) {
        free(fusionSums);
        free(fusionPartialSumsTmp);
        free(fusionPartialSums);
    }

    if (type >= COX) {
        free(d);
        free(status);
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

void RegressionDataScheme::regressionDataSchemeShallowCopy(
    const RegressionDataScheme& source) {
    x = source.x;
    yOrg = source.yOrg;
    v = source.v;
    u = source.u;

    foldid = source.foldid;
    nFold = source.nFold;

    lambdaSeq = source.lambdaSeq;
    gammaSeq = source.gammaSeq;

    lengthLambda = source.lengthLambda;
    lengthGamma = source.lengthGamma;

    type = source.type;
    isFusion = source.isFusion;
    isZeroSum = source.isZeroSum;

    N = source.N;
    P = source.P;
    K = source.K;
    nc = source.nc;

    memory_N = source.memory_N;
    memory_P = source.memory_P;
    memory_nc = source.memory_nc;

    cSum = source.cSum;
    alpha = source.alpha;
    lambda = source.lambda;
    gamma = source.gamma;
    downScaler = source.downScaler;
    precision = source.precision;

    diagonalMoves = source.diagonalMoves;
    useOffset = source.useOffset;
    useApprox = source.useApprox;
    algorithm = source.algorithm;
    polish = source.polish;
    verbose = source.verbose;
    cores = source.cores;
    cvStop = source.cvStop;

    loglikelihood = source.loglikelihood;
    lasso = source.lasso;
    ridge = source.ridge;
    fusion = source.fusion;
    cost = source.cost;

    fusionKernel = source.fusionKernel;
}

void RegressionDataScheme::regressionDataSchemeDeepCopy(
    const RegressionDataScheme& source) {
    memcpy(y, source.y, memory_N * K * sizeof(double));
    memcpy(w, source.w, memory_N * sizeof(double));
    memcpy(wOrg, source.wOrg, memory_N * sizeof(double));
    memcpy(tmp_array1, source.tmp_array1, memory_N * sizeof(double));
    memcpy(tmp_array2, source.tmp_array2, memory_N * sizeof(double));
    memcpy(xTimesBeta, source.xTimesBeta, memory_N * K * sizeof(double));
    memcpy(beta, source.beta, memory_P * K * sizeof(double));
    memcpy(offset, source.offset, K * sizeof(double));

    if (type >= COX) {
        memcpy(status, source.status, memory_N * 2 * sizeof(int));
        memcpy(d, source.d, memory_N * sizeof(double));
    }
    if (isFusion) {
        memcpy(fusionPartialSums, source.fusionPartialSums,
               memory_nc * K * sizeof(double));
        memcpy(fusionPartialSumsTmp, source.fusionPartialSumsTmp,
               nc * sizeof(double));
        memcpy(fusionSums, source.fusionSums, K * sizeof(double));
    }
}

void RegressionDataScheme::regressionDataSchemePointerMove(
    RegressionDataScheme& source) {
    y = source.y;
    d = source.d;
    status = source.status;
    w = source.w;
    wOrg = source.wOrg;
    tmp_array1 = source.tmp_array1;
    tmp_array2 = source.tmp_array2;
    xTimesBeta = source.xTimesBeta;
    beta = source.beta;
    offset = source.offset;
    fusionPartialSums = source.fusionPartialSums;
    fusionPartialSumsTmp = source.fusionPartialSumsTmp;
    fusionSums = source.fusionSums;

    source.y = nullptr;
    source.d = nullptr;
    source.status = nullptr;
    source.w = nullptr;
    source.wOrg = nullptr;
    source.tmp_array1 = nullptr;
    source.tmp_array2 = nullptr;
    source.xTimesBeta = nullptr;
    source.beta = nullptr;
    source.offset = nullptr;
    source.fusionPartialSums = nullptr;
    source.fusionPartialSumsTmp = nullptr;
    source.fusionSums = nullptr;
}

RegressionDataScheme::RegressionDataScheme() {
    x = nullptr;
    yOrg = nullptr;
    status = nullptr;
    d = nullptr;
    v = nullptr;
    u = nullptr;

    foldid = nullptr;
    nFold = 0;

    lambdaSeq = nullptr;
    gammaSeq = nullptr;

    lengthLambda = 0;
    lengthGamma = 0;

    type = 0;
    isFusion = false;
    isZeroSum = false;

    N = 0;
    P = 0;
    K = 0;
    nc = 0;

    memory_N = 0;
    memory_P = 0;
    memory_nc = 0;

    fusionKernel = nullptr;
    fusionPartialSums = nullptr;
    fusionPartialSumsTmp = nullptr;
    fusionSums = nullptr;

    cSum = 0.0;
    alpha = 0.0;
    lambda = 0.0;
    gamma = 0.0;
    downScaler = 0.0;
    precision = 0.0;

    diagonalMoves = 0;
    useOffset = 0;
    useApprox = 0;
    algorithm = 0;
    polish = 0;
    verbose = 0;
    cores = 0;
    cvStop = 0;

    loglikelihood = 0.0;
    lasso = 0.0;
    ridge = 0.0;
    fusion = 0.0;
    cost = 0.0;
}

RegressionDataScheme::RegressionDataScheme(int _N,
                                           int _P,
                                           int _K,
                                           int _nc,
                                           int _type)
    : RegressionDataScheme() {
    type = _type;
    if (type == FUSION_GAUSSIAN || type == FUSION_GAUSSIAN_ZS ||
        type == FUSION_BINOMIAL || type == FUSION_BINOMIAL_ZS ||
        type == FUSION_MULTINOMIAL || type == FUSION_MULTINOMIAL_ZS ||
        type == FUSION_COX || type == FUSION_COX_ZS) {
        isFusion = TRUE;
    } else {
        isFusion = FALSE;
    }

    if (type % 2 == 0)
        isZeroSum = TRUE;
    else
        isZeroSum = FALSE;

    N = _N;
    P = _P;
    K = _K;
    nc = _nc;

    memory_N = N;

    memory_P = P;
    memory_nc = nc;

#ifdef AVX_VERSION
    if (memory_N % ALIGNED_DOUBLES != 0)
        memory_N += ALIGNED_DOUBLES - memory_N % ALIGNED_DOUBLES;

    if (memory_P % ALIGNED_DOUBLES != 0)
        memory_P += ALIGNED_DOUBLES - memory_P % ALIGNED_DOUBLES;

    if (memory_nc % ALIGNED_DOUBLES != 0)
        memory_nc += ALIGNED_DOUBLES - memory_nc % ALIGNED_DOUBLES;
#endif

    regressionDataSchemeAlloc();

    memset(y, 0.0, memory_N * K * sizeof(double));
    memset(w, 0.0, memory_N * sizeof(double));
    memset(wOrg, 0.0, memory_N * sizeof(double));
    memset(tmp_array1, 0.0, memory_N * sizeof(double));
    memset(tmp_array2, 0.0, memory_N * sizeof(double));
    memset(xTimesBeta, 0.0, memory_N * K * sizeof(double));
    memset(beta, 0.0, memory_P * K * sizeof(double));
    memset(offset, 0.0, K * sizeof(double));

    if (type >= COX) {
        memset(status, 0.0, memory_N * 2 * sizeof(int));
        memset(d, 0.0, memory_N * sizeof(double));
    }

    if (isFusion) {
        memset(fusionPartialSums, 0.0, memory_nc * K * sizeof(double));
        memset(fusionPartialSumsTmp, 0.0, nc * sizeof(double));
        memset(fusionSums, 0.0, K * sizeof(double));
    }
}

RegressionDataScheme::~RegressionDataScheme() {
    regressionDataSchemeFree();
}

RegressionDataScheme::RegressionDataScheme(const RegressionDataScheme& source) {
    regressionDataSchemeShallowCopy(source);
    regressionDataSchemeAlloc();
    regressionDataSchemeDeepCopy(source);
}

RegressionDataScheme::RegressionDataScheme(RegressionDataScheme&& source) {
    regressionDataSchemeShallowCopy(source);
    regressionDataSchemePointerMove(source);
}

RegressionDataScheme& RegressionDataScheme::operator=(
    const RegressionDataScheme& source) {
    regressionDataSchemeFree();
    regressionDataSchemeShallowCopy(source);
    regressionDataSchemeAlloc();
    regressionDataSchemeDeepCopy(source);

    return *this;
}

RegressionDataScheme& RegressionDataScheme::operator=(
    RegressionDataScheme&& source) {
    regressionDataSchemeShallowCopy(source);
    regressionDataSchemePointerMove(source);
    return *this;
}
