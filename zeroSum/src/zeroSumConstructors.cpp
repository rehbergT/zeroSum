#include "zeroSum.h"

void zeroSum::allocData() {
    lambda = (double*)malloc(nFold1 * sizeof(double));
    loglikelihood = (double*)malloc(nFold1 * sizeof(double));
    lasso = (double*)malloc(nFold1 * sizeof(double));
    ridge = (double*)malloc(nFold1 * sizeof(double));
    fusion = (double*)malloc(nFold1 * sizeof(double));
    cost = (double*)malloc(nFold1 * sizeof(double));
    featureMean = (double*)malloc(memory_P * sizeof(double));
    ySD = (double*)malloc(nFold1 * sizeof(double));

    x = (double*)malloc(memory_N * memory_P * sizeof(double));
    y = (double*)malloc(memory_N * K * nFold1 * sizeof(double));
    yOrg = (double*)malloc(memory_N * K * sizeof(double));
    w = (double*)malloc(memory_N * K * nFold1 * sizeof(double));
    wCV = (double*)malloc(memory_N * nFold1 * sizeof(double));
    wOrg = (double*)malloc(memory_N * nFold1 * sizeof(double));
    tmp_array1 = (double*)malloc(memory_N * nFold1 * sizeof(double));
    tmp_array2 = (double*)malloc(memory_N * nFold1 * K * sizeof(double));
    xTimesBeta = (double*)malloc(memory_N * K * nFold1 * sizeof(double));
    beta = (double*)malloc(memory_P * K * nFold1 * sizeof(double));
    intercept = (double*)malloc(K * nFold1 * sizeof(double));
    v = (double*)malloc(memory_P * nFold1 * sizeof(double));
    u = (double*)malloc(memory_P * sizeof(double));

    memset(lambda, 0.0, nFold1 * sizeof(double));
    memset(loglikelihood, 0.0, nFold1 * sizeof(double));
    memset(lasso, 0.0, nFold1 * sizeof(double));
    memset(ridge, 0.0, nFold1 * sizeof(double));
    memset(cost, 0.0, nFold1 * sizeof(double));
    memset(featureMean, 0.0, memory_P * sizeof(double));

    for (uint32_t f = 0; f < nFold1; f++)
        ySD[f] = 1.0;

    memset(x, 0.0, memory_N * memory_P * sizeof(double));
    memset(y, 0.0, memory_N * K * nFold1 * sizeof(double));
    memset(yOrg, 0.0, memory_N * K * sizeof(double));
    memset(w, 0.0, memory_N * K * nFold1 * sizeof(double));
    memset(wCV, 0.0, memory_N * nFold1 * sizeof(double));
    memset(wOrg, 0.0, memory_N * nFold1 * sizeof(double));
    memset(tmp_array1, 0.0, memory_N * nFold1 * sizeof(double));
    memset(tmp_array2, 0.0, memory_N * nFold1 * K * sizeof(double));
    memset(xTimesBeta, 0.0, memory_N * K * nFold1 * sizeof(double));
    memset(beta, 0.0, memory_P * K * nFold1 * sizeof(double));
    memset(intercept, 0.0, K * nFold1 * sizeof(double));
    memset(v, 0.0, memory_P * nFold1 * sizeof(double));
    memset(u, 0.0, memory_P * sizeof(double));

    if (type == cox) {
        status = (uint32_t*)malloc(memory_N * sizeof(uint32_t));
        d = (double*)malloc(memory_N * nFold1 * sizeof(double));

        memset(status, 0, memory_N * sizeof(uint32_t));
        memset(d, 0.0, memory_N * nFold1 * sizeof(double));
    }

    if (useFusion) {
        fusionPartialSums =
            (double*)malloc(memory_nc * K * nFold1 * sizeof(double));
        fusionPartialSumsTmp =
            (double*)malloc(memory_nc * nFold1 * sizeof(double));
        fusionSums = (double*)malloc(K * nFold1 * sizeof(double));

        memset(fusion, 0.0, nFold1 * sizeof(double));
        memset(fusionPartialSums, 0.0, memory_nc * K * nFold1 * sizeof(double));
        memset(fusionPartialSumsTmp, 0.0, memory_nc * nFold1 * sizeof(double));
        memset(fusionSums, 0.0, K * nFold1 * sizeof(double));

        fusionKernel =
            (struct fusionKernel**)malloc(P * sizeof(struct fusionKernel*));
        for (uint32_t j = 0; j < P; j++)
            fusionKernel[j] = nullptr;
    }

    approxFailed.resize(nFold1);
    activeSet.resize(nFold1);
    parallelActiveSet.resize(nFold1);

    // some memory for saving the last good coefficients
    last_beta = (double*)malloc(memory_P * K * nFold1 * sizeof(double));
    last_intercept = (double*)malloc(K * nFold1 * sizeof(double));
}

zeroSum::zeroSum(uint32_t N,
                 uint32_t P,
                 uint32_t K,
                 uint32_t nc,
                 uint32_t type,
                 bool useZeroSum,
                 bool useFusion,
                 bool useIntercept,
                 bool useApprox,
                 bool useCentering,
                 bool useStandardization,
                 bool usePolish,
                 uint32_t rotatedUpdates,
                 double precision,
                 uint32_t algorithm,
                 uint32_t nFold,
                 uint32_t cvStop,
                 uint32_t verbose,
                 double cSum,
                 double alpha,
                 double downScaler,
                 uint32_t threads,
                 uint32_t seed)
    : N(N),
      P(P),
      K(K),
      nc(nc),
      type(type),
      useZeroSum(useZeroSum),
      useFusion(useFusion),
      useIntercept(useIntercept),
      useApprox(useApprox),
      useCentering(useCentering),
      useStandardization(useStandardization),
      usePolish(usePolish),
      rotatedUpdates(rotatedUpdates),
      precision(precision),
      algorithm(algorithm),
      nFold(nFold),
      cvStop(cvStop),
      verbose(verbose),
      cSum(cSum),
      alpha(alpha),
      downScaler(downScaler),
      threads(threads),
      parallel(threads),
      seed(seed) {
    nFold1 = nFold + 1;
    memory_N = N;
    memory_P = P;
    memory_nc = nc;

    uint32_t alignedDoubles = 0;
#ifdef BUILD_WITH_AVX2
    if (__builtin_cpu_supports("avx2")) {
        avxType = avx2;
        alignedDoubles = 4;
    }
#endif

#ifdef BUILD_WITH_AVX512
    if (__builtin_cpu_supports("avx512f")) {
        avxType = avx512;
        alignedDoubles = 8;
    }
#endif

#if defined _OPENMP
    omp_set_num_threads(parallel.maxThreads);
#endif

    if (avxType != fallback) {
        if (memory_N % alignedDoubles != 0)
            memory_N += alignedDoubles - memory_N % alignedDoubles;

        if (memory_P % alignedDoubles != 0)
            memory_P += alignedDoubles - memory_P % alignedDoubles;

        if (memory_nc % alignedDoubles != 0)
            memory_nc += alignedDoubles - memory_nc % alignedDoubles;
    }

    allocData();
}

void zeroSum::freeData() {
    free(last_beta);
    free(last_intercept);

    if (useFusion && fusionKernel != nullptr) {
        for (uint32_t j = 0; j < P; ++j) {
            struct fusionKernel* currEl = fusionKernel[j];
            struct fusionKernel* nextEl;

            while (currEl != NULL) {
                nextEl = currEl->next;
                free(currEl);
                currEl = nextEl;
            }
        }
        free(fusionKernel);
        free(fusionSums);
        free(fusionPartialSumsTmp);
        free(fusionPartialSums);
    }

    if (type == cox) {
        free(d);
        free(status);
    }

    free(u);
    free(v);
    free(intercept);
    free(beta);
    free(xTimesBeta);
    free(tmp_array2);
    free(tmp_array1);
    free(wCV);
    free(wOrg);
    free(w);
    free(yOrg);
    free(y);
    free(x);
    free(ySD);
    free(featureMean);
    free(cost);
    free(fusion);
    free(ridge);
    free(lasso);
    free(loglikelihood);
    free(lambda);
}

void zeroSum::shallowCopy(const zeroSum& source) {
    // the following just copy all uint32_t/doubles
    type = source.type;
    nFold = source.nFold;
    useZeroSum = source.useZeroSum;
    useFusion = source.useFusion;
    N = source.N;
    P = source.P;
    K = source.K;
    nc = source.nc;
    memory_N = source.memory_N;
    memory_P = source.memory_P;
    memory_nc = source.memory_nc;
    algorithm = source.algorithm;
    useApprox = source.useApprox;
    useCentering = source.useCentering;
    useStandardization = source.useStandardization;
    usePolish = source.usePolish;
    rotatedUpdates = source.rotatedUpdates;
    precision = source.precision;
    cvStop = source.cvStop;
    verbose = source.verbose;
    lambda = source.lambda;
    gamma = source.gamma;
    cSum = source.cSum;
    alpha = source.alpha;
    downScaler = source.downScaler;
    threads = source.threads;
    seed = source.seed;

    // copy std vectors
    lambdaSeq = source.lambdaSeq;
    gammaSeq = source.gammaSeq;
    foldid = source.foldid;
    approxFailed = source.approxFailed;
    activeSet = source.activeSet;
    parallelActiveSet = source.parallelActiveSet;
}

void zeroSum::deepCopy(const zeroSum& source) {
    memcpy(lambda, source.lambda, nFold1 * sizeof(double));
    memcpy(loglikelihood, source.loglikelihood, nFold1 * sizeof(double));
    memcpy(lasso, source.lasso, nFold1 * sizeof(double));
    memcpy(ridge, source.ridge, nFold1 * sizeof(double));
    memcpy(cost, source.cost, nFold1 * sizeof(double));
    memcpy(featureMean, source.featureMean, memory_P * sizeof(double));

    for (uint32_t f = 0; f < nFold1; f++)
        ySD[f] = 1.0;

    memcpy(x, source.x, memory_N * P * sizeof(double));
    memcpy(y, source.y, memory_N * K * nFold1 * sizeof(double));
    memcpy(w, source.w, memory_N * K * nFold1 * sizeof(double));
    memcpy(wCV, source.wCV, memory_N * nFold1 * sizeof(double));
    memcpy(wOrg, source.wOrg, memory_N * nFold1 * sizeof(double));

    memcpy(xTimesBeta, source.xTimesBeta,
           memory_N * K * nFold1 * sizeof(double));
    memcpy(beta, source.beta, memory_P * K * nFold1 * sizeof(double));
    memcpy(intercept, source.intercept, K * nFold1 * sizeof(double));

    memcpy(v, source.v, memory_P * nFold1 * sizeof(double));
    memcpy(u, source.u, memory_P * sizeof(double));

    memcpy(yOrg, source.yOrg, memory_N * K * sizeof(double));

    if (type == cox) {
        memcpy(status, source.status, memory_N * sizeof(uint32_t));
        memcpy(d, source.d, memory_N * nFold1 * sizeof(double));
    }
    if (useFusion) {
        memcpy(fusion, source.fusion, nFold1 * sizeof(double));
        memcpy(fusionPartialSums, source.fusionPartialSums,
               memory_nc * K * nFold1 * sizeof(double));
        memcpy(fusionPartialSumsTmp, source.fusionPartialSumsTmp,
               memory_nc * nFold1 * sizeof(double));
        memcpy(fusionSums, source.fusionSums, K * nFold1 * sizeof(double));

        for (uint32_t j = 0; j < P; j++) {
            struct fusionKernel* currEl = source.fusionKernel[j];

            while (currEl != nullptr) {
                fusionKernel[j] =
                    appendElement(fusionKernel[j], currEl->i, currEl->value);
                currEl = currEl->next;
            }
        }
    }
}

void zeroSum::pointerMove(zeroSum& source) {
    x = source.x;
    yOrg = source.yOrg;
    y = source.y;
    wOrg = source.wOrg;
    w = source.w;
    wCV = source.wCV;
    xTimesBeta = source.xTimesBeta;
    beta = source.beta;
    intercept = source.intercept;
    status = source.status;
    d = source.d;
    tmp_array1 = source.tmp_array1;
    tmp_array2 = source.tmp_array2;
    v = source.v;
    u = source.u;

    featureMean = source.featureMean;
    ySD = source.ySD;

    fusionKernel = source.fusionKernel;
    fusionPartialSums = source.fusionPartialSums;
    fusionPartialSumsTmp = source.fusionPartialSumsTmp;
    fusionSums = source.fusionSums;

    lambda = source.lambda;
    loglikelihood = source.loglikelihood;
    lasso = source.lasso;
    ridge = source.ridge;
    fusion = source.fusion;
    cost = source.cost;
    ySD = source.ySD;

    source.x = nullptr;
    source.yOrg = nullptr;
    source.y = nullptr;
    source.w = nullptr;
    source.wOrg = nullptr;
    source.wCV = nullptr;
    source.xTimesBeta = nullptr;
    source.beta = nullptr;
    source.intercept = nullptr;
    source.status = nullptr;
    source.d = nullptr;
    source.tmp_array1 = nullptr;
    source.tmp_array2 = nullptr;
    source.v = nullptr;
    source.u = nullptr;

    source.featureMean = nullptr;
    source.ySD = nullptr;

    source.fusionKernel = nullptr;
    source.fusionPartialSums = nullptr;
    source.fusionPartialSumsTmp = nullptr;
    source.fusionSums = nullptr;

    source.loglikelihood = nullptr;
    source.lasso = nullptr;
    source.ridge = nullptr;
    source.fusion = nullptr;
    source.cost = nullptr;
}

// copy constructor
zeroSum::zeroSum(const zeroSum& source) : parallel(source.threads) {
    shallowCopy(source);
    allocData();
    deepCopy(source);
}

// move constructor
zeroSum::zeroSum(zeroSum&& source) : parallel(source.threads) {
    shallowCopy(source);
    pointerMove(source);
}

// copy assignment operator
zeroSum& zeroSum::operator=(const zeroSum& source) {
    freeData();
    shallowCopy(source);
    allocData();
    deepCopy(source);
    return *this;
}

// move assignment operator
zeroSum& zeroSum::operator=(zeroSum&& source) {
    shallowCopy(source);
    pointerMove(source);
    return *this;
}

// destructor
zeroSum::~zeroSum() {
    freeData();
}
