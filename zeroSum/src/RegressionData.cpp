#include "RegressionData.h"

void RegressionData::regressionDataAlloc() {
#ifdef AVX_VERSION
    x = (double*)aligned_alloc(ALIGNMENT, memory_N * P * sizeof(double));
    v = (double*)aligned_alloc(ALIGNMENT, memory_P * sizeof(double));
    u = (double*)aligned_alloc(ALIGNMENT, memory_P * sizeof(double));
#else
    x = (double*)malloc(memory_N * P * sizeof(double));
    v = (double*)malloc(memory_P * sizeof(double));
    u = (double*)malloc(memory_P * sizeof(double));
#endif

    if (type > 4) {
#ifdef AVX_VERSION
        yOrg = (double*)aligned_alloc(ALIGNMENT, memory_N * K * sizeof(double));
#else
        yOrg = (double*)malloc(memory_N * K * sizeof(double));
#endif
    }
    if (isFusion) {
        fusionKernel =
            (struct fusionKernel**)malloc(P * sizeof(struct fusionKernel*));
        for (int j = 0; j < P; j++)
            fusionKernel[j] = NULL;
    }
}

void RegressionData::regressionDataFree() {
    free(x);
    free(v);
    free(u);

    if (type > 4)
        free(yOrg);

    if (isFusion) {
        for (int j = 0; j < P; ++j) {
            struct fusionKernel* currEl = fusionKernel[j];
            struct fusionKernel* nextEl;

            while (currEl != NULL) {
                nextEl = currEl->next;
                free(currEl);
                currEl = nextEl;
            }
        }
        free(fusionKernel);
    }
}

void RegressionData::regressionDataDeepCopy(const RegressionData& source) {
    memcpy(x, source.x, memory_N * P * sizeof(double));
    memcpy(v, source.v, memory_P * sizeof(double));
    memcpy(u, source.u, memory_P * sizeof(double));

    if (type > 4)
        memcpy(yOrg, source.y, memory_N * K * sizeof(double));

    if (isFusion) {
        for (int j = 0; j < P; j++) {
            struct fusionKernel* currEl = source.fusionKernel[j];

            while (currEl != NULL) {
                fusionKernel[j] =
                    appendElement(fusionKernel[j], currEl->i, currEl->value);
                currEl = currEl->next;
            }
        }
    }
}

void RegressionData::regressionDataPointerMove(RegressionData& source) {
    x = source.x;
    v = source.v;
    u = source.u;
    yOrg = source.yOrg;
    fusionKernel = source.fusionKernel;

    source.x = nullptr;
    source.v = nullptr;
    source.u = nullptr;
    source.yOrg = nullptr;
    source.fusionKernel = nullptr;
}

RegressionData::RegressionData() : RegressionDataScheme() {}

RegressionData::RegressionData(int _N, int _P, int _K, int _nc, int _type)
    : RegressionDataScheme(_N, _P, _K, _nc, _type) {
    regressionDataAlloc();
    memset(x, 0.0, memory_N * P * sizeof(double));
    memset(v, 0.0, memory_P * sizeof(double));
    memset(u, 0.0, memory_P * sizeof(double));

    if (type > 4)
        memset(yOrg, 0.0, memory_N * K * sizeof(double));
}

RegressionData::RegressionData(const RegressionData& source)
    : RegressionDataScheme(source) {
    regressionDataAlloc();
    regressionDataDeepCopy(source);
}

RegressionData::~RegressionData() {
    regressionDataFree();
}

RegressionData& RegressionData::operator=(const RegressionData& source) {
    regressionDataFree();
    regressionDataSchemeFree();

    regressionDataSchemeShallowCopy(source);
    regressionDataSchemeAlloc();
    regressionDataSchemeDeepCopy(source);

    regressionDataAlloc();
    regressionDataDeepCopy(source);

    return *this;
}

RegressionData& RegressionData::operator=(RegressionData&& source) {
    regressionDataSchemeShallowCopy(source);
    regressionDataSchemePointerMove(source);
    regressionDataPointerMove(source);

    return *this;
}
