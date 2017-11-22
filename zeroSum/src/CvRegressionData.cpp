#include "CvRegressionData.h"

void CvRegressionData::cvRegressionDataAlloc() {
#ifdef AVX_VERSION
    wCV = (double*)aligned_alloc(ALIGNMENT, memory_N * sizeof(double));
#else
    wCV = (double*)malloc(memory_N * sizeof(double));
#endif
}

void CvRegressionData::cvRegressionDataFree() {
    free(wCV);
}

void CvRegressionData::cvRegressionDataDeepCopy(
    const CvRegressionData& source) {
    memcpy(wCV, source.wCV, memory_N * sizeof(double));
}

CvRegressionData::CvRegressionData(RegressionData& source)
    : RegressionDataScheme(source.N,
                           source.P,
                           source.K,
                           source.nc,
                           source.type) {
    // PRINT("special copy constructor\n");
    regressionDataSchemeShallowCopy(source);
    regressionDataSchemeDeepCopy(source);
    cvRegressionDataAlloc();
    memcpy(wCV, source.wOrg, memory_N * sizeof(double));
}

// copy constructor
CvRegressionData::CvRegressionData(const CvRegressionData& source)
    : RegressionDataScheme(source.N,
                           source.P,
                           source.K,
                           source.nc,
                           source.type) {
    // PRINT("copy constructor\n");
    regressionDataSchemeShallowCopy(source);
    regressionDataSchemeDeepCopy(source);

    cvRegressionDataAlloc();
    cvRegressionDataDeepCopy(source);
}

// move constructor
CvRegressionData::CvRegressionData(CvRegressionData&& source) {
    // PRINT("move constructor\n");
    regressionDataSchemeShallowCopy(source);
    regressionDataSchemePointerMove(source);

    wCV = source.wCV;
    source.wCV = nullptr;
}

// copy assignment operator
CvRegressionData& CvRegressionData::operator=(const CvRegressionData& source) {
    // PRINT("copy assignment\n");
    cvRegressionDataFree();
    regressionDataSchemeFree();

    regressionDataSchemeShallowCopy(source);
    regressionDataSchemeAlloc();
    regressionDataSchemeDeepCopy(source);

    cvRegressionDataAlloc();
    cvRegressionDataDeepCopy(source);

    return *this;
}

// move assignment operator
CvRegressionData& CvRegressionData::operator=(CvRegressionData&& source) {
    // PRINT("move assignment\n");
    regressionDataSchemeShallowCopy(source);
    regressionDataSchemePointerMove(source);

    wCV = source.wCV;
    source.wCV = nullptr;

    return *this;
}

// destructor
CvRegressionData::~CvRegressionData() {
    // PRINT("DESTRUCTOR CV!\n");
    cvRegressionDataFree();
}
