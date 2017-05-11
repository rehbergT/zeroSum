#include "CvRegressionData.h"


void CvRegressionData::cvRegressionDataAlloc()
{
    #ifdef AVX_VERSION
        wCV = (double*)aligned_alloc( ALIGNMENT, memory_N * sizeof(double) );
    #else
        wCV = (double*)malloc( memory_N * sizeof(double) );
    #endif
}

void CvRegressionData::cvRegressionDataFree()
{
    free(wCV);
}

void CvRegressionData::cvRegressionDataDeepCopy(const CvRegressionData& source)
{
    memcpy( wCV, source.wCV, memory_N * sizeof(double) );
}

CvRegressionData::CvRegressionData( RegressionData& source ) :
    RegressionDataScheme( source.N, source.P, source.K, source.nc, source.type )
{
    regressionDataSchemeShallowCopy(source);
    regressionDataSchemeDeepCopy(source);

    cvRegressionDataAlloc();
    memcpy( wCV, source.wOrg, memory_N * sizeof(double) );
}

CvRegressionData::CvRegressionData( const CvRegressionData& source ) :
    RegressionDataScheme( source.N, source.P, source.K, source.nc, source.type )
{
    regressionDataSchemeShallowCopy(source);
    regressionDataSchemeDeepCopy(source);

    cvRegressionDataAlloc();
    cvRegressionDataDeepCopy(source);
}


CvRegressionData::CvRegressionData( CvRegressionData&& source )
{
    regressionDataSchemeShallowCopy(source);
    regressionDataSchemePointerMove(source);

    wCV = source.wCV;
    source.wCV = nullptr;
}

CvRegressionData& CvRegressionData::operator=( const CvRegressionData& source )
{
    cvRegressionDataFree();
    regressionDataSchemeFree();

    regressionDataSchemeShallowCopy(source);
    regressionDataSchemeAlloc();
    regressionDataSchemeDeepCopy(source);

    cvRegressionDataAlloc();
    cvRegressionDataDeepCopy(source);

    return *this;
}

CvRegressionData& CvRegressionData::operator=( CvRegressionData&& source )
{
    regressionDataSchemeShallowCopy(source);
    regressionDataSchemePointerMove(source);

    wCV = source.wCV;
    source.wCV = nullptr;

    return *this;
}

CvRegressionData::~CvRegressionData()
{
    cvRegressionDataFree();
}
