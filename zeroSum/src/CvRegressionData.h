#ifndef CVREGRESSIONDATA_H
#define CVREGRESSIONDATA_H

#include "RegressionDataScheme.h"
#include "RegressionData.h"
#include "settings.h"
#include <string.h>

class CvRegressionData : public RegressionDataScheme
{

private:
    void cvRegressionDataAlloc();
    void cvRegressionDataFree();
    void cvRegressionDataDeepCopy( const CvRegressionData& source );

public:
    CvRegressionData( RegressionData& data );
    CvRegressionData( const CvRegressionData& source );
    CvRegressionData( CvRegressionData&& source );

    ~CvRegressionData();

    CvRegressionData& operator=( const CvRegressionData& source );
    CvRegressionData& operator=( CvRegressionData&& source );

    double* wCV;
};


#endif /* CVREGRESSIONDATA_H */
