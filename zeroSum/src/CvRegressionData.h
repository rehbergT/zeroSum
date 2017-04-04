#ifndef CVREGRESSIONDATA_H
#define CVREGRESSIONDATA_H

#include "RegressionDataScheme.h"
#include "RegressionData.h"
#include "settings.h"
#include <string.h>

class CvRegressionData : public RegressionDataScheme
{
public:
    CvRegressionData( RegressionData* data );
    CvRegressionData( const CvRegressionData& source );
    ~CvRegressionData();
    double* wCV;
};


#endif /* CVREGRESSIONDATA_H */
