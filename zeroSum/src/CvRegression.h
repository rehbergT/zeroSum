#ifndef CV_REGRESSIONDATA_H
#define CV_REGRESSIONDATA_H

#include <vector>
#include "CvRegressionData.h"
#include "RegressionData.h"

class CvRegression
{
public:
    std::vector<CvRegressionData> cv_data;

    std::vector<double> cv_tmp;
    std::vector<double> cv_predict;

    CvRegression( RegressionData* data );
};



#endif /* CV_REGRESSIONDATA_H */
