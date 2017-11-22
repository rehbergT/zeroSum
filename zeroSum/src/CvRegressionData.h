#ifndef CVREGRESSIONDATA_H
#define CVREGRESSIONDATA_H

#include <string.h>
#include "RegressionData.h"
#include "RegressionDataScheme.h"
#include "settings.h"

class CvRegressionData : public RegressionDataScheme {
   private:
    void cvRegressionDataAlloc();
    void cvRegressionDataFree();
    void cvRegressionDataDeepCopy(const CvRegressionData& source);

   public:
    CvRegressionData(RegressionData& data);
    CvRegressionData(const CvRegressionData& source);
    CvRegressionData(CvRegressionData&& source);

    ~CvRegressionData();

    CvRegressionData& operator=(const CvRegressionData& source);
    CvRegressionData& operator=(CvRegressionData&& source);

    double* wCV;
};

#endif /* CVREGRESSIONDATA_H */
