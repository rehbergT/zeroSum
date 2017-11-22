#ifndef REGRESSIONDATA_H
#define REGRESSIONDATA_H

#include <string.h>
#include "RegressionDataScheme.h"
#include "settings.h"

class RegressionData : public RegressionDataScheme {
   private:
    void regressionDataAlloc();
    void regressionDataFree();
    void regressionDataDeepCopy(const RegressionData& source);
    void regressionDataPointerMove(RegressionData& source);

   public:
    RegressionData();
    RegressionData(int _N, int _P, int _K, int _nc, int _type);
    RegressionData(const RegressionData& source);
    ~RegressionData();

    RegressionData& operator=(const RegressionData& source);
    RegressionData& operator=(RegressionData&& source);
};

#endif /* REGRESSIONDATA_H */
