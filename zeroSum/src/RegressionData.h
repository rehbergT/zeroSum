#ifndef REGRESSIONDATA_H
#define REGRESSIONDATA_H

#include "RegressionDataScheme.h"
#include "settings.h"
#include <string.h>

class RegressionData : public RegressionDataScheme
{
public:
    RegressionData( int _N, int _P, int _K, int _nc, int _type );
    RegressionData( const RegressionData& source );
    ~RegressionData();
};


#endif /* REGRESSIONDATA_H */
