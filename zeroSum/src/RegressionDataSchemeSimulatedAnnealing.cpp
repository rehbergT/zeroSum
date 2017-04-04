#include "RegressionDataScheme.h"

void RegressionDataScheme::simulatedAnnealing( int seed )
{

#ifdef DEBUG
    double timet, costStart, costEnd;
    struct timespec ts0, ts1;
    clock_gettime(CLOCK_REALTIME , &ts0);
    PRINT("Starting SA\n");
#endif

    // init every variable especially xTimesBeta
    costFunction();

#ifdef DEBUG
    costStart = cost;
#endif


    PRINT("Loglikelihood: %e lasso: %e ridge: %e cost: %e\n",
           loglikelihood, lasso, ridge, cost );

#ifdef DEBUG
    costFunction();
    costEnd = cost;

    PRINT("Cost start: %e cost end: %e diff %e\n",
            costStart, costEnd, costStart-costEnd );

    clock_gettime(CLOCK_REALTIME , &ts1);
    timet = (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;
    PRINT("time taken = %e s\n", timet);
#endif


}
