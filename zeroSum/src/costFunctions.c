#include "costFunctions.h"


void vectorElNetCostFunction(   
                struct regressionData *data,
                double* res,
                double* energy,
                double* residum,
                double* ridge,
                double* lasso )
{    
    double* restrict x = (*data).x;
    double* restrict y = (*data).y;
    double* restrict beta = (*data).beta;
    
    const int N = (*data).N;
    const int P = (*data).P;
    
    *ridge = squaresum( &beta[1], P-1 );
    *lasso = abssum( &beta[1], P-1 );
   
    memcpy ( res, y, sizeof(double)*N );
    for(int i=0; i<N; i++)
    {
        for(int j=0; j<P; j++)
            res[i] -= x[ INDEX(i,j,N) ] * beta[j];        
    }
  
    *residum = squaresum(res, N) / N;
 
    *energy = (*residum) / ( 2.0 ) 
        + (*data).lambda * ( (1.0 - (*data).alpha) * (*ridge) / 2.0 
        + (*data).alpha * (*lasso)   );
}




