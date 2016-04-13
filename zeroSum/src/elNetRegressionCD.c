// RBioC CMD SHLIB fit.c -lgsl -lgslcblas -Wall -Wextra
#include "regressions.h"

#define REFRESH 1000

//   #define DEBUG
//   # define DEBUG2

#ifdef DEBUG
#include <time.h>
#endif

double elnet_gamma;

int calcElNetGradient(  struct regressionData *data,
                        const int j,                        
                        double* restrict betasX, 
                        double* restrict denominators)
{
    double* restrict x = (*data).x;
    double* restrict beta = (*data).beta;    
    const int N = (*data).N;
    
    double nominator = 0.0;

    int i1 = INDEX(0,j,N);

    for( int i=0; i<N; ++i, ++i1 )
    {
        nominator +=  x[ i1 ] * ( betasX[i] + x[ i1 ] * beta[j] );
    }
 
    double betaj = 0.0;
    if( nominator > 0.0  && nominator > elnet_gamma )
    {
        betaj = ( nominator - elnet_gamma ) / denominators[j];
    }
    else if( nominator < 0.0  && -nominator > elnet_gamma )
    {
        betaj = ( nominator + elnet_gamma ) / denominators[j];
    }

    double diff = beta[j] - betaj;
    
    i1 = INDEX(0,j,N);
    for( int i=0; i<N; ++i, ++i1 )
        betasX[i] += x[ i1 ] * diff;
    beta[j] = betaj;
    
    if(fabs( diff) < DBL_EPSILON * 100 )
        return 0;
    else
        return 1;

}

void calcOffsetElNetGradient(   struct regressionData *data,
                                double* restrict betasX)
{
    double* restrict beta = (*data).beta;    
    const int N = (*data).N;
    
    double oldbeta0 = beta[0];
    beta[0] = 0.0;
    for( int i=0; i<N; ++i )
    {
        beta[0] += betasX[i] + oldbeta0;
    }
    beta[0] /= N;
    
    double tmp = oldbeta0 - beta[0];
    for( int i=0; i<N; ++i )
    {
        betasX[i] += tmp; 
    }
}


void elNetRefresh(  struct regressionData *data,
                    double* restrict betasX )
{
    double* restrict x = (*data).x;
    double* restrict y = (*data).y;
    double* restrict beta = (*data).beta;
    
    const int N = (*data).N;
    const int P = (*data).P;    
    
    for( int i=0; i<N; ++i )
    {
        betasX[i] = y[i] - beta[0];
        for( int j=1; j<P; ++j )
        {
            betasX[i] -= x[ INDEX(i,j,N) ] * beta[j];
        }
    }
}



void elNetRegressionCD( struct regressionData data )
{
    
    #ifdef DEBUG2
    double timet;
    struct timespec ts0, ts1;
    clock_gettime(CLOCK_REALTIME , &ts0);
    #endif
     
    // random numer generation
    #ifdef R_PACKAGE
    GetRNGstate();
    #endif
    
    const int P = data.P;
    
    elnet_gamma = data.lambda * data.alpha * data.N;    
    
    double* denominators = (double*)malloc( P * sizeof(double));
    memset( denominators, 0, P * sizeof(double) );

    double tmp;
    double tmp2 = data.lambda * ( 1.0 - data.alpha ) * data.N;
    for( int j=1; j<P; ++j )
    {
        for( int i=0; i<data.N; ++i )
        {
            tmp = data.x[ INDEX(i,j,data.N) ];
            denominators[j] += tmp * tmp;
        }
        denominators[j] += tmp2;
    }

    double* betasX = (double*)malloc( data.N * sizeof(double));
    elNetRefresh( &data, betasX );

    double* res = (double*)malloc( data.N * sizeof(double));
    double ridge = 0.0;
    double lasso = 0.0;
    double residum = 0.0;

    #ifdef DEBUG2
    double energy1, energy2, energy3;
    vectorElNetCostFunction( &data, res, &energy1, &residum, &ridge, &lasso);
    PRINT("Initial Energy: %e  res=%e l=%e r=%e\n", energy1, residum,lasso, ridge);

    double betaold;
    #endif

    double energynew;
    double energyold;

    size_t activeSetSize = (size_t)ceil( P/32.0 );
    int* activeset = (int*) malloc( activeSetSize * sizeof(int) );
    memset( activeset, 0, activeSetSize * sizeof(int));

    int activesetChange = 0;
    int step=0;

    int* ind = (int*) malloc( P * sizeof(int) );
    int refreshCounter = 0;
    while( 1 )
    {
        #ifdef DEBUG
        PRINT("Step: %d\nFind active set\n", step);
        #endif

        if( data.offset == TRUE )
        {
            calcOffsetElNetGradient( &data, betasX);
            ++refreshCounter;
        }            
        
        activesetChange = 0;       
        fisherYates(ind, P);

        for( int k=1; k<P; ++k )
        {                        
            int j = ind[k];

            #ifdef DEBUG
            vectorElNetCostFunction( &data, res, &energyold, &residum, &ridge, &lasso);
            #endif
            
            int change = calcElNetGradient( &data, j, betasX, denominators);
            
            if( change == 1 && TestBit( activeset, j ) == 0 )
            {
                activesetChange = 1;
                SetBit( activeset, j );
                
                if( data.offset == TRUE )
                    calcOffsetElNetGradient( &data, betasX);
                
                ++refreshCounter;
            }
            
            #ifdef DEBUG
            vectorElNetCostFunction( &data, res, &energynew, &residum, &ridge, &lasso);
            PRINT("Vorher:j=%d  activesetChange: %d activeset: %d energy: %e deltaE: %e\n",
                    j, activesetChange, TestBit( activeset, j )!=0, energynew, energynew-energyold );
            #endif
        }
                
 
        if( activesetChange == 0  ) break;
               
        
        #ifdef DEBUG
        PRINT("converge\n");
        #endif
        
        int convergence = 0;
        // cycle on active set until convergence
        while( convergence == 0 )
        {
            vectorElNetCostFunction( &data, res, &energyold, &residum, &ridge, &lasso);
            int test = 0;
            
            fisherYates(ind, P);  

            for( int k=1; k<P; ++k )
            {
                int j = ind[k];
                if( TestBit( activeset, j ) == 0 ) continue;
                
                int change = calcElNetGradient( &data, j, betasX, denominators);
                
                if( change == 1)
                {
                    if( data.offset == TRUE )
                        calcOffsetElNetGradient( &data, betasX);
                    
                    ++test;
                }
                
            }
                       
            vectorElNetCostFunction( &data, res, &energynew, &residum, &ridge, &lasso);

            #ifdef DEBUG
            PRINT("Energy before: %e\t  later %e\tDif %e\n", energyold, energynew, energynew-energyold );
            #endif
            
            if( test == 0 || (energyold-energynew)/(energynew * (double)test) < data.precision )
                convergence = 1;

            refreshCounter += test;
            if(refreshCounter >= REFRESH)
            {
                elNetRefresh( &data, betasX );
                refreshCounter = 0;
            }
          
          
            #ifdef R_PACKAGE
            R_CheckUserInterrupt();
            #endif
        }

        
        if(step==100) break;
        ++step;

    }

    #ifdef DEBUG2
    vectorElNetCostFunction( &data, res, &energynew, &residum, &ridge, &lasso);

    PRINT("Energy before: %e\t  later %e\tDif %e\nTEST %e\n",
            energy1, energynew, energynew-energy1 );

    clock_gettime(CLOCK_REALTIME , &ts1);
    timet = (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;
    PRINT("DONE\tDauer in Sekunden: = %e\n", timet);
    #endif

    free(activeset);
    free(res);
    free(betasX);
    free(denominators);
    
    #ifdef R_PACKAGE
    PutRNGstate();
    #endif
}
