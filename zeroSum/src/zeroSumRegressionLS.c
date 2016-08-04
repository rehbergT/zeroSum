// RBioC CMD SHLIB sa.c -lgsl -lgslcblas -Wall -Wextra
#include "regressions.h"

#define MOVE_SCALE 0.05
#define STEP_SIZE 5000

//  #define DEBUG

#ifdef DEBUG
#include <time.h>
#endif


int moveLS( struct regressionData *data,
            double* restrict res,
            double* restrict energy,
            double* restrict residum, 
            double* restrict ridge,
            double* restrict lasso,
            const int from,
            const int to,
            const double amount,
            double* restrict tmp      )
{
    double* restrict x = (*data).x;
    double* restrict beta = (*data).beta;
    const int N = (*data).N;
    
    // copy residums to tmp array in order to reject the move if necessary
    memcpy( tmp, res, sizeof(double) * N );

    int colFrom = INDEX(0, from, N);
    int colTo = INDEX(0, to, N);

    for( int i=0; i<N; ++i )
    {
        tmp[i] +=  amount * ( x[colFrom+i] - x[colTo+i] );
    }

    double tmp_residum = squaresum( tmp, N ) / N;
    double tmp_energy = tmp_residum / 2.0;
    
    double tmp_ridge = *ridge + 2.0 * ( amount * amount + amount * ( beta[to] - beta[from] ) ); 
    double tmp_lasso = *lasso - fabs(beta[from]) - fabs(beta[to]) 
                              + fabs(beta[from]-amount) + fabs(beta[to]+amount);
                              
    tmp_energy += (*data).lambda * ( (1.0 - (*data).alpha) * tmp_ridge / 2.0 + (*data).alpha * tmp_lasso );
    
    double deltaE = tmp_energy - *energy;    

    if( deltaE <= 0.0 )
    {
         memcpy ( res, tmp, sizeof(double) * N );
         *ridge = tmp_ridge;
         *lasso = tmp_lasso;
         *energy = tmp_energy;
         *residum = tmp_residum;
         beta[from] -= amount;
         beta[to] += amount;        
         return 1;
    }
    else{
        return 0;
    }
}

int moveLSOffset(   struct regressionData *data,
                    double* restrict res,
                    double* restrict energy,
                    double* restrict residum, 
                    double* restrict ridge,
                    double* restrict lasso,
                    const double amount,
                    double* restrict tmp     )
{
    double* restrict x = (*data).x;
    const int N = (*data).N;
    
    memcpy ( tmp, res, sizeof(double) * N );

    for( int i = 0; i < N; ++i )
    {
        tmp[i] -=  amount *  x[i];
    }

    double tmp_residum = squaresum( tmp, N ) / N;
    double tmp_energy = tmp_residum / 2.0;
                             
    tmp_energy += (*data).lambda * ( (1.0 - (*data).alpha) * *ridge / 2.0 + (*data).alpha *  *lasso );
    
    double deltaE = tmp_energy - *energy;    

    if( deltaE <= 0.0 )
    {
         memcpy ( res, tmp, sizeof(double) * N );
         *energy = tmp_energy;
         *residum = tmp_residum;
         (*data).beta[0] += amount;        
         return 1;
    }
    else
    {
        return 0;
    }
}





void zeroSumRegressionLS(  
                struct regressionData data,
                const int steps   )
{    
    #ifdef DEBUG
    double timet;
    struct timespec ts0, ts1;
    clock_gettime(CLOCK_REALTIME , &ts0);
    #endif   
        
    // random numer generation
    #ifdef R_PACKAGE
    GetRNGstate();
    #endif   

    const int P = data.P;
    
    // moves
    int numMoves = (P-1)*(P-2);
    int rng, tmpi, tmpj;
    
    // temporal for storing some data (forwarded to energy calculation)
    double* tmp = (double*)malloc( data.N * sizeof(double));
    
    // array for residuals        
    double* res = (double*)malloc( data.N * sizeof(double));
        
    double ridge = 0.0;
    double lasso = 0.0;
    double residum = 0.0;
    double energy;
    
    vectorElNetCostFunction( &data, res, &energy, &residum, &ridge, &lasso);

    #ifdef DEBUG
    double energy1 = energy; 
    PRINT("Initial Energy: %e (residum: %e ridge term: %e, lasso: %e)\n",
            energy, residum, ridge, lasso); 
    #endif 

    
    double energy_start;    
    int repeats = steps > 0 ? steps : STEP_SIZE;

    if( data.offset == TRUE )
        calcOffsetElNetGradientUpdate( &data, res, &energy, 
            &residum, &ridge, &lasso);
    
    do{
        int counter = 0;
        double amount;   

        energy_start = energy;

        for( int i = 0; i < repeats * P; ++i )
        {                            
            rng = ceil( MY_RND * (double) numMoves );
            tmpi = ceil( (double)rng/(double)(P-2) );        
            tmpj = rng - (tmpi-1) * (P-2);
            if(tmpj >= tmpi) tmpj++;
            
            if(i%10 == 0)
                amount = data.beta[ tmpi ];
            else
                amount = (MY_RND-0.5) * MOVE_SCALE;
              
            counter += moveLS( &data, res, &energy, &residum, &ridge,
                               &lasso, tmpi, tmpj, amount, tmp );
            
            if( data.offset == TRUE )
                calcOffsetElNetGradientUpdate( &data, res, &energy, 
                                &residum, &ridge, &lasso);
        }
        
        #ifdef R_PACKAGE
        R_CheckUserInterrupt();
        #endif
        
        if(steps > 0) break;        
 
        #ifdef DEBUG
        double acceptrate = counter / ( (double) (repeats * P)  );
        
        PRINT("Energy before: %e Energy  now: %e Diff: %e  accept: %f\n",
              energy_start, energy, energy_start - energy, acceptrate );
        #endif
      
        vectorElNetCostFunction( &data, res, &energy, &residum, &ridge, &lasso);
       
        
    }while( energy < energy_start && fabs(energy - energy_start) > data.precision);
    

    
    
    #ifdef DEBUG
    PRINT("Energy before: %e\t  later %e\tDif %e\n",
            energy1, energy, energy-energy1  );
    clock_gettime(CLOCK_REALTIME , &ts1);
    timet = (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;
    PRINT("DONE\tDauer in Sekunden: = %e\n", timet);
    #endif 
    
    free(res); 
    free(tmp);    
    
    #ifdef R_PACKAGE
    PutRNGstate();
    #endif
}
