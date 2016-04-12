// RBioC CMD SHLIB sa.c -lgsl -lgslcblas -Wall -Wextra
#include "regressions.h"

#define MOVE_SCALE 0.05
#define ANZ_WERTE 2

#define T_STEPS 10000
#define T_START 0.1

#define COOLING_FAKTOR 0.8

#define THERMALIZE 350
#define MEASURE 200


//#define DEBUG
//#define DEBUG2

#ifdef DEBUG
#include <time.h>
#endif


int move(   struct regressionData *data,
            double* restrict res,
            double* restrict energy,
            double* restrict residum, 
            double* restrict ridge,
            double* restrict lasso,
            const int from,
            const int to,
            const double amount,
            const double rng,
            double* restrict tmp,
            const double temperature      )
{
    double* restrict x = (*data).x;
    double* restrict beta = (*data).beta;
    const int N = (*data).N;
    
    // copy residums to tmp array in order to reject the move if necessary
    memcpy( tmp, res, sizeof(double) * N );

    int colFrom = INDEX(0, from, N);
    int colTo = INDEX(0, to, N);

    for( int i=0; i<N; ++i ){
        tmp[i] +=  amount * ( x[colFrom+i] - x[colTo+i] );
    }

    double tmp_residum = squaresum( tmp, N ) / N;
    double tmp_energy = tmp_residum / 2.0;
    
    double tmp_ridge = *ridge + 2.0 * ( amount * amount + amount * ( beta[to] - beta[from] ) ); 
    double tmp_lasso = *lasso - fabs(beta[from]) - fabs(beta[to]) 
                              + fabs(beta[from]-amount) + fabs(beta[to]+amount);
                              
    tmp_energy += (*data).lambda * ( (1.0 - (*data).alpha) * tmp_ridge / 2.0 + (*data).alpha * tmp_lasso );
    
    double deltaE = tmp_energy - *energy;    

    if( deltaE <= 0.0   || rng < exp(-deltaE / temperature) )
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
    else
    {
        return 0;
    }
    return -1;
}

int moveOffset( struct regressionData *data,
                double* restrict res,
                double* restrict energy,
                double* restrict residum, 
                double* restrict ridge,
                double* restrict lasso,
                const double amount,
                const double rng,
                double* restrict tmp,
                const double temperature      )
{
    double* restrict x = (*data).x;
    const int N = (*data).N;
    
    memcpy ( tmp, res, sizeof(double) * N );

    for( int i = 0; i < N; ++i ){
        tmp[i] -=  amount *  x[i];
    }

    double tmp_residum = squaresum( tmp, N ) / N;
    double tmp_energy = tmp_residum / 2.0;
                             
    tmp_energy += (*data).lambda * ( (1.0 - (*data).alpha) * *ridge / 2.0 + (*data).alpha *  *lasso );
    
    double deltaE = tmp_energy - *energy;    

    if( deltaE <= 0.0   || rng < exp(-deltaE / temperature) )
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
    return -1;
}





void zeroSumRegressionSA( struct regressionData data )
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
    
    // moves
    int numMoves = (P-1)*(P-2);
    int rng, tmpi, tmpj;
    
    // temporal for storing some data (forwarded to energy calculation)
    double* tmp = (double*)malloc( data.N * sizeof(double));
    
    // array for residuals        
    double* res = (double*)malloc( data.N * sizeof(double));
    
    // array for storing best solution
    double best_energy = DBL_MAX;
    double* best_beta = (double*)malloc( P * sizeof(double));
    
    // initialize beta ( sum will be conserved!)
    memset( data.beta, 0, P * sizeof(double) );
    
    double ridge = 0.0;
    double lasso = 0.0;
    double residum = 0.0;
    double energy;
    
    vectorElNetCostFunction( &data, res, &energy, &residum, &ridge, &lasso);   

    #ifdef DEBUG2
    double energy1 = energy; 
    #endif 
    
    double amount;
    double dtemp;
   
    double temperature = 0.0;
    int counter = 0;
    
    for( int i = 0; i < T_STEPS; i++)
    {
        dtemp = -energy;         
        
        rng = ceil( MY_RND * (double) numMoves );
        tmpi = ceil( (double)rng/(double)(P-2) );        
        tmpj = rng - (tmpi-1) * (P-2);
        if(tmpj >= tmpi) tmpj++;
        
//         PRINT("rng=%d\t\ti=%d\tj=%d\t", rng, tmpi, tmpj);
        
        
        if(i%25 == 0)
        {
            amount = data.beta[ tmpi ];
            move( &data, res, &energy,&residum, &ridge,
                 &lasso, tmpi, tmpj, amount, 0.0, tmp, DBL_MAX );
        }
        else if( data.offset == TRUE && i%15 == 0)
        {
            amount = (MY_RND-0.5) * MOVE_SCALE;
            moveOffset( &data, res, &energy, &residum, &ridge, &lasso,                                             
                           amount, 0.0, tmp, DBL_MAX );
        }
        else
        {
            amount = (MY_RND-0.5) * MOVE_SCALE;
            move( &data, res, &energy, &residum, &ridge,
                 &lasso, tmpi, tmpj, amount, 0.0, tmp, DBL_MAX );
        }        
                
        dtemp += energy;
        
        if( dtemp > 0.0 )
        {
            temperature += dtemp;
            ++counter;
        }        
        
        if( energy < best_energy )
        {
            best_energy = energy;
            memcpy ( best_beta, data.beta, sizeof(double) * P );
        }
    }

    temperature /= (double)( counter );  
    temperature = -temperature / (  log(T_START) );
    
    // reset beta (better for sparse solution)
    memset( data.beta, 0, P*sizeof(double) );
    vectorElNetCostFunction( &data, res, &energy, &residum, &ridge, &lasso);
    
    
    #ifdef DEBUG    
    PRINT("Estimated Temperature: %e\n\n", temperature );
    PRINT("Initial Energy: %e (residum: %e ridge term: %e, lasso: %e)\n",
            energy, residum, ridge, lasso); 
    #endif 

    double average_Eng[MEASURE],   comp_energy[ANZ_WERTE];

    do{
        counter = 0;
        int k = 0;
        
        for( int i = 0; i < THERMALIZE + MEASURE; ++i )
        {            
            for( int j = 0; j < 4*P; ++j )
            {                
                rng = ceil( MY_RND * (double) numMoves );
                tmpi = ceil( (double)rng/(double)(P-2) );        
                tmpj = rng - (tmpi-1) * (P-2);
                if(tmpj >= tmpi) tmpj++;
                
                if(j%25 == 0)
                {
                    amount = data.beta[ tmpi ];
                    counter += move( &data, res, &energy, &residum, &ridge, &lasso, tmpi, tmpj, 
                                     amount, MY_RND, tmp, temperature );
                }
                else if( data.offset == TRUE && j%15 == 0)
                {
                    amount = (MY_RND-0.5) * MOVE_SCALE;
                    counter += moveOffset( &data, res, &energy, &residum, &ridge, &lasso,                                             
                           amount, MY_RND, tmp, temperature );
                }
                else
                {
                    amount = (MY_RND-0.5) * MOVE_SCALE;
                    counter += move( &data, res, &energy, &residum, &ridge, &lasso, tmpi, tmpj,
                                     amount, MY_RND, tmp, temperature );
                }
                 
                
                if( energy < best_energy )
                {
                    best_energy = energy;
                    memcpy ( best_beta, data.beta, sizeof(double) * P );
                }
            }            
            
            if(i >= THERMALIZE )
            {
                average_Eng[k] = energy;
                k++;
            }
            
            #ifdef R_PACKAGE
            R_CheckUserInterrupt();
            #endif
        }
           
        // neaded measures
        MeanVar(average_Eng, MEASURE, comp_energy);
        
        #ifdef DEBUG
        double acceptrate = counter / ( (double) ((THERMALIZE + MEASURE ) * 4 * P)  ); 
        PRINT("temperatur %e energy: %e var eng %e  accept: %f best E: %e, crit=%e   test=%d\n",
                temperature, comp_energy[0],comp_energy[1], acceptrate, best_energy,
                sqrt( comp_energy[1] ) / comp_energy[0],
                sqrt( comp_energy[1] ) / comp_energy[0] > data.precision
             );
        #endif
        
        
        energy = best_energy;
        memcpy ( data.beta, best_beta, sizeof(double) * P );
        vectorElNetCostFunction( &data, res, &energy, &residum, &ridge, &lasso);
        
        temperature *= COOLING_FAKTOR;
        
    }while( sqrt( comp_energy[1] ) / comp_energy[0] > data.precision );
    

    
    
    #ifdef DEBUG2
    PRINT("Energy before: %e\t  later %e\tDif %e\n",
            energy1, energy, energy-energy1  );
    clock_gettime(CLOCK_REALTIME , &ts1);
    timet = (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;
    PRINT("DONE\tDauer in Sekunden: = %e\n", timet);
    #endif 
    
    free(best_beta);
    free(res); 
    free(tmp);    
    
    #ifdef R_PACKAGE
    PutRNGstate();
    #endif
}
