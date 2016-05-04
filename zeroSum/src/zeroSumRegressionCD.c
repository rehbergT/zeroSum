#include "regressions.h"

#define REFRESH 1000
#define MIN_SUCCESS_RATE 0.90

//        #define DEBUG
//        #define DEBUG2

#ifdef DEBUG
#include <time.h>
#endif

double lambdaAlpha;
double lambdaAlpha2;

int calcGradient( struct regressionData *data,
                  const int s,
                  const int k,
                  double* restrict betas,
                  double* restrict betasX,
                  double* restrict tmparray)
{
    double* restrict x = (*data).x;
    double* restrict beta = (*data).beta;    
    const int N = (*data).N;    
    
    double betaPartSum = *betas - beta[k];
    *betas = betaPartSum - beta[s];

    int i1 = INDEX(0,s,N);
    int i2 = INDEX(0,k,N);    
    
    for( int i=0; i<N; ++i, ++i1, ++i2 )
        tmparray[i] = ( x[ i1 ] - x[ i2 ] );    

    double denominator = squaresum( tmparray, N ) / (double)N  + 2.0 * lambdaAlpha;
    double nominator = 0.0;
    
    i1 = INDEX(0,s,N);
    i2 = INDEX(0,k,N);    
    
    for( int i=0; i<N; ++i, ++i1, ++i2 )
    {
        tmparray[i] = tmparray[i] * ( betasX[i]  + x[ i2 ] * beta[k] 
                        + x[ i1 ] * betaPartSum  );

    }
    nominator = - sum( tmparray, N ) / N - lambdaAlpha * (*betas);
    
    double beta1 = nominator  / denominator;
    double beta2 = ( nominator - lambdaAlpha2 ) / denominator;   
    double beta3 = ( nominator + lambdaAlpha2 ) / denominator; 
    double betatmp = NAN;
    
    if( beta1 > 0 && beta1 < -(*betas) )
    {
        betatmp = beta1;
    }
    else if( beta2 > 0 && beta2 > -(*betas) )
    {
        betatmp = beta2;
    }
    else if( beta3 < 0 && beta3 < -(*betas) )
    {
        betatmp = beta3;
    }
    else if( beta1 < 0 && beta1 > -(*betas) )
    {
        betatmp = beta1;
    }
    
    double diff = 0.0;
    if( !isnan(betatmp) )
    {
        diff = betatmp - beta[k];
        beta[s] -= diff;

        i1 = INDEX(0,s,N);
        i2 = INDEX(0,k,N);
        for( int i=0; i<N; ++i, ++i1, ++i2 )
            betasX[i] -= diff * ( x[ i2 ] - x[i1]  );

        beta[k] = betatmp;
    }

    *betas += ( beta[k] + beta[s] );
    
    if(fabs( diff) < DBL_EPSILON * 100 )
        return 0;
    else
        return 1;
}


int calcRTGradient(  struct regressionData *data,
                     const int n, 
                     const int m, 
                     const int s, 
                     const double theta,
                     double* restrict betas, 
                     double* restrict betasX, 
                     double* restrict tmparray)
{ 
    double* restrict x = (*data).x;
    double* restrict beta = (*data).beta;    
    const int N = (*data).N;
    
    double cosT = cos(theta);
    double sinT = sin(theta);
    double sinTMcosT = sinT-cosT;

    double c1 = beta[n];
    double c2 = beta[m];

    int i1 = INDEX(0,n,N);
    int i2 = INDEX(0,m,N);   
    int i3 = INDEX(0,s,N);

    for( int i=0; i<N; ++i, ++i1, ++i2, ++i3 )
    {
        tmparray[i] = x[ i2 ] * sinT - x[ i1 ] * cosT  - x[ i3 ] * sinTMcosT;
    }
    double a_nm = squaresum( tmparray, N ) / ( (double)N ) 
                    + ( lambdaAlpha * (2.0 - 2.0*cosT*sinT));
    
    double betaPartSum = *betas - beta[n] - beta[m];    
    *betas = betaPartSum - beta[s];   
    
    i1 = INDEX(0,n,N);
    i2 = INDEX(0,m,N);
    i3 = INDEX(0,s,N);
    
    for( int i=0; i<N; ++i, ++i1, ++i2, ++i3 )
    {
        tmparray[i] = tmparray[i] * ( betasX[i] 
                        + x[i1] * beta[n] + x[i2] * beta[m] 
                        + x[ i3 ] * betaPartSum
                        + c1 * ( x[i3] - x[i1] ) + c2 * ( x[i3] - x[i2] )  );
    }
    double b_nm = sum( tmparray, N );
    
    double betaC1C2 = (*betas) + c1 + c2;
    b_nm =  - b_nm / N - lambdaAlpha 
                * ( c1 * cosT - c2 * sinT - betaC1C2 * sinTMcosT );    
    
    double beta_n1 = b_nm / a_nm;
    double beta_n2 = ( b_nm + lambdaAlpha2 * sinTMcosT )  / a_nm;
    double beta_n3 = ( b_nm - lambdaAlpha2 * sinT ) / a_nm;
    double beta_n4 = ( b_nm - lambdaAlpha2 * cosT ) / a_nm;
    
    double beta_n5 = ( b_nm + lambdaAlpha2 * cosT ) / a_nm;
    double beta_n6 = ( b_nm + lambdaAlpha2 * sinT ) / a_nm;
    double beta_n7 = ( b_nm - lambdaAlpha2 * sinTMcosT ) / a_nm;
    
    double betatmp = NAN;

    if(      beta_n1 * cosT  > -c1 && beta_n1 * sinT < c2 && beta_n1 * sinTMcosT > betaC1C2 )
    {
        betatmp = beta_n1;
    }
    else if( beta_n2 * cosT  > -c1 && beta_n2 * sinT < c2 && beta_n2 * sinTMcosT < betaC1C2 )
    {
        betatmp = beta_n2;
    }
    else if( beta_n3 * cosT  > -c1 && beta_n3 * sinT > c2 && beta_n3 * sinTMcosT > betaC1C2 )
    {
        betatmp = beta_n3;
    }
    else if( beta_n4 * cosT  > -c1 && beta_n4 * sinT > c2 && beta_n4 * sinTMcosT < betaC1C2 )
    {
        betatmp = beta_n4;
    }
    else if( beta_n5 * cosT  < -c1 && beta_n5 * sinT < c2 && beta_n5 * sinTMcosT > betaC1C2 )
    {
        betatmp = beta_n5;
    }
    else if( beta_n6 * cosT  < -c1 && beta_n6 * sinT < c2 && beta_n6 * sinTMcosT < betaC1C2 )
    {
        betatmp = beta_n6;
    }
    else if( beta_n7 * cosT  < -c1 && beta_n7 * sinT > c2 && beta_n7 * sinTMcosT > betaC1C2 )
    {
        betatmp = beta_n7;
    }
    else if( beta_n1 * cosT  < -c1 && beta_n1 * sinT > c2 && beta_n1 * sinTMcosT < betaC1C2 )
    {
        betatmp = beta_n1;
    }
    
    i3 = 0;
    if( !isnan(betatmp) )
    {
        double betaNold = beta[n];
        double betaMold = beta[m];

        beta[n] =  betatmp * cosT + c1;
        beta[m] = -betatmp * sinT + c2;  
        
        double tmp1 = betaNold - beta[n]; 
        double tmp2 = betaMold - beta[m]; 
        double tmp3 = tmp1 + tmp2;
        
        beta[s] += tmp3;
        
        i1 = INDEX(0,n,N);
        i2 = INDEX(0,m,N);
        i3 = INDEX(0,s,N);

        for( int i=0; i<N; ++i, ++i1, ++i2, ++i3 )
            betasX[i] += x[i1] * tmp1 + x[i2] * tmp2 - x[i3] * tmp3;
        
        i3 = 1;
    }

    *betas += ( beta[n] + beta[m] + beta[s] );
    return i3;
}


inline void calcOffsetGradient( struct regressionData *data,
                                double* restrict betasX )
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
    
    double tmp = oldbeta0-beta[0];
    for( int i=0; i<N; ++i )
    {
        betasX[i] += tmp; 
    }
}


void refresh( struct regressionData *data,
              double* restrict betas, 
              double* restrict betasX )
{
    double* restrict x = (*data).x;
    double* restrict y = (*data).y;
    double* restrict beta = (*data).beta;
    
    const int N = (*data).N;
    const int P = (*data).P;
    
    *betas = 0.0;
    for( int j=1; j<P; ++j )
        *betas += beta[j];

    for(int i=0; i<N; ++i )
    {
        betasX[i] = y[i] - beta[0];
        for( int j=1; j<P; ++j )
        {
            betasX[i] -= x[ INDEX(i,j,N) ] * beta[j];
        }
    }
}


void zeroSumRegressionCD( struct regressionData data,
                          const int verticalMoves  )
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

    lambdaAlpha = data.lambda * ( 1.0 - data.alpha );
    lambdaAlpha2 = 2.0 * data.lambda * data.alpha;    

    const int P = data.P; 
    double betas;
    double* betasX = (double*)malloc( data.N * sizeof(double));

    refresh( &data, &betas, betasX );

    double* res = (double*)malloc( data.N * sizeof(double));
    double ridge = 0.0;
    double lasso = 0.0;
    double residum = 0.0;

    #ifdef DEBUG2
    double energy1, energy2, energy3;
    vectorElNetCostFunction( &data, res, &energy1, &residum, &ridge, &lasso);
    PRINT("Initial Energy: %e  (sum: %e)\n", energy1, sum(&data.beta[1], P-1));
    #endif

    double energynew;
    double energyold;

    size_t activeSetSize = (size_t)ceil( P/32.0 );
    int* activeset = (int*) malloc( activeSetSize * sizeof(int) );
    memset( activeset, 0, activeSetSize * sizeof(int));

    int activesetChange = 0;
    int step=0;

    int* ind1 = (int*) malloc( P * sizeof(int) );
    int* ind2 = (int*) malloc( P * sizeof(int) );
    
    int refreshCounter = 0;
    while( 1 )
    {        
        #ifdef DEBUG
        PRINT("Step: %d\nFind active set\n", step);
        #endif
        
        if( data.offset == TRUE )
        {
            calcOffsetGradient( &data, betasX);
            ++refreshCounter;
        }        
        
        activesetChange = 0;
        
        fisherYates(ind1, P);
        for( int l=1; l<P; ++l )
        { 
            fisherYates(ind2, P);
            for( int k=1; k<(int)ceil(P*0.1)+1; ++k )
            {   
                int s = ind1[l];
                int j = ind2[k];

                if( s == j ) continue;
                
                #ifdef DEBUG
                vectorElNetCostFunction( &data, res, &energyold, &residum, &ridge, &lasso);
                #endif
                            
                int change = calcGradient( &data, s, j, &betas, betasX, res);                
                
                if( change == 1 ) 
                {
                    if( TestBit( activeset, j ) == 0 || TestBit( activeset, s ) == 0  ) 
                    {
                        activesetChange = 1;
                        SetBit( activeset, j );
                        SetBit( activeset, s );
                    }
                    
                    if( data.offset == TRUE )
                        calcOffsetGradient( &data, betasX);
                    
                    ++refreshCounter;                    
                }
                
                #ifdef DEBUG
                vectorElNetCostFunction( &data, res, &energynew, &residum, &ridge, &lasso);
                  PRINT("Vorher:j=%d Beta[j]=%e Beta[1]=%e activesetChange: %d activeset: %d energy: %e deltaE: %e  (sum %e)\n",
                         j, data.beta[j], data.beta[1], activesetChange, TestBit( activeset, j )!=0, energynew, energynew-energyold,
                          sum(&(data.beta[1]), P-1)
                       );
                if( energynew-energyold > 10000 * DBL_EPSILON ){
                        PRINT("ENERGY BREAK (ACTIVESET SEARCH)!\tDeltaE=%e, E_NEW: %e  E_old: %e\n",
                              energynew-energyold, energynew, energyold );
                }
                #endif           
            }
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
            int counter = 0;
            fisherYates(ind1, P);            
            for( int i=1; i<P; ++i )
            {                
                int s = ind1[i];
                if( TestBit( activeset, s ) == 0 ) continue;
                
                for( int j=1; j<s; ++j )
                {      
                    if( TestBit( activeset, j ) == 0 ) continue;
                          
                    #ifdef DEBUG
                    vectorElNetCostFunction( &data, res, &energy2, &residum, &ridge, &lasso);
                    #endif                    
                    
                    int change = calcGradient( &data, s, j, &betas, betasX, res);
                    
                    if( change == 1 ) 
                    {
                        if( data.offset == TRUE )
                            calcOffsetGradient( &data, betasX);
                        
                        ++test;
                    }
                    
                    #ifdef DEBUG
                    vectorElNetCostFunction( &data, res, &energy3, &residum, &ridge, &lasso); 
                    if( energy3-energy2 > 10000 * DBL_EPSILON ){
                        PRINT("ENERGY BREAK (CONVERGE)!\tDeltaE=%e, E_NEW: %e  E_old: %e\n",
                              energy3-energy2, energy3, energy2 );
                    }
                    #endif
                    ++counter;
                }
            }
            
            
            
            #ifdef DEBUG
            PRINT("Test: %d  counter: %d  Failraite: %e <= %e  -> %d\n",  test, counter, (double)test/(double)counter, MIN_SUCCESS_RATE,
                    ((double)test/(double)counter <= MIN_SUCCESS_RATE)  );
            #endif
            
            if( verticalMoves == 1  && (double)test/(double)counter <= MIN_SUCCESS_RATE)
            {
                #ifdef DEBUG
                PRINT("DIAGONAL MOVES!\n");
                #endif   
                
                fisherYates(ind1, P);
                for( int j=1; j<P; ++j )
                {   
                    int s = ind1[j];
                    
                    if( s == j || TestBit( activeset, j ) == 0 || TestBit( activeset, s ) == 0 ) continue;                    
                                                   
                    
                    fisherYates(ind2, P);
                    
                    for( int k=1; k<j; ++k )
                    {
                        if( k == s || TestBit( activeset, k ) == 0 ) continue;

                        #ifdef DEBUG
                        double energy2, energy3;
                        vectorElNetCostFunction( &data, res, &energy2, &residum, &ridge, &lasso);                        
                        #endif
                        
                        int change = calcRTGradient( &data, k, j, s, MY_RND * M_PI, &betas, betasX, res);
  
                        if( change == 1 ) 
                        {
                            if( data.offset == TRUE )
                                calcOffsetGradient( &data, betasX);
                            
                            ++test;
                        }
                        
                        #ifdef DEBUG
                        vectorElNetCostFunction( &data, res, &energy3, &residum, &ridge, &lasso);
                        if( energy3-energy2 > 10000.0 * DBL_EPSILON ){
                            PRINT("ENERGY BREAK (DIAGONAL)!\tDeltaE=%e, E_NEW: %e  E_old: %e  sum=%e\n",
                                  energy3-energy2, energy3, energy2,
                                   sum(&data.beta[1], P-1)
                                 );
                        }               
                        #endif
                     }
                }
            }
            
            vectorElNetCostFunction( &data, res, &energynew, &residum, &ridge, &lasso);

//             #ifdef DEBUG
//             PRINT("Energy before: %e\t  later %e\tDif %e     %e   test: %d\n", energyold, energynew,
//                   (energyold-energynew)/(energynew * (double)test), 
//                   (energyold-energynew)/(energynew * (double)test) < data.precision
//             );
//             #endif

            if( test == 0 || (energyold-energynew)/(energynew * (double)test) < data.precision )
                convergence = 1;

            refreshCounter += test;
            if( refreshCounter >= REFRESH)
            {
                refresh( &data, &betas, betasX );
                refreshCounter = 0;
            }
            
            
            #ifdef R_PACKAGE
            R_CheckUserInterrupt();
            #endif
        }


        if( step==100 ) break;
        step++;

    }


    #ifdef DEBUG2
    vectorElNetCostFunction( &data, res, &energynew, &residum, &ridge, &lasso);

    PRINT("Energy before: %e\t  later %e\tDif %e\n",
            energy1, energynew, energynew-energy1);

    clock_gettime(CLOCK_REALTIME , &ts1);
    timet = (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;
    PRINT("DONE\tDauer in Sekunden: = %e\n", timet);
    #endif

    free(ind2);
    free(ind1);
    free(activeset);
    free(res);
    free(betasX);
    
    #ifdef R_PACKAGE
    PutRNGstate();
    #endif
}
