#include "RegressionDataScheme.h"

void RegressionDataScheme::localSearch( int seed )
{

#ifdef DEBUG
    double timet, costStart, costEnd;
    struct timespec ts0, ts1;
    clock_gettime(CLOCK_REALTIME , &ts0);
    PRINT("Starting LS\n");
    costFunction();
    costStart = cost;

    PRINT("Loglikelihood: %e lasso: %e ridge: %e cost: %e sum=%e\n",
           loglikelihood, lasso, ridge, cost, sum_a_times_b(beta, u, P) );
#endif

    double delta_k, e1=0.0, e2=0.0;

    std::mt19937_64 mt(seed);
    std::uniform_real_distribution<double> rng( 0.0, 1.0 );

    int k=0, s=0, t;

    for( int l=0; l<K; l++ )
    {
        for( int j=0; j<P; j++ )
            checkActiveSet(j);
    }

    double intervalSize = INTERVAL_SIZE;

    int maxSteps = polish > 0 ? polish : MAX_STEPS;

    for(int step=1; step<=maxSteps; step++)
    {
        long counter  = 0;
        long attempts = 0;

        costFunction();
        e1 = cost;

        // random sweeps -> looking for new coefficients
        if( polish != 0 )
        for( int sr=0; sr<SWEEPS_RANDOM; sr++ )
        {
            if( isZeroSum )
            {
                for( int i=0; i<P*downScaler; i++ )
                {
                    for( int l=0; l<K; l++ )
                    {
                        if( useApprox )
                            refreshApproximation( l, TRUE );

                        if( useOffset )
                            lsSaOffsetMove( l );

                        for( int j=0; j<P; j++ )
                        {
                            // choose one (two) random coefficients
                            k = floor( rng(mt) * P );
                            s = floor( rng(mt) * P );
                            if( s == k ) continue;

                            // choose a random amount
                            delta_k = rng(mt) * intervalSize - intervalSize * 0.5;

                            attempts++;
                            t = lsSaMove( k, s, l, delta_k );

                            if( t != 0 )
                            {
                                counter++;
                                checkActiveSet(k);
                                checkActiveSet(s);
                            }
                        }
                    }
                }
            }
            else
            {
                for( int l=0; l<K; l++ )
                {
                    if( useApprox )
                        refreshApproximation( l, TRUE );

                    if( useOffset )
                        lsSaOffsetMove( l );

                    for( int i=0; i<P*downScaler; i++ )
                    {
                        // choose one random coefficients
                        k = floor( rng(mt) * P );

                        // choose a random amount
                        delta_k = rng(mt) * intervalSize - intervalSize * 0.5;

                        attempts++;
                        t = lsSaMove( k, 0, l, delta_k );

                        if( t != 0 )
                        {
                            counter++;
                            checkActiveSet(k);
                        }
                    }
                }
            }
        }

        // active set sweeps -> adjust coeffiecnts
        for( int sa=0; sa<SWEEPS_ACTIVESET; sa++ )
        {
            for( int l=0; l<K; l++ )
            {
                if( useApprox )
                    refreshApproximation( l, TRUE );

                if( useOffset )
                    lsSaOffsetMove( l );

                if( isZeroSum )
                {
                    for( const int &s : activeSet )
                    {
                        for( const int &k : activeSet )
                        {
                            if( s == k || rng(mt) > downScaler ) continue;

                            // choose a random amount
                            delta_k = rng(mt) * intervalSize - intervalSize * 0.5;
                            t = lsSaMove( k, s, l, delta_k );

                            attempts++;
                            if( t != 0 ) counter++;
                        }
                    }
                }
                else
                {
                    for( const int &k : activeSet )
                    {
                        // choose a random amount
                        delta_k = rng(mt) * intervalSize - intervalSize * 0.5;
                        t = lsSaMove( k, 0, l, delta_k );

                        attempts++;
                        if( t != 0 ) counter++;
                    }
                }
            }
        }

        // null moves -> looking for coefficients which can be set to 0
        for( int sn=0; sn<SWEEPS_NULL; sn++ )
        {
            for( int l=0; l<K; l++ )
            {
                double* betaPtr = &beta[ INDEX(0,l,memory_N) ];

                if( useApprox )
                    refreshApproximation( l, TRUE );

                if( useOffset )
                    lsSaOffsetMove( l );

                if( isZeroSum )
                {
                    for( const int &s : activeSet )
                    {
                        for( const int &k : activeSet )
                        {
                            if( s == k || rng(mt) > downScaler ) continue;

                            // choose a random amount
                            delta_k = -betaPtr[k];
                            t = lsSaMove( k, s, l, delta_k );

                            attempts++;
                            if( t != 0 ) counter++;
                        }
                    }
                }
                else
                {
                    for( const int &k : activeSet )
                    {
                        if( rng(mt) > downScaler ) continue;

                        // choose a random amount
                        delta_k = -betaPtr[k];
                        t = lsSaMove( k, 0, l, delta_k );

                        attempts++;
                        if( t != 0 ) counter++;
                    }
                }
            }
        }
        checkWholeActiveSet();

        // fused moves should attempt to bring neighbouring betas on the same value
        // only make sense if isFused
        if( isFused )
        {
            for( int ms=0; ms<SWEEPS_FUSED; ms++ )
            {
                for( int l=0; l<K; l++ )
                {
                    double* betaPtr = &beta[ INDEX(0,l,memory_N) ];

                    if( useApprox )
                        refreshApproximation( l, TRUE );

                    if( useOffset )
                        lsSaOffsetMove( l );

                    for( const int &s : activeSet )
                    {
                        // choose randomly if s+1 or s-1 is attempted
                        // to bring on them same value
                        if( rng(mt) < 0.5 )
                            s < P-2 ? k = s+1 : k = s-1;
                        else
                            s > 0 ? k = s-1 : k = s+1;

                        // choose beta so that beta[k] == beta[s]
                        delta_k = ( betaPtr[k] - betaPtr[s] ) * 0.5;
                        t = lsSaMove( k, s, l, delta_k );

                        attempts++;
                        if( t != 0 ) counter++;
                    }
                }
            }
        }

        #ifdef DEBUG
        PRINT("1: Loglikelihood: %e lasso: %e ridge: %e fusion: %e cost: %e sum=%e t1=%d t2=%d i_size: %e\n",
                        loglikelihood, lasso, ridge, fusion,
                        cost, sum_a_times_b(beta, u, P),
                        checkXtimesBeta(), checkYsubXtimesBeta(), intervalSize );
        #endif

        costFunction();
        e2 = cost;

        #ifdef DEBUG
        if( useApprox )
            refreshApproximation( K-1, TRUE );

        PRINT("2: Loglikelihood: %e lasso: %e ridge: %e fusion: %e cost: %e sum=%e t1=%d t2=%d\n",
                        loglikelihood, lasso, ridge, fusion,
                        cost, sum_a_times_b(beta, u, P),
                        checkXtimesBeta(), checkYsubXtimesBeta() );
        PRINT("e2-e1: %e %e\n", e2 - e1 , -precision);

        PRINT("step:%d size=%e accreptrate: %e  deltaE: %e\n", step, intervalSize, (double)counter/(double)(attempts),
           e2 - e1  );
        #endif

        intervalSize *= INTERVAL_SHRINK;

        if( e2 - e1 > -precision )
            break;
    }

#ifdef DEBUG
    costFunction();
    costEnd = cost;

    PRINT("Cost start: %e cost end: %e diff %e\n",
            costStart, costEnd, costEnd-costStart );

    clock_gettime(CLOCK_REALTIME , &ts1);
    timet = (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;
    PRINT("time taken = %e s\n", timet);
#endif

}
