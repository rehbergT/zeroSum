#include "RegressionDataScheme.h"

#define MIN_SUCCESS_RATE 0.90

void RegressionDataScheme::coordinateDescent( int seed )
{
    #ifdef DEBUG
    double timet, costStart, costEnd;
    struct timespec ts0, ts1;
    clock_gettime(CLOCK_REALTIME , &ts0);
    #endif

    costFunction();

    #ifdef DEBUG
    costStart = cost;
    PRINT("Loglikelihood: %e lasso: %e ridge: %e cost: %e sum=%e sum=%e\n",
           loglikelihood, lasso, ridge, cost, sum_a(beta,P), sum_a_times_b(beta, u, P) );
    double e3;
    #endif

    activeSet.clear();
    int activeSetChange = 0;

    double e1, e2, rn;
    int change, success, attempts;

    std::mt19937_64 mt(seed);
    std::uniform_real_distribution<double> rng( 0.0, 1.0 );

    for( int steps=0; steps<100; steps++)
    {
        #ifdef DEBUG
        PRINT("Step: %d\nFind active set\n", steps);
        #endif

        activeSetChange = 0;

        if( type == GAUSSIAN || type == BINOMIAL )
        {
            if( type == BINOMIAL )
                refreshApproximation(0);

            if( useOffset )
                offsetMove(0);

            for( int s=0; s<P; s++)
            {
                #ifdef DEBUG
                costFunction();
                e3 = cost;
                #endif

                change = cdMove( s, 0);

                if( change )
                    activeSetChange += checkActiveSet(s);

                #ifdef DEBUG
                costFunction();
                if( cost - e3 > 1000 * DBL_EPSILON )
                        PRINT("ENERGY BREAK (ACTIVESET SEARCH)! DeltaE=%e, E_NEW: %e  E_old: %e\n",
                            cost - e3, cost, e3);
                #endif
            }
        }
        if( type == FUSED_GAUSSIAN || type == FUSED_BINOMIAL )
        {
            if( type == FUSED_BINOMIAL )
                refreshApproximation(0);

            if( useOffset )
                offsetMove(0);

            for( int s=0; s<P; s++)
            {
                #ifdef DEBUG
                costFunction();
                e3 = cost;
                #endif

                change = cdMoveFused( s, 0);

                if( change )
                    activeSetChange += checkActiveSet(s);

                #ifdef DEBUG
                costFunction();
                if( cost - e3 > 1000 * DBL_EPSILON )
                        PRINT("ENERGY BREAK (ACTIVESET SEARCH)! DeltaE=%e, E_NEW: %e  E_old: %e\n",
                            cost - e3, cost, e3);
                #endif
            }
        }
        else if( type == GAUSSIAN_ZS || type == BINOMIAL_ZS )
        {
            if( type == BINOMIAL_ZS )
                refreshApproximation(0);

            if( useOffset )
                offsetMove(0);

            for( int s=0; s<P; s++)
            {
                for( int k=0; k<s; k++ )
                {
                    if( s == k || rng(mt) > downScaler ) continue;

                    #ifdef DEBUG
                    costFunction();
                    e3 = cost;
                    #endif

                    change = cdMoveZS( s, k, 0);

                    if( change )
                    {
                        activeSetChange += checkActiveSet(k);
                        activeSetChange += checkActiveSet(s);
                    }

                    #ifdef DEBUG
                    costFunction();
                    if( cost - e3 > 1000 * DBL_EPSILON )
                    {
                        PRINT("ENERGY BREAK (ACTIVESET SEARCH)! DeltaE=%e, E_NEW: %e  E_old: %e check=%d sum=%e\n",
                            cost - e3, cost, e3, checkYsubXtimesBeta(),
                              sum_a_times_b(beta, u, P) );
                    }
                    #endif
                }
            }
        }
        else if( type == MULTINOMIAL || type == MULTINOMIAL_ZS )
        {
            for( int l=0; l<K; l++ )
            {
                #ifdef DEBUG
                e1 = cost;
                PRINT("l=%d\n",l);
                PRINT("Loglikelihood: %e lasso: %e ridge: %e cost: %e\n",
                    loglikelihood, lasso, ridge, cost );
                #endif

                refreshApproximation(l);

                if( useOffset )
                    offsetMove(l);

                if( type == MULTINOMIAL )
                {
                    for( int s=0; s<P; s++)
                    {
                        change = cdMove( s, l);

                        if( change )
                            activeSetChange += checkActiveSet(s);
                    }
                }
                else
                {
                    for( int s=0; s<P; s++)
                    {
                        for( int k=0; k<s; k++ )
                        {
                            if( s == k ) continue;

                            change = cdMoveZS( s, k, l);

                            if( change )
                            {
                                activeSetChange += checkActiveSet(s);
                                activeSetChange += checkActiveSet(k);
                            }
                        }
                    }
                }

                #ifdef DEBUG
                costFunction();

                PRINT("Loglikelihood: %e lasso: %e ridge: %e cost: %e\n",
                    loglikelihood, lasso, ridge, cost );
                #endif
            }
        }

        if( activeSetChange == 0 || activeSet.empty() )
        {
            costFunction();
            break;
        }
        #ifdef DEBUG
        PRINT("converge\n");
        #endif

        while( TRUE )
        {
            costFunction();
            e1 = cost;
            success  = 0;

            if( type == GAUSSIAN || type == BINOMIAL )
            {
                if( type == BINOMIAL )
                    refreshApproximation(0);

                for(const int &s : activeSet )
                {
                    #ifdef DEBUG
                    costFunction();
                    e3 = cost;
                    #endif

                    change = cdMove( s, 0);

                    if( change )
                        success++;

                    #ifdef DEBUG
                    costFunction();
                    if( cost - e3 > 1000 * DBL_EPSILON )
                        PRINT("ENERGY BREAK (CONVERGE)! DeltaE=%e, E_NEW: %e  E_old: %e\n",
                            cost - e3, cost, e3);
                    #endif
                }
            }
            if( type == FUSED_GAUSSIAN || type == FUSED_BINOMIAL )
            {
                if( type == BINOMIAL )
                    refreshApproximation(0);

                for( const int &s : activeSet )
                {
                    #ifdef DEBUG
                    costFunction();
                    e3 = cost;
                    #endif

                    change = cdMoveFused( s, 0);

                    if( change )
                        success++;

                    #ifdef DEBUG
                    costFunction();
                    if( cost - e3 > 1000 * DBL_EPSILON )
                        PRINT("ENERGY BREAK (CONVERGE)! DeltaE=%e, E_NEW: %e  E_old: %e\n",
                            cost - e3, cost, e3);
                    #endif
                }
            }
            else if( type == GAUSSIAN_ZS || type == BINOMIAL_ZS )
            {
                if( type == BINOMIAL_ZS )
                    refreshApproximation(0);

                for(const int &s : activeSet )
                {
                    for(const int &k : activeSet )
                    {
                        if( s == k ) continue;

                        #ifdef DEBUG
                        costFunction();
                        e3 = cost;
                        #endif

                        change = cdMoveZS( s, k, 0);

                        if( change )
                            success++;

                        #ifdef DEBUG
                        costFunction();
                        if( cost - e3 > 1000 * DBL_EPSILON )
                            PRINT("ENERGY BREAK (CONVERGE)! DeltaE=%e, E_NEW: %e  E_old: %e\n",
                                cost - e3, cost, e3);
                        #endif
                    }
                }

                attempts = activeSet.size() * ( activeSet.size() - 1 );

                if( diagonalMoves && (double)success/(double)attempts <= MIN_SUCCESS_RATE )
                {
                    for(const int &s : activeSet )
                    {
                        for(const int &k : activeSet )
                        {
                            if( s == k ) continue;

                            for( int l : activeSet )
                            {
                                if( l == s || l == k ) continue;

                                #ifdef DEBUG
                                costFunction();
                                e3 = cost;
                                #endif

                                rn = rng(mt);

                                change = cdMoveZSRotated( s, k, l, 0, rn * M_PI);

                                #ifdef DEBUG
                                costFunction();
                                if( cost - e3 > 1000 * DBL_EPSILON )
                                    PRINT("ENERGY BREAK (CONVERGE)! DeltaE=%e, E_NEW: %e  E_old: %e\n",
                                        cost - e3, cost, e3);

                                #endif
                            }
                        }
                    }
                }
            }
            else if( type == MULTINOMIAL || type == MULTINOMIAL_ZS )
            {
                for( int l=0; l<K; l++ )
                {
                    if( useOffset )
                    {
                        offsetMove(l);
                        refreshApproximation(l);
                    }

                    refreshApproximation(l);

                    if(type == MULTINOMIAL)
                    {
                        for(const int &s : activeSet )
                        {
                            change = cdMove( s, l);

                            if( change )
                                success++;
                        }
                    }
                    else
                    {
                        for(const int &s : activeSet )
                        {
                            for(const int &k : activeSet )
                            {
                                if( s == k ) continue;

                                change = cdMoveZS( s, k, l);

                                if( change )
                                    success++;
                            }
                        }
                    }
                }
            }

            costFunction();
            e2 = cost;

            #ifdef DEBUG
            PRINT("Loglikelihood: %e lasso: %e ridge: %e cost: %e sum=%e sum=%e\tChange: e1=%e e2=%e %e %e (success:%d)\n",
                  loglikelihood, lasso, ridge, cost, sum_a(beta,P), sum_a_times_b(beta, u, P), e1, e2,
                  fabs(e2 - e1), precision, success );
            #endif


            if( success == 0 || e1 < e2 || fabs(e2 - e1) < precision )
                break;

        }
    }

    if( type == MULTINOMIAL || type == MULTINOMIAL_ZS )
    {
        optimizeParameterAmbiguity(100);
        costFunction();
    }

    for(int j=0; j<K * memory_P; j++)
        if( fabs(beta[j]) < 10* DBL_EPSILON )
            beta[j] = 0.0;

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
