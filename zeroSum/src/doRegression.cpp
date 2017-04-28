#include "regressions.h"

void doRegression( RegressionDataScheme *data, int seed )
{
    if( data->algorithm == 1 )
    {
        data->coordinateDescent( seed );
        if( data->polish != FALSE )
        {
            data->localSearch( seed );
        }
    }
    else if( data->algorithm == 2 )
    {
        data->simulatedAnnealing( seed );
    }
    else if( data->algorithm == 3 )
    {
        data->localSearch( seed );
    }
    else if( data->algorithm == 4 )
    {
        data->coordinateDescent( seed );
        data->localSearch( seed );
    }

}


void doCVRegression( RegressionData* data, double* gammaSeq,
        int gammaLength, double* lambdaSeq, int lambdaLength,
        double* cv_stats, int cv_cols, char* path, char* name,
        int mpi_rank, int seed )
{
    std::vector<CvRegression> cv_data;
    cv_data.reserve(gammaLength);

    for( int i=0; i<gammaLength; i++ )
    {
        CvRegression tmp = CvRegression(data);
        cv_data.push_back(tmp);
    }

    #ifdef AVX_VERSION
        double* lastCV = (double*)aligned_alloc( ALIGNMENT, gammaLength * sizeof(double));
    #else
        double* lastCV = (double*)malloc( gammaLength * sizeof(double));
    #endif

    int worseLambdaSteps=0;

    for( int j=0; j<lambdaLength; j++ )
    {
        #ifdef R_PACKAGE
        R_CheckUserInterrupt();
        #endif

        #pragma omp parallel for collapse(2) schedule(dynamic)
        for( int f=0; f < data->nFold+1; f++ )
        for( int i=gammaLength-1; i >= 0; i-- )
        {
            cv_data[i].cv_data[f].lambda = lambdaSeq[j];
            cv_data[i].cv_data[f].gamma  = gammaSeq[i];

            int cvSeed = f + seed + (int)((lambdaSeq[j] + gammaSeq[i]) * 1e3);
            doRegression( &(cv_data[i].cv_data[f]), cvSeed );

            double* tmp = cv_data[i].cv_data[f].wOrg;
            cv_data[i].cv_data[f].wOrg = cv_data[i].cv_data[f].wCV;

            cv_data[i].cv_data[f].costFunction();

            cv_data[i].cv_data[f].wOrg = tmp;
            cv_data[i].cv_tmp[f] = cv_data[i].cv_data[f].loglikelihood;

            if( f < data->nFold )
            {
                int N  = cv_data[i].cv_data[f].N;
                int mN = cv_data[i].cv_data[f].memory_N;
                int K  = cv_data[i].cv_data[f].K;

                cv_data[i].cv_data[f].predict();

                for( int ii=0; ii<N; ii++)
                {
                    if( cv_data[i].cv_data[f].wCV[ii] != 0.0 )
                    {
                        for( int l=0; l<K; ++l)
                            cv_data[i].cv_predict[INDEX(ii,l,mN)] = cv_data[i].cv_data[f].xTimesBeta[INDEX(ii,l,mN)];
                    }
                }
            }
        }

        int cvImproving = 0;
        for( int i=0; i < gammaLength; i++ )
        {
            double trainingError = cv_data[i].cv_tmp[data->nFold];

            double cvError = 0.0;
            for( int ff=0; ff<data->nFold; ff++ )
                cvError += cv_data[i].cv_tmp[ff];

            cvError /= data->nFold;

            double cvErrorSD  = 0.0;
            for( int ff=0; ff<data->nFold; ff++ )
                cvErrorSD += cv_data[i].cv_tmp[ff] * cv_data[i].cv_tmp[ff];


            // printf("test <- c( %e ", cv_data[i].cv_tmp[0]);
            // for( int ff=1; ff<data->nFold; ff++ )
            //     printf(", %e ", cv_data[i].cv_tmp[ff] );
            // printf(")\n");

           cvErrorSD /= (double)data->nFold;

           cvErrorSD -= cvError * cvError;
           cvErrorSD /= (double)data->nFold - 1.0;
           cvErrorSD  = sqrt( cvErrorSD );

            if(std::isnan(cvErrorSD)) cvErrorSD = -1.0;

            if( path != NULL )
            {
                char outPath[MAX_PATH_LENGTH] = "";
                FILE *file;

                snprintf( outPath, sizeof(outPath), "%s%s_%d_stats.csv",
                                path, name, mpi_rank  );

                file = fopen( outPath, "a");

                fprintf(file, "%a,%a,%a,%a,%a,", gammaSeq[i], lambdaSeq[j],
                        trainingError, cvError, cvErrorSD );

                for( int k=0; k<data->K; k++ )
                {
                    fprintf( file, "%a,", cv_data[i].cv_data[data->nFold].offset[k] );

                    for( int jj=0; jj<data->P; jj++ )
                        fprintf( file, "%a,", cv_data[i].cv_data[data->nFold].beta[INDEX(jj,k,data->memory_P)] );
                }

                for( int k=0; k< data->N*data->K; k++)
                    fprintf( file, "%a,", cv_data[i].cv_predict[k] );

                fprintf( file, "\n" );
                fclose(file);
            }
            else
            {
                double* cv_stats_row = &cv_stats[ INDEX_ROW( INDEX_ROW(i,j,lambdaLength), 0, cv_cols)  ];

                cv_stats_row[0] = gammaSeq[i];
                cv_stats_row[1] = lambdaSeq[j];
                cv_stats_row[2] = trainingError;
                cv_stats_row[3] = cvError;
                cv_stats_row[4] = cvErrorSD;

                int col = 5;
                double* beta;
                for( int k=0; k<data->K; k++ )
                {
                    cv_stats_row[col++] = cv_data[i].cv_data[data->nFold].offset[k];
                    beta = &( cv_data[i].cv_data[data->nFold].beta[INDEX(0,k,data->memory_P)] );
                    memcpy( &cv_stats_row[col], beta, data->P * sizeof(double) );
                    col += data->P;
                }

                for( int k=0; k< data->N*data->K; k++ )
                    cv_stats_row[col++] = cv_data[i].cv_predict[k];
            }

            if(data->verbose)
            {
                PRINT("Step: %3d ", j);

                if( data->lengthGamma != 1 )
                    PRINT("G: %e ", gammaSeq[i]);

                PRINT("L: %e T-E %e CV-E %e CV-SD %e\n",
                   lambdaSeq[j], trainingError, cvError, cvErrorSD );
            }

            if( cvError - lastCV[i] < DBL_EPSILON )
                cvImproving++;

            lastCV[i] = cvError;
        }

        if( j>1 && cvImproving !=0 )
            worseLambdaSteps++;
        else
            worseLambdaSteps=0;

        if( worseLambdaSteps > data->cvStop ) break;
    }


    free(lastCV);
}
