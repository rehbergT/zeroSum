#include "CvRegression.h"

CvRegression::CvRegression( RegressionData* data )
{
    cv_data.reserve(data->nFold+1);

    for( int f=0; f<data->nFold+1; f++ )
    {
        CvRegressionData* cvData = new CvRegressionData(data);
        cv_data.push_back(*cvData);
    }

    cv_predict.resize( data->K * data->N );
    cv_tmp.resize( data->nFold+1 );

    for( int f=0; f<data->nFold; f++ )
    {
        int foldSize =0;
        for( int i=0; i<data->N; i++)
        {
            if( (data->foldid[i]-1) == f || data->foldid[i] == -1 )
                foldSize++;
        }

        double scaler1 = (double)data->N / (double)( data->N - foldSize );
        double scaler2 = (double)data->N / (double)foldSize;

        for( int n=0; n<data->N; n++)
        {
            if( (data->foldid[n]-1) == f )
            {
                cv_data[f].w[n]    = 0.0;
                cv_data[f].wOrg[n] = 0.0;
                cv_data[f].wCV[n] *= scaler2;
            }
            else
            {
                cv_data[f].w[n]    *= scaler1;
                cv_data[f].wOrg[n] *= scaler1;
                cv_data[f].wCV[n]   = 0.0;
            }
        }
    }
}
