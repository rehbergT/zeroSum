#ifndef REGRESSIONDATA_H
#define REGRESSIONDATA_H


struct regressionData
{
    double* restrict x;
    double* restrict y;
    const int N; 
    const int P; 
    double* restrict beta;
    const double lambda;
    const double alpha;
    const int offset;
    const double precision;
};



#endif /* REGRESSIONDATA_H */