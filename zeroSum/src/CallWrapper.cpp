#include "zeroSum.h"

// R header must be included after zeroSum.h!
#include <R.h>
#include <Rdefines.h>

SEXP getElementFromRList(SEXP& RList, const char* name) {
    SEXP element = R_NilValue;
    SEXP names = getAttrib(RList, R_NamesSymbol);
    for (uint32_t i = 0; i < (uint32_t)length(RList); i++) {
        if (strcmp(CHAR(STRING_ELT(names, i)), name) == 0) {
            element = VECTOR_ELT(RList, i);
            break;
        }
    }
    return element;
}

zeroSum rListToRegressionData(SEXP& _dataObjects) {
    SEXP _x = getElementFromRList(_dataObjects, "x");
    SEXP _y = getElementFromRList(_dataObjects, "y");
    SEXP _beta = getElementFromRList(_dataObjects, "beta");
    SEXP _w = getElementFromRList(_dataObjects, "w");
    SEXP _v = getElementFromRList(_dataObjects, "v");
    SEXP _u = getElementFromRList(_dataObjects, "u");
    SEXP _cSum = getElementFromRList(_dataObjects, "cSum");
    SEXP _alpha = getElementFromRList(_dataObjects, "alpha");
    SEXP _lambdaSeq = getElementFromRList(_dataObjects, "lambda");
    SEXP _gammaSeq = getElementFromRList(_dataObjects, "gamma");
    SEXP _nc = getElementFromRList(_dataObjects, "nc");
    SEXP _fusion = getElementFromRList(_dataObjects, "fusionC");
    SEXP _precision = getElementFromRList(_dataObjects, "precision");
    SEXP _useIntercept = getElementFromRList(_dataObjects, "useIntercept");
    SEXP _useCentering = getElementFromRList(_dataObjects, "center");
    SEXP _standardize = getElementFromRList(_dataObjects, "standardize");
    SEXP _useApprox = getElementFromRList(_dataObjects, "useApprox");
    SEXP _downScaler = getElementFromRList(_dataObjects, "downScaler");
    SEXP _type = getElementFromRList(_dataObjects, "type");
    SEXP _algorithm = getElementFromRList(_dataObjects, "algorithm");
    SEXP _rotatedUpdates = getElementFromRList(_dataObjects, "rotatedUpdates");
    SEXP _usePolish = getElementFromRList(_dataObjects, "usePolish");
    SEXP _foldid = getElementFromRList(_dataObjects, "foldid");
    SEXP _nFold = getElementFromRList(_dataObjects, "nFold");
    SEXP _threads = getElementFromRList(_dataObjects, "threads");
    SEXP _seed = getElementFromRList(_dataObjects, "seed");
    SEXP _verbose = getElementFromRList(_dataObjects, "verbose");
    SEXP _cvStop = getElementFromRList(_dataObjects, "cvStop");
    SEXP _useFusion = getElementFromRList(_dataObjects, "useFusion");
    SEXP _isZerosum = getElementFromRList(_dataObjects, "useZeroSum");

    uint32_t N = (uint32_t)INTEGER(GET_DIM(_x))[0];
    uint32_t P = (uint32_t)INTEGER(GET_DIM(_x))[1];
    uint32_t K = (uint32_t)INTEGER(GET_DIM(_y))[1];
    uint32_t nc = (uint32_t)INTEGER(_nc)[0];
    uint32_t type = (uint32_t)INTEGER(_type)[0];
    bool useZeroSum = (uint32_t)INTEGER(_isZerosum)[0];
    bool useFusion = (uint32_t)INTEGER(_useFusion)[0];
    bool useIntercept = (uint32_t)INTEGER(_useIntercept)[0];
    bool useApprox = (uint32_t)INTEGER(_useApprox)[0];
    bool useCentering = (uint32_t)INTEGER(_useCentering)[0];
    bool useStandardization = (uint32_t)INTEGER(_standardize)[0];
    uint32_t algorithm = (uint32_t)INTEGER(_algorithm)[0];
    bool usePolish = (uint32_t)INTEGER(_usePolish)[0];
    bool rotatedUpdates = (uint32_t)INTEGER(_rotatedUpdates)[0];
    double precision = REAL(_precision)[0];
    uint32_t nFold = (uint32_t)INTEGER(_nFold)[0];
    uint32_t cvStop = (uint32_t)INTEGER(_cvStop)[0];
    uint32_t verbose = (uint32_t)INTEGER(_verbose)[0];
    double cSum = REAL(_cSum)[0];
    double alpha = REAL(_alpha)[0];
    double downScaler = REAL(_downScaler)[0];

    uint32_t threads = (uint32_t)INTEGER(_threads)[0];
    uint32_t seed = (uint32_t)INTEGER(_seed)[0];

    zeroSum data(N, P, K, nc, type, useZeroSum, useFusion, useIntercept,
                 useApprox, useCentering, useStandardization, usePolish,
                 rotatedUpdates, precision, algorithm, nFold, cvStop, verbose,
                 cSum, alpha, downScaler, threads, seed);

    double* lambdaSeq = REAL(_lambdaSeq);

    for (uint32_t j = 0; j < (uint32_t)length(_lambdaSeq); j++) {
        data.lambdaSeq.push_back(lambdaSeq[j]);
    }
    for (uint32_t f = 0; f < nFold + 1; f++)
        data.lambda[f] = data.lambdaSeq[0];

    double* gammaSeq = REAL(_gammaSeq);
    for (uint32_t j = 0; j < (uint32_t)length(_gammaSeq); j++) {
        data.gammaSeq.push_back(gammaSeq[j]);
    }

    // initialize gamma with the first value of the lambda sequence
    data.gamma = data.gammaSeq[0];

    uint32_t* foldid = (uint32_t*)INTEGER(_foldid);
    // foldid must have N elements otherwise the following fails, but checked in
    // R!
    for (uint32_t i = 0; i < N; i++)
        data.foldid.push_back(foldid[i]);

    double* xR = REAL(_x);
    for (uint32_t j = 0; j < data.P; ++j)
        memcpy(&(data.x[INDEX_COL(j, data.memory_N)]),
               &xR[INDEX_COL(j, data.N)], data.N * sizeof(double));

    if (data.useFusion) {
        double* fusionListR = REAL(_fusion);
        uint32_t rows = (uint32_t)INTEGER(GET_DIM(_fusion))[0];

        uint32_t i, j;
        double x;

        for (uint32_t row = 0; row < rows; row++) {
            i = (uint32_t)fusionListR[INDEX(row, 0, rows)];
            j = (uint32_t)fusionListR[INDEX(row, 1, rows)];
            x = fusionListR[INDEX(row, 2, rows)];

            data.fusionKernel[j] = appendElement(data.fusionKernel[j], i, x);
        }
    }

    double* yR = REAL(_y);
    double* betaR = REAL(_beta);
    double* wR = REAL(_w);

    if (type == zeroSum::types::cox) {
        SEXP _status = getElementFromRList(_dataObjects, "status");
        memcpy(data.status, (uint32_t*)INTEGER(_status),
               data.N * sizeof(uint32_t));
    }

    for (uint32_t f = 0; f < data.nFold1; f++) {
        for (uint32_t l = 0; l < data.K; ++l) {
            uint32_t iiN = INDEX_TENSOR_COL(l, f, data.memory_N, data.K);
            uint32_t iiP = INDEX_TENSOR_COL(l, f, data.memory_P, data.K);
            // copy y from R to C memory
            memcpy(&data.y[iiN], &yR[INDEX_COL(l, data.N)],
                   data.N * sizeof(double));

            // copy beta from R to C memory
            double* betaRF = &betaR[INDEX_TENSOR_COL(l, f, data.P + 1, data.K)];
            data.intercept[INDEX(l, f, K)] = betaRF[0];
            memcpy(&(data.beta[iiP]), &betaRF[1], data.P * sizeof(double));

            // copy the weights from R to C memory
            memcpy(&data.w[iiN], wR, data.N * sizeof(double));
        }

        uint32_t iiF = INDEX_COL(f, data.memory_N);
        memcpy(&data.wOrg[iiF], wR, data.N * sizeof(double));
        memcpy(&data.wCV[iiF], wR, data.N * sizeof(double));

        memcpy(&data.v[INDEX_COL(f, data.memory_P)], REAL(_v),
               data.P * sizeof(double));
    }

    memcpy(data.u, REAL(_u), data.P * sizeof(double));
    memcpy(data.yOrg, data.y, data.memory_N * data.K * sizeof(double));

    if (type == zeroSum::types::cox)
        data.calcCoxRegressionD();

    return data;
}

SEXP CV(SEXP _dataObjects) {
    PROTECT(_dataObjects = AS_LIST(_dataObjects));
    zeroSum data = rListToRegressionData(_dataObjects);

    std::vector<double> cv_stats = data.doCVRegression();
    SEXP measures;
    PROTECT(measures = allocVector(REALSXP, cv_stats.size()));
    double* m = REAL(measures);

    for (size_t i = 0; i < cv_stats.size(); i++)
        m[i] = cv_stats[i];

    UNPROTECT(2);
    return measures;
}

SEXP checkMoves(SEXP _dataObjects,
                SEXP _number,
                SEXP _k,
                SEXP _s,
                SEXP _t,
                SEXP _l) {
    PROTECT(_dataObjects = AS_LIST(_dataObjects));
    PROTECT(_number = AS_INTEGER(_number));
    uint32_t num = (uint32_t)INTEGER(_number)[0];

    PROTECT(_k = AS_INTEGER(_k));
    uint32_t k = (uint32_t)INTEGER(_k)[0];

    PROTECT(_s = AS_INTEGER(_s));
    uint32_t s = (uint32_t)INTEGER(_s)[0];

    PROTECT(_t = AS_INTEGER(_t));
    uint32_t t = (uint32_t)INTEGER(_t)[0];

    PROTECT(_l = AS_INTEGER(_l));
    uint32_t l = (uint32_t)INTEGER(_l)[0];

    zeroSum data = rListToRegressionData(_dataObjects);

    data.costFunction(0);
    data.approxFailed[0] = FALSE;

    SEXP result = R_NilValue;

    if (num == 0) {
        data.cdMove(0, k, l);

        PROTECT(result = allocVector(REALSXP, 1));
        REAL(result)[0] = data.beta[INDEX(k, l, data.memory_P)];
    } else if (num == 1) {
        data.interceptMove(0, l);

        PROTECT(result = allocVector(REALSXP, 1));
        REAL(result)[0] = data.intercept[l];
    } else if (num == 2) {
        data.cdMoveZS(0, k, s, l);

        PROTECT(result = allocVector(REALSXP, 2));
        REAL(result)[0] = data.beta[INDEX(k, l, data.memory_P)];
        REAL(result)[1] = data.beta[INDEX(s, l, data.memory_P)];
    } else if (num == 3) {
        data.cdMoveZSRotated(0, k, s, t, l, 37.32);

        PROTECT(result = allocVector(REALSXP, 3));
        REAL(result)[0] = data.beta[INDEX(k, l, data.memory_P)];
        REAL(result)[1] = data.beta[INDEX(s, l, data.memory_P)];
        REAL(result)[2] = data.beta[INDEX(t, l, data.memory_P)];
    }

    UNPROTECT(7);
    return result;
}

SEXP costFunctionWrapper(SEXP _dataObjects) {
    PROTECT(_dataObjects = AS_LIST(_dataObjects));
    zeroSum data = rListToRegressionData(_dataObjects);

    // printf("N: %d K: %d P: %d\n", data.N, data.K, data.P);
    // printf("x: \n");
    // printMatrix(data.x, data.N, data.P);
    // printf("y: \n");
    // printMatrix(data.y, data.N, data.K);
    // printf("data is zeroSum: %d\n", data.useZeroSum);

    data.costFunction(0);
    SEXP returnList, names;
    PROTECT(returnList = allocVector(VECSXP, 5));
    PROTECT(names = allocVector(STRSXP, 5));

    SET_STRING_ELT(names, 0, mkChar("loglikelihood"));
    SET_STRING_ELT(names, 1, mkChar("lasso"));
    SET_STRING_ELT(names, 2, mkChar("ridge"));
    SET_STRING_ELT(names, 3, mkChar("fusion"));
    SET_STRING_ELT(names, 4, mkChar("cost"));

    setAttrib(returnList, R_NamesSymbol, names);

    SEXP loglikelihood, lasso, ridge, fusion, cost;
    PROTECT(loglikelihood = allocVector(REALSXP, 1));
    PROTECT(lasso = allocVector(REALSXP, 1));
    PROTECT(ridge = allocVector(REALSXP, 1));
    PROTECT(fusion = allocVector(REALSXP, 1));
    PROTECT(cost = allocVector(REALSXP, 1));
    REAL(loglikelihood)[0] = data.loglikelihood[0];
    REAL(lasso)[0] = data.lasso[0];
    REAL(ridge)[0] = data.ridge[0];
    REAL(fusion)[0] = data.fusion[0];
    REAL(cost)[0] = data.cost[0];

    SET_VECTOR_ELT(returnList, 0, loglikelihood);
    SET_VECTOR_ELT(returnList, 1, lasso);
    SET_VECTOR_ELT(returnList, 2, ridge);
    SET_VECTOR_ELT(returnList, 3, fusion);
    SET_VECTOR_ELT(returnList, 4, cost);

    UNPROTECT(8);
    return returnList;
}

SEXP lambdaMax(SEXP _X, SEXP _res, SEXP _u, SEXP _v, SEXP _alpha) {
    PROTECT(_X = AS_NUMERIC(_X));
    double* x = REAL(_X);

    uint32_t* dimX = (uint32_t*)INTEGER(GET_DIM(_X));
    uint32_t N = dimX[0];
    uint32_t P = dimX[1];

    PROTECT(_res = AS_NUMERIC(_res));

    uint32_t* dimRes = (uint32_t*)INTEGER(GET_DIM(_res));
    uint32_t K = dimRes[1];
    uint32_t memory_N = N;
    double* res = REAL(_res);

    PROTECT(_u = AS_NUMERIC(_u));
    double* u = REAL(_u);

    PROTECT(_v = AS_NUMERIC(_v));
    double* v = REAL(_v);

    PROTECT(_alpha = AS_NUMERIC(_alpha));
    double alpha = REAL(_alpha)[0];

    double tmp1, tmp2;

    double lambdaMax = DBL_MIN;
    double lambda;
    for (uint32_t l = 0; l < K; ++l) {
        for (uint32_t k = 2; k < P; ++k) {
            if (u[k] == 0)
                continue;
            for (uint32_t s = 1; s < k; ++s) {
                if (u[s] == 0)
                    continue;
                tmp1 = u[k] / u[s];
                tmp2 = alpha * (v[k] + v[s] * tmp1);
                if (fabs(tmp2) < 1000 * DBL_EPSILON)
                    continue;

                lambda = 0.0;
                for (uint32_t i = 0; i < N; ++i)
                    lambda += (x[INDEX(i, s, N)] * tmp1 - x[INDEX(i, k, N)]) *
                              res[INDEX(i, l, memory_N)];

                lambda = fabs(lambda);
                lambda /= tmp2;

                if (lambda > lambdaMax)
                    lambdaMax = lambda;
            }
        }
    }

    UNPROTECT(5);
    return ScalarReal(lambdaMax);
}

extern "C" {

static const R_CallMethodDef callMethods[] = {
    {"CV", (DL_FUNC)(void*)&CV, 1},
    {"costFunctionWrapper", (DL_FUNC)(void*)&costFunctionWrapper, 1},
    {"checkMoves", (DL_FUNC)(void*)&checkMoves, 6},
    {"lambdaMax", (DL_FUNC)(void*)&lambdaMax, 5},
    {NULL, NULL, 0}};

void R_init_zeroSum(DllInfo* info) {
    R_registerRoutines(info, NULL, callMethods, NULL, NULL);
    R_useDynamicSymbols(info, FALSE);
}
}
