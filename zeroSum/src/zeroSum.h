/*!
 * Note that this software follows the approach proposed by J. Friedman et al.
 * in "Regularization paths for generalized linear models via coordinate
 * descent" and N. Simon et al. in "Regularization paths for cox’s proportional
 * hazards model via coordinate descent" as implemented in the R package glmnet.
 * However, this software allows to additionally enforce the zero-sum constraint
 * as described in M. Altenbuchinger et al. "Reference point insensitive
 * molecular data analysis".
 *
 */

#ifndef REGRESSION_CD_H
#define REGRESSION_CD_H

#include <x86intrin.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <future>
#include <mutex>
#include <random>
#include <unordered_set>
#include <vector>
#include "Parallel.h"
#include "fusionKernel.h"
#include "vectorizableKernels.h"

// #define DEBUG
#ifdef DEBUG
#include <ctime>
#endif

#define INDEX(i, j, N) ((i) + ((j) * (N)))
#define INDEX_COL(j, N) ((j) * (N))
#define INDEX_TENSOR(i, j, k, N, M) ((i) + (((j) + (k) * (M)) * (N)))
#define INDEX_TENSOR_COL(j, k, N, M) ((((j) + (k) * (M)) * (N)))
#define MAX_LINE_LENGTH 1000

#define BETA_CHANGE_PRECISION 1e-14
#define COX_MIN_PRECISION 1e-10

#ifdef R_PACKAGE
#include <R.h>
#include <R_ext/Rdynload.h>
#include <Rinternals.h>
#define PRINT Rprintf
#else
#define PRINT printf
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

// declare used BLAS functions -> no header file required -> can be directly
// linked to the blas version used by R
extern "C" {
// C = alpha * a %*% n + beta * C
extern void dgemm_(const char* transa,
                   const char* transb,
                   const uint32_t* m,
                   const uint32_t* n,
                   const uint32_t* k,
                   const double* alpha,
                   const double* a,
                   const uint32_t* lda,
                   const double* b,
                   const uint32_t* ldb,
                   const double* beta,
                   double* c,
                   const uint32_t* ldc);
// dy = alpha * dx + dy
extern void daxpy_(const uint32_t* n,
                   const double* alpha,
                   const double* dx,
                   const uint32_t* incx,
                   double* dy,
                   const uint32_t* incy);
//  dx * dy
extern double ddot_(const uint32_t* n,
                    const double* dx,
                    const uint32_t* incx,
                    const double* dy,
                    const uint32_t* incy);

// alpha * a %*% x + beta * y
extern double dgemv_(const char* trans,
                     const uint32_t* m,
                     const uint32_t* n,
                     const double* alpha,
                     const double* a,
                     const uint32_t* lda,
                     const double* x,
                     const uint32_t* incx,
                     const double* beta,
                     double* y,
                     const uint32_t* incy);

// abs sum
extern double dasum_(const uint32_t* n, const double* dx, const uint32_t* incx);
}

/**
 * A class for storing all data which can be accessed by all cross validation
 * folds
 *
 * @author  Thorsten Rehberg <thorsten.rehberg@ur.de>
 *
 */
class zeroSum {
   private:
    /** A function for allocating all necessary memory. Should only be called
     * from the different constructors.
     */
    void allocData();

    /** A function for freeing all allocated memory. Should only be called
     * from the different constructors.
     */
    void freeData();

    /** A function for copying all member variables but NOT array contents.
     */
    void shallowCopy(const zeroSum& source);

    /** A function for copying all array contents but no other variables.
     */
    void deepCopy(const zeroSum& source);

    /** A function for copying all array points from source and setting all
     * pointer in source to nullptr
     */
    void pointerMove(zeroSum& source);

    /** A function for adjusting the sample weights of each fold of the cross
     * validation to remove samples from a fold (set weight to zero) and to
     * adjust the size correction */
    void adjustWeights();

    /** A function for printing the active set (only used in debug mode)
     */
    void activeSetPrint(uint32_t fold);

    /** A function for inserting beta_k in the active set of fold fold
     */
    bool activeSetInsert(uint32_t fold, uint32_t k);

    /** A function for checking if beta_k is in the active set of fold fold
     */
    bool activeSetContains(uint32_t fold, uint32_t k);

    /** A function getting the element k of the active set. Note that it is an
     * unordered_set -> only used for getting random elements
     */
    uint32_t activeSetGetElement(uint32_t fold, uint32_t k);

    /** A function for removing beta which are zero from the active set of fold
     * fold
     */
    void activeSetRemoveZeros(uint32_t fold);

    /** A function for calculating the loglikelihood
     */
    double calcLogLikelihood(uint32_t fold, double* xb);

    /** A function for mean-centering x */
    void meanCentering();

    /** A function for standardizing x. However, x is not really standardized
     * but the regularization weights v are divided by the the standard
     * deviation. Note that this function uses beta as intermediate storage and
     * causes that beta is complelty set to 0. */
    void standardizeData();

    /** A function for standardizing y. However, y is not really standardized
     * but the regularization weights v are divided by the the standard
     * deviation. */
    void standardizeResponse();

    /** A function for calculating the sum of a double array of length n.
     * However, the memory array has to aliged and the padding space filled with
     * zeros
     */
    double arraySumAvx(double* a, uint32_t n);

    /** A function for calculating the sum of a double array of length n */
    double arraySum(double* a, uint32_t n);

    /** A function for calculating the weighted abs sum of a double array of
     * length n */
    double weightedAbsSum(double* a, double* b, uint32_t n);

    /** A function for calculating the weighted square sum of a double
     * array of length n */
    double weightedSquareSum(double* a, double* b, uint32_t n);

    /** A function for calculating the weighted residual square sum
     * a * ( b - c)^2 */
    double weightedResidualSquareSum(double* a,
                                     double* b,
                                     double* c,
                                     uint32_t n);

    /** A function for calculating b*a^2 */
    double squareWeightedSum(double* a, double* b, uint32_t n);

    /** A function for calculating a - b and stores the result in c */
    void a_sub_b(double* a, double* b, double* c, uint32_t n);

    /** A function for adding the double b to the every element of the
    double array a */
    void a_add_scalar_b(double* a, double b, uint32_t n);

    /** Some constants used for BLAS calls */
    const char BLAS_NO = 'N';
    const char BLAS_T = 'T';
    const double BLAS_D_ONE = 1.0;
    const double BLAS_D_ZERO = 0.0;
    const double BLAS_D_MINUS_ONE = -1.0;
    const uint32_t BLAS_I_ONE = 1.0;

   public:
    /** The number of samples */
    uint32_t N;

    /** The number of features */
    uint32_t P;

    /** The number of classes of a multinomial regression else K is set to 1 */
    uint32_t K;

    /** The number of rows of the fusionKernel */
    uint32_t nc;

    /** The specified regression type (numbers used as in R) */
    enum types { gaussian = 1, binomial = 2, multinomial = 3, cox = 4 };
    uint32_t type;

    /** variable which specifies if the zero-sum constraint should be used */
    bool useZeroSum;

    /** variable which specifies if the generalized lasso regularization should
     * be used */
    bool useFusion;

    /** variable which specifies if the intercept should be used */
    bool useIntercept;

    /** variable which specifies if the approximation should be used, only valid
     * for simulated annealing and local search */
    bool useApprox;

    /** variable which specifies if the predictor matrix x should be centered */
    bool useCentering;

    /** variable which specifies if the predictor matrix x should be
     * standardized. However, x is not really standardized but the
     * regularization weights v are divided by the the standard deviation.  */
    bool useStandardization;

    /** variable which specifies if the polish of the coordinate decent should
     * be used */
    bool usePolish;

    /** variable which specifies if the rotated updates of the coordinate decent
     * should be used */
    uint32_t rotatedUpdates;

    /** variable which specifies the convergence precision */
    double precision;

    /** Specify which algorithm should be used (same numbers used as in R) */
    enum algorithms {
        coodinateDescent = 1,
        simulatedAnnealing = 2,
        localSearch = 3,
        coodinateDescentParallel = 4
    };
    uint32_t algorithm;

    /** The number of cross validation folds */
    uint32_t nFold;
    uint32_t nFold1;  // = nFold+1

    /** The number of allowed successive worsening cv steps */
    uint32_t cvStop;

    /** Flag if runtime details should be printed */
    uint32_t verbose;

    /** The number of rows of x (= samples + placeholder lines for alignment) */
    uint32_t memory_N;

    /** The number of cols of x (= features + placeholder lines for alignment)
     */
    uint32_t memory_P;

    /** The number of rows of the fusionKernel in memory (= The number of rows
     * of the fusionKernel + placeholder lines for alignment) */
    uint32_t memory_nc;

    /** The supplied lambda sequence (elastic net regularization weight). Has to
     * be stored per fold since standardization scales lambda */
    std::vector<double> lambdaSeq;

    /** weight of the elastic net regularization with 1=lasso, 0=ridge. Has to
     * be stored per fold since standardization scales lambda. */
    double* lambda;

    /** The supplied gamma sequence (fusion regularization weight) */
    std::vector<double> gammaSeq;

    /** weight of the generalized lasso regularization (fusion kernel).  */
    double gamma;

    /** The sum of coefficients should add up to cSum. Only different from zero
     * in the multinomial case due to the parameter ambiguity problem */
    double cSum;

    /** weight within the elastic net regularization.  1=lasso, 0=ridge */
    double alpha;

    /** variable for reducing the amount of iterations of the for loops of local
     * search and simulated annealing */
    double downScaler;

    /** Defines the number of threads, which should be used  */
    uint32_t threads = 1;

    /** vector of length N specifing the fold of each samples */
    std::vector<uint32_t> foldid;

    /** parallel execution funtions */
    Parallel parallel;

    /** The feature matrix with rows=samples, cols=features. The memory of the
     * columns has to be 32-byte (AVX/AVX2) or 64-byte (AVX512) aligned. Thus,
     * additional rows with zero have to be appended so that the second, third,
     * ... columns starts aligned */
    double* x;

    /** The response vector or matrix (in multinomial case with rows=samples,
     * columns=classes). The memory of the columns has to be 32-byte (AVX/AVX2)
     * or 64-byte (AVX512) aligned. Thus, additional rows with zero have to be
     * appended so that the second, third, ... columns starts aligned */
    double* yOrg;

    /** The working response vector = the same as y in the gaussian case, but
     * otherwise the local approximation otherwise -> every fold needs its own
     * working response. ( N*K*NFold matrix/tensor in the multinomial case)
     */
    double* y;

    /** Sample weights for the contribution to the loglikelihood (vector length
     * of P) */
    double* wOrg;

    /** The working weights vector = the same as w in the gaussian case, but
     * otherwise the local approximation -> every fold needs its own weights. (
     * N * K * nFold matrix/tensor in the multinomial case) */
    double* w;

    /** The sample weights used to set weights to zero to remove them from the
     * cross validation */
    double* wCV;

    /** Matrix saving the residuals (y - beta_0 - x * beta) in the gaussian case
        or the evaluation of the linear model (beta_0 + x * beta). (N *
       K * nFold matrix/tensor in the multinomial case) */
    double* xTimesBeta;

    /** Matrix saving the coefficients (P * K * nFold matrix/tensor in the
     * multinomial case) */
    double* beta;

    /** Matrix saving the intercepts (K*nFold matrix) */
    double* intercept;

    /** array for saving the censoring status and d of the cox regression
     * (N*nFold matrix) */
    uint32_t* status;
    double* d;

    /** N x NFold array for storing intermediate results only allowed to be
     * used within a function */
    double* tmp_array1;

    /** N x K x NFold array for storing intermediate results only allowed to be
     * used within a function */
    double* tmp_array2;

    /** weights of the elastic-net regularization (vector length of P) */
    double* v;

    /** weights of the zero-sum constraint (vector length P)*/
    double* u;

    /** Sparse matrix (linked list) containing the entries of the fusion matrix
     * with nc rows and P columns */
    struct fusionKernel** fusionKernel;

    /** arrays for storing intermediate sum of the fusion kernel calculations */
    double* fusionPartialSums;
    double* fusionPartialSumsTmp;
    double* fusionSums;

    /**
    Vector containing the loglikelihood of each cross-validation fold and of the
    full sample fit
    */
    double* loglikelihood;

    /** Vector containing the lasso regularization of each cross-validation fold
     * and of the full sample fit */
    double* lasso;

    /** Vector containing the ridge regularization of each cross-validation fold
     * and of the full sample fit */
    double* ridge;

    /** Vector containing the fusion regularization of each cross-validation
     * fold and of the full sample fit */
    double* fusion;

    /** Vector containing the costs of each cross-validation fold and of the
     * full sample fit */
    double* cost;

    /** Vector containing the mean of each feature (length P) */
    double* featureMean;

    /** Store the standard deviation of y */
    double* ySD;

    /** Enum to store if AVX2, AVX512 or fallback should be used
     *  (detected at runtime) */
    enum avxTypes { fallback, avx2, avx512 };
    uint32_t avxType = fallback;

    /** Seed used for the random number generator */
    uint32_t seed;

    /** array for storing if the approximation for an fold and class has failed
     */
    std::vector<uint32_t> approxFailed;

    /** array for storing the activeSet of fold and class */
    std::vector<std::unordered_set<uint32_t>> activeSet;
    std::vector<std::vector<std::array<uint32_t, 3>>> parallelActiveSet;

    /** array for storing the last best found beta configuration */
    double* last_beta;

    /** array for storing the last best found intercept configuration */
    double* last_intercept;

    void doFitUsingCoordinateDescent(uint32_t seed);
    void doFitUsingCoordinateDescentParallel(uint32_t seed);
    void doFitUsingLocalSearch(uint32_t seed);
    void doFitUsingSimulatedAnnealing(uint32_t seed);
    void calcCoxRegressionD();

    void costFunctionAllFolds();
    void costFunction(uint32_t f);
    void updateCost(uint32_t fold, uint32_t l);
    void predict();

    void refreshApproximation(uint32_t fold,
                              uint32_t l,
                              uint32_t _updateCost = false);
    void interceptMove(uint32_t fold, uint32_t l);
    uint32_t cdMove(uint32_t fold, uint32_t k, uint32_t l);
    uint32_t cdMoveZS(uint32_t fold, uint32_t k, uint32_t s, uint32_t l);
    void cdMove_parallel(uint32_t* improving, uint32_t steps);
    void cdMoveZS_parallel(uint32_t* improving, uint32_t steps);
    uint32_t cdMoveZSRotated(uint32_t fold,
                             uint32_t n,
                             uint32_t m,
                             uint32_t s,
                             uint32_t l,
                             double theta);

    double penaltyCost(double* coefs, double t);
    void optimizeParameterAmbiguity(uint32_t fold, uint32_t iterations = 10);

    void lsSaOffsetMove(uint32_t fold, uint32_t l);
    uint32_t lsSaMove(uint32_t fold,
                      uint32_t k,
                      uint32_t s,
                      uint32_t l,
                      double delta_k,
                      double* rng = nullptr,
                      double temperature = 0);

    /** A function for printing a N times M matrix stored in the double array x
     */
    void printMatrix(double* x, uint32_t N, uint32_t M);

    /** A function for printing a N times M matrix stored in the uint32_t array
     * x
     */
    void printIntMatrix(uint32_t* x, uint32_t N, uint32_t M);

    /** A function for printing a N times M sparse matrix stored in a fusion
     * kernel struct x
     */
    void printSparseMatrix(struct fusionKernel** x, uint32_t M);

    // constructor but the content of lambda, gamma, foldid, x, y, fusion,
    // status, w, v, u has to be filled additionally
    zeroSum(uint32_t N,
            uint32_t P,
            uint32_t K,
            uint32_t nc,
            uint32_t type,
            bool useZeroSum,
            bool useFusion,
            bool useIntercept,
            bool useApprox,
            bool useCentering,
            bool useStandardization,
            bool usePolish,
            uint32_t rotatedUpdates,
            double precision,
            uint32_t algorithm,
            uint32_t nFold,
            uint32_t cvStop,
            uint32_t verbose,
            double cSum,
            double alpha,
            double downScaler,
            uint32_t threads,
            uint32_t seed);

    // copy constructor
    zeroSum(const zeroSum& source);

    // move constructor
    zeroSum(zeroSum&& source);

    // copy assignment operator
    zeroSum& operator=(const zeroSum& source);

    // move assignment operator
    zeroSum& operator=(zeroSum&& source);

    // destructor
    ~zeroSum();

    std::vector<double> doCVRegression(char* path = nullptr,
                                       char* name = nullptr,
                                       uint32_t mpi_rank = 0);
};

#endif
