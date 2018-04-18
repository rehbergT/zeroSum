#ifndef SETTINGS_H
#define SETTINGS_H

#ifdef __AVX512F__

#define AVX_VERSION_512
#define FMA

#endif

#ifndef AVX_VERSION_512

#ifdef __AVX__
#define AVX_VERSION_256
#endif

#ifdef __AVX2__
#define FMA
#endif

#endif

#ifdef AVX_VERSION_512

#define AVX_VERSION
#define ALIGNMENT 64
#define ALIGNED_DOUBLES 8
#include <x86intrin.h>
#define FMA

#else

#define ALIGNMENT 32
#define ALIGNED_DOUBLES 4

#endif

#ifdef AVX_VERSION_256

#define AVX_VERSION
#include <x86intrin.h>

#endif

#ifdef R_PACKAGE

#include <R.h>
#include <R_ext/Rdynload.h>
#define PRINT Rprintf

#else

#define PRINT printf

#endif

// #define DEBUG
#ifdef DEBUG
#include <ctime>
#endif

#define INDEX(i, j, N) ((i) + ((j) * (N)))
#define INDEX_ROW(i, j, M) ((j) + ((i) * (M)))
#define MAX_PATH_LENGTH 1000

#define TRUE 1
#define FALSE 0

#define GAUSSIAN 1
#define GAUSSIAN_ZS 2
#define FUSION_GAUSSIAN 3
#define FUSION_GAUSSIAN_ZS 4
#define BINOMIAL 5
#define BINOMIAL_ZS 6
#define FUSION_BINOMIAL 7
#define FUSION_BINOMIAL_ZS 8
#define MULTINOMIAL 9
#define MULTINOMIAL_ZS 10
#define FUSION_MULTINOMIAL 11
#define FUSION_MULTINOMIAL_ZS 12

#define COX 13
#define COX_ZS 14
#define FUSION_COX 15
#define FUSION_COX_ZS 16


// localSearch setttings
#define MAX_STEPS 500
#define INTERVAL_SIZE 0.2
#define INTERVAL_SHRINK 0.9
#define LS_MIN_REPEATS 100

#define SWEEPS_RANDOM 15
#define SWEEPS_ACTIVESET 8
#define SWEEPS_NULL 1
#define SWEEPS_FUSED 10

// simulatedAnnealing settings
#define T_STEPS 10000
#define T_START 0.2
#define ACCEPTRATE 0.5

#define MEASURE 500
#define THERMALIZE 300
#define COOLING_FAKTOR 0.8

#define BETA_CHANGE_PRECISION 1e-14
#define COX_MIN_PRECISION 1e-10

#endif /* SETTINGS_H */
