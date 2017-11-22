#ifndef MATHHELPERS_H
#define MATHHELPERS_H

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "fusionKernel.h"
#include "settings.h"

int getMax(double* a, int n);
double median(double* x, int N);
double mean(double* a, int N);
double sd(double* a, int N, double* mean_ptr = nullptr);

double sum_a(double* a, int n);

double sum_square_a(double* a, int n);

double sum_abs_a(double* a, int n);

double sum_abs_a_times_b(double* a, double* b, int n);

double sum_square_a_times_b(double* a, double* b, int n);

double sum_a_times_b(double* a, double* b, int n);

double sum_a_sub_b_times_c(double* a, double* b, double* c, int n);

double sum_a_add_b_times_c(double* a, double* b, double* c, int n);

double sum_a_sub_b_mul_d_times_c(double* a,
                                 double* b,
                                 double* c,
                                 double d,
                                 int n);

void a_times_b(double* a, double* b, double* c, int n);

void a_add_b(double* a, double* b, double* c, int n);

void a_sub_b(double* a, double* b, double* c, int n);

double sum_a_times_b_times_c(double* a, double* b, double* c, int n);

void a_add_scalar_b(double* a, double b, double* c, int n);

void a_times_scalar_b(double* a, double b, double* c, int n);

void add_a_add_scalar_b(double* a, double b, double* c, int n);

void sub_a_times_scalar_b_sub_c_times_scalar_d(double* a,
                                               double b,
                                               double* c,
                                               double d,
                                               double* e,
                                               int n);

void sub_a_times_scalar_b_sub_c_times_scalar_d_sub_d_times_scalar_f(double* a,
                                                                    double b,
                                                                    double* c,
                                                                    double d,
                                                                    double* e,
                                                                    double f,
                                                                    double* res,
                                                                    int n);

void a_times_scalar_b_sub_c(double* a, double b, double* c, double* d, int n);

void a_add_scalar_b_times_c_sub_d_times_e(double* a,
                                          double b,
                                          double* c,
                                          double* d,
                                          double e,
                                          double* res,
                                          int n);

void a_times_scalar_b_add_c_times_scalar_d_add_d_times_scalar_f(double* a,
                                                                double b,
                                                                double* c,
                                                                double d,
                                                                double* e,
                                                                double f,
                                                                double* res,
                                                                int n);

int getMinIndex(double* a, int N);

void printMatrix(double* x, int N, int M);

#endif /* MATHHELPERS_H */
