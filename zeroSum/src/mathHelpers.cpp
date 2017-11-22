#include "mathHelpers.h"

int getMax(double* a, int n) {
    int max = 0;
    for (int i = 1; i < n; ++i) {
        if (a[max] < a[i])
            max = i;
    }
    return a[max];
}

int cmpfunc(const void* a, const void* b) {
    if (*(double*)a < *(double*)b)
        return -1;
    if (*(double*)a == *(double*)b)
        return 0;
    if (*(double*)a > *(double*)b)
        return 1;
    return 0;
}

double median(double* x, int N) {
    qsort(x, N, sizeof(double), cmpfunc);

    if (N % 2 == 0)
        return ((x[N / 2] + x[N / 2 - 1]) / 2.0);
    else
        return x[N / 2];
}

double mean(double* a, int N) {
    double sum = sum_a(a, N);
    sum /= (double)N;

    return sum;
}

double sd(double* a, int N, double* mean_ptr) {
    double m = mean_ptr == nullptr ? *mean_ptr : mean(a, N);
    double s = sum_square_a(a, N) / (double)N;
    s = s - m * m;
    return sqrt(s);
}

#ifdef AVX_VERSION

double sum_a(double* a, int n) {
    if (n < 4) {
        double sum = 0.0;
        for (int i = 0; i < n; ++i)
            sum += a[i];

        return sum;
    } else {
        __m256d _a;
        __m256d _res = _mm256_setzero_pd();
        int i = 0;
        for (; i < n - (n % 4); i += 4) {
            _a = _mm256_load_pd(&a[i]);
            _res = _mm256_add_pd(_a, _res);
        }

        double* res = (double*)&_res;
        double sum = res[0] + res[1] + res[2] + res[3];

        for (; i < n; ++i)
            sum += a[i];
        return sum;
    }
}

double sum_square_a(double* a, int n) {
    if (n < 4) {
        double sum = 0.0;
        for (int i = 0; i < n; ++i)
            sum += a[i] * a[i];

        return sum;
    } else {
        __m256d _a;
        __m256d _res = _mm256_setzero_pd();

        int i = 0;
        for (; i < n - (n % 4); i += 4) {
            _a = _mm256_load_pd(&a[i]);

#ifdef FMA
            _res = _mm256_fmadd_pd(_a, _a, _res);
#else
            _a = _mm256_mul_pd(_a, _a);
            _res = _mm256_add_pd(_a, _res);
#endif
        }

        double* res = (double*)&_res;
        double sum = res[0] + res[1] + res[2] + res[3];

        for (; i < n; ++i)
            sum += a[i] * a[i];
        return sum;
    }
}

double sum_abs_a(double* a, int n) {
    if (n < 4) {
        double sum = 0.0;
        for (int i = 0; i < n; ++i)
            sum += fabs(a[i]);

        return sum;
    } else {
        __m256d _a;
        __m256d _res = _mm256_setzero_pd();
        __m256d mask = _mm256_castsi256_pd(
            _mm256_setr_epi64x(0x7FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF,
                               0x7FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF));

        int i = 0;
        for (; i < n - (n % 4); i += 4) {
            _a = _mm256_load_pd(&a[i]);
            _a = _mm256_and_pd(_a, mask);
            _res = _mm256_add_pd(_a, _res);
        }

        double* res = (double*)&_res;
        double sum = res[0] + res[1] + res[2] + res[3];

        for (; i < n; ++i)
            sum += fabs(a[i]);
        return sum;
    }
}

double sum_abs_a_times_b(double* a, double* b, int n) {
    if (n < 4) {
        double sum = 0.0;
        for (int i = 0; i < n; ++i)
            sum += b[i] * fabs(a[i]);

        return sum;
    } else {
        __m256d _a, _b;
        __m256d _res = _mm256_setzero_pd();
        __m256d mask = _mm256_castsi256_pd(
            _mm256_setr_epi64x(0x7FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF,
                               0x7FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF));

        int i = 0;
        for (; i < n - (n % 4); i += 4) {
            _a = _mm256_load_pd(&a[i]);
            _a = _mm256_and_pd(_a, mask);

            _b = _mm256_load_pd(&b[i]);

#ifdef FMA
            _res = _mm256_fmadd_pd(_a, _b, _res);
#else
            _a = _mm256_mul_pd(_a, _b);
            _res = _mm256_add_pd(_a, _res);
#endif
        }

        double* res = (double*)&_res;
        double sum = res[0] + res[1] + res[2] + res[3];

        for (; i < n; ++i)
            sum += b[i] * fabs(a[i]);
        return sum;
    }
}

double sum_square_a_times_b(double* a, double* b, int n) {
    if (n < 4) {
        double sum = 0.0;
        for (int i = 0; i < n; ++i)
            sum += a[i] * a[i] * b[i];

        return sum;
    } else {
        __m256d _a, _b;
        __m256d _res = _mm256_setzero_pd();

        int i = 0;
        for (; i < n - (n % 4); i += 4) {
            _a = _mm256_load_pd(&a[i]);
            _a = _mm256_mul_pd(_a, _a);

            _b = _mm256_load_pd(&b[i]);

#ifdef FMA
            _res = _mm256_fmadd_pd(_a, _b, _res);
#else
            _a = _mm256_mul_pd(_a, _b);
            _res = _mm256_add_pd(_a, _res);
#endif
        }

        double* res = (double*)&_res;
        double sum = res[0] + res[1] + res[2] + res[3];

        for (; i < n; ++i)
            sum += a[i] * a[i] * b[i];
        return sum;
    }
}

double sum_a_times_b(double* a, double* b, int n) {
    if (n < 4) {
        double sum = 0.0;
        for (int i = 0; i < n; ++i)
            sum += a[i] * b[i];

        return sum;
    } else {
        __m256d _a, _b;
        __m256d _res = _mm256_setzero_pd();

        int i = 0;
        for (; i < n - (n % 4); i += 4) {
            _a = _mm256_load_pd(&a[i]);
            _b = _mm256_load_pd(&b[i]);

#ifdef FMA
            _res = _mm256_fmadd_pd(_a, _b, _res);
#else
            _a = _mm256_mul_pd(_a, _b);
            _res = _mm256_add_pd(_a, _res);
#endif
        }

        double* res = (double*)&_res;
        double sum = res[0] + res[1] + res[2] + res[3];

        for (; i < n; ++i)
            sum += a[i] * b[i];
        return sum;
    }
}

double sum_a_sub_b_times_c(double* a, double* b, double* c, int n) {
    if (n < 4) {
        double sum = 0.0;
        for (int i = 0; i < n; ++i)
            sum += (a[i] - b[i]) * c[i];

        return sum;
    } else {
        __m256d _a, _b, _c;
        __m256d _res = _mm256_setzero_pd();

        int i = 0;
        for (; i < n - (n % 4); i += 4) {
            _a = _mm256_load_pd(&a[i]);
            _b = _mm256_load_pd(&b[i]);
            _a = _mm256_sub_pd(_a, _b);

            _c = _mm256_load_pd(&c[i]);

#ifdef FMA
            _res = _mm256_fmadd_pd(_a, _c, _res);
#else
            _c = _mm256_mul_pd(_a, _c);
            _res = _mm256_add_pd(_c, _res);
#endif
        }

        double* res = (double*)&_res;
        double sum = res[0] + res[1] + res[2] + res[3];

        for (; i < n; ++i)
            sum += (a[i] - b[i]) * c[i];
        return sum;
    }
}

double sum_a_add_b_times_c(double* a, double* b, double* c, int n) {
    if (n < 4) {
        double sum = 0.0;
        for (int i = 0; i < n; ++i)
            sum += (a[i] + b[i]) * c[i];

        return sum;
    } else {
        __m256d _a, _b, _c;
        __m256d _res = _mm256_setzero_pd();

        int i = 0;
        for (; i < n - (n % 4); i += 4) {
            _a = _mm256_load_pd(&a[i]);
            _b = _mm256_load_pd(&b[i]);
            _a = _mm256_add_pd(_a, _b);

            _c = _mm256_load_pd(&c[i]);

#ifdef FMA
            _res = _mm256_fmadd_pd(_a, _c, _res);
#else
            _c = _mm256_mul_pd(_a, _c);
            _res = _mm256_add_pd(_c, _res);
#endif
        }

        double* res = (double*)&_res;
        double sum = res[0] + res[1] + res[2] + res[3];

        for (; i < n; ++i)
            sum += (a[i] + b[i]) * c[i];
        return sum;
    }
}

void a_times_b(double* a, double* b, double* c, int n) {
    if (n < 4) {
        for (int i = 0; i < n; ++i)
            c[i] = a[i] * b[i];
    } else {
        __m256d _a, _b;

        int i = 0;
        for (; i < n - (n % 4); i += 4) {
            _a = _mm256_load_pd(&a[i]);
            _b = _mm256_load_pd(&b[i]);
            _a = _mm256_mul_pd(_a, _b);

            _mm256_store_pd(&c[i], _a);
        }

        for (; i < n; ++i)
            c[i] = a[i] * b[i];
    }
}

void a_add_b(double* a, double* b, double* c, int n) {
    if (n < 4) {
        for (int i = 0; i < n; ++i)
            c[i] = a[i] + b[i];
    } else {
        __m256d _a, _b;

        int i = 0;
        for (; i < n - (n % 4); i += 4) {
            _a = _mm256_load_pd(&a[i]);
            _b = _mm256_load_pd(&b[i]);
            _a = _mm256_add_pd(_a, _b);

            _mm256_store_pd(&c[i], _a);
        }

        for (; i < n; ++i)
            c[i] = a[i] + b[i];
    }
}

void a_sub_b(double* a, double* b, double* c, int n) {
    if (n < 4) {
        for (int i = 0; i < n; ++i)
            c[i] = a[i] - b[i];
    } else {
        __m256d _a, _b;

        int i = 0;
        for (; i < n - (n % 4); i += 4) {
            _a = _mm256_load_pd(&a[i]);
            _b = _mm256_load_pd(&b[i]);
            _a = _mm256_sub_pd(_a, _b);

            _mm256_store_pd(&c[i], _a);
        }

        for (; i < n; ++i)
            c[i] = a[i] - b[i];
    }
}

double sum_a_times_b_times_c(double* a, double* b, double* c, int n) {
    if (n < 4) {
        double sum = 0.0;
        for (int i = 0; i < n; ++i)
            sum += a[i] * b[i] * c[i];

        return sum;
    } else {
        __m256d _a, _b, _c;
        __m256d _res = _mm256_setzero_pd();
        int i = 0;
        for (; i < n - (n % 4); i += 4) {
            _a = _mm256_load_pd(&a[i]);
            _b = _mm256_load_pd(&b[i]);
            _b = _mm256_mul_pd(_a, _b);

            _c = _mm256_load_pd(&c[i]);

#ifdef FMA
            _res = _mm256_fmadd_pd(_b, _c, _res);
#else
            _c = _mm256_mul_pd(_b, _c);
            _res = _mm256_add_pd(_c, _res);
#endif
        }

        double* res = (double*)&_res;
        double sum = res[0] + res[1] + res[2] + res[3];

        for (; i < n; ++i)
            sum += a[i] * b[i] * c[i];
        return sum;
    }
}

void a_add_scalar_b(double* a, double b, double* c, int n) {
    if (n < 4) {
        for (int i = 0; i < n; ++i)
            c[i] = a[i] + b;
    } else {
        __m256d _a;
        __m256d _b = _mm256_set_pd(b, b, b, b);

        int i = 0;
        for (; i < n - (n % 4); i += 4) {
            _a = _mm256_load_pd(&a[i]);
            _a = _mm256_add_pd(_a, _b);
            _mm256_store_pd(&c[i], _a);
        }

        for (; i < n; ++i)
            c[i] = a[i] + b;
    }
}

void a_times_scalar_b(double* a, double b, double* c, int n) {
    if (n < 4) {
        for (int i = 0; i < n; ++i)
            c[i] = a[i] * b;
    } else {
        __m256d _a;
        __m256d _b = _mm256_set_pd(b, b, b, b);

        int i = 0;
        for (; i < n - (n % 4); i += 4) {
            _a = _mm256_load_pd(&a[i]);
            _a = _mm256_mul_pd(_a, _b);
            _mm256_store_pd(&c[i], _a);
        }

        for (; i < n; ++i)
            c[i] = a[i] * b;
    }
}

void add_a_add_scalar_b(double* a, double b, double* c, int n) {
    if (n < 4) {
        for (int i = 0; i < n; ++i)
            c[i] += a[i] * b;
    } else {
        __m256d _a, _c;
        __m256d _b = _mm256_set_pd(b, b, b, b);

        int i = 0;
        for (; i < n - (n % 4); i += 4) {
            _a = _mm256_load_pd(&a[i]);
            _c = _mm256_load_pd(&c[i]);

#ifdef FMA
            _c = _mm256_fmadd_pd(_a, _b, _c);
#else
            _a = _mm256_mul_pd(_a, _b);
            _c = _mm256_add_pd(_c, _a);
#endif

            _mm256_store_pd(&c[i], _c);
        }

        for (; i < n; ++i)
            c[i] += a[i] * b;
    }
}

void sub_a_times_scalar_b_sub_c_times_scalar_d(double* a,
                                               double b,
                                               double* c,
                                               double d,
                                               double* e,
                                               int n) {
    if (n < 4) {
        for (int i = 0; i < n; ++i)
            e[i] -= a[i] * b + c[i] * d;
    } else {
        __m256d _a, _c, _e;
        __m256d _b = _mm256_set_pd(b, b, b, b);
        __m256d _d = _mm256_set_pd(d, d, d, d);

        int i = 0;
        for (; i < n - (n % 4); i += 4) {
            _a = _mm256_load_pd(&a[i]);
            _c = _mm256_load_pd(&c[i]);
            _e = _mm256_load_pd(&e[i]);

#ifdef FMA
            _e = _mm256_fnmadd_pd(_a, _b, _e);
            _e = _mm256_fnmadd_pd(_c, _d, _e);
#else
            _a = _mm256_mul_pd(_a, _b);
            _c = _mm256_mul_pd(_c, _d);

            _a = _mm256_add_pd(_a, _c);
            _e = _mm256_sub_pd(_e, _a);
#endif

            _mm256_store_pd(&e[i], _e);
        }

        for (; i < n; ++i)
            e[i] -= a[i] * b + c[i] * d;
    }
}

void sub_a_times_scalar_b_sub_c_times_scalar_d_sub_d_times_scalar_f(double* a,
                                                                    double b,
                                                                    double* c,
                                                                    double d,
                                                                    double* e,
                                                                    double f,
                                                                    double* res,
                                                                    int n) {
    if (n < 4) {
        for (int i = 0; i < n; ++i)
            res[i] -= a[i] * b + c[i] * d + e[i] * f;
    } else {
        __m256d _a, _c, _e, _res;
        __m256d _b = _mm256_set_pd(b, b, b, b);
        __m256d _d = _mm256_set_pd(d, d, d, d);
        __m256d _f = _mm256_set_pd(f, f, f, f);

        int i = 0;
        for (; i < n - (n % 4); i += 4) {
            _a = _mm256_load_pd(&a[i]);
            _c = _mm256_load_pd(&c[i]);
            _e = _mm256_load_pd(&e[i]);

            _res = _mm256_load_pd(&res[i]);

#ifdef FMA
            _res = _mm256_fnmadd_pd(_a, _b, _res);
            _res = _mm256_fnmadd_pd(_c, _d, _res);
            _res = _mm256_fnmadd_pd(_e, _f, _res);

#else
            _a = _mm256_mul_pd(_a, _b);
            _c = _mm256_mul_pd(_c, _d);
            _e = _mm256_mul_pd(_e, _f);

            _c = _mm256_add_pd(_a, _c);
            _e = _mm256_add_pd(_e, _c);

            _res = _mm256_sub_pd(_res, _e);
#endif

            _mm256_store_pd(&res[i], _res);
        }

        for (; i < n; ++i)
            res[i] -= a[i] * b + c[i] * d + e[i] * f;
    }
}

void a_times_scalar_b_sub_c(double* a, double b, double* c, double* d, int n) {
    if (n < 4) {
        for (int i = 0; i < n; ++i)
            d[i] = a[i] * b - c[i];
    } else {
        __m256d _a, _c;
        __m256d _b = _mm256_set_pd(b, b, b, b);

        int i = 0;
        for (; i < n - (n % 4); i += 4) {
            _a = _mm256_load_pd(&a[i]);
            _c = _mm256_load_pd(&c[i]);

#ifdef FMA
            _c = _mm256_fmsub_pd(_a, _b, _c);
#else
            _a = _mm256_mul_pd(_a, _b);
            _c = _mm256_sub_pd(_a, _c);
#endif

            _mm256_store_pd(&d[i], _c);
        }

        for (; i < n; ++i)
            d[i] = a[i] * b - c[i];
    }
}

void a_add_scalar_b_times_c_sub_d_times_e(double* a,
                                          double b,
                                          double* c,
                                          double* d,
                                          double e,
                                          double* res,
                                          int n) {
    if (n < 4) {
        for (int i = 0; i < n; ++i)
            res[i] = a[i] + b * (c[i] - d[i] * e);
    } else {
        __m256d _a, _c, _d;
        __m256d _b = _mm256_set_pd(b, b, b, b);
        __m256d _e = _mm256_set_pd(e, e, e, e);

        int i = 0;
        for (; i < n - (n % 4); i += 4) {
            _a = _mm256_load_pd(&a[i]);
            _c = _mm256_load_pd(&c[i]);
            _d = _mm256_load_pd(&d[i]);

#ifdef FMA
            _c = _mm256_fnmadd_pd(_d, _e, _c);
            _a = _mm256_fmadd_pd(_c, _b, _a);
#else
            _d = _mm256_mul_pd(_d, _e);
            _c = _mm256_sub_pd(_c, _d);
            _c = _mm256_mul_pd(_c, _b);
            _a = _mm256_add_pd(_a, _c);
#endif

            _mm256_store_pd(&res[i], _a);
        }

        for (; i < n; ++i)
            res[i] = a[i] + b * (c[i] - d[i] * e);
    }
}

void a_times_scalar_b_add_c_times_scalar_d_add_d_times_scalar_f(double* a,
                                                                double b,
                                                                double* c,
                                                                double d,
                                                                double* e,
                                                                double f,
                                                                double* res,
                                                                int n) {
    if (n < 4) {
        for (int i = 0; i < n; ++i)
            res[i] = a[i] * b + c[i] * d + e[i] * f;
    } else {
        __m256d _a, _c, _e;

        __m256d _b = _mm256_set_pd(b, b, b, b);
        __m256d _d = _mm256_set_pd(d, d, d, d);
        __m256d _f = _mm256_set_pd(f, f, f, f);

        int i = 0;
        for (; i < n - (n % 4); i += 4) {
            _a = _mm256_load_pd(&a[i]);
            _c = _mm256_load_pd(&c[i]);
            _e = _mm256_load_pd(&e[i]);

#ifdef FMA
            _a = _mm256_mul_pd(_a, _b);
            _c = _mm256_fmadd_pd(_c, _d, _a);
            _e = _mm256_fmadd_pd(_e, _f, _c);
#else
            _a = _mm256_mul_pd(_a, _b);
            _c = _mm256_mul_pd(_c, _d);
            _e = _mm256_mul_pd(_e, _f);

            _c = _mm256_add_pd(_a, _c);
            _e = _mm256_add_pd(_e, _c);
#endif

            _mm256_store_pd(&res[i], _e);
        }

        for (; i < n; ++i)
            res[i] = a[i] * b + c[i] * d + e[i] * f;
    }
}

#elif AVX_VERSION_512

double sum_a(double* a, int n) {
    if (n < 8) {
        double sum = 0.0;
        for (int i = 0; i < n; ++i)
            sum += a[i];

        return sum;
    } else {
        __m512d _a;
        __m512d _res = _mm512_setzero_pd();
        int i = 0;
        for (; i < n - (n % 8); i += 8) {
            _a = _mm512_load_pd(&a[i]);
            _res = _mm512_add_pd(_a, _res);
        }

        double* res = (double*)&_res;
        double sum = res[0] + res[1] + res[2] + res[3] + res[4] + res[5] +
                     res[6] + res[7];

        for (; i < n; ++i)
            sum += a[i];
        return sum;
    }
}

double sum_square_a(double* a, int n) {
    if (n < 8) {
        double sum = 0.0;
        for (int i = 0; i < n; ++i)
            sum += a[i] * a[i];

        return sum;
    } else {
        __m512d _a;
        __m512d _res = _mm512_setzero_pd();

        int i = 0;
        for (; i < n - (n % 8); i += 8) {
            _a = _mm512_load_pd(&a[i]);

#ifdef FMA
            _res = _mm512_fmadd_pd(_a, _a, res);
#else
            _a = _mm512_mul_pd(_a, _a);
            _res = _mm512_add_pd(_a, _res);
#endif
        }

        double* res = (double*)&_res;
        double sum = res[0] + res[1] + res[2] + res[3] + res[4] + res[5] +
                     res[6] + res[7];

        for (; i < n; ++i)
            sum += a[i] * a[i];
        return sum;
    }
}

double sum_abs_a(double* restrict a, int n) {
    double sum = 0.0;
#pragma omp simd aligned(a : 64) reduction(+ : sum)
    for (int i = 0; i < n; ++i)
        sum += fabs(a[i]);

    return sum;
}

double sum_abs_a_times_b(double* restrict a, double* restrict b, int n) {
    double sum = 0.0;
#pragma omp simd aligned(a, b : 64) reduction(+ : sum)
    for (int i = 0; i < n; ++i)
        sum += b[i] * fabs(a[i]);

    return sum;
}

double sum_square_a_times_b(double* a, double* b, int n) {
    if (n < 8) {
        double sum = 0.0;
        for (int i = 0; i < n; ++i)
            sum += a[i] * a[i] * b[i];

        return sum;
    } else {
        __m512d _a, _b;
        __m512d _res = _mm512_setzero_pd();

        int i = 0;
        for (; i < n - (n % 8); i += 8) {
            _a = _mm512_load_pd(&a[i]);
            _a = _mm512_mul_pd(_a, _a);

            _b = _mm512_load_pd(&b[i]);

#ifdef FMA
            _res = _mm512_fmadd_pd(_a, _b, _res);
#else
            _a = _mm512_mul_pd(_a, _b);
            _res = _mm512_add_pd(_a, _res);
#endif
        }

        double* res = (double*)&_res;
        double sum = res[0] + res[1] + res[2] + res[3] + res[4] + res[5] +
                     res[6] + res[7];

        for (; i < n; ++i)
            sum += a[i] * a[i] * b[i];
        return sum;
    }
}

double sum_a_times_b(double* a, double* b, int n) {
    if (n < 8) {
        double sum = 0.0;
        for (int i = 0; i < n; ++i)
            sum += a[i] * b[i];

        return sum;
    } else {
        __m512d _a, _b;
        __m512d _res = _mm512_setzero_pd();

        int i = 0;
        for (; i < n - (n % 8); i += 8) {
            _a = _mm512_load_pd(&a[i]);
            _b = _mm512_load_pd(&b[i]);

#ifdef FMA
            _res = _mm512_fmadd_pd(_a, _b, _res);
#else
            _a = _mm512_mul_pd(_a, _b);
            _res = _mm512_add_pd(_a, _res);
#endif
        }

        double* res = (double*)&_res;
        double sum = res[0] + res[1] + res[2] + res[3] + res[4] + res[5] +
                     res[6] + res[7];

        for (; i < n; ++i)
            sum += a[i] * b[i];
        return sum;
    }
}

double sum_a_sub_b_times_c(double* a, double* b, double* c, int n) {
    if (n < 8) {
        double sum = 0.0;
        for (int i = 0; i < n; ++i)
            sum += (a[i] - b[i]) * c[i];

        return sum;
    } else {
        __m512d _a, _b, _c;
        __m512d _res = _mm512_setzero_pd();

        int i = 0;
        for (; i < n - (n % 8); i += 8) {
            _a = _mm512_load_pd(&a[i]);
            _b = _mm512_load_pd(&b[i]);
            _a = _mm512_sub_pd(_a, _b);

            _c = _mm512_load_pd(&c[i]);

#ifdef FMA
            _res = _mm512_fmadd_pd(_a, _c, _res);
#else
            _c = _mm512_mul_pd(_a, _c);
            _res = _mm512_add_pd(_c, _res);
#endif
        }

        double* res = (double*)&_res;
        double sum = res[0] + res[1] + res[2] + res[3] + res[4] + res[5] +
                     res[6] + res[7];

        for (; i < n; ++i)
            sum += (a[i] - b[i]) * c[i];
        return sum;
    }
}

double sum_a_add_b_times_c(double* a, double* b, double* c, int n) {
    if (n < 8) {
        double sum = 0.0;
        for (int i = 0; i < n; ++i)
            sum += (a[i] + b[i]) * c[i];

        return sum;
    } else {
        __m512d _a, _b, _c;
        __m512d _res = _mm512_setzero_pd();

        int i = 0;
        for (; i < n - (n % 8); i += 8) {
            _a = _mm512_load_pd(&a[i]);
            _b = _mm512_load_pd(&b[i]);
            _a = _mm512_add_pd(_a, _b);

            _c = _mm512_load_pd(&c[i]);

#ifdef FMA
            _res = _mm512_fmadd_pd(_a, _c, _res);
#else
            _c = _mm512_mul_pd(_a, _c);
            _res = _mm512_add_pd(_c, _res);
#endif
        }

        double* res = (double*)&_res;
        double sum = res[0] + res[1] + res[2] + res[3] + res[4] + res[5] +
                     res[6] + res[7];

        for (; i < n; ++i)
            sum += (a[i] + b[i]) * c[i];
        return sum;
    }
}

void a_times_b(double* a, double* b, double* c, int n) {
    if (n < 8) {
        for (int i = 0; i < n; ++i)
            c[i] = a[i] * b[i];
    } else {
        __m512d _a, _b;

        int i = 0;
        for (; i < n - (n % 8); i += 8) {
            _a = _mm512_load_pd(&a[i]);
            _b = _mm512_load_pd(&b[i]);
            _a = _mm512_mul_pd(_a, _b);

            _mm512_store_pd(&c[i], _a);
        }

        for (; i < n; ++i)
            c[i] = a[i] * b[i];
    }
}

void a_add_b(double* a, double* b, double* c, int n) {
    if (n < 8) {
        for (int i = 0; i < n; ++i)
            c[i] = a[i] + b[i];
    } else {
        __m512d _a, _b;

        int i = 0;
        for (; i < n - (n % 8); i += 8) {
            _a = _mm512_load_pd(&a[i]);
            _b = _mm512_load_pd(&b[i]);
            _a = _mm512_add_pd(_a, _b);

            _mm512_store_pd(&c[i], _a);
        }

        for (; i < n; ++i)
            c[i] = a[i] + b[i];
    }
}

void a_sub_b(double* a, double* b, double* c, int n) {
    if (n < 8) {
        for (int i = 0; i < n; ++i)
            c[i] = a[i] - b[i];
    } else {
        __m512d _a, _b;

        int i = 0;
        for (; i < n - (n % 8); i += 8) {
            _a = _mm512_load_pd(&a[i]);
            _b = _mm512_load_pd(&b[i]);
            _a = _mm512_sub_pd(_a, _b);

            _mm512_store_pd(&c[i], _a);
        }

        for (; i < n; ++i)
            c[i] = a[i] - b[i];
    }
}

double sum_a_times_b_times_c(double* a, double* b, double* c, int n) {
    if (n < 8) {
        double sum = 0.0;
        for (int i = 0; i < n; ++i)
            sum += a[i] * b[i] * c[i];

        return sum;
    } else {
        __m512d _a, _b, _c;
        __m512d _res = _mm512_setzero_pd();
        int i = 0;
        for (; i < n - (n % 8); i += 8) {
            _a = _mm512_load_pd(&a[i]);
            _b = _mm512_load_pd(&b[i]);
            _b = _mm512_mul_pd(_a, _b);

            _c = _mm512_load_pd(&c[i]);

#ifdef FMA
            _res = _mm512_fmadd_pd(_b, _c, _res);
#else
            _c = _mm512_mul_pd(_b, _c);
            _res = _mm512_add_pd(_c, _res);
#endif
        }

        double* res = (double*)&_res;
        double sum = res[0] + res[1] + res[2] + res[3] + res[4] + res[5] +
                     res[6] + res[7];

        for (; i < n; ++i)
            sum += a[i] * b[i] * c[i];
        return sum;
    }
}

void a_add_scalar_b(double* a, double b, double* c, int n) {
    if (n < 8) {
        for (int i = 0; i < n; ++i)
            c[i] = a[i] + b;
    } else {
        __m512d _a;
        __m512d _b = _mm512_set_pd(b, b, b, b, b, b, b, b);

        int i = 0;
        for (; i < n - (n % 8); i += 8) {
            _a = _mm512_load_pd(&a[i]);
            _a = _mm512_add_pd(_a, _b);
            _mm512_store_pd(&c[i], _a);
        }

        for (; i < n; ++i)
            c[i] = a[i] + b;
    }
}

void a_times_scalar_b(double* a, double b, double* c, int n) {
    if (n < 8) {
        for (int i = 0; i < n; ++i)
            c[i] = a[i] * b;
    } else {
        __m512d _a;
        __m512d _b = _mm512_set_pd(b, b, b, b, b, b, b, b);

        int i = 0;
        for (; i < n - (n % 8); i += 8) {
            _a = _mm512_load_pd(&a[i]);
            _a = _mm512_mul_pd(_a, _b);
            _mm512_store_pd(&c[i], _a);
        }

        for (; i < n; ++i)
            c[i] = a[i] * b;
    }
}

void add_a_add_scalar_b(double* a, double b, double* c, int n) {
    if (n < 8) {
        for (int i = 0; i < n; ++i)
            c[i] += a[i] * b;
    } else {
        __m512d _a, _c;
        __m512d _b = _mm512_set_pd(b, b, b, b, b, b, b, b);

        int i = 0;
        for (; i < n - (n % 8); i += 8) {
            _a = _mm512_load_pd(&a[i]);
            _c = _mm512_load_pd(&c[i]);

#ifdef FMA
            _c = _mm512_fmadd(_a, _b, _c);
#else
            _a = _mm512_mul_pd(_a, _b);
            _c = _mm512_add_pd(_c, _a);
#endif

            _mm512_store_pd(&c[i], _c);
        }

        for (; i < n; ++i)
            c[i] += a[i] * b;
    }
}

void sub_a_times_scalar_b_sub_c_times_scalar_d(double* a,
                                               double b,
                                               double* c,
                                               double d,
                                               double* e,
                                               int n) {
    if (n < 8) {
        for (int i = 0; i < n; ++i)
            e[i] -= a[i] * b + c[i] * d;
    } else {
        __m512d _a, _c, _e;
        __m512d _b = _mm512_set_pd(b, b, b, b, b, b, b, b);
        __m512d _d = _mm512_set_pd(d, d, d, d, d, d, d, d);

        int i = 0;
        for (; i < n - (n % 8); i += 8) {
            _a = _mm512_load_pd(&a[i]);
            _c = _mm512_load_pd(&c[i]);
            _e = _mm512_load_pd(&e[i]);

#ifdef FMA
            _e = _mm512_fnmadd(_a, _b, _e);
            _e = _mm512_fnmadd(_c, _d, _e);
#else
            _a = _mm512_mul_pd(_a, _b);
            _c = _mm512_mul_pd(_c, _d);

            _a = _mm512_add_pd(_a, _c);
            _e = _mm512_sub_pd(_e, _a);
#endif

            _mm512_store_pd(&e[i], _e);
        }

        for (; i < n; ++i)
            e[i] -= a[i] * b + c[i] * d;
    }
}

void sub_a_times_scalar_b_sub_c_times_scalar_d_sub_d_times_scalar_f(double* a,
                                                                    double b,
                                                                    double* c,
                                                                    double d,
                                                                    double* e,
                                                                    double f,
                                                                    double* res,
                                                                    int n) {
    if (n < 8) {
        for (int i = 0; i < n; ++i)
            res[i] -= a[i] * b + c[i] * d + e[i] * f;
    } else {
        __m512d _a, _c, _e, _res;
        __m512d _b = _mm512_set_pd(b, b, b, b, b, b, b, b);
        __m512d _d = _mm512_set_pd(d, d, d, d, d, d, d, d);
        __m512d _f = _mm512_set_pd(f, f, f, f, f, f, f, f);

        int i = 0;
        for (; i < n - (n % 8); i += 8) {
            _a = _mm512_load_pd(&a[i]);
            _c = _mm512_load_pd(&c[i]);
            _e = _mm512_load_pd(&e[i]);

#ifdef FMA
            _a = _mm512_mul_pd(_a, _b);
            _c = _mm512_fmadd_pd(_c, _d, _a);
            _e = _mm512_fmadd_pd(_e, _f, _c);
#else
            _a = _mm512_mul_pd(_a, _b);
            _c = _mm512_mul_pd(_c, _d);
            _e = _mm512_mul_pd(_e, _f);

            _c = _mm512_add_pd(_a, _c);
            _e = _mm512_add_pd(_e, _c);
#endif

            _res = _mm512_load_pd(&res[i]);
            _res = _mm512_sub_pd(_res, _e);

            _mm512_store_pd(&res[i], _res);
        }

        for (; i < n; ++i)
            res[i] -= a[i] * b + c[i] * d + e[i] * f;
    }
}

void a_times_scalar_b_sub_c(double* a, double b, double* c, double* d, int n) {
    if (n < 8) {
        for (int i = 0; i < n; ++i)
            d[i] = a[i] * b - c[i];
    } else {
        __m512d _a, _c;
        __m512d _b = _mm512_set_pd(b, b, b, b, b, b, b, b);

        int i = 0;
        for (; i < n - (n % 8); i += 8) {
            _a = _mm512_load_pd(&a[i]);
            _c = _mm512_load_pd(&c[i]);

#ifdef FMA
            _c = _mm512_fmsub_pd(_a, _b, _c);
#else
            _a = _mm512_mul_pd(_a, _b);
            _c = _mm512_sub_pd(_a, _c);
#endif

            _mm512_store_pd(&d[i], _c);
        }

        for (; i < n; ++i)
            d[i] = a[i] * b - c[i];
    }
}

void a_add_scalar_b_times_c_sub_d_times_e(double* a,
                                          double b,
                                          double* c,
                                          double* d,
                                          double e,
                                          double* res,
                                          int n) {
    if (n < 8) {
        for (int i = 0; i < n; ++i)
            res[i] = a[i] + b * (c[i] - d[i] * e);
    } else {
        __m512d _a, _c, _d;
        __m512d _b = _mm512_set_pd(b, b, b, b, b, b, b, b);
        __m512d _e = _mm512_set_pd(e, e, e, e, e, e, e, e);

        int i = 0;
        for (; i < n - (n % 8); i += 8) {
            _a = _mm512_load_pd(&a[i]);
            _c = _mm512_load_pd(&c[i]);
            _d = _mm512_load_pd(&d[i]);

#ifdef FMA
            _c = _mm512_fnmadd_pd(_d, _e, _c);
            _a = _mm512_fmadd_pd(_c, _b, _a);
#else
            _d = _mm512_mul_pd(_d, _e);
            _c = _mm512_sub_pd(_c, _d);
            _c = _mm512_mul_pd(_c, _b);
            _a = _mm512_add_pd(_a, _c);
#endif

            _mm512_store_pd(&res[i], _a);
        }

        for (; i < n; ++i)
            res[i] = a[i] + b * (c[i] - d[i] * e);
    }
}

void a_times_scalar_b_add_c_times_scalar_d_add_d_times_scalar_f(double* a,
                                                                double b,
                                                                double* c,
                                                                double d,
                                                                double* e,
                                                                double f,
                                                                double* res,
                                                                int n) {
    if (n < 8) {
        for (int i = 0; i < n; ++i)
            res[i] = a[i] * b + c[i] * d + e[i] * f;
    } else {
        __m512d _a, _c, _e;

        __m512d _b = _mm512_set_pd(b, b, b, b, b, b, b, b);
        __m512d _d = _mm512_set_pd(d, d, d, d, d, d, d, d);
        __m512d _f = _mm512_set_pd(f, f, f, f, f, f, f, f);

        int i = 0;
        for (; i < n - (n % 8); i += 8) {
            _a = _mm512_load_pd(&a[i]);
            _c = _mm512_load_pd(&c[i]);
            _e = _mm512_load_pd(&e[i]);

#ifdef FMA
            _a = _mm512_mul_pd(_a, _b);
            _c = _mm512_fmadd_pd(_c, _d, _a);
            _e = _mm512_fmadd_pd(_e, _f, _c);
#else
            _a = _mm512_mul_pd(_a, _b);
            _c = _mm512_mul_pd(_c, _d);
            _e = _mm512_mul_pd(_e, _f);

            _c = _mm512_add_pd(_a, _c);
            _e = _mm512_add_pd(_e, _c);
#endif

            _mm512_store_pd(&res[i], _e);
        }

        for (; i < n; ++i)
            res[i] = a[i] * b + c[i] * d + e[i] * f;
    }
}

#else

double sum_a(double* a, int n) {
    double x = 0.0;
    for (int i = 0; i < n; i++) {
        x += a[i];
    }
    return x;
}

double sum_square_a(double* a, int n) {
    double x = 0.0;
    for (int i = 0; i < n; i++) {
        x += a[i] * a[i];
    }
    return x;
}

double sum_abs_a(double* a, int n) {
    double x = 0.0;
    for (int i = 0; i < n; i++) {
        x += fabs(a[i]);
    }
    return x;
}

double sum_abs_a_times_b(double* a, double* b, int n) {
    double x = 0.0;
    for (int i = 0; i < n; i++) {
        x += b[i] * fabs(a[i]);
    }
    return x;
}

double sum_square_a_times_b(double* a, double* b, int n) {
    double x = 0.0;
    for (int i = 0; i < n; i++) {
        x += a[i] * a[i] * b[i];
    }
    return x;
}

double sum_a_times_b(double* a, double* b, int n) {
    double x = 0.0;
    for (int i = 0; i < n; i++) {
        x += a[i] * b[i];
    }
    return x;
}

double sum_a_sub_b_times_c(double* a, double* b, double* c, int n) {
    double x = 0.0;
    for (int i = 0; i < n; i++) {
        x += (a[i] - b[i]) * c[i];
    }
    return x;
}

double sum_a_add_b_times_c(double* a, double* b, double* c, int n) {
    double x = 0.0;
    for (int i = 0; i < n; i++) {
        x += (a[i] + b[i]) * c[i];
    }
    return x;
}

void a_times_b(double* a, double* b, double* c, int n) {
    for (int i = 0; i < n; ++i)
        c[i] = a[i] * b[i];
}

void a_add_b(double* a, double* b, double* c, int n) {
    for (int i = 0; i < n; ++i)
        c[i] = a[i] + b[i];
}

void a_sub_b(double* a, double* b, double* c, int n) {
    for (int i = 0; i < n; ++i)
        c[i] = a[i] - b[i];
}

double sum_a_times_b_times_c(double* a, double* b, double* c, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i)
        sum += a[i] * b[i] * c[i];

    return sum;
}

void a_add_scalar_b(double* a, double b, double* c, int n) {
    for (int i = 0; i < n; ++i)
        c[i] = a[i] + b;
}

void a_times_scalar_b(double* a, double b, double* c, int n) {
    for (int i = 0; i < n; ++i)
        c[i] = a[i] * b;
}

void add_a_add_scalar_b(double* a, double b, double* c, int n) {
    for (int i = 0; i < n; ++i)
        c[i] += a[i] * b;
}

void a_times_scalar_b_sub_c(double* a, double b, double* c, double* d, int n) {
    for (int i = 0; i < n; ++i)
        d[i] = a[i] * b - c[i];
}

void a_add_scalar_b_times_c_sub_d_times_e(double* a,
                                          double b,
                                          double* c,
                                          double* d,
                                          double e,
                                          double* res,
                                          int n) {
    for (int i = 0; i < n; ++i)
        res[i] = a[i] + b * (c[i] - d[i] * e);
}

void sub_a_times_scalar_b_sub_c_times_scalar_d(double* a,
                                               double b,
                                               double* c,
                                               double d,
                                               double* e,
                                               int n) {
    for (int i = 0; i < n; ++i)
        e[i] -= a[i] * b + c[i] * d;
}

void sub_a_times_scalar_b_sub_c_times_scalar_d_sub_d_times_scalar_f(double* a,
                                                                    double b,
                                                                    double* c,
                                                                    double d,
                                                                    double* e,
                                                                    double f,
                                                                    double* res,
                                                                    int n) {
    for (int i = 0; i < n; ++i)
        res[i] -= a[i] * b + c[i] * d + e[i] * f;
}

void a_times_scalar_b_add_c_times_scalar_d_add_d_times_scalar_f(double* a,
                                                                double b,
                                                                double* c,
                                                                double d,
                                                                double* e,
                                                                double f,
                                                                double* res,
                                                                int n) {
    for (int i = 0; i < n; ++i)
        res[i] = a[i] * b + c[i] * d + e[i] * f;
}

#endif

int getMinIndex(double* a, int N) {
    int min = 0;
    for (int i = 1; i < N; ++i)
        if (a[min] > a[i])
            min = i;

    return min;
}

void printMatrix(double* x, int N, int M) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            std::printf("%+.3e ", x[INDEX(i, j, N)]);
        }
        std::printf("\n");
    }
}

double sum_a_sub_b_mul_d_times_c(double* a,
                                 double* b,
                                 double* c,
                                 double d,
                                 int n) {
    double sum = 0.0;
#pragma omp simd
    for (int i = 0; i < n; ++i)
        sum += (a[i] * d - b[i]) * c[i];

    return sum;
}
