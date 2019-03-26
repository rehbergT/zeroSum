#include "vectorizableKernels.h"

void interceptMoveKernelAVX2(double* y,
                             double* xb,
                             double* w,
                             uint32_t N,
                             double* result) {
    __m256d _y, _xb, _w;
    __m256d _res = _mm256_setzero_pd();

    for (uint32_t i = 0; i < N; i += 4) {
        _y = _mm256_load_pd(&y[i]);
        _xb = _mm256_load_pd(&xb[i]);

        _y = _mm256_sub_pd(_y, _xb);

        _w = _mm256_load_pd(&w[i]);
        _res = _mm256_fmadd_pd(_y, _w, _res);
    }

    double* res = (double*)&_res;
    *result = res[0] + res[1] + res[2] + res[3];
}

void cvMoveKernelAVX2(double* y,
                      double* xb,
                      double* w,
                      double* xk,
                      double* betak,
                      uint32_t N,
                      double* ak,
                      double* bk) {
    __m256d _ak = _mm256_setzero_pd();
    __m256d _bk = _mm256_setzero_pd();
    __m256d _betak = _mm256_set_pd(*betak, *betak, *betak, *betak);
    __m256d _w, _xk, _xb, _y;

    for (uint32_t i = 0; i < N; i += 4) {
        _w = _mm256_load_pd(&w[i]);
        _xk = _mm256_load_pd(&xk[i]);

        _w = _mm256_mul_pd(_w, _xk);
        _ak = _mm256_fmadd_pd(_w, _xk, _ak);

        _xb = _mm256_load_pd(&xb[i]);
        _xb = _mm256_fnmadd_pd(_betak, _xk, _xb);

        _y = _mm256_load_pd(&y[i]);
        _xb = _mm256_sub_pd(_y, _xb);

        _bk = _mm256_fmadd_pd(_w, _xb, _bk);
    }

    double* __ak = (double*)&_ak;
    *ak = __ak[0] + __ak[1] + __ak[2] + __ak[3];
    double* __bk = (double*)&_bk;
    *bk = __bk[0] + __bk[1] + __bk[2] + __bk[3];
}

void cvMoveZSKernelAVX2(double* y,
                        double* xb,
                        double* w,
                        double* xk,
                        double* xs,
                        double* betak,
                        double* ukus,
                        uint32_t N,
                        double* ak,
                        double* bk) {
    __m256d _ak = _mm256_setzero_pd();
    __m256d _bk = _mm256_setzero_pd();
    __m256d _betak = _mm256_set_pd(*betak, *betak, *betak, *betak);
    __m256d _ukus = _mm256_set_pd(*ukus, *ukus, *ukus, *ukus);
    __m256d _y, _w, _xk, _xs, _xb;

    for (uint32_t i = 0; i < N; i += 4) {
        _xk = _mm256_load_pd(&xk[i]);
        _xs = _mm256_load_pd(&xs[i]);
        _xk = _mm256_fmsub_pd(_xs, _ukus, _xk);

        _w = _mm256_load_pd(&w[i]);
        _w = _mm256_mul_pd(_w, _xk);
        _ak = _mm256_fmadd_pd(_w, _xk, _ak);

        _xb = _mm256_load_pd(&xb[i]);
        _xb = _mm256_fmadd_pd(_betak, _xk, _xb);

        _y = _mm256_load_pd(&y[i]);
        _xb = _mm256_sub_pd(_y, _xb);

        _bk = _mm256_fmadd_pd(_w, _xb, _bk);
    }

    double* __ak = (double*)&_ak;
    *ak += __ak[0] + __ak[1] + __ak[2] + __ak[3];
    double* __bk = (double*)&_bk;
    *bk -= __bk[0] + __bk[1] + __bk[2] + __bk[3];
}

void cvMoveZSKernel2AVX2(double* xb,
                         double* diffk,
                         double* xk,
                         double* diffs,
                         double* xs,
                         uint32_t N) {
    __m256d _xb, _xk, _xs;
    __m256d _diffk = _mm256_set_pd(*diffk, *diffk, *diffk, *diffk);
    __m256d _diffs = _mm256_set_pd(*diffs, *diffs, *diffs, *diffs);

    for (uint32_t i = 0; i < N; i += 4) {
        _xb = _mm256_load_pd(&xb[i]);
        _xk = _mm256_load_pd(&xk[i]);
        _xb = _mm256_fnmadd_pd(_xk, _diffk, _xb);
        _xs = _mm256_load_pd(&xs[i]);
        _xb = _mm256_fnmadd_pd(_xs, _diffs, _xb);
        _mm256_store_pd(&xb[i], _xb);
    }
}

void cvMoveZSParallelKernelAVX2(double* xb,
                                double* w,
                                double* xk,
                                double* xs,
                                double* betak,
                                double* ukus,
                                uint32_t N,
                                double* ak,
                                double* bk) {
    __m256d _ak = _mm256_setzero_pd();
    __m256d _bk = _mm256_setzero_pd();
    __m256d _betak = _mm256_set_pd(*betak, *betak, *betak, *betak);
    __m256d _ukus = _mm256_set_pd(*ukus, *ukus, *ukus, *ukus);
    __m256d _w, _xk, _xs, _xb;

    for (uint32_t i = 0; i < N; i += 4) {
        _xk = _mm256_load_pd(&xk[i]);
        _xs = _mm256_load_pd(&xs[i]);
        _xk = _mm256_fmsub_pd(_xs, _ukus, _xk);

        _w = _mm256_load_pd(&w[i]);
        _w = _mm256_mul_pd(_w, _xk);
        _ak = _mm256_fmadd_pd(_w, _xk, _ak);

        _xb = _mm256_load_pd(&xb[i]);
        _xb = _mm256_fmadd_pd(_betak, _xk, _xb);

        _bk = _mm256_fmadd_pd(_w, _xb, _bk);
    }

    double* __ak = (double*)&_ak;
    *ak += __ak[0] + __ak[1] + __ak[2] + __ak[3];
    double* __bk = (double*)&_bk;
    *bk -= __bk[0] + __bk[1] + __bk[2] + __bk[3];
}

void cdMoveZSRotatedKernelAVX2(double* xm,
                               double* xn,
                               double* xs,
                               double* sinT,
                               double* cosT,
                               double* unum2,
                               uint32_t N,
                               double* res) {
    __m256d _xm, _xn, _xs;
    __m256d _sinT = _mm256_set_pd(*sinT, *sinT, *sinT, *sinT);
    __m256d _MinusCosT = _mm256_set_pd(-*cosT, -*cosT, -*cosT, -*cosT);
    __m256d _unum2 = _mm256_set_pd(*unum2, *unum2, *unum2, *unum2);

    for (uint32_t i = 0; i < N; i += 4) {
        _xm = _mm256_load_pd(&xm[i]);
        _xm = _mm256_mul_pd(_xm, _sinT);

        _xn = _mm256_load_pd(&xn[i]);
        _xn = _mm256_fmadd_pd(_xn, _MinusCosT, _xm);

        _xs = _mm256_load_pd(&xs[i]);
        _xs = _mm256_fmadd_pd(_xs, _unum2, _xn);

        _mm256_store_pd(&res[i], _xs);
    }
}

void cdMoveZSRotatedKernel2AVX2(double* y,
                                double* xb,
                                double* w,
                                double* tmp,
                                uint32_t N,
                                double* bk) {
    __m256d _y, _ww, _xb, _tmp;
    __m256d _res = _mm256_setzero_pd();
    for (uint32_t i = 0; i < N; i += 4) {
        _xb = _mm256_load_pd(&xb[i]);
        _y = _mm256_load_pd(&y[i]);
        _xb = _mm256_sub_pd(_y, _xb);

        _ww = _mm256_load_pd(&w[i]);
        _xb = _mm256_mul_pd(_ww, _xb);

        _tmp = _mm256_load_pd(&tmp[i]);
        _res = _mm256_fmadd_pd(_xb, _tmp, _res);
    }
    double* res = (double*)&_res;
    *bk = res[0] + res[1] + res[2] + res[3];
}

void cvMoveZSRotatedKernel3AVX2(double* xb,
                                double* diffn,
                                double* xn,
                                double* diffm,
                                double* xm,
                                double* diffs,
                                double* xs,
                                uint32_t N) {
    __m256d _x, _xb;
    __m256d _diffn = _mm256_set_pd(*diffn, *diffn, *diffn, *diffn);
    __m256d _diffm = _mm256_set_pd(*diffm, *diffm, *diffm, *diffm);
    __m256d _diffs = _mm256_set_pd(*diffs, *diffs, *diffs, *diffs);

    for (uint32_t i = 0; i < N; i += 4) {
        _xb = _mm256_load_pd(&xb[i]);

        _x = _mm256_load_pd(&xn[i]);
        _xb = _mm256_fnmadd_pd(_x, _diffn, _xb);

        _x = _mm256_load_pd(&xm[i]);
        _xb = _mm256_fnmadd_pd(_x, _diffm, _xb);

        _x = _mm256_load_pd(&xs[i]);
        _xb = _mm256_fnmadd_pd(_x, _diffs, _xb);

        _mm256_store_pd(&xb[i], _xb);
    }
}

void arraySumKernelAVX2(double* a, uint32_t N, double* res) {
    __m256d _a;
    __m256d _res = _mm256_setzero_pd();
    uint32_t i = 0;
    for (; i < N; i += 4) {
        _a = _mm256_load_pd(&a[i]);
        _res = _mm256_add_pd(_a, _res);
    }

    double* ptr = (double*)&_res;
    *res = ptr[0] + ptr[1] + ptr[2] + ptr[3];
}

void weightedSquareSumKernelAVX2(double* a, double* b, uint32_t N, double* res) {
    __m256d _a, _b;
    __m256d _res = _mm256_setzero_pd();

    for (uint32_t i = 0; i < N; i += 4) {
        _a = _mm256_load_pd(&a[i]);
        _a = _mm256_mul_pd(_a, _a);
        _b = _mm256_load_pd(&b[i]);
        _res = _mm256_fmadd_pd(_a, _b, _res);
    }

    double* ptr = (double*)&_res;
    *res = ptr[0] + ptr[1] + ptr[2] + ptr[3];
}

void weightedResidualSquareSumKernelAVX2(double* a,
                                         double* b,
                                         double* c,
                                         uint32_t N,
                                         double* res) {
    __m256d _a, _b, _c;
    __m256d _res = _mm256_setzero_pd();

    for (uint32_t i = 0; i < N; i += 4) {
        _b = _mm256_load_pd(&b[i]);
        _c = _mm256_load_pd(&c[i]);
        _b = _mm256_sub_pd(_b, _c);
        _b = _mm256_mul_pd(_b, _b);

        _a = _mm256_load_pd(&a[i]);
        _res = _mm256_fmadd_pd(_a, _b, _res);
    }

    double* ptr = (double*)&_res;
    *res = ptr[0] + ptr[1] + ptr[2] + ptr[3];
}

void squareWeightedSumKernelAVX2(double* a, double* b, uint32_t N, double* res) {
    __m256d _a, _b;
    __m256d _res = _mm256_setzero_pd();

    for (uint32_t i = 0; i < N; i += 4) {
        _a = _mm256_load_pd(&a[i]);
        _b = _mm256_load_pd(&b[i]);
        _a = _mm256_mul_pd(_a, _b);
        _res = _mm256_fmadd_pd(_a, _a, _res);
    }

    double* ptr = (double*)&_res;
    *res = ptr[0] + ptr[1] + ptr[2] + ptr[3];
}

void a_sub_bKernelAVX2(double* a, double* b, double* c, uint32_t N) {
    __m256d _a, _b;
    for (uint32_t i = 0; i < N; i += 4) {
        _a = _mm256_load_pd(&a[i]);
        _b = _mm256_load_pd(&b[i]);
        _a = _mm256_sub_pd(_a, _b);

        _mm256_store_pd(&c[i], _a);
    }
}

void a_add_scalar_bKernelAVX2(double* a, double* b, uint32_t N) {
    __m256d _a;
    __m256d _b = _mm256_set_pd(*b, *b, *b, *b);

    for (uint32_t i = 0; i < N; i += 4) {
        _a = _mm256_load_pd(&a[i]);
        _a = _mm256_add_pd(_a, _b);
        _mm256_store_pd(&a[i], _a);
    }
}