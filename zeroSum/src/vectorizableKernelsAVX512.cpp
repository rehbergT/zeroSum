#if !defined(__APPLE__) || !defined(__arm64__)

#include "vectorizableKernels.h"

void interceptMoveKernelAVX512(double* y,
                               double* xb,
                               double* w,
                               uint32_t N,
                               double* result) {
    __m512d _y, _xb, _w;
    __m512d _res = _mm512_setzero_pd();

    for (uint32_t i = 0; i < N; i += 8) {
        _y = _mm512_loadu_pd(&y[i]);
        _xb = _mm512_loadu_pd(&xb[i]);

        _y = _mm512_sub_pd(_y, _xb);

        _w = _mm512_loadu_pd(&w[i]);
        _res = _mm512_fmadd_pd(_y, _w, _res);
    }

    double* res = (double*)&_res;
    *result =
        res[0] + res[1] + res[2] + res[3] + res[4] + res[5] + res[6] + res[7];
}

void cvMoveKernelAVX512(double* y,
                        double* xb,
                        double* w,
                        double* xk,
                        double* betak,
                        uint32_t N,
                        double* ak,
                        double* bk) {
    __m512d _ak = _mm512_setzero_pd();
    __m512d _bk = _mm512_setzero_pd();
    __m512d _betak = _mm512_set_pd(*betak, *betak, *betak, *betak, *betak,
                                   *betak, *betak, *betak);
    __m512d _w, _xk, _xb, _y;

    for (uint32_t i = 0; i < N; i += 8) {
        _w = _mm512_loadu_pd(&w[i]);
        _xk = _mm512_loadu_pd(&xk[i]);

        _w = _mm512_mul_pd(_w, _xk);
        _ak = _mm512_fmadd_pd(_w, _xk, _ak);

        _xb = _mm512_loadu_pd(&xb[i]);
        _xb = _mm512_fnmadd_pd(_betak, _xk, _xb);

        _y = _mm512_loadu_pd(&y[i]);
        _xb = _mm512_sub_pd(_y, _xb);

        _bk = _mm512_fmadd_pd(_w, _xb, _bk);
    }
    double* __ak = (double*)&_ak;
    *ak = __ak[0] + __ak[1] + __ak[2] + __ak[3] + __ak[4] + __ak[5] + __ak[6] +
          __ak[7];
    double* __bk = (double*)&_bk;
    *bk = __bk[0] + __bk[1] + __bk[2] + __bk[3] + __bk[4] + __bk[5] + __bk[6] +
          __bk[7];
}

void cvMoveZSKernelAVX512(double* y,
                          double* xb,
                          double* w,
                          double* xk,
                          double* xs,
                          double* betak,
                          double* ukus,
                          uint32_t N,
                          double* ak,
                          double* bk) {
    __m512d _ak = _mm512_setzero_pd();
    __m512d _bk = _mm512_setzero_pd();
    __m512d _betak = _mm512_set_pd(*betak, *betak, *betak, *betak, *betak,
                                   *betak, *betak, *betak);
    __m512d _ukus =
        _mm512_set_pd(*ukus, *ukus, *ukus, *ukus, *ukus, *ukus, *ukus, *ukus);
    __m512d _y, _w, _xk, _xs, _xb;

    for (uint32_t i = 0; i < N; i += 8) {
        _xk = _mm512_loadu_pd(&xk[i]);
        _xs = _mm512_loadu_pd(&xs[i]);
        _xk = _mm512_fmsub_pd(_xs, _ukus, _xk);

        _w = _mm512_loadu_pd(&w[i]);
        _w = _mm512_mul_pd(_w, _xk);
        _ak = _mm512_fmadd_pd(_w, _xk, _ak);

        _xb = _mm512_loadu_pd(&xb[i]);
        _xb = _mm512_fmadd_pd(_betak, _xk, _xb);

        _y = _mm512_loadu_pd(&y[i]);
        _xb = _mm512_sub_pd(_y, _xb);

        _bk = _mm512_fmadd_pd(_w, _xb, _bk);
    }
    double* __ak = (double*)&_ak;
    *ak += __ak[0] + __ak[1] + __ak[2] + __ak[3] + __ak[4] + __ak[5] + __ak[6] +
           __ak[7];
    double* __bk = (double*)&_bk;
    *bk -= __bk[0] + __bk[1] + __bk[2] + __bk[3] + __bk[4] + __bk[5] + __bk[6] +
           __bk[7];
}

void cvMoveZSKernel2AVX512(double* xb,
                           double* diffk,
                           double* xk,
                           double* diffs,
                           double* xs,
                           uint32_t N) {
    __m512d _xb, _xk, _xs;
    __m512d _diffk = _mm512_set_pd(*diffk, *diffk, *diffk, *diffk, *diffk,
                                   *diffk, *diffk, *diffk);
    __m512d _diffs = _mm512_set_pd(*diffs, *diffs, *diffs, *diffs, *diffs,
                                   *diffs, *diffs, *diffs);

    for (uint32_t i = 0; i < N; i += 8) {
        _xb = _mm512_loadu_pd(&xb[i]);
        _xk = _mm512_loadu_pd(&xk[i]);
        _xb = _mm512_fnmadd_pd(_xk, _diffk, _xb);
        _xs = _mm512_loadu_pd(&xs[i]);
        _xb = _mm512_fnmadd_pd(_xs, _diffs, _xb);
        _mm512_storeu_pd(&xb[i], _xb);
    }
}

void cvMoveZSParallelKernelAVX512(double* xb,
                                  double* w,
                                  double* xk,
                                  double* xs,
                                  double* betak,
                                  double* ukus,
                                  uint32_t N,
                                  double* ak,
                                  double* bk) {
    __m512d _ak = _mm512_setzero_pd();
    __m512d _bk = _mm512_setzero_pd();
    __m512d _betak = _mm512_set_pd(*betak, *betak, *betak, *betak, *betak,
                                   *betak, *betak, *betak);
    __m512d _ukus =
        _mm512_set_pd(*ukus, *ukus, *ukus, *ukus, *ukus, *ukus, *ukus, *ukus);
    __m512d _w, _xk, _xs, _xb;

    for (uint32_t i = 0; i < N; i += 8) {
        _xk = _mm512_loadu_pd(&xk[i]);
        _xs = _mm512_loadu_pd(&xs[i]);
        _xk = _mm512_fmsub_pd(_xs, _ukus, _xk);

        _w = _mm512_loadu_pd(&w[i]);
        _w = _mm512_mul_pd(_w, _xk);
        _ak = _mm512_fmadd_pd(_w, _xk, _ak);

        _xb = _mm512_loadu_pd(&xb[i]);
        _xb = _mm512_fmadd_pd(_betak, _xk, _xb);

        _bk = _mm512_fmadd_pd(_w, _xb, _bk);
    }
    double* __ak = (double*)&_ak;
    *ak += __ak[0] + __ak[1] + __ak[2] + __ak[3] + __ak[4] + __ak[5] + __ak[6] +
           __ak[7];
    double* __bk = (double*)&_bk;
    *bk -= __bk[0] + __bk[1] + __bk[2] + __bk[3] + __bk[4] + __bk[5] + __bk[6] +
           __bk[7];
}

void cdMoveZSRotatedKernelAVX512(double* xm,
                                 double* xn,
                                 double* xs,
                                 double* sinT,
                                 double* cosT,
                                 double* unum2,
                                 uint32_t N,
                                 double* res) {
    __m512d _xm, _xn, _xs;
    __m512d _sinT =
        _mm512_set_pd(*sinT, *sinT, *sinT, *sinT, *sinT, *sinT, *sinT, *sinT);
    __m512d _MinusCosT = _mm512_set_pd(-*cosT, -*cosT, -*cosT, -*cosT, -*cosT,
                                       -*cosT, -*cosT, -*cosT);
    __m512d _unum2 = _mm512_set_pd(*unum2, *unum2, *unum2, *unum2, *unum2,
                                   *unum2, *unum2, *unum2);

    for (uint32_t i = 0; i < N; i += 8) {
        _xm = _mm512_loadu_pd(&xm[i]);
        _xm = _mm512_mul_pd(_xm, _sinT);

        _xn = _mm512_loadu_pd(&xn[i]);
        _xn = _mm512_fmadd_pd(_xn, _MinusCosT, _xm);

        _xs = _mm512_loadu_pd(&xs[i]);
        _xs = _mm512_fmadd_pd(_xs, _unum2, _xn);

        _mm512_storeu_pd(&res[i], _xs);
    }
}

void cdMoveZSRotatedKernel2AVX512(double* y,
                                  double* xb,
                                  double* w,
                                  double* tmp,
                                  uint32_t N,
                                  double* bk) {
    __m512d _y, _ww, _xb, _tmp;
    __m512d _res = _mm512_setzero_pd();
    for (uint32_t i = 0; i < N; i += 8) {
        _xb = _mm512_loadu_pd(&xb[i]);
        _y = _mm512_loadu_pd(&y[i]);
        _xb = _mm512_sub_pd(_y, _xb);

        _ww = _mm512_loadu_pd(&w[i]);
        _xb = _mm512_mul_pd(_ww, _xb);

        _tmp = _mm512_loadu_pd(&tmp[i]);
        _res = _mm512_fmadd_pd(_xb, _tmp, _res);
    }
    double* res = (double*)&_res;
    *bk = res[0] + res[1] + res[2] + res[3] + res[4] + res[5] + res[6] + res[7];
}

void cvMoveZSRotatedKernel3AVX512(double* xb,
                                  double* diffn,
                                  double* xn,
                                  double* diffm,
                                  double* xm,
                                  double* diffs,
                                  double* xs,
                                  uint32_t N) {
    __m512d _x, _xb;
    __m512d _diffn = _mm512_set_pd(*diffn, *diffn, *diffn, *diffn, *diffn,
                                   *diffn, *diffn, *diffn);
    __m512d _diffm = _mm512_set_pd(*diffm, *diffm, *diffm, *diffm, *diffm,
                                   *diffm, *diffm, *diffm);
    __m512d _diffs = _mm512_set_pd(*diffs, *diffs, *diffs, *diffs, *diffs,
                                   *diffs, *diffs, *diffs);

    for (uint32_t i = 0; i < N; i += 8) {
        _xb = _mm512_loadu_pd(&xb[i]);

        _x = _mm512_loadu_pd(&xn[i]);
        _xb = _mm512_fnmadd_pd(_x, _diffn, _xb);

        _x = _mm512_loadu_pd(&xm[i]);
        _xb = _mm512_fnmadd_pd(_x, _diffm, _xb);

        _x = _mm512_loadu_pd(&xs[i]);
        _xb = _mm512_fnmadd_pd(_x, _diffs, _xb);

        _mm512_storeu_pd(&xb[i], _xb);
    }
}

void arraySumKernelAVX512(double* a, uint32_t N, double* res) {
    __m512d _a;
    __m512d _res = _mm512_setzero_pd();
    uint32_t i = 0;
    for (; i < N; i += 8) {
        _a = _mm512_loadu_pd(&a[i]);
        _res = _mm512_add_pd(_a, _res);
    }

    double* ptr = (double*)&_res;
    *res =
        ptr[0] + ptr[1] + ptr[2] + ptr[3] + ptr[4] + ptr[5] + ptr[6] + ptr[7];
}

void weightedSquareSumKernelAVX512(double* a,
                                   double* b,
                                   uint32_t N,
                                   double* res) {
    __m512d _a, _b;
    __m512d _res = _mm512_setzero_pd();

    for (uint32_t i = 0; i < N; i += 8) {
        _a = _mm512_loadu_pd(&a[i]);
        _a = _mm512_mul_pd(_a, _a);
        _b = _mm512_loadu_pd(&b[i]);
        _res = _mm512_fmadd_pd(_a, _b, _res);
    }

    double* ptr = (double*)&_res;
    *res =
        ptr[0] + ptr[1] + ptr[2] + ptr[3] + ptr[4] + ptr[5] + ptr[6] + ptr[7];
}

void weightedResidualSquareSumKernelAVX512(double* a,
                                           double* b,
                                           double* c,
                                           uint32_t N,
                                           double* res) {
    __m512d _a, _b, _c;
    __m512d _res = _mm512_setzero_pd();

    for (uint32_t i = 0; i < N; i += 8) {
        _b = _mm512_loadu_pd(&b[i]);
        _c = _mm512_loadu_pd(&c[i]);
        _b = _mm512_sub_pd(_b, _c);
        _b = _mm512_mul_pd(_b, _b);

        _a = _mm512_loadu_pd(&a[i]);
        _res = _mm512_fmadd_pd(_a, _b, _res);
    }

    double* ptr = (double*)&_res;
    *res =
        ptr[0] + ptr[1] + ptr[2] + ptr[3] + ptr[4] + ptr[5] + ptr[6] + ptr[7];
}

void squareWeightedSumKernelAVX512(double* a,
                                   double* b,
                                   uint32_t N,
                                   double* res) {
    __m512d _a, _b;
    __m512d _res = _mm512_setzero_pd();

    for (uint32_t i = 0; i < N; i += 8) {
        _a = _mm512_loadu_pd(&a[i]);
        _b = _mm512_loadu_pd(&b[i]);
        _a = _mm512_mul_pd(_a, _b);
        _res = _mm512_fmadd_pd(_a, _a, _res);
    }

    double* ptr = (double*)&_res;
    *res =
        ptr[0] + ptr[1] + ptr[2] + ptr[3] + ptr[4] + ptr[5] + ptr[6] + ptr[7];
}

void a_sub_bKernelAVX512(double* a, double* b, double* c, uint32_t N) {
    __m512d _a, _b;
    for (uint32_t i = 0; i < N; i += 8) {
        _a = _mm512_loadu_pd(&a[i]);
        _b = _mm512_loadu_pd(&b[i]);
        _a = _mm512_sub_pd(_a, _b);

        _mm512_storeu_pd(&c[i], _a);
    }
}

void a_add_scalar_bKernelAVX512(double* a, double* b, uint32_t N) {
    __m512d _a;
    __m512d _b = _mm512_set_pd(*b, *b, *b, *b, *b, *b, *b, *b);

    for (uint32_t i = 0; i < N; i += 8) {
        _a = _mm512_loadu_pd(&a[i]);
        _a = _mm512_add_pd(_a, _b);
        _mm512_storeu_pd(&a[i], _a);
    }
}

#endif
