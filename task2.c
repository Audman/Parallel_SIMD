#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#include <emmintrin.h>

#define SIZE_MB  256
#define THREADS  4
#define MB       (1024ULL * 1024ULL)

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void fill_buffer(char *buf, size_t n)
{
    static const char charset[] =
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789 !.,;:?-_()[]{}";
    int len = sizeof(charset) - 1;
    srand(42);
    for (size_t i = 0; i < n; i++)
        buf[i] = charset[rand() % len];
}

typedef struct {
    char  *buf;
    size_t start;
    size_t end;
} thread_arg;

static void *mt_worker(void *arg)
{
    thread_arg *th = (thread_arg *)arg;
    for (size_t i = th->start; i < th->end; i++)
    {
        if (th->buf[i] >= 'a' && th->buf[i] <= 'z')
            th->buf[i] -= 32;
    }
    return NULL;
}

static void toupper_mt(char *buf, size_t n, int nthreads)
{
    pthread_t  threads[nthreads];
    thread_arg args[nthreads];
    size_t chunk = n / nthreads;

    for (int i = 0; i < nthreads; i++)
    {
        args[i].buf   = buf;
        args[i].start = (size_t)i * chunk;
        args[i].end   = (i == nthreads - 1) ? n : args[i].start + chunk;
        pthread_create(&threads[i], NULL, mt_worker, &args[i]);
    }

    for (int i = 0; i < nthreads; i++)
        pthread_join(threads[i], NULL);
}

static void toupper_simd(char *buf, size_t n)
{
    const __m128i low_a  = _mm_set1_epi8('a' - 1);
    const __m128i thresh = _mm_set1_epi8(127);
    const __m128i offset = _mm_set1_epi8((char)(128 - 'z' - 1));
    const __m128i bit5   = _mm_set1_epi8(0x20);

    size_t vec_n = n & ~(size_t)15;
    size_t i = 0;

    for (; i < vec_n; i += 16)
    {
        __m128i data    = _mm_loadu_si128((const __m128i *)(buf + i));
        __m128i ge_a    = _mm_cmpgt_epi8(data, low_a);
        __m128i shifted = _mm_add_epi8(data, offset);
        __m128i le_z    = _mm_cmpgt_epi8(thresh, shifted);
        __m128i mask    = _mm_and_si128(ge_a, le_z);
        __m128i flip    = _mm_and_si128(mask, bit5);
        data            = _mm_sub_epi8(data, flip);
        _mm_storeu_si128((__m128i *)(buf + i), data);
    }

    for (; i < n; i++)
    {
        if (buf[i] >= 'a' && buf[i] <= 'z')
            buf[i] -= 32;
    }
}

static void *simd_mt_worker(void *arg)
{
    thread_arg *th = (thread_arg *)arg;
    toupper_simd(th->buf + th->start, th->end - th->start);
    return NULL;
}

static void toupper_simd_mt(char *buf, size_t n, int nthreads)
{
    pthread_t  threads[nthreads];
    thread_arg args[nthreads];
    size_t chunk = n / nthreads;

    for (int i = 0; i < nthreads; i++)
    {
        args[i].buf   = buf;
        args[i].start = (size_t)i * chunk;
        args[i].end   = (i == nthreads - 1) ? n : args[i].start + chunk;
        pthread_create(&threads[i], NULL, simd_mt_worker, &args[i]);
    }

    for (int i = 0; i < nthreads; i++)
        pthread_join(threads[i], NULL);
}

int main(void)
{
    size_t n = (size_t)SIZE_MB * MB;

    char *src         = malloc(n);
    char *buf_mt      = malloc(n);
    char *buf_simd    = malloc(n);
    char *buf_simd_mt = malloc(n);

    if (!src || !buf_mt || !buf_simd || !buf_simd_mt)
    {
        perror("malloc");
        return 1;
    }

    fill_buffer(src, n);
    memcpy(buf_mt,      src, n);
    memcpy(buf_simd,    src, n);
    memcpy(buf_simd_mt, src, n);

    printf("Buffer size:  %d MB\n", SIZE_MB);
    printf("Threads used: %d\n\n", THREADS);

    double t0, t1;

    t0 = now_sec();
    toupper_mt(buf_mt, n, THREADS);
    t1 = now_sec();

    printf("MT time:        %.3f sec\n", t1 - t0);

    t0 = now_sec();
    toupper_simd(buf_simd, n);
    t1 = now_sec();

    printf("SIMD time:      %.3f sec\n", t1 - t0);

    t0 = now_sec();
    toupper_simd_mt(buf_simd_mt, n, THREADS);
    t1 = now_sec();

    printf("SIMD + MT time: %.3f sec\n", t1 - t0);

    free(src);
    free(buf_mt);
    free(buf_simd);
    free(buf_simd_mt);

    return 0;
}
