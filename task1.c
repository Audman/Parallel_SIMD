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

static void generate_dna(char *buf, size_t n)
{
    static const char nucleotides[4] = {'A', 'C', 'G', 'T'};
    for (size_t i = 0; i < n; i++)
        buf[i] = nucleotides[rand() % 4];
}

typedef struct {
    const char *buf;
    size_t      start;
    size_t      end;
} MT_Args;

static long long   mt_A, mt_C, mt_G, mt_T;
static pthread_mutex_t mt_mutex = PTHREAD_MUTEX_INITIALIZER;

static void *mt_worker(void *arg)
{
    MT_Args *a = (MT_Args *)arg;
    long long la = 0, lc = 0, lg = 0, lt = 0;
    for (size_t i = a->start; i < a->end; i++)
    {
        char ch = a->buf[i];
        if      (ch == 'A') la++;
        else if (ch == 'C') lc++;
        else if (ch == 'G') lg++;
        else if (ch == 'T') lt++;
    }

    pthread_mutex_lock(&mt_mutex);
    mt_A += la;
    mt_C += lc;
    mt_G += lg;
    mt_T += lt;
    pthread_mutex_unlock(&mt_mutex);

    return NULL;
}

static void count_multithreaded(const char *buf, size_t n, int nthreads, long long *a, long long *c, long long *g, long long *t)
{
    mt_A = mt_C = mt_G = mt_T = 0;

    pthread_t  *threads = malloc(nthreads * sizeof(pthread_t));
    MT_Args    *args    = malloc(nthreads * sizeof(MT_Args));
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

    *a = mt_A;
    *c = mt_C;
    *g = mt_G;
    *t = mt_T;

    free(threads);
    free(args);
}

static void count_simd(const char *buf, size_t n, long long *ra, long long *rc, long long *rg, long long *rt)
{
    const __m128i vA = _mm_set1_epi8('A');
    const __m128i vC = _mm_set1_epi8('C');
    const __m128i vG = _mm_set1_epi8('G');
    const __m128i vT = _mm_set1_epi8('T');

    __m128i sumA32 = _mm_setzero_si128();
    __m128i sumC32 = _mm_setzero_si128();
    __m128i sumG32 = _mm_setzero_si128();
    __m128i sumT32 = _mm_setzero_si128();

    __m128i accA = _mm_setzero_si128();
    __m128i accC = _mm_setzero_si128();
    __m128i accG = _mm_setzero_si128();
    __m128i accT = _mm_setzero_si128();

    size_t i       = 0;
    size_t vec_end = n - (n % 16);
    int    cycle   = 0;

    while (i < vec_end)
    {
        __m128i data = _mm_loadu_si128((const __m128i *)(buf + i));

        accA = _mm_sub_epi8(accA, _mm_cmpeq_epi8(data, vA));
        accC = _mm_sub_epi8(accC, _mm_cmpeq_epi8(data, vC));
        accG = _mm_sub_epi8(accG, _mm_cmpeq_epi8(data, vG));
        accT = _mm_sub_epi8(accT, _mm_cmpeq_epi8(data, vT));

        i += 16;
        cycle++;

        if (cycle == 255)
        {
            sumA32 = _mm_add_epi32(sumA32, _mm_sad_epu8(accA, _mm_setzero_si128()));
            sumC32 = _mm_add_epi32(sumC32, _mm_sad_epu8(accC, _mm_setzero_si128()));
            sumG32 = _mm_add_epi32(sumG32, _mm_sad_epu8(accG, _mm_setzero_si128()));
            sumT32 = _mm_add_epi32(sumT32, _mm_sad_epu8(accT, _mm_setzero_si128()));
            accA = accC = accG = accT = _mm_setzero_si128();
            cycle = 0;
        }
    }

    sumA32 = _mm_add_epi32(sumA32, _mm_sad_epu8(accA, _mm_setzero_si128()));
    sumC32 = _mm_add_epi32(sumC32, _mm_sad_epu8(accC, _mm_setzero_si128()));
    sumG32 = _mm_add_epi32(sumG32, _mm_sad_epu8(accG, _mm_setzero_si128()));
    sumT32 = _mm_add_epi32(sumT32, _mm_sad_epu8(accT, _mm_setzero_si128()));

    int32_t tmp[4];
    long long la = 0, lc = 0, lg = 0, lt = 0;

#define HSUM32(vec, out) do {                              \
    _mm_storeu_si128((__m128i *)tmp, (vec));               \
    (out) = (long long)tmp[0] + tmp[1] + tmp[2] + tmp[3]; \
} while(0)

    HSUM32(sumA32, la);
    HSUM32(sumC32, lc);
    HSUM32(sumG32, lg);
    HSUM32(sumT32, lt);

    for (; i < n; i++)
    {
        char ch = buf[i];
        if      (ch == 'A') la++;
        else if (ch == 'C') lc++;
        else if (ch == 'G') lg++;
        else if (ch == 'T') lt++;
    }

    *ra = la;
    *rc = lc;
    *rg = lg;
    *rt = lt;
}

typedef struct {
    const char *buf;
    size_t      start;
    size_t      end;
    long long   a, c, g, t;
} SIMD_MT_Args;

static void *simd_mt_worker(void *arg) {
    SIMD_MT_Args *a = (SIMD_MT_Args *)arg;
    count_simd(a->buf + a->start, a->end - a->start,
               &a->a, &a->c, &a->g, &a->t);
    return NULL;
}

static void count_simd_mt(const char *buf, size_t n, int nthreads, long long *ra, long long *rc, long long *rg, long long *rt)
{
    pthread_t    *threads = malloc(nthreads * sizeof(pthread_t));
    SIMD_MT_Args *args    = malloc(nthreads * sizeof(SIMD_MT_Args));
    size_t chunk = n / nthreads;

    for (int i = 0; i < nthreads; i++)
    {
        args[i].buf   = buf;
        args[i].start = (size_t)i * chunk;
        args[i].end   = (i == nthreads - 1) ? n : args[i].start + chunk;
        args[i].a = args[i].c = args[i].g = args[i].t = 0;
        pthread_create(&threads[i], NULL, simd_mt_worker, &args[i]);
    }

    long long ta = 0, tc = 0, tg = 0, tt = 0;
    for (int i = 0; i < nthreads; i++)
    {
        pthread_join(threads[i], NULL);
        ta += args[i].a;
        tc += args[i].c;
        tg += args[i].g;
        tt += args[i].t;
    }

    *ra = ta;
    *rc = tc;
    *rg = tg;
    *rt = tt;

    free(threads);
    free(args);
}


int main()
{
    int    size_mb  = SIZE_MB;
    int    nthreads = THREADS;
    size_t n        = (size_t)size_mb * MB;

    printf("Allocating %d MB ...\n", size_mb);
    char *buf = malloc(n);
    if (!buf) { perror("malloc"); return 1; }

    printf("Generating random DNA sequence ...\n");
    generate_dna(buf, n);

    printf("\nDNA size:     %d MB  (%zu bytes)\n", size_mb, n);
    printf("Threads used: %d\n\n", nthreads);

    long long a, c, g, t;
    double t0, t1;

    t0 = now_sec();
    count_multithreaded(buf, n, 1, &a, &c, &g, &t);
    t1 = now_sec();
    printf("Counts (A C G T):\n%lld %lld %lld %lld\n\n", a, c, g, t);
    printf("Scalar time:    %.3f sec\n", t1 - t0);

    t0 = now_sec();
    count_multithreaded(buf, n, nthreads, &a, &c, &g, &t);
    t1 = now_sec();
    printf("MT time:        %.3f sec\n", t1 - t0);

    t0 = now_sec();
    count_simd(buf, n, &a, &c, &g, &t);
    t1 = now_sec();
    printf("SIMD time:      %.3f sec\n", t1 - t0);

    t0 = now_sec();
    count_simd_mt(buf, n, nthreads, &a, &c, &g, &t);
    t1 = now_sec();
    printf("SIMD + MT time: %.3f sec\n", t1 - t0);

    free(buf);
    return 0;
}
