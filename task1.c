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
    srand(42);
    for (size_t i = 0; i < n; i++)
        buf[i] = nucleotides[rand() % 4];
}

typedef struct {
    const char *buf;
    size_t      start;
    size_t      end;
    long long   a, c, g, t;
} thread_arg;

static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

static void *mt_worker(void *arg)
{
    thread_arg *th = (thread_arg *)arg;
    long long la = 0, lc = 0, lg = 0, lt = 0;

    for (size_t i = th->start; i < th->end; i++)
    {
        switch (th->buf[i])
        {
            case 'A': la++; break;
            case 'C': lc++; break;
            case 'G': lg++; break;
            case 'T': lt++; break;
        }
    }

    pthread_mutex_lock(&mutex);
    th->a += la;
    th->c += lc;
    th->g += lg;
    th->t += lt;
    pthread_mutex_unlock(&mutex);

    return NULL;
}

static void count_mt(const char *buf, size_t n, int nthreads, long long *ra, long long *rc, long long *rg, long long *rt)
{
    pthread_t  threads[nthreads];
    thread_arg args[nthreads];
    size_t chunk = n / nthreads;

    long long ga = 0, gc = 0, gg = 0, gt = 0;

    for (int i = 0; i < nthreads; i++)
    {
        args[i].buf   = buf;
        args[i].start = (size_t)i * chunk;
        args[i].end   = (i == nthreads - 1) ? n : args[i].start + chunk;
        args[i].a = args[i].c = args[i].g = args[i].t = 0;
        pthread_create(&threads[i], NULL, mt_worker, &args[i]);
    }

    for (int i = 0; i < nthreads; i++)
    {
        pthread_join(threads[i], NULL);
        ga += args[i].a;
        gc += args[i].c;
        gg += args[i].g;
        gt += args[i].t;
    }

    *ra = ga;
    *rc = gc;
    *rg = gg;
    *rt = gt;
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
        switch (buf[i])
        {
            case 'A': la++; break;
            case 'C': lc++; break;
            case 'G': lg++; break;
            case 'T': lt++; break;
        }
    }

    *ra = la;
    *rc = lc;
    *rg = lg;
    *rt = lt;
}

/* ── SIMD + Multithreading ──────────────────────────────────────────────── */
static void *simd_mt_worker(void *arg)
{
    thread_arg *th = (thread_arg *)arg;
    count_simd(th->buf + th->start, th->end - th->start,
               &th->a, &th->c, &th->g, &th->t);
    return NULL;
}

static void count_simd_mt(const char *buf, size_t n, int nthreads, long long *ra, long long *rc, long long *rg, long long *rt)
{
    pthread_t  threads[nthreads];
    thread_arg args[nthreads];
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
        ta += args[i].a; tc += args[i].c;
        tg += args[i].g; tt += args[i].t;
    }

    *ra = ta;
    *rc = tc;
    *rg = tg;
    *rt = tt;
}

int main(void)
{
    size_t n = (size_t)SIZE_MB * MB;

    printf("Allocating %d MB ...\n", SIZE_MB);
    char *buf = malloc(n);
    if (!buf) { perror("malloc"); return 1; }

    printf("Generating random DNA sequence ...\n");
    generate_dna(buf, n);

    printf("\nDNA size:     %d MB  (%zu bytes)\n", SIZE_MB, n);
    printf("Threads used: %d\n\n", THREADS);

    long long a, c, g, t;
    double t0, t1;

    t0 = now_sec();
    count_mt(buf, n, 1, &a, &c, &g, &t);
    t1 = now_sec();

    printf("Counts (A C G T):\n%lld %lld %lld %lld\n\n", a, c, g, t);
    printf("Scalar time:    %.3f sec\n", t1 - t0);

    t0 = now_sec();
    count_mt(buf, n, THREADS, &a, &c, &g, &t);
    t1 = now_sec();

    printf("MT time:        %.3f sec\n", t1 - t0);

    t0 = now_sec();
    count_simd(buf, n, &a, &c, &g, &t);
    t1 = now_sec();

    printf("SIMD time:      %.3f sec\n", t1 - t0);

    t0 = now_sec();
    count_simd_mt(buf, n, THREADS, &a, &c, &g, &t);
    t1 = now_sec();

    printf("SIMD + MT time: %.3f sec\n", t1 - t0);

    free(buf);
    return 0;
}
