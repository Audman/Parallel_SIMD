#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#include <emmintrin.h>

#define THREADS 4

typedef struct {
    int     width, height;
    uint8_t *data;
} Image;

static double now_sec(void) 
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static Image read_ppm(const char *path) 
{
    Image img = {0};
    FILE *f = fopen(path, "rb");
    if (!f)
    {
        perror(path);
        exit(1);
    }

    char magic[3];
    int  maxval;
    fscanf(f, "%2s %d %d %d", magic, &img.width, &img.height, &maxval);
    if (magic[0] != 'P' || magic[1] != '6') {
        fprintf(stderr, "Only P6 PPM supported\n"); exit(1);
    }
    fgetc(f);

    size_t pixels = (size_t)img.width * img.height * 3;
    img.data = malloc(pixels);
    if (!img.data)
    {
        perror("malloc");
        exit(1);
    }
    
    fread(img.data, 1, pixels, f);
    fclose(f);
    return img;
}

static void write_ppm(const char *path, const Image *img) 
{
    FILE *f = fopen(path, "wb");
    if (!f) 
    {
        perror(path);
        exit(1);
    }
    fprintf(f, "P6\n%d %d\n255\n", img->width, img->height);
    fwrite(img->data, 1, (size_t)img->width * img->height * 3, f);
    fclose(f);
}

static void grayscale_scalar(const uint8_t *src, uint8_t *dst, size_t npixels) 
{
    for (size_t i = 0; i < npixels; i++) 
    {
        uint8_t r = src[i*3 + 0];
        uint8_t g = src[i*3 + 1];
        uint8_t b = src[i*3 + 2];
        uint8_t gray = (uint8_t)(0.299f * r + 0.587f * g + 0.114f * b);
        dst[i*3 + 0] = gray;
        dst[i*3 + 1] = gray;
        dst[i*3 + 2] = gray;
    }
}

static void grayscale_simd(const uint8_t *src, uint8_t *dst, size_t npixels) 
{
    const int16_t kR = 9798, kG = 19235, kB = 3735;

    size_t i     = 0;
    size_t pairs = npixels / 2;

    for (size_t p = 0; p < pairs; p++, i += 2) 
    {
        const uint8_t *s = src + i * 3;

        __m128i v = _mm_set_epi16(0, 0,
            (int16_t)s[5], (int16_t)s[4],
            (int16_t)s[3], (int16_t)s[2],
            (int16_t)s[1], (int16_t)s[0]);
        __m128i coeff = _mm_set_epi16(0, 0, kB, kG, kR, kB, kG, kR);
        __m128i prod  = _mm_madd_epi16(v, coeff);

        // prod[0] = R0*kR + G0*kG,  prod[1] = B0*kB + 0
        // prod[2] = R1*kR + G1*kG,  prod[3] = B1*kB + 0
        int32_t t[4];
        _mm_storeu_si128((__m128i *)t, prod);

        uint8_t g0 = (uint8_t)((t[0] + t[1]) >> 15);
        uint8_t g1 = (uint8_t)((t[2] + t[3]) >> 15);

        dst[i*3+0] = dst[i*3+1] = dst[i*3+2] = g0;
        dst[i*3+3] = dst[i*3+4] = dst[i*3+5] = g1;
    }

    /* Scalar tail for odd pixel count */
    for (; i < npixels; i++) 
    {
        uint8_t gray = (uint8_t)(0.299f*src[i*3] + 0.587f*src[i*3+1] + 0.114f*src[i*3+2]);
        dst[i*3+0] = dst[i*3+1] = dst[i*3+2] = gray;
    }
}

typedef struct {
    const uint8_t *src;
    uint8_t       *dst;
    size_t         start;
    size_t         end;
} ChunkArgs;

static void *mt_worker(void *arg) 
{
    ChunkArgs *a = (ChunkArgs *)arg;
    grayscale_scalar(a->src + a->start * 3, a->dst + a->start * 3,
                     a->end - a->start);
    return NULL;
}

static void grayscale_mt(const uint8_t *src, uint8_t *dst, size_t npixels, int nthreads) 
{
    pthread_t threads[nthreads];
    ChunkArgs args[nthreads];
    size_t chunk = npixels / nthreads;
    for (int t = 0; t < nthreads; t++) {
        args[t].src   = src;
        args[t].dst   = dst;
        args[t].start = (size_t)t * chunk;
        args[t].end   = (t == nthreads - 1) ? npixels : args[t].start + chunk;
        pthread_create(&threads[t], NULL, mt_worker, &args[t]);
    }
    for (int t = 0; t < nthreads; t++) pthread_join(threads[t], NULL);
}

static void *simd_mt_worker(void *arg) 
{
    ChunkArgs *a = (ChunkArgs *)arg;
    grayscale_simd(a->src + a->start * 3, a->dst + a->start * 3, a->end - a->start);
    return NULL;
}

static void grayscale_simd_mt(const uint8_t *src, uint8_t *dst, size_t npixels, int nthreads) 
{
    pthread_t threads[nthreads];
    ChunkArgs args[nthreads];
    size_t chunk = npixels / nthreads;
    for (int t = 0; t < nthreads; t++) 
    {
        args[t].src   = src;
        args[t].dst   = dst;
        args[t].start = (size_t)t * chunk;
        args[t].end   = (t == nthreads - 1) ? npixels : args[t].start + chunk;
        pthread_create(&threads[t], NULL, simd_mt_worker, &args[t]);
    }
    
    for (int t = 0; t < nthreads; t++)
        pthread_join(threads[t], NULL);
}

static int verify(const uint8_t *ref, const uint8_t *out, size_t nbytes) 
{
    for (size_t i = 0; i < nbytes; i++)
        if (ref[i] != out[i]) 
            return 0;
    return 1;
}

int main(int argc, char *argv[]) 
{
    if (argc < 2) 
    {
        fprintf(stderr, "Usage: %s input.ppm\n", argv[0]);
        return 1;
    }

    Image src = read_ppm(argv[1]);
    size_t npixels = (size_t)src.width * src.height;
    size_t nbytes  = npixels * 3;

    uint8_t *out_scalar  = malloc(nbytes);
    uint8_t *out_simd    = malloc(nbytes);
    uint8_t *out_mt      = malloc(nbytes);
    uint8_t *out_simd_mt = malloc(nbytes);

    if (!out_scalar || !out_simd || !out_mt || !out_simd_mt) 
    {
        perror("malloc"); 
        return 1;
    }

    printf("Image size: %d x %d\n", src.width, src.height);
    printf("Threads used: %d\n", THREADS);

    double t0, t1;

    t0 = now_sec();
    grayscale_scalar(src.data, out_scalar, npixels);
    t1 = now_sec();
    printf("Scalar time:    %.3f sec\n", t1 - t0);

    t0 = now_sec();
    grayscale_simd(src.data, out_simd, npixels);
    t1 = now_sec();
    printf("SIMD time:      %.3f sec\n", t1 - t0);

    t0 = now_sec();
    grayscale_mt(src.data, out_mt, npixels, THREADS);
    t1 = now_sec();
    printf("MT time:        %.3f sec\n", t1 - t0);

    t0 = now_sec();
    grayscale_simd_mt(src.data, out_simd_mt, npixels, THREADS);
    t1 = now_sec();
    printf("MT + SIMD time: %.3f sec\n", t1 - t0);

    int ok = verify(out_scalar, out_simd,    nbytes) &&
             verify(out_scalar, out_mt,      nbytes) &&
             verify(out_scalar, out_simd_mt, nbytes);

    printf("Verification: %s\n", ok ? "PASSED" : "FAILED");

    Image out_img = { src.width, src.height, out_scalar };
    write_ppm("gray_output.ppm", &out_img);
    printf("Output image: gray_output.ppm\n");

    free(src.data);
    free(out_scalar); free(out_simd); free(out_mt); free(out_simd_mt);

    return 0;
}
