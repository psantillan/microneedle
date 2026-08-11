/* Where the encoder's projection time actually goes.
 *
 * bench/vector says the PIE row kernel runs at 0.236 cycles/MAC. The engine's
 * projections measure 1.5 core-cycles/MAC on the same silicon, running the same
 * kernel. This benchmark takes the engine's inner loop apart one factor at a
 * time to find the other 6x.
 *
 * Every stage reports cycles/MAC and CPI. CPI is the discriminator: the kernel
 * issues a fixed number of instructions per MAC, so if cycles/MAC rises while
 * instructions/MAC stays flat, the machine is stalling, and the stage that
 * introduces the stall is the answer. mcycle and minstret are read directly --
 * a wall clock cannot tell those two apart.
 *
 * Shape matches ne_gemm_rows: one weight row swept against a block of NB
 * quantized token activations, output written to y[t * LDY + row].
 */

#include <stdio.h>
#include <string.h>
#include <inttypes.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <math.h>
#include "esp_heap_caps.h"

#define COLS 512               /* NE_DMODEL                                  */
#define ROWS 128               /* 64 KB of int8 weights: fits internal SRAM  */
#define NB   64                /* NE_TBLK                                    */
#define LDY  512               /* NE_DMODEL: the real output row stride      */
#define NCHUNK (COLS / 32)

extern int32_t pie_rowsum_i8(const int8_t *xq, const int8_t *w, int nchunks);

static inline uint32_t rd_cycle(void)
{ uint32_t v; __asm__ volatile("csrr %0, mcycle" : "=r"(v)); return v; }
static inline uint32_t rd_instret(void)
{ uint32_t v; __asm__ volatile("csrr %0, minstret" : "=r"(v)); return v; }

static uint32_t rng = 0x1234567;
static int8_t rnd8(void) { rng = rng * 1103515245 + 12345; return (int8_t)(rng >> 16); }

static double g_base_cpm;

static void report(const char *label, uint32_t cyc, uint32_t ins, double macs)
{
    double cpm = (double)cyc / macs;
    printf("  %-42s %7.3f cyc/MAC  %6.4f ins/MAC  CPI %5.2f",
           label, cpm, (double)ins / macs, (double)cyc / (double)ins);
    if (g_base_cpm > 0) printf("  %5.2fx", cpm / g_base_cpm);
    printf("\n");
}

/* Stage 1: the kernel and nothing else. Both operands internal, result thrown
 * away. This is the floor the rest is measured against. */
static double stage_kernel(const int8_t *x, const int8_t *W)
{
    volatile int32_t sink = 0;
    uint32_t c0 = rd_cycle(), i0 = rd_instret();
    for (int r = 0; r < ROWS; r++)
        for (int t = 0; t < NB; t++)
            sink += pie_rowsum_i8(x + (size_t)t * COLS, W + (size_t)r * COLS, NCHUNK);
    uint32_t cyc = rd_cycle() - c0, ins = rd_instret() - i0;
    double macs = (double)ROWS * NB * COLS;
    g_base_cpm = 0;
    report("kernel only, both operands internal", cyc, ins, macs);
    g_base_cpm = (double)cyc / macs;
    (void)sink;
    return g_base_cpm;
}

/* Stages 2-4: add the float epilogue the engine actually runs, varying only
 * where the output goes and how it is strided. */
static void stage_epilogue(const char *label, const int8_t *x, const int8_t *W,
                           float *y, size_t ldy, const float *sc32, const float *sx)
{
    uint32_t c0 = rd_cycle(), i0 = rd_instret();
    for (int r = 0; r < ROWS; r++)
        for (int t = 0; t < NB; t++) {
            int32_t s = pie_rowsum_i8(x + (size_t)t * COLS, W + (size_t)r * COLS, NCHUNK);
            y[(size_t)t * ldy + r] = sc32[r] * sx[t] * (float)s;
        }
    uint32_t cyc = rd_cycle() - c0, ins = rd_instret() - i0;
    report(label, cyc, ins, (double)ROWS * NB * COLS);
}

/* Stage 5: ne_gemm_rows as the engine actually runs it -- int4 weights read
 * from PSRAM, unpacked once per row into a stack buffer, then swept against the
 * whole token block. The unpack is the point: it converts the streamed operand
 * into an internal-SRAM one, so the PSRAM traffic drops to 256 bytes per 32768
 * MACs. If this stage is fast, the engine's cost is not in ne_gemm_rows. */
static void unpack_i4_row(const uint8_t *w, int nchunks, int8_t *out)
{
    for (int c = 0; c < nchunks; c++) {
        const uint8_t *src = w + c * 16;
        int8_t *dst = out + c * 32;
        for (int k = 0; k < 16; k++) {
            const uint8_t b = src[k];
            dst[k]      = (int8_t)((int)(b & 0xF) - ((b & 0x08) ? 16 : 0));
            dst[k + 16] = (int8_t)((int)(b >> 4)  - ((b & 0x80) ? 16 : 0));
        }
    }
}

static void stage_gemm_i4(const int8_t *x, const uint8_t *W4, int rows,
                          float *y, const float *sc32, const float *sx)
{
    int8_t rowbuf[COLS + 16] __attribute__((aligned(16)));
    uint32_t c0 = rd_cycle(), i0 = rd_instret();
    for (int r = 0; r < rows; r++) {
        unpack_i4_row(W4 + (size_t)r * (COLS / 2), NCHUNK, rowbuf);
        for (int t = 0; t < NB; t++) {
            int32_t s = pie_rowsum_i8(x + (size_t)t * COLS, rowbuf, NCHUNK);
            y[(size_t)t * LDY + r] = sc32[r & (ROWS - 1)] * sx[t] * (float)s;
        }
    }
    uint32_t cyc = rd_cycle() - c0, ins = rd_instret() - i0;
    char lab[64];
    snprintf(lab, sizeof lab, "ne_gemm_rows replica, %d rows int4 from PSRAM", rows);
    report(lab, cyc, ins, (double)rows * NB * COLS);
}

/* Stage 6: the activation quantization project_block runs before the sweep,
 * over PSRAM-resident normed activations. The engine runs it once per tensor,
 * i.e. three times per block on identical input. */
static void stage_quant(const float *hb, int8_t *xq, float *sxo)
{
    uint32_t c0 = rd_cycle(), i0 = rd_instret();
    for (int t = 0; t < NB; t++) {
        const float *xx = hb + (size_t)t * COLS;
        float m = 0.f;
        for (int k = 0; k < COLS; k++) { float a = __builtin_fabsf(xx[k]); if (a > m) m = a; }
        float sc = (m > 0.f) ? m / 127.0f : 1.0f, inv = 1.0f / sc;
        sxo[t] = sc;
        for (int k = 0; k < COLS; k++)
            xq[(size_t)t * COLS + k] = (int8_t)(int32_t)(xx[k] * inv + (xx[k] >= 0.f ? 0.5f : -0.5f));
    }
    uint32_t cyc = rd_cycle() - c0, ins = rd_instret() - i0;
    /* charged against the MACs it enables: one sweep of a 512-row tensor */
    report("activation quantize (per tensor, 512-row sweep)", cyc, ins,
           (double)512 * NB * COLS);
}

/* Stage 7: everything else inside the engine's projection timer -- the norms
 * and the rotary embedding. Per token per layer that is one rmsnorm over 512,
 * then twelve heads of rmsnorm-over-64 plus rope-over-64. rope calls cosf and
 * sinf per dimension pair. */
static float g_freq[32];
static void bench_rope(float *h, int pos)
{
    for (int i = 0; i < 32; i++) {
        float a = (float)pos * g_freq[i];
        float c = cosf(a), s = sinf(a);
        float x1 = h[i], x2 = h[i + 32];
        h[i] = x1 * c - x2 * s;
        h[i + 32] = x2 * c + x1 * s;
    }
}
static void bench_rmsnorm(const float *x, const float *w, int n, float *y)
{
    float ss = 0.f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / (float)n + 1e-6f);
    for (int i = 0; i < n; i++) y[i] = (1.0f + w[i]) * x[i] * inv;
}

static void stage_norm_rope(int tokens, int layers)
{
    static float h[512], w[512], y[512];
    for (int i = 0; i < 512; i++) { h[i] = 0.01f * (float)(i - 256); w[i] = 0.001f * (float)i; }
    for (int i = 0; i < 32; i++) g_freq[i] = powf(10000.0f, -2.0f * (float)i / 64.0f);

    const int REP = 200;
    uint32_t c0 = rd_cycle();
    for (int r = 0; r < REP; r++) bench_rmsnorm(h, w, 512, y);
    uint32_t n512 = (rd_cycle() - c0) / REP;

    c0 = rd_cycle();
    for (int r = 0; r < REP; r++) bench_rmsnorm(h, w, 64, y);
    uint32_t n64 = (rd_cycle() - c0) / REP;

    c0 = rd_cycle();
    for (int r = 0; r < REP; r++) bench_rope(h, r);
    uint32_t rp = (rd_cycle() - c0) / REP;

    /* per token per layer: 1 rmsnorm(512) + 12 heads of rmsnorm(64) + rope(64) */
    double per_tok_layer = (double)n512 + 12.0 * ((double)n64 + (double)rp);
    double total = per_tok_layer * tokens * layers;
    printf("    rmsnorm(512)  %5" PRIu32 " cyc     rmsnorm(64) %5" PRIu32 " cyc"
           "     rope(64) %5" PRIu32 " cyc  (cosf+sinf x32)\n", n512, n64, rp);
    printf("    per token per layer %6.0f cyc -> %d tokens x %d layers = %.0f Mcyc = %.2f s\n",
           per_tok_layer, tokens, layers, total / 1e6, total / 360e6);
    printf("      of which rope: %.0f Mcyc = %.2f s\n",
           12.0 * rp * tokens * layers / 1e6, 12.0 * rp * tokens * layers / 360e6);
}

/* Stage 8: how far the kernel degrades as the streamed operand outgrows L2.
 * Weight bytes per sweep = rows * COLS; L2 is 128 KB. */
static void stage_working_set(const int8_t *x, const int8_t *Wp)
{
    printf("\n  weight working set vs L2 (128 KB), operand streamed from PSRAM:\n");
    for (int kb = 8; kb <= 2048; kb *= 2) {
        int rows = kb * 1024 / COLS;
        int reps = (2 * 1024 * 1024) / (rows * COLS);   /* equal bytes touched */
        if (reps < 1) reps = 1;
        volatile int32_t sink = 0;
        uint32_t c0 = rd_cycle(), i0 = rd_instret();
        for (int rep = 0; rep < reps; rep++)
            for (int r = 0; r < rows; r++)
                sink += pie_rowsum_i8(x, Wp + (size_t)r * COLS, NCHUNK);
        uint32_t cyc = rd_cycle() - c0, ins = rd_instret() - i0;
        char lab[64];
        snprintf(lab, sizeof lab, "  %5d KB of weights%s", kb, kb == 128 ? "  <- L2 size" : "");
        report(lab, cyc, ins, (double)reps * rows * COLS);
        (void)sink;
    }
}

/* Attention, and whether the head-split pushed it over the L2 cliff.
 *
 * Per head at S=271, K is S*64*2 = 34.7 KB and V is 64*Sp*2 = 36.9 KB: one head is a
 * 71.5 KB working set, comfortably inside the 128 KB L2. Two cores on two different heads
 * make it 143 KB, over the edge.
 *
 * That is testable on one core: interleaving two heads per query keeps both K/V sets live
 * simultaneously, which is the same pressure the two-core split creates. If heads_live=2
 * is much worse per MAC than heads_live=1, the split is thrashing L2 and the fix is to put
 * both cores on the same head. */
extern void pie_dots_s16(const int16_t *a, const int16_t *B, int len, int nrows, int32_t *out);

static void stage_attn(const char *label, const int16_t *kq, const int16_t *vq,
                       const int16_t *qq, int16_t *pq, int32_t *iw,
                       int S, int Sp, int heads_live, int blocked)
{
    const size_t kstride = (size_t)S * 64, vstride = (size_t)64 * Sp;
    const int QB = blocked ? 16 : 1;
    uint32_t c0 = rd_cycle(), i0 = rd_instret();
    for (int h = 0; h < heads_live; h++)
        for (int tb = 0; tb < S; tb += QB) {
            const int nb = (S - tb < QB) ? S - tb : QB;
            /* Blocked: every QK for the block, then every AV. Only K is live,
             * then only V. Fused: they alternate and both are live. */
            for (int i = 0; i < nb; i++)
                pie_dots_s16(qq, kq + (size_t)h * kstride, 64, S, iw);
            for (int i = 0; i < nb; i++)
                pie_dots_s16(pq, vq + (size_t)h * vstride, Sp, 64, iw);
        }
    uint32_t cyc = rd_cycle() - c0, ins = rd_instret() - i0;
    double macs = (double)S * heads_live * (64.0 * S + (double)Sp * 64.0);
    report(label, cyc, ins, macs);
}

/* Two cores on two heads is not the same as one core alternating two heads when
 * the loop is blocked: each core is only touching K, or only V, at any moment.
 * heads_live models the concurrent pressure; blocked models the loop shape. */
extern void qacc_probe(const int16_t *bc, const int16_t *vec, int n, void *out64);

/* Read the QACC lane layout off the hardware. Two cases: one step with distinct
 * vector lanes tells us where each lane lands and how wide it is; two steps
 * with a different broadcast tells us the accumulate works and that .incp
 * advances the broadcast pointer by one 16-bit element. */
static void stage_qacc_probe(void)
{
    static int16_t bc[8]  __attribute__((aligned(16)));
    static int16_t vec[16] __attribute__((aligned(16)));
    static int32_t out[16] __attribute__((aligned(16)));

    printf("\n  QACC broadcast MAC probe:\n");

    bc[0] = 1; bc[1] = 0;
    for (int k = 0; k < 8; k++) vec[k] = (int16_t)(k + 1);
    for (int k = 0; k < 8; k++) vec[8 + k] = 0;
    memset(out, 0xEE, sizeof out);
    qacc_probe(bc, vec, 1, out);
    printf("    n=1 bc=[1] vec=[1..8], expect lanes 1..8 somewhere:\n     ");
    for (int i = 0; i < 16; i++) printf(" %ld", (long)out[i]);
    printf("\n");

    bc[0] = 1; bc[1] = 1000;
    for (int k = 0; k < 8; k++) { vec[k] = (int16_t)(k + 1); vec[8 + k] = (int16_t)(k + 1); }
    memset(out, 0xEE, sizeof out);
    qacc_probe(bc, vec, 2, out);
    printf("    n=2 bc=[1,1000] vec rows both [1..8], expect lanes 1001*k:\n     ");
    for (int i = 0; i < 16; i++) printf(" %ld", (long)out[i]);
    printf("\n");

    /* widest case the attention kernel would hand it: 64 terms at +-4096 */
    bc[0] = 4096;
    for (int k = 0; k < 8; k++) vec[k] = 4096;
    memset(out, 0xEE, sizeof out);
    qacc_probe(bc, vec, 1, out);
    printf("    n=1 bc=[4096] vec=[4096 x8], expect 16777216 per lane:\n     ");
    for (int i = 0; i < 16; i++) printf(" %ld", (long)out[i]);
    printf("\n");
}

extern void pie_qk8_s16(const int16_t *QT, const int16_t *K, int nkeys, void *out);

/* The whole point, measured: eight query dot-products from one pass over K
 * against eight passes. Correctness first -- both must reproduce the scalar
 * result exactly before either timing means anything. */
static void stage_qk8(const int16_t *kq, int S)
{
    static int16_t QT[64 * 8]  __attribute__((aligned(16)));
    static int16_t Q8[8][64 + 8] __attribute__((aligned(16)));
    int32_t *out8 = heap_caps_aligned_alloc(16, (size_t)S * 64, MALLOC_CAP_INTERNAL);
    int32_t *out1 = heap_caps_aligned_alloc(16, (size_t)S * 8 * 4, MALLOC_CAP_INTERNAL);
    if (!out8 || !out1) { printf("    alloc failed\n"); return; }

    for (int q = 0; q < 8; q++)
        for (int d = 0; d < 64; d++) {
            int16_t v = (int16_t)(((q * 37 + d * 11) % 8193) - 4096);
            Q8[q][d] = v;
            QT[d * 8 + q] = v;
        }
    for (int q = 0; q < 8; q++) for (int k = 64; k < 64 + 8; k++) Q8[q][k] = 0;

    printf("\n  eight queries per pass over K, S=%d:\n", S);

    pie_qk8_s16(QT, kq, S, out8);
    for (int q = 0; q < 8; q++) pie_dots_s16(Q8[q], kq, 64, S, out1 + (size_t)q * S);
    int bad = 0;
    for (int j = 0; j < S && bad < 3; j++)
        for (int q = 0; q < 8; q++)
            if (out8[j * 16 + 2 * q] != out1[(size_t)q * S + j]) {
                if (bad++ < 3)
                    printf("    MISMATCH key %d lane %d: qacc %ld accx %ld\n",
                           j, q, (long)out8[j * 16 + 2 * q], (long)out1[(size_t)q * S + j]);
            }
    printf("    agreement with the accx kernel: %s\n", bad ? "FAIL" : "exact on all lanes");
    if (bad) return;

    const int REP = 20;
    double macs = (double)REP * S * 8 * 64;
    uint32_t c0 = rd_cycle(), i0 = rd_instret();
    for (int r = 0; r < REP; r++)
        for (int q = 0; q < 8; q++) pie_dots_s16(Q8[q], kq, 64, S, out1 + (size_t)q * S);
    uint32_t cyc = rd_cycle() - c0, ins = rd_instret() - i0;
    g_base_cpm = 0; report("accx: eight passes over K", cyc, ins, macs);
    g_base_cpm = (double)cyc / macs;

    c0 = rd_cycle(); i0 = rd_instret();
    for (int r = 0; r < REP; r++) pie_qk8_s16(QT, kq, S, out8);
    cyc = rd_cycle() - c0; ins = rd_instret() - i0;
    report("qacc: one pass, eight lanes", cyc, ins, macs);
    g_base_cpm = 0;
}

extern void pie_dots_s8(const int8_t *a, const int8_t *B, int len, int nrows,
                        int32_t *out);

/* The s8 kernel on the engine's two attention shapes, against the s16 kernel
 * on the same values. Correctness first: both kernels and a scalar loop must
 * agree exactly on identical integer inputs (the s8 values fit both types) --
 * only then is the timing a comparison and not two different computations.
 *
 * Expectation, not yet measured on hardware: 16 lanes per vector against 8
 * and half the bytes streamed puts the target at ~0.12-0.15 cyc/MAC against
 * the s16 kernel's 0.24 in bench/vector conditions; on the S=271 value shape
 * the PSRAM/L2 traffic halving is the part the engine actually feels. */
static void stage_dots_s8(int S)
{
    const int Sp8 = (S + 63) & ~63;          /* the s8 path rounds Sp to 64  */
    static int8_t  qa8[64 + 16]  __attribute__((aligned(16)));
    static int16_t qa16[64 + 8]  __attribute__((aligned(16)));
    int8_t  *K8  = heap_caps_aligned_alloc(16, 4u * S * 64, MALLOC_CAP_SPIRAM);
    int16_t *K16 = heap_caps_aligned_alloc(16, 4u * S * 64 * 2, MALLOC_CAP_SPIRAM);
    int8_t  *P8  = heap_caps_aligned_alloc(16, (size_t)Sp8 + 16, MALLOC_CAP_INTERNAL);
    int16_t *P16 = heap_caps_aligned_alloc(16, ((size_t)Sp8 + 8) * 2, MALLOC_CAP_INTERNAL);
    int8_t  *V8  = heap_caps_aligned_alloc(16, 64u * Sp8, MALLOC_CAP_SPIRAM);
    int16_t *V16 = heap_caps_aligned_alloc(16, 64u * Sp8 * 2, MALLOC_CAP_SPIRAM);
    int32_t *o8  = heap_caps_aligned_alloc(16, (size_t)S * 4, MALLOC_CAP_INTERNAL);
    int32_t *o16 = heap_caps_aligned_alloc(16, (size_t)S * 4, MALLOC_CAP_INTERNAL);
    if (!K8 || !K16 || !P8 || !P16 || !V8 || !V16 || !o8 || !o16) {
        printf("    s8 stage: alloc failed\n");
        return;
    }
    for (int i = 0; i < 64; i++) { qa8[i] = (int8_t)((i * 37 % 255) - 127); qa16[i] = qa8[i]; }
    for (int i = 64; i < 64 + 16; i++) qa8[i] = 0;
    for (int i = 64; i < 64 + 8; i++)  qa16[i] = 0;
    for (size_t i = 0; i < 4u * S * 64; i++) { K8[i] = (int8_t)((i % 255) - 127); K16[i] = K8[i]; }
    for (int i = 0; i < Sp8; i++) { P8[i] = (int8_t)(i % 128); P16[i] = P8[i]; }
    for (int i = Sp8; i < Sp8 + 16; i++) P8[i] = 0;
    for (int i = Sp8; i < Sp8 + 8; i++)  P16[i] = 0;
    for (size_t i = 0; i < 64u * Sp8; i++) { V8[i] = (int8_t)((i * 7 % 255) - 127); V16[i] = V8[i]; }

    printf("\n  s8 dots kernel vs s16, identical +-127 values:\n");

    int bad = 0;
    pie_dots_s8(qa8, K8, 64, S, o8);
    pie_dots_s16(qa16, K16, 64, S, o16);
    for (int r = 0; r < S; r++) {
        int32_t ref = 0;
        for (int d = 0; d < 64; d++) ref += (int32_t)qa8[d] * K8[(size_t)r * 64 + d];
        if (o8[r] != ref || o16[r] != ref) {
            if (bad++ < 3)
                printf("    MISMATCH row %d: scalar %ld s8 %ld s16 %ld\n",
                       r, (long)ref, (long)o8[r], (long)o16[r]);
        }
    }
    pie_dots_s8(P8, V8, Sp8, 64, o8);
    pie_dots_s16(P16, V16, Sp8, 64, o16);
    for (int r = 0; r < 64; r++) {
        int32_t ref = 0;
        for (int d = 0; d < Sp8; d++) ref += (int32_t)P8[d] * V8[(size_t)r * Sp8 + d];
        if (o8[r] != ref || o16[r] != ref) {
            if (bad++ < 3)
                printf("    MISMATCH vrow %d: scalar %ld s8 %ld s16 %ld\n",
                       r, (long)ref, (long)o8[r], (long)o16[r]);
        }
    }
    printf("    agreement (scalar = s8 = s16): %s\n", bad ? "FAIL" : "exact on both shapes");
    if (bad) return;

    const int REP = 40;
    double macs = (double)REP * S * 64;
    uint32_t c0 = rd_cycle(), i0 = rd_instret();
    for (int r = 0; r < REP; r++) pie_dots_s16(qa16, K16, 64, S, o16);
    uint32_t cyc = rd_cycle() - c0, ins = rd_instret() - i0;
    g_base_cpm = 0; report("scores s16: q against K, S rows", cyc, ins, macs);
    g_base_cpm = (double)cyc / macs;
    c0 = rd_cycle(); i0 = rd_instret();
    for (int r = 0; r < REP; r++) pie_dots_s8(qa8, K8, 64, S, o8);
    cyc = rd_cycle() - c0; ins = rd_instret() - i0;
    report("scores s8:  q against K, S rows", cyc, ins, macs);

    macs = (double)REP * 64 * Sp8;
    c0 = rd_cycle(); i0 = rd_instret();
    for (int r = 0; r < REP; r++) pie_dots_s16(P16, V16, Sp8, 64, o16);
    cyc = rd_cycle() - c0; ins = rd_instret() - i0;
    g_base_cpm = 0; report("values s16: p against V, 64 rows", cyc, ins, macs);
    g_base_cpm = (double)cyc / macs;
    c0 = rd_cycle(); i0 = rd_instret();
    for (int r = 0; r < REP; r++) pie_dots_s8(P8, V8, Sp8, 64, o8);
    cyc = rd_cycle() - c0; ins = rd_instret() - i0;
    report("values s8:  p against V, 64 rows", cyc, ins, macs);
    g_base_cpm = 0;

    heap_caps_free(K8); heap_caps_free(K16); heap_caps_free(P8); heap_caps_free(P16);
    heap_caps_free(V8); heap_caps_free(V16); heap_caps_free(o8); heap_caps_free(o16);
}

void app_main(void)
{
    printf("\nWhere the projection time goes. ESP32-P4 @ %d MHz, "
           "%dx%d int8 weights, %d-token block\n",
           CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ, ROWS, COLS, NB);

    /* mcycle is standard; minstret is not guaranteed. Prove both move before
     * believing any CPI printed below. */
    uint32_t c0 = rd_cycle(), i0 = rd_instret();
    for (volatile int i = 0; i < 1000; i++) { }
    uint32_t dc = rd_cycle() - c0, di = rd_instret() - i0;
    printf("counter check: mcycle +%" PRIu32 "  minstret +%" PRIu32 "%s\n",
           dc, di, di ? "" : "   <-- minstret DEAD, CPI below is meaningless");

    int8_t *x  = heap_caps_aligned_alloc(16, (size_t)NB * COLS + 16, MALLOC_CAP_INTERNAL);
    int8_t *Wi = heap_caps_aligned_alloc(16, (size_t)ROWS * COLS, MALLOC_CAP_INTERNAL);
    int8_t *Wp = heap_caps_aligned_alloc(16, 2u * 1024 * 1024, MALLOC_CAP_SPIRAM);
    float  *yi = heap_caps_aligned_alloc(16, (size_t)NB * LDY * sizeof(float), MALLOC_CAP_SPIRAM);
    float  *yc = heap_caps_aligned_alloc(16, (size_t)NB * ROWS * sizeof(float), MALLOC_CAP_INTERNAL);
    uint8_t *W4 = heap_caps_aligned_alloc(16, 512u * (COLS / 2), MALLOC_CAP_SPIRAM);
    float   *hbp = heap_caps_aligned_alloc(16, (size_t)NB * COLS * sizeof(float), MALLOC_CAP_SPIRAM);
    int8_t  *xq2 = heap_caps_aligned_alloc(16, (size_t)NB * COLS + 16, MALLOC_CAP_INTERNAL);
    static float sc32[ROWS], sx[NB], sx2[NB];
    if (!x || !Wi || !Wp || !yi || !yc || !W4 || !hbp || !xq2) { printf("alloc failed\n"); return; }
    for (size_t i = 0; i < 512u * (COLS / 2); i++) W4[i] = (uint8_t)rnd8();
    for (size_t i = 0; i < (size_t)NB * COLS; i++) hbp[i] = 0.001f * (float)((int)(i % 977) - 488);

    for (size_t i = 0; i < (size_t)NB * COLS + 16; i++) x[i] = rnd8();
    for (size_t i = 0; i < (size_t)ROWS * COLS; i++) Wi[i] = rnd8();
    for (size_t i = 0; i < 2u * 1024 * 1024; i++) Wp[i] = rnd8();
    for (int i = 0; i < ROWS; i++) sc32[i] = 0.001f * (float)(i + 1);
    for (int i = 0; i < NB;   i++) sx[i]   = 0.01f  * (float)(i + 1);

    printf("\n  x internal (32 KB, the engine's g_xqb), weights internal unless stated\n");
    stage_kernel(x, Wi);
    stage_epilogue("+ float epilogue, y internal, contiguous", x, Wi, yc, ROWS, sc32, sx);
    stage_epilogue("+ y in PSRAM, contiguous", x, Wi, yi, ROWS, sc32, sx);
    stage_epilogue("+ y in PSRAM, strided by LDY (the engine)", x, Wi, yi, LDY, sc32, sx);

    printf("\n  same ladder with weights streamed from PSRAM:\n");
    g_base_cpm = 0;
    stage_kernel(x, Wp);
    stage_epilogue("+ y in PSRAM, strided by LDY", x, Wp, yi, LDY, sc32, sx);

    printf("\n  the engine's actual inner loop:\n");
    g_base_cpm = 0;
    stage_kernel(x, Wi);
    stage_gemm_i4(x, W4, 512, yi, sc32, sx);
    stage_quant(hbp, xq2, sx2);

    stage_qacc_probe();

    {   /* attention working-set test: S=271 as in the long fixture */
        const int S = 271, Sp = (271 + 31) & ~31;
        int16_t *kqb = heap_caps_aligned_alloc(16, 4u * S * 64 * 2, MALLOC_CAP_SPIRAM);
        int16_t *vqb = heap_caps_aligned_alloc(16, 4u * 64 * Sp * 2, MALLOC_CAP_SPIRAM);
        int16_t *qqb = heap_caps_aligned_alloc(16, 64 * 2 + 16, MALLOC_CAP_INTERNAL);
        int16_t *pqb = heap_caps_aligned_alloc(16, (size_t)Sp * 2 + 16, MALLOC_CAP_INTERNAL);
        int32_t *iwb = heap_caps_aligned_alloc(16, 512 * 4, MALLOC_CAP_INTERNAL);
        if (kqb && vqb && qqb && pqb && iwb) {
            for (size_t i = 0; i < 4u * S * 64; i++)      kqb[i] = (int16_t)((int)(i % 8193) - 4096);
            for (size_t i = 0; i < 4u * 64 * Sp; i++)     vqb[i] = (int16_t)((int)(i % 8193) - 4096);
            for (int i = 0; i < 64 + 8; i++)              qqb[i] = (int16_t)(4096 - 128 * i);
            for (int i = 0; i < Sp + 8; i++)              pqb[i] = (int16_t)(i % 16384);
            printf("\n  attention at S=271. K is 34.7 KB per head, V is 36.9 KB.\n");
            printf("  blocked keeps only one of them live at a time; fused keeps both.\n");
            g_base_cpm = 0;
            stage_qk8(kqb, S);
            g_base_cpm = 0;
            stage_dots_s8(S);
            g_base_cpm = 0;
            stage_attn("blocked, 1 head  (~35 KB live)  <- the engine", kqb, vqb, qqb, pqb, iwb, S, Sp, 1, 1);
            double one = g_base_cpm;
            g_base_cpm = one;
            stage_attn("blocked, 2 heads (~70 KB live)  <- + head split", kqb, vqb, qqb, pqb, iwb, S, Sp, 2, 1);
            stage_attn("fused,   1 head  (71 KB live)", kqb, vqb, qqb, pqb, iwb, S, Sp, 1, 0);
            stage_attn("fused,   2 heads (143 KB live)", kqb, vqb, qqb, pqb, iwb, S, Sp, 2, 0);
            /* Same one-head case but with the kernel's own scratch in PSRAM, which
             * is where the engine's arena actually puts it. Sizes what moving it to
             * internal SRAM is worth, independent of the head-split question. */
            int16_t *qqp = heap_caps_aligned_alloc(16, 64 * 2 + 16, MALLOC_CAP_SPIRAM);
            int16_t *pqp = heap_caps_aligned_alloc(16, (size_t)Sp * 2 + 16, MALLOC_CAP_SPIRAM);
            int32_t *iwp = heap_caps_aligned_alloc(16, 512 * 4, MALLOC_CAP_SPIRAM);
            if (qqp && pqp && iwp) {
                memcpy(qqp, qqb, 64 * 2); memcpy(pqp, pqb, (size_t)Sp * 2);
                g_base_cpm = one;
                stage_attn("blocked, 1 head, scratch in PSRAM",
                           kqb, vqb, qqp, pqp, iwp, S, Sp, 1, 1);
            }
            g_base_cpm = 0;
        }
    }

    printf("\n  the rest of the projection timer, at 271 tokens:\n");
    stage_norm_rope(271, 12);

    g_base_cpm = 0;
    stage_working_set(x, Wp);

    printf("done\n");
    while (1) vTaskDelay(pdMS_TO_TICKS(10000));
}
