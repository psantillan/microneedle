/* Forward pass for Needle. See needle_engine.h for the contract.
 *
 * Model semantics this implementation must obey:
 *   1. cross-attention gets NO RoPE (DecoderBlock passes rope only to self_attn)
 *   2. inside attention: project -> split heads -> ZCRMSNorm on q,k over
 *      head_dim -> GQA pairing -> RoPE -> softmax(QK/sqrt(64))V
 *   3. RoPE is half-split (x1=dims 0..31, x2=dims 32..63), theta 10000
 *   4. norm weights are stored raw; apply (1 + w)
 *   5. residual gates are raw logits; apply sigmoid, x = res + sig(g)*sub(x)
 *   6. embedding lookup is scaled by sqrt(512) -- on both encoder and decoder
 *   7. logits = final_norm(x) @ embed_tokens^T, embedding tied and stored once
 *   8. encoder has a final_norm; decoder has its own final norm before logits
 */

#include "needle_engine.h"

#include <stdint.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------- fp16 (IEEE binary16) -> fp32, portable ---------- */
static inline float f16_to_f32(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t man  = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) { bits = sign; }
        else {                                    /* subnormal */
            exp = 127 - 15 + 1;
            while (!(man & 0x400)) { man <<= 1; exp--; }
            man &= 0x3FF;
            bits = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);  /* inf/nan */
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

/* exp() for softmax, whose argument is always <= 0 after the max subtraction.
 * Range-reduce x*log2(e) into an integer power of two and a fraction, evaluate
 * 2^f with a degree-6 polynomial, and assemble the result by writing the
 * exponent field. Worst relative error over [-87, 0] is 1.6e-5, against
 * probabilities that get quantized to 1/16384; newlib's expf is a real call on
 * this toolchain and was 4.4 s of a 17 s prefill. */
static inline float ne_expf(float x)
{
    if (x < -87.0f) return 0.f;                /* exp(-87) = 1.6e-38 */
    float t  = x * 1.44269504088896f;          /* log2(e) */
    float fi = (float)(int)t;
    if (fi > t) fi -= 1.0f;                    /* floor: t is negative here */
    const int k = (int)fi;
    const float f = t - fi;                    /* [0, 1) */
    float p = 0.0001355350f;
    p = p * f + 0.0013422634f;
    p = p * f + 0.0096180896f;
    p = p * f + 0.0555034621f;
    p = p * f + 0.2402265069f;
    p = p * f + 0.6931471805f;
    p = p * f + 1.0f;
    union { uint32_t u; float f; } pow2;
    pow2.u = (uint32_t)(k + 127) << 23;
    return p * pow2.f;
}

/* Round to nearest, ties away from zero. lrintf is a call on this toolchain and
 * this sits in the inner loop of every quantizer. */
static inline int32_t ne_rnd(float x)
{
    return (int32_t)(x + (x >= 0.f ? 0.5f : -0.5f));
}

/* ---------- weight lookup ---------- */
static int g_missing;                 /* set by find(); checked once in ne_load */
static ne_rec_t g_null_rec;           /* zeroed stand-in so callers never deref NULL */

static const ne_rec_t *find(const ne_weights_t *w, const char *name)
{
    for (uint32_t i = 0; i < w->n_recs; i++)
        if (!strcmp(w->recs[i].name, name)) return &w->recs[i];
    /* Do not abort: on the board that is a panic-reboot loop with no protocol
     * error line, and it makes ne_load's documented -1 unreachable for the
     * likeliest failure (a truncated or mismatched image with a valid header). */
    g_missing++;
    return &g_null_rec;
}

static const ne_rec_t *findf(const ne_weights_t *w, const char *fmt, int layer)
{
    char buf[96];
    snprintf(buf, sizeof buf, fmt, layer);
    return find(w, buf);
}

/* Read a 1-D fp16 tensor into out[n]. */
/* `cap` is the destination length: the .npk is a flash partition, i.e. a trust
 * boundary, and a record claiming more rows than the caller allocated would
 * otherwise write past it. */
static void load_vec(const ne_weights_t *w, const ne_rec_t *r, float *out, uint32_t cap)
{
    if (r->rows == 0) return;                  /* stand-in for a missing tensor */
    uint32_t n = r->rows < cap ? r->rows : cap;
    if (r->rows > cap) g_missing++;            /* refuse the image in ne_load */
    const uint16_t *h = (const uint16_t *)(w->blob + r->data_off);
    for (uint32_t i = 0; i < n; i++) out[i] = f16_to_f32(h[i]);
}

static float load_scalar(const ne_weights_t *w, const ne_rec_t *r)
{
    if (r->rows == 0) return 0.f;              /* stand-in for a missing tensor */
    return f16_to_f32(*(const uint16_t *)(w->blob + r->data_off));
}

void (*ne_split_for)(void (*fn)(void *, int, int), void *arg, int n) = NULL;

/* Below this many indices the cross-core round trip costs more than the work. */
#define NE_FOR_MIN 16

static void run_for(void (*fn)(void *, int, int), void *arg, int n)
{
#ifdef NE_PIE
    if (ne_split_for && n >= NE_FOR_MIN) { ne_split_for(fn, arg, n); return; }
#endif
    fn(arg, 0, n);
}

/* Loop bodies for run_for. Forward declarations of what they wrap; the
 * definitions sit with the rest of the math below. */
static void rmsnorm(const float *x, const float *w_raw, int n, float *y);
static void norm_rope_heads(float *x, const float *nw, int nheads, int pos);

typedef struct { const float *x; const float *w; float *y; } nr_arg_t;
static void rmsnorm_range(void *a, int i0, int i1)
{
    const nr_arg_t *c = (const nr_arg_t *)a;
    for (int i = i0; i < i1; i++)
        rmsnorm(c->x + (size_t)i * NE_DMODEL, c->w, NE_DMODEL,
                c->y + (size_t)i * NE_DMODEL);
}

typedef struct { float *base; const float *nw; int nheads; size_t stride; int rope; } hr_arg_t;
static void headnorm_range(void *a, int t0, int t1)
{
    const hr_arg_t *c = (const hr_arg_t *)a;
    for (int t = t0; t < t1; t++)
        norm_rope_heads(c->base + (size_t)t * c->stride, c->nw, c->nheads,
                        c->rope ? t : -1);
}

/* ---------- PIE fast path (board only) ----------
 * The scalar path below is the reference implementation and the host build.
 * With NE_PIE, gemv() quantizes the activation vector to int8 per group of 32
 * (scale = max|x|/127, round to nearest) and runs the PIE row
 * kernels; weight scales are precomputed to fp32 at load, since a per-group
 * software fp16 decode costs as much as the MACs themselves. This path carries
 * activation quantization that the JAX oracle does not, so its outputs are
 * close to but not bit-identical with the scalar path. */
/* Vector path. It is not numerically identical to the scalar path: activations
 * are quantized to int8 per group (scale = max|x|/127, round-to-nearest) before
 * the kernels see them, so logits differ from the scalar reference by a small
 * margin. Weight scales are decoded to fp32 once at load; decoding fp16 per
 * group inside the row loop costs about as much as the multiplies do.
 *
 * g_xq must stay 16-byte aligned: esp.vld.128 mis-loads silently otherwise. */
#ifdef NE_PIE
/* +16: the fused row kernel's load runs a chunk ahead of its accumulate and
 * reads one vector past the end. See engine/pie_rows.S. */
static int8_t g_xq[NE_DMODEL + 16] __attribute__((aligned(16)));
static float  g_sx[NE_DMODEL / 32];

/* The quantized activation block for the batched projections, and its scales.
 *
 * These are static rather than carved from the scratch arena so they land in
 * internal SRAM instead of PSRAM. ne_gemm_rows re-reads the whole 32 KB block
 * once per output row -- 512 times per tensor -- and from PSRAM that read is
 * what the projections are bound by, not the multiplies. */
static int8_t g_xqb[(size_t)NE_TBLK * NE_DMODEL + 16] __attribute__((aligned(16)));
static float  g_sxb[(size_t)NE_TBLK * (NE_DMODEL / NE_PACK)];
int ne_force_scalar = 0;
void (*ne_split_rows)(const ne_gemv_job_t *job, uint32_t rows) = NULL;

void (*ne_split_gemm)(const ne_gemm_job_t *job, uint32_t rows) = NULL;


/* Compute y[r0..r1) of one quantized GEMV. Row ranges are independent -- the
 * activation is already quantized and read-only -- so they may run on
 * different cores.
 *
 * Weight rows are read straight from PSRAM: they are consumed sequentially, so
 * the cache streams them and staging through internal SRAM would only touch
 * every byte twice.
 */
void ne_gemv_rows(const ne_gemv_job_t *j, uint32_t r0, uint32_t r1)
{
    int32_t gs[NE_DMODEL / 32];
    /* One scale for the whole row: single-drain kernels, no per-group tail. */
    const int single = (j->ngroups == 1);
    const int nchunk = single
        ? (int)(j->stride / (size_t)(j->i8 ? NE_PACK : NE_PACK / 2)) : 0;

    const uint8_t *base = j->qd;
    if (single) {
        const float sx0 = j->sx[0];
        for (uint32_t row = r0; row < r1; row++) {
            const uint8_t *rp = base + (size_t)row * j->stride;
            int32_t sum = j->i8 ? pie_rowsum_i8(j->xq, (const int8_t *)rp, nchunk)
                                : pie_rowsum_i4(j->xq, rp, nchunk);
            j->y[row] = j->sc32[row] * sx0 * (float)sum;
        }
    } else {
        for (uint32_t row = r0; row < r1; row++) {
            const uint8_t *rp = base + (size_t)row * j->stride;
            if (j->i8) pie_row_i8_g32(j->xq, (const int8_t *)rp, j->ngroups, gs);
            else       pie_row_i4_g32(j->xq, rp, j->ngroups, gs);
            const float *sr = j->sc32 + (size_t)row * j->ngroups;
            float acc = 0.f;
            for (uint32_t g = 0; g < j->ngroups; g++)
                acc += sr[g] * j->sx[g] * (float)gs[g];
            j->y[row] = acc;
        }
    }
}

/* Unpack one int4 weight row to int8, NPK2 nibble order: byte j of a 16-byte
 * group carries element j in the low nibble and element j+16 in the high one,
 * both two's complement. */
static void unpack_i4_row(const uint8_t *w, int nchunks, int8_t *out)
{
    for (int c = 0; c < nchunks; c++) {
        const uint8_t *src = w + c * (NE_PACK / 2);
        int8_t *dst = out + c * NE_PACK;
        for (int k = 0; k < NE_PACK / 2; k++) {
            const uint8_t b = src[k];
            dst[k]                = (int8_t)((int)(b & 0xF) - ((b & 0x08) ? 16 : 0));
            dst[k + NE_PACK / 2]  = (int8_t)((int)(b >> 4)  - ((b & 0x80) ? 16 : 0));
        }
    }
}

/* Same arithmetic as ne_gemv_rows, swept once for a block of tokens.
 *
 * int4 rows are unpacked to int8 once per row rather than once per row-token
 * pair. The int4 kernel spends three instructions unpacking nibbles for every
 * one it spends multiplying, so re-deriving the same row for each of 64 tokens
 * costs about as much as the multiplies; unpacking once and running the int8
 * kernel is the same arithmetic in the same order, exactly. */
#ifdef NE_PIE
void ne_gemm_rows(const ne_gemm_job_t *j, uint32_t r0, uint32_t r1)
{
    int32_t gs[NE_DMODEL / 32];
    int8_t  rowbuf[NE_DMODEL] __attribute__((aligned(16)));
    const int single = (j->ngroups == 1);
    const int nchunk = (int)(j->stride / (size_t)(j->i8 ? NE_PACK : NE_PACK / 2));

    for (uint32_t row = r0; row < r1; row++) {
        const uint8_t *rp = j->qd + (size_t)row * j->stride;
        const int8_t *w8;
        if (j->i8) {
            w8 = (const int8_t *)rp;
        } else {
            unpack_i4_row(rp, nchunk, rowbuf);
            w8 = rowbuf;
        }
        for (int t = 0; t < j->nb; t++) {
            const int8_t *xt = j->xq + (size_t)t * NE_DMODEL;
            const float  *st = j->sx + (size_t)t * j->ngroups;
            float acc;
            if (single) {
                acc = j->sc32[row] * st[0] * (float)pie_rowsum_i8(xt, w8, nchunk);
            } else {
                pie_row_i8_g32(xt, w8, j->ngroups, gs);
                const float *sr = j->sc32 + (size_t)row * j->ngroups;
                acc = 0.f;
                for (uint32_t g = 0; g < j->ngroups; g++)
                    acc += sr[g] * st[g] * (float)gs[g];
            }
            j->y[(size_t)t * j->ldy + row] = acc;
        }
    }
}
#endif /* NE_PIE */
#endif
/* Optional phase timing. The engine has no clock of its own; the host or
 * firmware installs one, and the counters below accumulate per-phase
 * microseconds. All NULL/zero means timing is off. */
void (*ne_critical_enter)(void) = NULL;
void (*ne_critical_exit)(void)  = NULL;
uint64_t (*ne_now_us)(void) = NULL;
uint64_t ne_us_logits, ne_us_proj;
uint64_t ne_us_enc_proj, ne_us_enc_attn, ne_us_enc_xkv, ne_us_enc_smax, ne_us_enc_core, ne_us_enc_kvq;
int ne_align_fault;
int ne_grammar_force = 1;
int ne_grammar_skips = 0;

static float *g_scales32[NE_MAX_RECS];   /* fp32 weight scales per record index */

static void ne_free_scales(void)
{
    for (int i = 0; i < NE_MAX_RECS; i++) { free(g_scales32[i]); g_scales32[i] = NULL; }
}

/* One packed int4 byte holds elem k in the low nibble and elem k + NE_PACK/2 in
 * the high nibble, both signed 4-bit. Shared by gemv's scalar path and
 * dequant_row so the split-nibble layout is spelled out exactly once; the PIE
 * assembly stays an independent implementation of the same layout. */
static inline void unpack_i4(uint8_t b, int *lo, int *hi)
{
    *lo = (int)(b & 0xF) - ((b & 0x8) ? 16 : 0);
    *hi = (int)(b >> 4) - ((b & 0x80) ? 16 : 0);
}

/* ---------- quantized GEMV ----------
 * y[r] = sum_k W[r,k] x[k]. W carries either one scale per group of 32 along
 * k, or one per row (the shipped layout); see ne_load for the check.
 * Hot loop for the projection path: decoder q/k/v/o and the vocab logits all
 * stream quantized weight rows through here. It is not the whole of decode
 * (int16 self/cross attention and the scalar tail sit outside), and on a
 * grammar-forced step the 8192-row logits GEMV is skipped entirely. Structured
 * row -> group -> lane so the PIE kernels replace the inner per-group lane
 * loop without changing callers. */
#ifdef NE_PIE
/* Quantize one activation row into g_xq/g_sx. Separate from gemv_xq because
 * decoder self-attention runs q, k and v against the same normed row: one
 * quantize serves all three, the same hoist the encoder's quant_block does for
 * a token block. Identical bytes either way -- only the work is done once. */
static void quant_act(const float *x, uint32_t group, uint32_t ngroups)
{
    for (uint32_t g = 0; g < ngroups; g++) {
        float m = 0.f;
        const float *xx = x + g * group;
        for (uint32_t k = 0; k < group; k++) {
            float a = fabsf(xx[k]);
            if (a > m) m = a;
        }
        float sxg = (m > 0.f) ? m / 127.0f : 1.0f;
        float inv = 1.0f / sxg;
        g_sx[g] = sxg;
        int8_t *q = g_xq + g * group;
        for (uint32_t k = 0; k < group; k++)
            q[k] = (int8_t)ne_rnd(xx[k] * inv);
    }
}

/* GEMV against the activation already sitting in g_xq/g_sx. */
static void gemv_xq(const ne_weights_t *w, const ne_rec_t *r, float *y)
{
    const uint32_t rows = r->rows, cols = r->cols, group = r->group;
    const uint32_t ngroups = cols / group;
    const float *sc32 = g_scales32[r - w->recs];
    ne_gemv_job_t job = {
        .qd      = w->blob + r->data_off,
        .sc32    = sc32,
        .xq      = g_xq,
        .sx      = g_sx,
        .stride  = (r->dtype == NE_DT_I8) ? cols : cols / 2,
        .ngroups = ngroups,
        .i8      = (r->dtype == NE_DT_I8),
        .y       = y,
    };
    /* Splitting costs a cross-core round trip, so only large row loops pay
     * for it; below the threshold the handoff dominates the work. */
    if (ne_critical_enter) ne_critical_enter();
    if (ne_split_rows && rows >= NE_SPLIT_MIN_ROWS) {
        ne_split_rows(&job, rows);
    } else {
        ne_gemv_rows(&job, 0, rows);
    }
    if (ne_critical_exit) ne_critical_exit();
}
#endif

static void gemv(const ne_weights_t *w, const ne_rec_t *r,
                 const float *x, float *y)
{
    const uint32_t rows = r->rows, cols = r->cols, group = r->group;
    const uint32_t ngroups = cols / group;
    const float *sc32 = g_scales32[r - w->recs];

#ifdef NE_PIE
    if (ne_force_scalar == 0) {
        quant_act(x, group, ngroups);
        gemv_xq(w, r, y);
        return;
    }
#endif

    if (r->dtype == NE_DT_I8) {
        const int8_t *q = (const int8_t *)(w->blob + r->data_off);
        for (uint32_t row = 0; row < rows; row++) {
            const int8_t *qr = q + (size_t)row * cols;
            const float *sr = sc32 + (size_t)row * ngroups;
            float acc = 0.f;
            for (uint32_t g = 0; g < ngroups; g++) {
                const int8_t *qq = qr + g * group;
                const float *xx = x + g * group;
                float dot = 0.f;
                for (uint32_t k = 0; k < group; k++) dot += (float)qq[k] * xx[k];
                acc += sr[g] * dot;
            }
            y[row] = acc;
        }
    } else {  /* NE_DT_I4: split-nibble, NE_PACK elements per 16-byte chunk */
        const uint8_t *q = (const uint8_t *)(w->blob + r->data_off);
        const uint32_t bpr = cols / 2, nchunk = cols / NE_PACK;
        const uint32_t cpg = group / NE_PACK;      /* chunks sharing one scale */
        for (uint32_t row = 0; row < rows; row++) {
            const uint8_t *qr = q + (size_t)row * bpr;
            const float *sr = sc32 + (size_t)row * ngroups;
            float acc = 0.f;
            for (uint32_t c = 0; c < nchunk; c++) {
                const uint8_t *qq = qr + c * (NE_PACK / 2);
                const float *xx = x + c * NE_PACK;
                float dot = 0.f;
                for (uint32_t k = 0; k < NE_PACK / 2; k++) {
                    int lo, hi;
                    unpack_i4(qq[k], &lo, &hi);
                    dot += (float)lo * xx[k] + (float)hi * xx[k + NE_PACK / 2];
                }
                acc += sr[c / cpg] * dot;
            }
            y[row] = acc;
        }
    }
}

/* Project a block of `nb` activation rows (hb[nb][NE_DMODEL]) through one
 * weight tensor, writing y[t * ldy + row].
 *
 * The vector build quantizes the whole block first and then sweeps the weights
 * once for all of it. Sweeping per token instead costs one full pass over the
 * tensor per token, which at 271 tokens is 1.3 GB of PSRAM traffic across the
 * encoder. Every output is still one dot product accumulated in the same
 * order, so the result does not depend on the block size. */
/* Quantize a block of activation rows to int8, per group of `group`.
 *
 * Separate from project_block because the projections that share a block share
 * its input: q, k and v all read the same normed rows, and so do the two
 * cross-attention tensors. Quantizing inside project_block did this three times
 * over identical data. Every quantized tensor in an .npk carries the same group
 * (ne_load rejects an image where they differ), so one result serves them all. */
#ifdef NE_PIE
typedef struct { const float *hb; uint32_t group; int8_t *xqb; float *sxb; } qb_arg_t;

static void quant_block_range(void *a, int t0, int t1)
{
    const qb_arg_t *c = (const qb_arg_t *)a;
    const float *hb = c->hb; const uint32_t group = c->group;
    int8_t *xqb = c->xqb; float *sxb = c->sxb;
    const uint32_t ng = NE_DMODEL / group;
    for (int t = t0; t < t1; t++) {
        const float *x = hb + (size_t)t * NE_DMODEL;
        int8_t *xq = xqb + (size_t)t * NE_DMODEL;
        float  *sx = sxb + (size_t)t * ng;
        for (uint32_t g = 0; g < ng; g++) {
            const float *xx = x + g * group;
            float m = 0.f;
            for (uint32_t k = 0; k < group; k++) {
                float a = fabsf(xx[k]);
                if (a > m) m = a;
            }
            float s = (m > 0.f) ? m / 127.0f : 1.0f, inv = 1.0f / s;
            sx[g] = s;
            for (uint32_t k = 0; k < group; k++)
                xq[g * group + k] = (int8_t)ne_rnd(xx[k] * inv);
        }
    }
}

static void quant_block(const float *hb, int nb, uint32_t group,
                        int8_t *xqb, float *sxb)
{
    qb_arg_t a = { hb, group, xqb, sxb };
    run_for(quant_block_range, &a, nb);
}
#endif

static void project_block(const ne_weights_t *w, const ne_rec_t *r,
                          const float *hb, int nb, float *y, size_t ldy,
                          int8_t *xqb, float *sxb)
{
#ifdef NE_PIE
    const uint32_t ng = r->cols / r->group;
    (void)hb;
    ne_gemm_job_t job = {
        .qd = w->blob + r->data_off, .sc32 = g_scales32[r - w->recs],
        .xq = xqb, .sx = sxb,
        .stride = (r->dtype == NE_DT_I8) ? r->cols : r->cols / 2,
        .ngroups = ng, .i8 = (r->dtype == NE_DT_I8),
        .nb = nb, .y = y, .ldy = ldy,
    };
    if (ne_critical_enter) ne_critical_enter();
    if (ne_split_gemm && r->rows >= NE_SPLIT_MIN_ROWS) ne_split_gemm(&job, r->rows);
    else ne_gemm_rows(&job, 0, r->rows);
    if (ne_critical_exit) ne_critical_exit();
#else
    (void)xqb; (void)sxb;
    for (int t = 0; t < nb; t++)
        gemv(w, r, hb + (size_t)t * NE_DMODEL, y + (size_t)t * ldy);
#endif
}

/* Dequantize one row of a 2-D tensor (embedding gather). */
static void dequant_row(const ne_weights_t *w, const ne_rec_t *r,
                        uint32_t row, float *out)
{
    const uint32_t cols = r->cols, group = r->group, ngroups = cols / group;
    const float *sr = g_scales32[r - w->recs] + (size_t)row * ngroups;
    if (r->dtype == NE_DT_I8) {
        const int8_t *qr = (const int8_t *)(w->blob + r->data_off)
                           + (size_t)row * cols;
        for (uint32_t g = 0; g < ngroups; g++) {
            float s = sr[g];
            for (uint32_t k = 0; k < group; k++)
                out[g * group + k] = s * (float)qr[g * group + k];
        }
    } else {
        const uint8_t *qr = (const uint8_t *)(w->blob + r->data_off)
                            + (size_t)row * (cols / 2);
        const uint32_t cpg = group / NE_PACK;
        for (uint32_t c = 0; c < cols / NE_PACK; c++) {
            float s = sr[c / cpg];
            for (uint32_t k = 0; k < NE_PACK / 2; k++) {
                int lo, hi;
                unpack_i4(qr[c * (NE_PACK / 2) + k], &lo, &hi);
                out[c * NE_PACK + k]                  = s * (float)lo;
                out[c * NE_PACK + k + NE_PACK / 2]    = s * (float)hi;
            }
        }
    }
}

/* ---------- primitive ops ---------- */

/* ZCRMSNorm: y = (1 + w) * x / sqrt(mean(x^2) + eps). fp32 accumulation --
 * the JAX source upcasts the mean-of-squares to f32 explicitly. */
static void rmsnorm(const float *x, const float *w_raw, int n, float *y)
{
    float ss = 0.f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / (float)n + NE_EPS);
    for (int i = 0; i < n; i++) y[i] = (1.0f + w_raw[i]) * x[i] * inv;
}

/* In-place half-split RoPE on one 64-dim head at position pos.
 *
 * The angle depends only on position and dimension -- not on the head, not on
 * the layer, not on the input -- and position is bounded by the context, so the
 * whole model uses NE_ROPE_POS * 32 distinct (cos, sin) pairs. ne_load computes
 * all of them once. Computing them in the hot path instead evaluated those same
 * 12288 pairs 1.25 million times per 271-token encode, every head repeating the
 * work of the head before it: 0.9 s of a 5 s encoder, in cosf and sinf.
 *
 * A row is [32 cos][32 sin], so one position is 256 contiguous bytes and all
 * twelve heads at that position read the same line before moving on.
 *
 * The values are the same values, so this is bit-identical to computing them
 * inline, not an approximation of it. */
#define NE_ROPE_POS ((NE_MAX_ENC > NE_MAX_GEN) ? NE_MAX_ENC : NE_MAX_GEN)
static float *g_rope_cs;                  /* [NE_ROPE_POS][NE_HEADDIM] */

static void rope(float *h, int pos)
{
    const float *cs = g_rope_cs + (size_t)pos * NE_HEADDIM;
    for (int i = 0; i < NE_HEADDIM / 2; i++) {
        const float c = cs[i], s = cs[i + NE_HEADDIM / 2];
        const float x1 = h[i], x2 = h[i + NE_HEADDIM / 2];
        h[i]                  = x1 * c - x2 * s;
        h[i + NE_HEADDIM / 2] = x2 * c + x1 * s;
    }
}

/* softmax stopped one step early: leaves exp(s - max) in s and returns 1/sum.
 * The int16 attention path folds that reciprocal into its quantization scale,
 * which saves a pass over the row. */
static float softmax_exp(float *s, int n)
{
    float m = s[0];
    for (int i = 1; i < n; i++) if (s[i] > m) m = s[i];
    float z = 0.f;
    for (int i = 0; i < n; i++) { s[i] = ne_expf(s[i] - m); z += s[i]; }
    return 1.0f / z;
}

static void softmax(float *s, int n)
{
    const float inv = softmax_exp(s, n);
    for (int i = 0; i < n; i++) s[i] *= inv;
}

static inline float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

/* ---------- per-layer weight handles, resolved once at load ---------- */
typedef struct {
    const ne_rec_t *pre_norm, *q, *k, *v, *o, *q_norm, *k_norm, *gate;
} attn_wts_t;

typedef struct {
    attn_wts_t enc[NE_ENC_LAYERS];
    attn_wts_t dec_self[NE_DEC_LAYERS];
    attn_wts_t dec_cross[NE_DEC_LAYERS];   /* pre_norm = encoder_attn_layer_norm */
    const ne_rec_t *embed, *enc_final, *dec_final;
    /* small tensors cached as floats */
    float enc_pre[NE_ENC_LAYERS][NE_DMODEL], enc_qn[NE_ENC_LAYERS][NE_HEADDIM],
          enc_kn[NE_ENC_LAYERS][NE_HEADDIM], enc_gate[NE_ENC_LAYERS];
    float ds_pre[NE_DEC_LAYERS][NE_DMODEL], ds_qn[NE_DEC_LAYERS][NE_HEADDIM],
          ds_kn[NE_DEC_LAYERS][NE_HEADDIM], ds_gate[NE_DEC_LAYERS];
    float dc_pre[NE_DEC_LAYERS][NE_DMODEL], dc_qn[NE_DEC_LAYERS][NE_HEADDIM],
          dc_kn[NE_DEC_LAYERS][NE_HEADDIM], dc_gate[NE_DEC_LAYERS];
    float enc_final_w[NE_DMODEL], dec_final_w[NE_DMODEL];
} model_wts_t;

static model_wts_t M;  /* single-model engine; the board runs exactly one */

#ifdef NE_TRACE
/* Latent-path tracing (host only; see the header note). The record writer and
 * the two globals the hooks communicate through: the decode step, set at the
 * top of ne_step, and the cross-attention layer, set only around the decoder's
 * cross attend() call so the shared attend() knows when its softmax rows are
 * the ones worth keeping. */
FILE *ne_trace_f = NULL;
static int ne_trace_step = -1;
static int ne_trace_xlayer = -1;
int ne_ablate_kind = 0, ne_ablate_layer = -1, ne_ablate_head = -1;

/* Zeroing the 64-float slice AFTER attention is exactly "this head said
 * nothing": o_proj sees zeros for that head and the gated residual carries
 * the other seven unchanged. */
static void trace_ablate(float *headcat, int rows, size_t stride, int kind, int layer)
{
    if (ne_ablate_kind != kind || ne_ablate_layer != layer) return;
    for (int t = 0; t < rows; t++)
        memset(headcat + (size_t)t * stride + (size_t)ne_ablate_head * NE_HEADDIM,
               0, NE_HEADDIM * sizeof(float));
}

static void trace_emit(const char *tag, int layer, int head, int n, const float *v)
{
    if (!ne_trace_f) return;
    char t[8] = {0};
    strncpy(t, tag, 7);
    int32_t hdr[4] = { layer, head, ne_trace_step, n };
    fwrite(t, 1, 8, ne_trace_f);
    fwrite(hdr, sizeof hdr[0], 4, ne_trace_f);
    fwrite(v, sizeof *v, (size_t)n, ne_trace_f);
}

/* Encoder states are pooled here rather than dumped whole: the analyses need
 * the query mean, the <tools> (id 5) position and the sequence mean, and a
 * 247-token prompt at 14 checkpoints is 7 MB of text nobody reads. */
static void trace_enc_pool(const float *x, int n, const int32_t *ids, int layer)
{
    if (!ne_trace_f) return;
    int tp = 0;
    while (tp < n && ids[tp] != 5) tp++;
    float mq[NE_DMODEL] = {0}, ma[NE_DMODEL] = {0};
    for (int t = 0; t < n; t++)
        for (int d = 0; d < NE_DMODEL; d++) {
            float v = x[(size_t)t * NE_DMODEL + d];
            ma[d] += v;
            if (t < tp) mq[d] += v;
        }
    for (int d = 0; d < NE_DMODEL; d++) {
        ma[d] /= (float)n;
        if (tp > 0) mq[d] /= (float)tp;
    }
    trace_emit("encq", layer, -1, NE_DMODEL, mq);
    trace_emit("enca", layer, -1, NE_DMODEL, ma);
    if (tp < n) trace_emit("enct", layer, -1, NE_DMODEL, x + (size_t)tp * NE_DMODEL);
}

static void dump_named(ne_ctx_t *ctx, FILE *f, const char *name,
                       const ne_rec_t *r, const float *vec, uint32_t vrows, uint32_t vcols)
{
    uint32_t namelen = (uint32_t)strlen(name);
    uint32_t rows = r ? r->rows : vrows, cols = r ? r->cols : vcols;
    fwrite(&namelen, 4, 1, f);
    fwrite(name, 1, namelen, f);
    fwrite(&rows, 4, 1, f);
    fwrite(&cols, 4, 1, f);
    if (r) {
        float row[NE_DMODEL];
        for (uint32_t i = 0; i < rows; i++) {
            dequant_row(&ctx->w, r, i, row);
            fwrite(row, sizeof *row, cols, f);
        }
    } else {
        fwrite(vec, sizeof *vec, (size_t)vrows * vcols, f);
    }
}

void ne_trace_dump_weights(ne_ctx_t *ctx, FILE *f)
{
    char nm[64];
    for (int L = 0; L < NE_ENC_LAYERS; L++) {
        const attn_wts_t *a = &M.enc[L];
        snprintf(nm, sizeof nm, "enc.%d.q", L); dump_named(ctx, f, nm, a->q, NULL, 0, 0);
        snprintf(nm, sizeof nm, "enc.%d.k", L); dump_named(ctx, f, nm, a->k, NULL, 0, 0);
        snprintf(nm, sizeof nm, "enc.%d.v", L); dump_named(ctx, f, nm, a->v, NULL, 0, 0);
        snprintf(nm, sizeof nm, "enc.%d.o", L); dump_named(ctx, f, nm, a->o, NULL, 0, 0);
        snprintf(nm, sizeof nm, "enc.%d.qn", L); dump_named(ctx, f, nm, NULL, M.enc_qn[L], 1, NE_HEADDIM);
        snprintf(nm, sizeof nm, "enc.%d.kn", L); dump_named(ctx, f, nm, NULL, M.enc_kn[L], 1, NE_HEADDIM);
        snprintf(nm, sizeof nm, "enc.%d.gate", L); dump_named(ctx, f, nm, NULL, &M.enc_gate[L], 1, 1);
    }
    for (int L = 0; L < NE_DEC_LAYERS; L++) {
        const attn_wts_t *s = &M.dec_self[L], *c = &M.dec_cross[L];
        snprintf(nm, sizeof nm, "dself.%d.q", L); dump_named(ctx, f, nm, s->q, NULL, 0, 0);
        snprintf(nm, sizeof nm, "dself.%d.k", L); dump_named(ctx, f, nm, s->k, NULL, 0, 0);
        snprintf(nm, sizeof nm, "dself.%d.v", L); dump_named(ctx, f, nm, s->v, NULL, 0, 0);
        snprintf(nm, sizeof nm, "dself.%d.o", L); dump_named(ctx, f, nm, s->o, NULL, 0, 0);
        snprintf(nm, sizeof nm, "dself.%d.gate", L); dump_named(ctx, f, nm, NULL, &M.ds_gate[L], 1, 1);
        snprintf(nm, sizeof nm, "dcross.%d.q", L); dump_named(ctx, f, nm, c->q, NULL, 0, 0);
        snprintf(nm, sizeof nm, "dcross.%d.k", L); dump_named(ctx, f, nm, c->k, NULL, 0, 0);
        snprintf(nm, sizeof nm, "dcross.%d.v", L); dump_named(ctx, f, nm, c->v, NULL, 0, 0);
        snprintf(nm, sizeof nm, "dcross.%d.o", L); dump_named(ctx, f, nm, c->o, NULL, 0, 0);
        snprintf(nm, sizeof nm, "dcross.%d.qn", L); dump_named(ctx, f, nm, NULL, M.dc_qn[L], 1, NE_HEADDIM);
        snprintf(nm, sizeof nm, "dcross.%d.kn", L); dump_named(ctx, f, nm, NULL, M.dc_kn[L], 1, NE_HEADDIM);
        snprintf(nm, sizeof nm, "dcross.%d.gate", L); dump_named(ctx, f, nm, NULL, &M.dc_gate[L], 1, 1);
    }
    dump_named(ctx, f, "dec_final_norm", NULL, M.dec_final_w, 1, NE_DMODEL);
}

/* Full per-position encoder state. step holds S so the reader can reshape
 * v into [S][DMODEL] without scanning the prompt. Emitted alongside the
 * pools at every layer checkpoint (tools/measure_schema_drift.py). */
static void trace_enc_full(const float *x, int n, int layer)
{
    if (!ne_trace_f) return;
    char t[8] = {0};
    strncpy(t, "encx", 7);
    int32_t hdr[4] = { layer, -1, n, n * NE_DMODEL };
    fwrite(t, 1, 8, ne_trace_f);
    fwrite(hdr, sizeof hdr[0], 4, ne_trace_f);
    fwrite(x, sizeof *x, (size_t)n * NE_DMODEL, ne_trace_f);
}

void ne_trace_dump_model(ne_ctx_t *ctx)
{
    if (!ne_trace_f) return;
    trace_emit("dfnw", -1, -1, NE_DMODEL, M.dec_final_w);
    char t[8] = {0};
    strncpy(t, "embw", 7);
    int32_t hdr[4] = { -1, -1, -1, NE_VOCAB * NE_DMODEL };
    fwrite(t, 1, 8, ne_trace_f);
    fwrite(hdr, sizeof hdr[0], 4, ne_trace_f);
    float row[NE_DMODEL];
    for (uint32_t r = 0; r < NE_VOCAB; r++) {
        dequant_row(&ctx->w, M.embed, r, row);
        fwrite(row, sizeof *row, NE_DMODEL, ne_trace_f);
    }
}
#endif

#ifdef NE_WARMSCHEMA
/* Product F=4 schema cache. fp32 only: host acceptance is byte-identical to
 * the scalar exact path; int16 mirrors would force a requantize of the spliced
 * K/V and break that gate. Board PIE can requantize from these fp32 rows at
 * splice time later without changing the cache layout. */
int ne_warm_nq, ne_warm_ns, ne_warm_ntot;
int ne_warm_row_layers, ne_warm_row_layers_full;

struct ne_schema_cache {
    int     ns;
    size_t  bytes;          /* measured payload: ids + k + v + x */
    int32_t *schema_ids;    /* [ns] cache key */
    float   *k;             /* [NE_WARM_F][ns][KVDIM] */
    float   *v;             /* [NE_WARM_F][ns][KVDIM] */
    float   *x;             /* [ns][DMODEL] residual after layer NE_WARM_F-1 */
};

/* Capture target written by hooks inside ne_encode. */
static ne_schema_cache_t *ws_cap;   /* non-NULL while capturing */
static int ws_cap_nq;

static int ws_tools_pos(const int32_t *ids, int n)
{
    for (int i = 0; i < n; i++)
        if (ids[i] == 5) return i;
    return -1;
}

static size_t ws_payload_bytes(int ns)
{
    size_t b = 0;
    b += (size_t)ns * sizeof(int32_t);
    b += (size_t)NE_WARM_F * (size_t)ns * NE_KVDIM * sizeof(float); /* k */
    b += (size_t)NE_WARM_F * (size_t)ns * NE_KVDIM * sizeof(float); /* v */
    b += (size_t)ns * NE_DMODEL * sizeof(float);                    /* x */
    return b;
}

static ne_schema_cache_t *ws_cache_alloc(int ns)
{
    ne_schema_cache_t *c = calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->ns = ns;
    c->bytes = ws_payload_bytes(ns);
    c->schema_ids = malloc((size_t)ns * sizeof(int32_t));
    c->k = malloc((size_t)NE_WARM_F * (size_t)ns * NE_KVDIM * sizeof(float));
    c->v = malloc((size_t)NE_WARM_F * (size_t)ns * NE_KVDIM * sizeof(float));
    c->x = malloc((size_t)ns * NE_DMODEL * sizeof(float));
    if (!c->schema_ids || !c->k || !c->v || !c->x) {
        ne_schema_cache_free(c);
        return NULL;
    }
    return c;
}

void ne_schema_cache_free(ne_schema_cache_t *cache)
{
    if (!cache) return;
    free(cache->schema_ids);
    free(cache->k);
    free(cache->v);
    free(cache->x);
    free(cache);
}

size_t ne_schema_cache_bytes(const ne_schema_cache_t *cache)
{
    return cache ? cache->bytes : 0;
}

int ne_schema_cache_ns(const ne_schema_cache_t *cache)
{
    return cache ? cache->ns : 0;
}

const int32_t *ne_schema_cache_key(const ne_schema_cache_t *cache, int *ns_out)
{
    if (!cache) { if (ns_out) *ns_out = 0; return NULL; }
    if (ns_out) *ns_out = cache->ns;
    return cache->schema_ids;
}

int ne_schema_cache_matches(const ne_schema_cache_t *cache,
                            const int32_t *enc_ids, int n)
{
    if (!cache || !enc_ids || n <= 0) return 0;
    int nq = ws_tools_pos(enc_ids, n);
    if (nq < 0 || nq >= n) return 0;
    int ns = n - nq;
    if (ns != cache->ns) return 0;
    return memcmp(enc_ids + nq, cache->schema_ids, (size_t)ns * sizeof(int32_t)) == 0;
}

float *ne_schema_cache_k_row(ne_schema_cache_t *cache, int layer, int row)
{
    if (!cache || layer < 0 || layer >= NE_WARM_F || row < 0 || row >= cache->ns)
        return NULL;
    return cache->k + ((size_t)layer * (size_t)cache->ns + (size_t)row) * NE_KVDIM;
}

float *ne_schema_cache_x_row(ne_schema_cache_t *cache, int row)
{
    if (!cache || row < 0 || row >= cache->ns) return NULL;
    return cache->x + (size_t)row * NE_DMODEL;
}

/* Hooks from ne_encode: only layers 0..NE_WARM_F-1 K/V and residual after
 * layer NE_WARM_F-1 (checkpoint NE_WARM_F). */
static void ws_save_kv(int layer, const float *k, const float *v, int nq, int n)
{
    if (!ws_cap || layer < 0 || layer >= NE_WARM_F) return;
    const int ns = n - nq;
    float *dk = ws_cap->k + (size_t)layer * (size_t)ws_cap->ns * NE_KVDIM;
    float *dv = ws_cap->v + (size_t)layer * (size_t)ws_cap->ns * NE_KVDIM;
    memcpy(dk, k + (size_t)nq * NE_KVDIM, (size_t)ns * NE_KVDIM * sizeof(float));
    memcpy(dv, v + (size_t)nq * NE_KVDIM, (size_t)ns * NE_KVDIM * sizeof(float));
}

static void ws_save_x_splice(const float *x, int nq, int n)
{
    if (!ws_cap) return;
    const int ns = n - nq;
    memcpy(ws_cap->x, x + (size_t)nq * NE_DMODEL, (size_t)ns * NE_DMODEL * sizeof(float));
}

static void ws_load_kv(const ne_schema_cache_t *c, int layer,
                       float *k, float *v, int nq, int ns)
{
    const float *sk = c->k + (size_t)layer * (size_t)c->ns * NE_KVDIM;
    const float *sv = c->v + (size_t)layer * (size_t)c->ns * NE_KVDIM;
    memcpy(k + (size_t)nq * NE_KVDIM, sk, (size_t)ns * NE_KVDIM * sizeof(float));
    memcpy(v + (size_t)nq * NE_KVDIM, sv, (size_t)ns * NE_KVDIM * sizeof(float));
}

static void ws_load_x(const ne_schema_cache_t *c, float *x, int nq, int ns)
{
    memcpy(x + (size_t)nq * NE_DMODEL, c->x, (size_t)ns * NE_DMODEL * sizeof(float));
}
#endif

int ne_load(ne_ctx_t *ctx, const uint8_t *npk, size_t len)
{
    if (len < 16 || *(const uint32_t *)npk != 0x324B504Eu) return -1;  /* NPK2: split-nibble int4 */
    uint32_t n = *(const uint32_t *)(npk + 4);
    uint64_t blob_bytes = *(const uint64_t *)(npk + 8);
    /* the scale cache is indexed by record number; more records than slots must
     * be a load error, not a silent cap that NULL-derefs in gemv later */
    if (n > NE_MAX_RECS) return -1;
    if (len < 16 + (size_t)n * sizeof(ne_rec_t) + blob_bytes) return -1;
    ctx->w.n_recs = n;
    ctx->w.recs = (const ne_rec_t *)(npk + 16);
    ctx->w.blob = npk + 16 + (size_t)n * sizeof(ne_rec_t);
    const ne_weights_t *w = &ctx->w;

    free(g_rope_cs);
    g_rope_cs = malloc((size_t)NE_ROPE_POS * NE_HEADDIM * sizeof(float));
    if (!g_rope_cs) return -1;
    for (int i = 0; i < NE_HEADDIM / 2; i++) {
        const float freq = powf(NE_ROPE_THETA, -(float)(2 * i) / (float)NE_HEADDIM);
        for (int p = 0; p < NE_ROPE_POS; p++) {
            const float a = (float)p * freq;
            g_rope_cs[(size_t)p * NE_HEADDIM + i]                  = cosf(a);
            g_rope_cs[(size_t)p * NE_HEADDIM + i + NE_HEADDIM / 2] = sinf(a);
        }
    }

    /* fp32 weight scales, decoded once -- the branchy software fp16 decode
     * must never sit in the per-row hot loop */
    ne_free_scales();
    uint32_t dmodel_group = 0;
    for (uint32_t i = 0; i < n; i++) {
        const ne_rec_t *r = &w->recs[i];
        /* gemv and dequant_row read any non-int8 matrix as packed int4, and
         * load_vec reads every 1-D tensor as fp16, so an unhandled dtype would
         * be mis-decoded into plausible numbers instead of failing. NE_DT_F32
         * is declared by the format but never packed. */
        if (r->cols == 0) {
            if (r->dtype != NE_DT_F16) return -1;
        } else if (r->dtype != NE_DT_I8 && r->dtype != NE_DT_I4) {
            return -1;
        }
        if (r->cols == 0 || r->scale_off == 0xFFFFFFFFFFFFFFFFull) continue;
        /* Only two scale layouts have kernels: group-32 (per-group drain) and
         * one-scale-per-row (single drain). Anything else would run the PIE
         * path over part of each row and return a plausible wrong answer, so
         * refuse the image instead of trusting it. */
        if (r->group == 0 || r->cols % r->group != 0) return -1;
        if (r->group != NE_PACK && r->group != r->cols) return -1;
        if (r->cols / r->group > NE_DMODEL / NE_PACK) return -1;
        /* One activation quantization is shared by every projection reading the
         * same block, which is only sound if they all group the same way. The
         * packer emits one group for the whole artifact; refuse an image that
         * does not, rather than silently feeding one tensor another's scales. */
        if (r->cols == NE_DMODEL) {
            if (dmodel_group == 0) dmodel_group = r->group;
            else if (r->group != dmodel_group) return -1;
        }
        uint32_t ng = r->cols / r->group;
        float *f = malloc((size_t)r->rows * ng * sizeof(float));
        if (!f) { ne_free_scales(); return -1; }
        const uint16_t *h = (const uint16_t *)(w->blob + r->scale_off);
        for (size_t j = 0; j < (size_t)r->rows * ng; j++) f[j] = f16_to_f32(h[j]);
        g_scales32[i] = f;
    }

    g_missing = 0;
    M.embed     = find(w, "embed_tokens");
    M.enc_final = find(w, "encoder.final_norm.weight");
    M.dec_final = find(w, "decoder.norm.weight");
    load_vec(w, M.enc_final, M.enc_final_w, NE_DMODEL);
    load_vec(w, M.dec_final, M.dec_final_w, NE_DMODEL);

    for (int i = 0; i < NE_ENC_LAYERS; i++) {
        attn_wts_t *a = &M.enc[i];
        a->pre_norm = findf(w, "encoder.layers.%d.input_layernorm.weight", i);
        a->q = findf(w, "encoder.layers.%d.self_attn.q_proj.weight", i);
        a->k = findf(w, "encoder.layers.%d.self_attn.k_proj.weight", i);
        a->v = findf(w, "encoder.layers.%d.self_attn.v_proj.weight", i);
        a->o = findf(w, "encoder.layers.%d.self_attn.out_proj.weight", i);
        a->q_norm = findf(w, "encoder.layers.%d.self_attn.q_norm.weight", i);
        a->k_norm = findf(w, "encoder.layers.%d.self_attn.k_norm.weight", i);
        a->gate = findf(w, "encoder.layers.%d.attn_gate", i);
        load_vec(w, a->pre_norm, M.enc_pre[i], NE_DMODEL);
        load_vec(w, a->q_norm, M.enc_qn[i], NE_HEADDIM);
        load_vec(w, a->k_norm, M.enc_kn[i], NE_HEADDIM);
        M.enc_gate[i] = sigmoidf(load_scalar(w, a->gate));
    }
    for (int i = 0; i < NE_DEC_LAYERS; i++) {
        attn_wts_t *s = &M.dec_self[i], *c = &M.dec_cross[i];
        s->pre_norm = findf(w, "decoder.layers.%d.input_layernorm.weight", i);
        s->q = findf(w, "decoder.layers.%d.self_attn.q_proj.weight", i);
        s->k = findf(w, "decoder.layers.%d.self_attn.k_proj.weight", i);
        s->v = findf(w, "decoder.layers.%d.self_attn.v_proj.weight", i);
        s->o = findf(w, "decoder.layers.%d.self_attn.out_proj.weight", i);
        s->q_norm = findf(w, "decoder.layers.%d.self_attn.q_norm.weight", i);
        s->k_norm = findf(w, "decoder.layers.%d.self_attn.k_norm.weight", i);
        s->gate = findf(w, "decoder.layers.%d.self_attn_gate", i);
        load_vec(w, s->pre_norm, M.ds_pre[i], NE_DMODEL);
        load_vec(w, s->q_norm, M.ds_qn[i], NE_HEADDIM);
        load_vec(w, s->k_norm, M.ds_kn[i], NE_HEADDIM);
        M.ds_gate[i] = sigmoidf(load_scalar(w, s->gate));

        c->pre_norm = findf(w, "decoder.layers.%d.encoder_attn_layer_norm.weight", i);
        c->q = findf(w, "decoder.layers.%d.encoder_attn.q_proj.weight", i);
        c->k = findf(w, "decoder.layers.%d.encoder_attn.k_proj.weight", i);
        c->v = findf(w, "decoder.layers.%d.encoder_attn.v_proj.weight", i);
        c->o = findf(w, "decoder.layers.%d.encoder_attn.out_proj.weight", i);
        c->q_norm = findf(w, "decoder.layers.%d.encoder_attn.q_norm.weight", i);
        c->k_norm = findf(w, "decoder.layers.%d.encoder_attn.k_norm.weight", i);
        c->gate = findf(w, "decoder.layers.%d.cross_attn_gate", i);
        load_vec(w, c->pre_norm, M.dc_pre[i], NE_DMODEL);
        load_vec(w, c->q_norm, M.dc_qn[i], NE_HEADDIM);
        load_vec(w, c->k_norm, M.dc_kn[i], NE_HEADDIM);
        M.dc_gate[i] = sigmoidf(load_scalar(w, c->gate));
    }
    if (g_missing) { ne_free_scales(); return -1; }
    return 0;
}

int ne_alloc(ne_ctx_t *ctx)
{
    size_t S = NE_MAX_ENC, T = NE_MAX_GEN;
    ctx->enc_x   = malloc(S * NE_DMODEL * sizeof(float));
    ctx->cross_k = malloc((size_t)NE_DEC_LAYERS * S * NE_KVDIM * sizeof(float));
    ctx->cross_v = malloc((size_t)NE_DEC_LAYERS * S * NE_KVDIM * sizeof(float));
    ctx->self_k  = malloc((size_t)NE_DEC_LAYERS * T * NE_KVDIM * sizeof(float));
    ctx->self_v  = malloc((size_t)NE_DEC_LAYERS * T * NE_KVDIM * sizeof(float));
#ifdef NE_PIE
    /* Decode reads the cross-attention K/V once per generated token, for every
     * token of the prompt: at long prompts that re-read dominates the
     * attention share of decode (board-certified decode is 94-130 ms/token
     * overall; the absolute split was last measured before int16 attention).
     * The encoder quantizes these ONCE and decode runs the same int16 kernels.
     * Sized for NE_MAX_ENC; the vq row stride is the maximal Sp so any actual
     * prompt length packs inside it. Raw pointers are kept for free(); the
     * ctx pointers are rounded up to the 16 bytes esp.vld.128 requires. */
    {
        const size_t vSp = ((size_t)NE_MAX_ENC + 31) & ~(size_t)31;
        ctx->cross_kq_raw = malloc((size_t)NE_DEC_LAYERS * NE_KVHEADS * S * NE_HEADDIM * sizeof(ne_aq_t) + 16);
        ctx->cross_ksc = malloc((size_t)NE_DEC_LAYERS * NE_KVHEADS * S * sizeof(float));
        ctx->cross_vq_raw = malloc((size_t)NE_DEC_LAYERS * NE_KVHEADS * NE_HEADDIM * vSp * sizeof(ne_aq_t) + 16);
        ctx->cross_vsc = malloc((size_t)NE_DEC_LAYERS * NE_KVDIM * sizeof(float));
        if (!ctx->cross_kq_raw || !ctx->cross_ksc || !ctx->cross_vq_raw || !ctx->cross_vsc)
            return -1;
        ctx->cross_kq = (ne_aq_t *)(((uintptr_t)ctx->cross_kq_raw + 15u) & ~(uintptr_t)15);
        ctx->cross_vq = (ne_aq_t *)(((uintptr_t)ctx->cross_vq_raw + 15u) & ~(uintptr_t)15);
    }
    /* Self-attention int16 mirrors. Capacity is NE_MAX_GEN (<=64), so Sp is
     * fixed at the rounded max and stored on the context -- pack and read
     * must agree on it (the cross-attn stride bugs were pack/read
     * disagreements). K layout [kv][T][64]; V layout [kv][64][Sp] with
     * per-dim scales like quant_vT. Aligned by construction for esp.vld.128;
     * pie_dots_s16 over-reads one vector of the A operand -- callers leave
     * slack on the query/prob scratch, not on these mirrors. */
    {
        const size_t sSp = ((size_t)NE_MAX_GEN + 31) & ~(size_t)31;
        ctx->self_sp = (int)sSp;
        ctx->self_kq_raw = malloc((size_t)NE_DEC_LAYERS * NE_KVHEADS * T * NE_HEADDIM * sizeof(int16_t) + 16);
        ctx->self_ksc = malloc((size_t)NE_DEC_LAYERS * NE_KVHEADS * T * sizeof(float));
        ctx->self_vq_raw = malloc((size_t)NE_DEC_LAYERS * NE_KVHEADS * NE_HEADDIM * sSp * sizeof(int16_t) + 16);
        ctx->self_vsc = malloc((size_t)NE_DEC_LAYERS * NE_KVDIM * sizeof(float));
        if (!ctx->self_kq_raw || !ctx->self_ksc || !ctx->self_vq_raw || !ctx->self_vsc)
            return -1;
        ctx->self_kq = (int16_t *)(((uintptr_t)ctx->self_kq_raw + 15u) & ~(uintptr_t)15);
        ctx->self_vq = (int16_t *)(((uintptr_t)ctx->self_vq_raw + 15u) & ~(uintptr_t)15);
    }
#endif
    /* One arena, partitioned in ne_encode: q[S][512] + k[S][256] + v[S][256] +
     * attn_out[S][512] + per-head kT + per-head vT + the NE_QBLK x S score
     * block, plus a few spare rows for the norm row. ne_step reuses the front
     * of it as its score row. */
    /* One arena, partitioned at the top of ne_encode. Per token: q, k, v, the
     * attention output, per-head kT and vT, and one score-block column. Fixed:
     * a norm row, the NE_TBLK projection buffers, and slack. Keep this in step
     * with that partition. */
    ctx->scratch = malloc(
        ((size_t)S * (NE_DMODEL + NE_KVDIM + NE_KVDIM + NE_DMODEL
                      + NE_KVDIM + NE_KVDIM + NE_QBLK)
         + NE_DMODEL                                   /* h            */
         + (size_t)NE_TBLK * NE_DMODEL                 /* oub          */
         + (size_t)NE_TBLK * NE_DMODEL                 /* hb           */
         /* int16 attention: kq and vq alias the kT/vT region, the rest is
          * additional. Sized for S even on the scalar build so one number
          * covers both. */
         + (size_t)NE_KVHEADS * S + NE_KVDIM           /* ksc, vsc          */
         + 2 * (S                                      /* sc                */
                + S                                    /* iw, int32         */
                + (NE_HEADDIM + S) / 2 + 16)           /* qq, pq, int16     */
         + (size_t)NE_KVHEADS * S * NE_HEADDIM         /* kq + vq, int16    */
         + 8 * NE_DMODEL                               /* slack             */
         + 16) * sizeof(float));                       /* base 64-alignment */
    ctx->logits  = malloc(NE_VOCAB * sizeof(float));
    if (!ctx->enc_x || !ctx->cross_k || !ctx->cross_v || !ctx->self_k ||
        !ctx->self_v || !ctx->scratch || !ctx->logits) return -1;
    return 0;
}

void ne_free(ne_ctx_t *ctx)
{
    free(ctx->enc_x); free(ctx->cross_k); free(ctx->cross_v);
    free(ctx->self_k); free(ctx->self_v); free(ctx->scratch); free(ctx->logits);
    free(ctx->cross_kq_raw); free(ctx->cross_ksc); free(ctx->cross_vq_raw); free(ctx->cross_vsc);
    free(ctx->self_kq_raw); free(ctx->self_ksc); free(ctx->self_vq_raw); free(ctx->self_vsc);
    free(g_rope_cs); g_rope_cs = NULL;
    ne_free_scales();
}

/* Attention core shared by encoder and decoder-self/cross paths.
 * q: [NE_HEADS][64] after norm+rope. keys/vals: [n][NE_KVDIM].
 * Query head h pairs with kv head h/2. Result accumulated into out[NE_DMODEL]. */
static void attend(const float *q, const float *keys, const float *vals,
                   int n, float *scores, float *out)
{
    const float inv_scale = 1.0f / sqrtf((float)NE_HEADDIM);
    for (int h = 0; h < NE_HEADS; h++) {
        const float *qh = q + h * NE_HEADDIM;
        int kv = h / (NE_HEADS / NE_KVHEADS);
        for (int t = 0; t < n; t++) {
            const float *kh = keys + (size_t)t * NE_KVDIM + kv * NE_HEADDIM;
            float dot = 0.f;
            for (int d = 0; d < NE_HEADDIM; d++) dot += qh[d] * kh[d];
            scores[t] = dot * inv_scale;
        }
        softmax(scores, n);
#ifdef NE_TRACE
        if (ne_trace_xlayer >= 0) trace_emit("xprob", ne_trace_xlayer, h, n, scores);
#endif
        float *oh = out + h * NE_HEADDIM;
        for (int d = 0; d < NE_HEADDIM; d++) oh[d] = 0.f;
        for (int t = 0; t < n; t++) {
            const float *vh = vals + (size_t)t * NE_KVDIM + kv * NE_HEADDIM;
            float a = scores[t];
            for (int d = 0; d < NE_HEADDIM; d++) oh[d] += a * vh[d];
        }
    }
}

/* Blocked whole-sequence attention: the encoder form of attend(), for the case
 * where every query in the sequence is available at once.
 *
 * Two properties keep the PSRAM traffic linear in S per block:
 *   1. K and V arrive transposed to per-head contiguous [kvhead][S][64], so the
 *      inner loops walk memory in a straight line instead of taking 64 useful
 *      floats out of each 256-float row.
 *   2. Queries run in blocks of NE_QBLK, so K and V are read once per block
 *      rather than once per query.
 *
 * The arithmetic is identical to attend(), and so is the accumulation order:
 * each dot accumulates over d ascending, and each context sum accumulates over
 * j ascending. That order is what makes the two interchangeable, so it must be
 * preserved.
 */
static void attend_seq(const float *q, const float *kT, const float *vT,
                       int S, float *sc, float *out)
{
    const float inv_scale = 1.0f / sqrtf((float)NE_HEADDIM);
    for (int h = 0; h < NE_HEADS; h++) {
        const int kv = h / (NE_HEADS / NE_KVHEADS);
        const float *K = kT + (size_t)kv * S * NE_HEADDIM;
        const float *V = vT + (size_t)kv * S * NE_HEADDIM;

        for (int tb = 0; tb < S; tb += NE_QBLK) {
            const int nb = (S - tb < NE_QBLK) ? S - tb : NE_QBLK;

            for (int j = 0; j < S; j++) {
                const float *kj = K + (size_t)j * NE_HEADDIM;
                for (int i = 0; i < nb; i++) {
                    const float *qh = q + (size_t)(tb + i) * NE_DMODEL + h * NE_HEADDIM;
                    float dot = 0.f;
                    for (int d = 0; d < NE_HEADDIM; d++) dot += qh[d] * kj[d];
                    sc[(size_t)i * S + j] = dot * inv_scale;
                }
            }
            for (int i = 0; i < nb; i++) softmax(sc + (size_t)i * S, S);

            for (int i = 0; i < nb; i++) {
                float *oh = out + (size_t)(tb + i) * NE_DMODEL + h * NE_HEADDIM;
                for (int d = 0; d < NE_HEADDIM; d++) oh[d] = 0.f;
            }
            for (int j = 0; j < S; j++) {
                const float *vj = V + (size_t)j * NE_HEADDIM;
                for (int i = 0; i < nb; i++) {
                    const float a = sc[(size_t)i * S + j];
                    float *oh = out + (size_t)(tb + i) * NE_DMODEL + h * NE_HEADDIM;
                    for (int d = 0; d < NE_HEADDIM; d++) oh[d] += a * vj[d];
                }
            }
        }
    }
}

#ifdef NE_WARMSCHEMA
/* Same arithmetic as attend_seq, but only writes output rows [t0, t1).
 * Queries still attend over the full key sequence of length S. Used by the
 * warm-schema path, which freezes schema rows and recomputes only the query. */
static void attend_seq_range(const float *q, const float *kT, const float *vT,
                             int S, int t0, int t1, float *sc, float *out)
{
    const float inv_scale = 1.0f / sqrtf((float)NE_HEADDIM);
    for (int h = 0; h < NE_HEADS; h++) {
        const int kv = h / (NE_HEADS / NE_KVHEADS);
        const float *K = kT + (size_t)kv * S * NE_HEADDIM;
        const float *V = vT + (size_t)kv * S * NE_HEADDIM;

        for (int tb = t0; tb < t1; tb += NE_QBLK) {
            const int nb = (t1 - tb < NE_QBLK) ? t1 - tb : NE_QBLK;

            for (int j = 0; j < S; j++) {
                const float *kj = K + (size_t)j * NE_HEADDIM;
                for (int i = 0; i < nb; i++) {
                    const float *qh = q + (size_t)(tb + i) * NE_DMODEL + h * NE_HEADDIM;
                    float dot = 0.f;
                    for (int d = 0; d < NE_HEADDIM; d++) dot += qh[d] * kj[d];
                    sc[(size_t)i * S + j] = dot * inv_scale;
                }
            }
            for (int i = 0; i < nb; i++) softmax(sc + (size_t)i * S, S);

            for (int i = 0; i < nb; i++) {
                float *oh = out + (size_t)(tb + i) * NE_DMODEL + h * NE_HEADDIM;
                for (int d = 0; d < NE_HEADDIM; d++) oh[d] = 0.f;
            }
            for (int j = 0; j < S; j++) {
                const float *vj = V + (size_t)j * NE_HEADDIM;
                for (int i = 0; i < nb; i++) {
                    const float a = sc[(size_t)i * S + j];
                    float *oh = out + (size_t)(tb + i) * NE_DMODEL + h * NE_HEADDIM;
                    for (int d = 0; d < NE_HEADDIM; d++) oh[d] += a * vj[d];
                }
            }
        }
    }
}
#endif

#ifdef NE_PIE
/* ---------- integer encoder attention (int16 default; int8 under NE_ATTN_I8) ----------
 *
 * attend_seq() above is fp32 scalar and, at 271 tokens, it is 80% of prefill:
 * 0.9 G MACs at about 12 cycles each. PIE has no floating-point multiply, so
 * going faster means doing attention in integers, and this path takes that
 * 30 s down to 6 s with byte-identical output on every fixture.
 *
 * Default is int16 (NE_AQ=4096, NE_PQ=16384): byte-identical to fp32 on every
 * fixture. NE_ATTN_I8 switches the same scheme to signed-int8 ranges AND to
 * int8 storage + the pie_dots_s8 kernel (ne_aq_t / ne_dots below): half the
 * K/V mirror memory, twice the lanes per vector. Accuracy was measured first
 * with int8 ranges in int16 storage (docs/notebook/RESULTS-INT8.md: fixtures 8/8, margins
 * ~12 vs worst shift -0.66). Decoder cross-attention shares these helpers,
 * so it follows.
 *
 * Ranges are chosen so every accumulator provably fits int32 and the kernel can
 * drain with a zero shift (see engine/pie_attn.S):
 *   q, k, v      symmetric per-row, +-NE_AQ
 *   probabilities  0..NE_PQ, and they sum to one, which is what bounds the
 *                  value sum at NE_PQ * NE_AQ rather than S * NE_PQ * NE_AQ
 *
 * Overflow bound (widest int32 sum the dots kernel can see):
 *   scores : HEADDIM * AQ * AQ
 *            int16: 64 * 4096 * 4096 = 1,073,741,824
 *            int8:  64 *  127 *  127 =     1,032,256
 *   values : PQ * AQ  (probs sum ≈ PQ after quantize; not S * PQ * AQ)
 *            int16: 16384 * 4096 = 67,108,864
 *            int8:    127 *  127 =     16,129
 * Both fits of either range leave headroom under INT32_MAX (2,147,483,647).
 *
 * V arrives transposed a second time, to [kvhead][dim][S], because the context
 * sum reduces over keys: that layout turns it into 64 contiguous dot products
 * per query, the same shape as the score pass. */
#ifdef NE_ATTN_I8
#define NE_AQ  127    /* signed int8 max; symmetric per-row q/k/v */
#define NE_PQ  127    /* non-neg probs in the same signed-int8 positive range */
/* pie_dots_s8 moves 16 lanes per vector and unrolls 4x, so its len must be a
 * multiple of 64; Sp rounds up to 64 (capacity strides use the maximal Sp,
 * (NE_MAX_ENC+31)&~31 = 384, which is already a multiple of 64). Storage is
 * ne_aq_t = int8_t: the K/V mirrors and quantized scratch halve. */
#define ne_dots     pie_dots_s8
#define NE_SPALIGN  63
/* Per-prompt Sp rounds by 63 here, but the mirror capacities and layer
 * strides in ne_alloc/ne_encode still round by 31. They agree only while
 * NE_MAX_ENC is a multiple of 64; anything else is a silent heap overrun
 * plus a pack/read stride disagreement -- the historical failure class. */
_Static_assert(NE_MAX_ENC % 64 == 0, "31/63 rounding split needs NE_MAX_ENC % 64 == 0");
#else
#define NE_AQ  4096
#define NE_PQ  16384
#define ne_dots     pie_dots_s16
#define NE_SPALIGN  31
#endif

/* Quantize n floats, stride apart, to +-NE_AQ. Returns the dequant scale. */
static float quant_s16(const float *x, int n, size_t stride, ne_aq_t *out)
{
    float amax = 0.f;
    for (int i = 0; i < n; i++) {
        float a = fabsf(x[(size_t)i * stride]);
        if (a > amax) amax = a;
    }
    if (amax == 0.f) {
        for (int i = 0; i < n; i++) out[i] = 0;
        return 0.f;
    }
    const float inv = (float)NE_AQ / amax;
    for (int i = 0; i < n; i++)
        out[i] = (ne_aq_t)ne_rnd(x[(size_t)i * stride] * inv);
    return amax / (float)NE_AQ;
}

/* k[S][NE_KVDIM] -> kq[NE_KVHEADS][S][NE_HEADDIM], one scale per key row. */
typedef struct { const float *k; int S; ne_aq_t *kq; float *ksc; } kT_arg_t;

static void quant_kT_range(void *a, int r0, int r1)
{
    const kT_arg_t *c = (const kT_arg_t *)a;
    for (int r = r0; r < r1; r++) {
        const int hh = r / c->S, t = r % c->S;
        c->ksc[r] = quant_s16(c->k + (size_t)t * NE_KVDIM + hh * NE_HEADDIM,
                              NE_HEADDIM, 1, c->kq + (size_t)r * NE_HEADDIM);
    }
}

static void quant_kT(const float *k, int S, ne_aq_t *kq, float *ksc)
{
    kT_arg_t a = { k, S, kq, ksc };
    run_for(quant_kT_range, &a, NE_KVHEADS * S);
}

/* v[S][NE_KVDIM] -> vq[NE_KVHEADS][NE_HEADDIM][Sp], one scale per dim.
 *
 * This is a transpose, and doing it a column at a time -- which is what calling
 * quant_s16 with a stride amounts to -- walks v with a 1024-byte stride and
 * touches a fresh cache line for every element, twice over, once to find the
 * maximum and once to quantize. At 271 tokens that was 997 ms of a 8.3 s
 * prefill.
 *
 * Instead: one sequential pass for the maxima, then a tiled pass for the
 * transpose. The tile is chosen so both sides stay resident -- 32 consecutive
 * floats read per row is two cache lines, and the 32 destination rows it feeds
 * are 32 lines that stay live for the whole tile.
 *
 * Same maxima over the same sets and the same rounding, so the output is
 * unchanged. */
#define NE_VT_TILE 32

typedef struct { const float *v; int S, Sp; ne_aq_t *vq; float *vsc; } vT_arg_t;

static void quant_vT_range(void *a, int i0, int i1)
{
    const vT_arg_t *c = (const vT_arg_t *)a;
    const float *v = c->v; const int S = c->S, Sp = c->Sp;
    float amax[NE_KVDIM], inv[NE_KVDIM];

    for (int i = i0; i < i1; i++) amax[i] = 0.f;
    for (int t = 0; t < S; t++) {
        const float *row = v + (size_t)t * NE_KVDIM;
        for (int i = i0; i < i1; i++) {
            const float a2 = fabsf(row[i]);
            if (a2 > amax[i]) amax[i] = a2;
        }
    }
    for (int i = i0; i < i1; i++) {
        c->vsc[i] = (amax[i] > 0.f) ? amax[i] / (float)NE_AQ : 0.f;
        inv[i]    = (amax[i] > 0.f) ? (float)NE_AQ / amax[i] : 0.f;
    }

    for (int ib = i0; ib < i1; ib += NE_VT_TILE) {
        const int ie = (i1 - ib < NE_VT_TILE) ? i1 : ib + NE_VT_TILE;
        for (int t0 = 0; t0 < S; t0 += NE_VT_TILE) {
            const int t1 = (S - t0 < NE_VT_TILE) ? S : t0 + NE_VT_TILE;
            for (int t = t0; t < t1; t++) {
                const float *row = v + (size_t)t * NE_KVDIM;
                for (int i = ib; i < ie; i++)
                    c->vq[(size_t)i * Sp + t] = (ne_aq_t)ne_rnd(row[i] * inv[i]);
            }
        }
    }
    for (int i = i0; i < i1; i++)                  /* padded lanes contribute 0 */
        for (int j = S; j < Sp; j++) c->vq[(size_t)i * Sp + j] = 0;
}

static void quant_vT(const float *v, int S, int Sp, ne_aq_t *vq, float *vsc)
{
    vT_arg_t a = { v, S, Sp, vq, vsc };
    run_for(quant_vT_range, &a, NE_KVDIM);
}

/* Same shape as attend_seq(), integer inner loops. Blocks of NE_QBLK queries
 * share one sweep of K and V; the scheduler stays suspended for a block, which
 * is the granularity the PIE context-save bug forces (see fw_main.c). */
/* The head interleave below is a bijection on [0, NE_HEADS) only for an even
 * head count, and its point is that heads 2m and 2m+1 share kv head m. With an
 * odd count or a different pairing the split would silently compute some heads
 * twice and others never, leaving those output slices holding whatever the
 * previous layer wrote. No fault, no assert, just wrong logits. */
_Static_assert(NE_HEADS % 2 == 0 && NE_HEADS / NE_KVHEADS == 2,
               "the head interleave in ne_attn_heads assumes an even head count "
               "and two query heads per kv head; change it with them");

/* Decoder self-attention keeps int16 STORAGE and pie_dots_s16 in both flag
 * states -- there is no s8 self-attention kernel and the cache is int16_t on
 * the context -- but its quantization RANGES follow the flag (NE_AQ/NE_PQ).
 * That matches the measured configuration exactly: the int8 battery ran with
 * the flag's 127 ranges reaching the self path through the shared helpers
 * (8/8 fixtures, 35/36 battery), and re-widening self to 4096/16384 under
 * the flag was tried and flipped the marginal no_matching_tool fixture.
 * When the flag retypes the shared quantizers to int8, these twins keep the
 * same arithmetic writing int16; without the flag they ARE the shared
 * helpers, so the default build's call chain is unchanged. */
#ifdef NE_ATTN_I8
#define NE_SELF_AQ NE_AQ
#define NE_SELF_PQ NE_PQ
static float quant_row_i16(const float *x, int n, size_t stride, int16_t *out)
{
    float amax = 0.f;
    for (int i = 0; i < n; i++) {
        float a = fabsf(x[(size_t)i * stride]);
        if (a > amax) amax = a;
    }
    if (amax == 0.f) {
        for (int i = 0; i < n; i++) out[i] = 0;
        return 0.f;
    }
    const float inv = (float)NE_SELF_AQ / amax;
    for (int i = 0; i < n; i++)
        out[i] = (int16_t)ne_rnd(x[(size_t)i * stride] * inv);
    return amax / (float)NE_SELF_AQ;
}
/* Serial mirror of quant_vT_range's arithmetic (same maxima sets, same
 * rounding), int16/4096 pinned. The self path quantizes S <= 64 rows per
 * step, far below NE_FOR_MIN's split threshold, so serial loses nothing. */
static void quant_vT_i16(const float *v, int S, int Sp, int16_t *vq, float *vsc)
{
    for (int i = 0; i < NE_KVDIM; i++) {
        float amax = 0.f;
        for (int t = 0; t < S; t++) {
            const float a = fabsf(v[(size_t)t * NE_KVDIM + i]);
            if (a > amax) amax = a;
        }
        vsc[i] = (amax > 0.f) ? amax / (float)NE_SELF_AQ : 0.f;
        const float inv = (amax > 0.f) ? (float)NE_SELF_AQ / amax : 0.f;
        for (int t = 0; t < S; t++)
            vq[(size_t)i * Sp + t] = (int16_t)ne_rnd(v[(size_t)t * NE_KVDIM + i] * inv);
        for (int j = S; j < Sp; j++) vq[(size_t)i * Sp + j] = 0;
    }
}
#else
#define NE_SELF_PQ NE_PQ
#define quant_row_i16 quant_s16
#define quant_vT_i16  quant_vT
#endif

/* Decode-side self-attention for one query row over the int16 self-KV
 * mirrors. int16 storage and pie_dots_s16 in both flag states; quantization
 * ranges follow NE_SELF_AQ/NE_SELF_PQ (above). S is the growing cache length
 * (<= NE_MAX_GEN) and Sp is the fixed pack stride stored on the context.
 *
 * Accumulator bounds at the widest (int16) ranges -- the int8 ranges only
 * shrink them (drain shift stays zero; see pie_attn.S):
 *   scores:  max |sum_d q_d k_d| = NE_HEADDIM * 4096 * 4096
 *            = 64 * 4096 * 4096 = 1,073,741,824 < 2^31
 *   values:  probabilities are non-negative and sum to one, quantized to
 *            0..16384, so max |sum_t p_t v_t| = 16384 * 4096
 *            = 67,108,864 < 2^31
 *   (S <= 64 is no tighter than the encoder's head-dim bound on scores;
 *    the value bound does not grow with S because of the unit-sum.)
 *
 * The caller holds the scheduler guard (PIE context-save bug; see fw_main.c).
 */
static void attend_self_q16(ne_ctx_t *ctx, int L, int S, const float *q,
                            float *out)
{
    static float   sq_sc[NE_MAX_GEN];
    static int32_t sq_iw[NE_MAX_GEN] __attribute__((aligned(16)));
    static int16_t sq_qq[NE_HEADDIM + 8] __attribute__((aligned(16)));
    static int16_t sq_pq[NE_MAX_GEN + 8] __attribute__((aligned(16)));

    const int Sp = ctx->self_sp;   /* fixed at alloc; pack and read agree */
    const int T  = NE_MAX_GEN;
    const float inv_scale = 1.0f / sqrtf((float)NE_HEADDIM);

    /* K: each key row was quantized once when appended (see ne_step), laid
     * out [kv][T][64] so a growing S is the leading prefix -- pie_dots walks
     * B as [S][64] from that base. V: per-dim scales need the live prefix,
     * so re-pack from the fp32 cache with quant_vT (S <= 64). */
    const int16_t *kq  = ctx->self_kq  + (size_t)L * NE_KVHEADS * T * NE_HEADDIM;
    const float   *ksc = ctx->self_ksc + (size_t)L * NE_KVHEADS * T;
    int16_t       *vq  = ctx->self_vq  + (size_t)L * NE_KVHEADS * NE_HEADDIM * Sp;
    float         *vsc = ctx->self_vsc + (size_t)L * NE_KVDIM;
    const float   *vc  = ctx->self_v  + (size_t)L * T * NE_KVDIM;
    quant_vT_i16(vc, S, Sp, vq, vsc);

    for (int h = 0; h < NE_HEADS; h++) {
        const int kv = h / (NE_HEADS / NE_KVHEADS);
        /* K stride is T, not S: rows were appended into a capacity-T slab. */
        const int16_t *K  = kq  + (size_t)kv * T * NE_HEADDIM;
        const float   *KS = ksc + (size_t)kv * T;
        const int16_t *V  = vq  + (size_t)kv * NE_HEADDIM * Sp;
        const float   *VS = vsc + kv * NE_HEADDIM;

        const float qs = quant_row_i16(q + h * NE_HEADDIM, NE_HEADDIM, 1, sq_qq);
        pie_dots_s16(sq_qq, K, NE_HEADDIM, S, sq_iw);
        const float f = qs * inv_scale;
        for (int c = 0; c < S; c++) sq_sc[c] = (float)sq_iw[c] * (f * KS[c]);
        const float pf = softmax_exp(sq_sc, S) * (float)NE_SELF_PQ;
        for (int c = 0; c < S; c++) sq_pq[c] = (int16_t)ne_rnd(sq_sc[c] * pf);
        for (int c = S; c < Sp; c++) sq_pq[c] = 0;
        pie_dots_s16(sq_pq, V, Sp, NE_HEADDIM, sq_iw);
        float *oh = out + h * NE_HEADDIM;
        for (int d = 0; d < NE_HEADDIM; d++)
            oh[d] = (float)sq_iw[d] * (VS[d] * (1.0f / (float)NE_SELF_PQ));
    }
}

/* Quantize the new self-attention key row at `pos` into the int16 mirror.
 * One scale per kv-head row, matching quant_kT; called once per step as the
 * fp32 cache entry is written. V is re-packed wholesale in attend_self_q16. */
static void quant_self_k_row(ne_ctx_t *ctx, int L, int pos, const float *k_row)
{
    const int T = NE_MAX_GEN;
    int16_t *kq  = ctx->self_kq  + (size_t)L * NE_KVHEADS * T * NE_HEADDIM;
    float   *ksc = ctx->self_ksc + (size_t)L * NE_KVHEADS * T;
    for (int hh = 0; hh < NE_KVHEADS; hh++) {
        ksc[(size_t)hh * T + pos] = quant_row_i16(
            k_row + hh * NE_HEADDIM, NE_HEADDIM, 1,
            kq + ((size_t)hh * T + pos) * NE_HEADDIM);
    }
}

/* Decode-side cross-attention for one query row, over the int16 K/V mirrors
 * quantized at encode time. Same kernels, same quantization ranges, same
 * accumulation order as the encoder path; the differences are that S is the
 * prompt length and the query is a single token, so there is no blocking and
 * no core split. Scratch is static -- 4 KB, and bench/gemm measured that
 * placement is irrelevant at this size.
 *
 * The caller holds the scheduler guard (PIE context-save bug; see fw_main.c).
 */
static void attend_cross_q16(const ne_ctx_t *ctx, int L, const float *q,
                             float *out)
{
    static float   cq_sc[NE_MAX_ENC];
    static int32_t cq_iw[NE_MAX_ENC] __attribute__((aligned(16)));
    static ne_aq_t cq_qq[NE_HEADDIM + 16] __attribute__((aligned(16)));
    static ne_aq_t cq_pq[NE_MAX_ENC + 16] __attribute__((aligned(16)));

    const int S = ctx->enc_len, Sp = ctx->cross_sp;   /* the stride vq was packed with */
    const size_t vLayer = ((size_t)NE_MAX_ENC + 31) & ~(size_t)31;  /* layer stride: max Sp */
    const float inv_scale = 1.0f / sqrtf((float)NE_HEADDIM);

    for (int h = 0; h < NE_HEADS; h++) {
        const int kv = h / (NE_HEADS / NE_KVHEADS);
        /* quant_kT packs by the actual prompt length: [4][S][64], scales [4][S] */
        const ne_aq_t *K  = ctx->cross_kq
                            + (size_t)L * NE_KVHEADS * NE_MAX_ENC * NE_HEADDIM
                            + (size_t)kv * S * NE_HEADDIM;
        const float   *KS = ctx->cross_ksc
                            + (size_t)L * NE_KVHEADS * NE_MAX_ENC + (size_t)kv * S;
        /* quant_vT packs by vSp rows: [4][64][vSp], scales [4][64] */
        /* layer stride uses the maximal Sp; inside a layer quant_vT packed
         * [4][64][Sp] with THIS encode's Sp, so the kv-head step is 64*Sp */
        const ne_aq_t *V  = ctx->cross_vq
                            + (size_t)L * NE_KVHEADS * NE_HEADDIM * vLayer
                            + (size_t)kv * NE_HEADDIM * Sp;
        const float   *VS = ctx->cross_vsc + (size_t)L * NE_KVDIM + kv * NE_HEADDIM;

        const float qs = quant_s16(q + h * NE_HEADDIM, NE_HEADDIM, 1, cq_qq);
        ne_dots(cq_qq, K, NE_HEADDIM, S, cq_iw);
        const float f = qs * inv_scale;
        for (int c = 0; c < S; c++) cq_sc[c] = (float)cq_iw[c] * (f * KS[c]);
        const float pf = softmax_exp(cq_sc, S) * (float)NE_PQ;
        for (int c = 0; c < S; c++) cq_pq[c] = (ne_aq_t)ne_rnd(cq_sc[c] * pf);
        for (int c = S; c < Sp; c++) cq_pq[c] = 0;
        ne_dots(cq_pq, V, Sp, NE_HEADDIM, cq_iw);
        float *oh = out + h * NE_HEADDIM;
        for (int d = 0; d < NE_HEADDIM; d++)
            oh[d] = (float)cq_iw[d] * (VS[d] * (1.0f / (float)NE_PQ));
    }
}

void (*ne_split_attn)(const ne_attn_job_t *job, uint32_t heads) = NULL;

void ne_attn_heads(const ne_attn_job_t *j, uint32_t h0, uint32_t h1)
{
    const int S = j->S, Sp = j->Sp;
    /* Scratch slot follows the head range: the split always hands the low
     * heads to the caller and the high heads to the worker. */
    const ne_attn_work_t *w = &j->work[h0 ? 1 : 0];
    /* Only the caller's side times the scalar tail. Both cores run the same
     * amount of it concurrently, so core 0's share is the part that is on the
     * wall clock -- and a shared += from two cores would be a race. */
    const int timed = (h0 == 0);
    const float inv_scale = 1.0f / sqrtf((float)NE_HEADDIM);

    for (uint32_t i = h0; i < h1; i++) {
        /* Interleave the head order: 0,2,4,6 then 1,3,5,7. Head h pairs with kv
         * head h/2, so a contiguous split hands the two cores DISJOINT kv heads
         * and they stream two K/V sets through one 128 KB L2. Taking every
         * other head puts both cores on the same kv head at the same time, so
         * one set is live. This is a bijection on [0, NE_HEADS) either way, and
         * heads are independent, so the order does not affect the result. */
        const uint32_t h = (i < NE_HEADS / 2) ? (i * 2u) : ((i - NE_HEADS / 2) * 2u + 1u);
        const int kv = (int)h / (NE_HEADS / NE_KVHEADS);
        const ne_aq_t *K  = j->kq  + (size_t)kv * S * NE_HEADDIM;
        const float   *KS = j->ksc + (size_t)kv * S;
        const ne_aq_t *V  = j->vq  + (size_t)kv * NE_HEADDIM * Sp;
        const float   *VS = j->vsc + (size_t)kv * NE_HEADDIM;

        for (int t = 0; t < S; t++) {
            const float qs = quant_s16(j->q + (size_t)t * NE_DMODEL + h * NE_HEADDIM,
                                       NE_HEADDIM, 1, w->qq);

            ne_dots(w->qq, K, NE_HEADDIM, S, w->iw);

            uint64_t _s0 = (timed && ne_now_us) ? ne_now_us() : 0;
            const float f = qs * inv_scale;
            for (int c = 0; c < S; c++) w->sc[c] = (float)w->iw[c] * (f * KS[c]);
            const float pf = softmax_exp(w->sc, S) * (float)NE_PQ;
            for (int c = 0; c < S; c++) w->pq[c] = (ne_aq_t)ne_rnd(w->sc[c] * pf);
            for (int c = S; c < Sp; c++) w->pq[c] = 0;
            if (timed && ne_now_us) ne_us_enc_smax += ne_now_us() - _s0;

            ne_dots(w->pq, V, Sp, NE_HEADDIM, w->iw);
            float *oh = j->out + (size_t)t * NE_DMODEL + h * NE_HEADDIM;
            for (int d = 0; d < NE_HEADDIM; d++)
                oh[d] = (float)w->iw[d] * (VS[d] * (1.0f / (float)NE_PQ));
        }
    }
}

#endif /* NE_PIE */

/* [S][NE_KVDIM] -> [NE_KVHEADS][S][NE_HEADDIM] */
static void transpose_kv(const float *src, int S, float *dst)
{
    for (int hh = 0; hh < NE_KVHEADS; hh++)
        for (int t = 0; t < S; t++)
            memcpy(dst + ((size_t)hh * S + t) * NE_HEADDIM,
                   src + (size_t)t * NE_KVDIM + hh * NE_HEADDIM,
                   NE_HEADDIM * sizeof(float));
}

/* ZCRMSNorm each 64-dim head of a q (8-head) or k (4-head) row in place with
 * the shared per-head norm weight, then RoPE at pos (pos<0 = no RoPE: the
 * cross-attention path). */
static void norm_rope_heads(float *x, const float *nw, int nheads, int pos)
{
    for (int h = 0; h < nheads; h++) {
        float *xh = x + h * NE_HEADDIM;
        float tmp[NE_HEADDIM];
        rmsnorm(xh, nw, NE_HEADDIM, tmp);
        memcpy(xh, tmp, sizeof tmp);
        if (pos >= 0) rope(xh, pos);
    }
}

/* Layers L0..11, the final norm, and the cross-attention tail (including the
 * integer decode mirrors under NE_PIE) over whatever ctx->enc_x holds. Split
 * out of ne_encode so the warm-schema path's full phase runs the identical
 * certified machinery; ne_encode calls it with L0 = 0 and is unchanged in
 * instruction order. enc_ids is only read by the NE_TRACE hooks. */
static int enc_from_layer(ne_ctx_t *ctx, const int32_t *enc_ids, int n, int L0)
{
    float *x = ctx->enc_x;
    (void)enc_ids;
    /* Arena layout for the sequence-parallel encoder pass.
     *
     * The base is rounded up to 16 bytes and every offset below is a whole
     * number of 4-float groups, so each buffer carved from it is 16-byte
     * aligned by construction. That is load-bearing: esp.vld.128 mis-loads
     * silently on a misaligned operand, malloc promises no more than 8 on this
     * target, and the failure looks like slightly wrong output rather than a
     * fault. ne_alloc reserves the sixteen floats this can consume.
     *
     * 64, not 16: 16 is all esp.vld.128 needs, but every offset carved below is
     * a whole number of 16-float groups, so a 64-byte base also lands each
     * buffer on a cache line. That matters for ao now that the head order is
     * interleaved -- the two cores write alternating 256-byte head slices of
     * every output row, and 256 is a multiple of 64, so a line-aligned base
     * means they share no line at all. */
    float *arena = (float *)(((uintptr_t)ctx->scratch + 63u) & ~(uintptr_t)63);
    float *q   = arena;                           /* [S][512] */
    float *k   = q + (size_t)NE_MAX_ENC * NE_DMODEL;    /* [S][256] */
    float *v   = k + (size_t)NE_MAX_ENC * NE_KVDIM;    /* [S][256] */
    float *ao  = v + (size_t)NE_MAX_ENC * NE_KVDIM;    /* [S][512] attn out         */
    float *h   = ao + (size_t)NE_MAX_ENC * NE_DMODEL;   /* [512] norm row      */
    float *kT  = h + NE_DMODEL;                   /* [4][S][64] per-head keys  */
    float *vT  = kT + (size_t)NE_MAX_ENC * NE_KVDIM;
    float *scb = vT + (size_t)NE_MAX_ENC * NE_KVDIM;    /* [NE_QBLK][S] scores */
    float *oub = scb + (size_t)NE_QBLK * NE_MAX_ENC;    /* [NE_TBLK][512] o out */
    float *hb  = oub + (size_t)NE_TBLK * NE_DMODEL;     /* [NE_TBLK][512] normed */
#ifdef NE_PIE
    float  *sxb = g_sxb;      /* internal SRAM; see the declaration */
    int8_t *xqb = g_xqb;
#else
    float  *sxb = NULL;       /* the scalar path never reads them */
    int8_t *xqb = NULL;
#endif
#ifdef NE_PIE
    /* int16 attention buffers, all carved 16-byte aligned from the tail of the
     * arena: esp.vld.128 mis-loads silently otherwise. Sp rounds the key count
     * up to the kernel's 32-lane block so every value and probability row also
     * starts aligned, and pq carries a vector of slack because the kernel's fused load
     * runs a chunk ahead of the accumulate. */
    const int Sp = (n + NE_SPALIGN) & ~NE_SPALIGN;
    float   *ksc = hb + (size_t)NE_TBLK * NE_DMODEL;   /* 16-aligned: see above */
    float   *vsc = ksc + (size_t)NE_KVHEADS * NE_MAX_ENC;
    ne_attn_job_t aj;
    float *wp = vsc + NE_KVDIM;
    for (int c = 0; c < 2; c++) {      /* one scratch set per core */
        aj.work[c].sc  = wp;             wp += NE_MAX_ENC;
        aj.work[c].iw  = (int32_t *)wp;  wp += NE_MAX_ENC;
        aj.work[c].qq  = (ne_aq_t *)wp;  wp += NE_HEADDIM * sizeof(ne_aq_t) / 4 + 8;
        aj.work[c].pq  = (ne_aq_t *)wp;  wp += NE_MAX_ENC * sizeof(ne_aq_t) / 4 + 8;
    }
    ne_aq_t *kq = (ne_aq_t *)wp;                         /* [4][S][64]  */
    ne_aq_t *vq = kq + (size_t)NE_KVHEADS * NE_MAX_ENC * NE_HEADDIM;
    /* The carve above is aligned by construction, and when that reasoning
     * breaks the symptom is slightly wrong output, not a fault. Record it so a
     * boot self-test can see it instead of the next person chasing it. */
    uintptr_t amask = (uintptr_t)kq | (uintptr_t)vq;
    for (int c = 0; c < 2; c++) amask |= (uintptr_t)aj.work[c].qq | (uintptr_t)aj.work[c].pq;
    if (amask & 15u) ne_align_fault = 1;
    aj.q = q; aj.out = ao; aj.S = n; aj.Sp = Sp;
    aj.kq = kq; aj.ksc = ksc; aj.vq = vq; aj.vsc = vsc;
#endif

    for (int L = L0; L < NE_ENC_LAYERS; L++) {
        const attn_wts_t *a = &M.enc[L];
        uint64_t _p0 = ne_now_us ? ne_now_us() : 0;
        for (int tb = 0; tb < n; tb += NE_TBLK) {
            const int nb = (n - tb < NE_TBLK) ? n - tb : NE_TBLK;
            nr_arg_t nr = { x + (size_t)tb * NE_DMODEL, M.enc_pre[L], hb };
            run_for(rmsnorm_range, &nr, nb);
#ifdef NE_PIE
            quant_block(hb, nb, a->q->group, xqb, sxb);   /* q, k and v share it */
#endif
            project_block(&ctx->w, a->q, hb, nb, q + (size_t)tb * NE_DMODEL, NE_DMODEL, xqb, sxb);
            project_block(&ctx->w, a->k, hb, nb, k + (size_t)tb * NE_KVDIM, NE_KVDIM, xqb, sxb);
            project_block(&ctx->w, a->v, hb, nb, v + (size_t)tb * NE_KVDIM, NE_KVDIM, xqb, sxb);
        }
        {
            hr_arg_t hq = { q, M.enc_qn[L], NE_HEADS,   NE_DMODEL, 1 };
            run_for(headnorm_range, &hq, n);
            hr_arg_t hk = { k, M.enc_kn[L], NE_KVHEADS, NE_KVDIM,  1 };
            run_for(headnorm_range, &hk, n);
        }
#ifdef NE_WARMSCHEMA
        /* Product F=4: only schema K/V for layers 0..3. */
        if (ws_cap && L < NE_WARM_F) ws_save_kv(L, k, v, ws_cap_nq, n);
#endif
        if (ne_now_us) ne_us_enc_proj += ne_now_us() - _p0;
        uint64_t _a0 = ne_now_us ? ne_now_us() : 0;
#ifdef NE_PIE
        if (!ne_force_scalar) {
            uint64_t _q0 = ne_now_us ? ne_now_us() : 0;
            quant_kT(k, n, kq, ksc);
            quant_vT(v, n, Sp, vq, vsc);
            if (ne_now_us) ne_us_enc_kvq += ne_now_us() - _q0;
            /* One suspension for the whole layer's attention: core 1 runs with
             * its scheduler already down, so it must not take the guard
             * itself. */
            uint64_t _c0 = ne_now_us ? ne_now_us() : 0;
            if (ne_critical_enter) ne_critical_enter();
            if (ne_split_attn) ne_split_attn(&aj, NE_HEADS);
            else               ne_attn_heads(&aj, 0, NE_HEADS);
            if (ne_critical_exit) ne_critical_exit();
            if (ne_now_us) ne_us_enc_core += ne_now_us() - _c0;
        } else
#endif
        {
            transpose_kv(k, n, kT);
            transpose_kv(v, n, vT);
            attend_seq(q, kT, vT, n, scb, ao);      /* ao = attention output */
        }
#ifdef NE_TRACE
        trace_ablate(ao, n, NE_DMODEL, 1, L);
#endif
        /* o_proj lands in oub because its input ao is still being read; the
         * gated residual then adds straight from oub into x. Staging back into
         * ao and re-reading it in a separate sequence-wide pass moved an extra
         * 1.6 MB of PSRAM per layer for nothing. Same additions in the same
         * t-ascending order, so the output is unchanged. */
        const float g = M.enc_gate[L];
        for (int tb = 0; tb < n; tb += NE_TBLK) {
            const int nb = (n - tb < NE_TBLK) ? n - tb : NE_TBLK;
#ifdef NE_PIE
            quant_block(ao + (size_t)tb * NE_DMODEL, nb, a->o->group, xqb, sxb);
#endif
            project_block(&ctx->w, a->o, ao + (size_t)tb * NE_DMODEL, nb,
                          oub, NE_DMODEL, xqb, sxb);
            for (int t = 0; t < nb; t++)
                for (int d = 0; d < NE_DMODEL; d++)
                    x[(size_t)(tb + t) * NE_DMODEL + d] += g * oub[(size_t)t * NE_DMODEL + d];
        }
        if (ne_now_us) ne_us_enc_attn += ne_now_us() - _a0;
#ifdef NE_TRACE
        trace_enc_pool(x, n, enc_ids, L + 1);  /* layers 1..12 = post-layer  */
        trace_enc_full(x, n, L + 1);
#endif
#ifdef NE_WARMSCHEMA
        /* Splice residual = state after layer (NE_WARM_F - 1) = checkpoint F. */
        if (ws_cap && L + 1 == NE_WARM_F) ws_save_x_splice(x, ws_cap_nq, n);
#endif
    }

    /* encoder final norm, in place: enc_x becomes the encoder output */
    for (int t = 0; t < n; t++) {
        rmsnorm(x + (size_t)t * NE_DMODEL, M.enc_final_w, NE_DMODEL, h);
        memcpy(x + (size_t)t * NE_DMODEL, h, NE_DMODEL * sizeof(float));
    }
#ifdef NE_TRACE
    trace_enc_pool(x, n, enc_ids, NE_ENC_LAYERS + 1);   /* 13 = final norm */
    trace_enc_full(x, n, NE_ENC_LAYERS + 1);
#endif

    /* Precompute cross-attention K/V per decoder layer: k_norm applied, NO rope.
     * Block-outer, layer-inner: all sixteen tensors read the same encoder
     * output, so one activation quantize serves every decoder layer. The
     * layer-outer order quantized the identical block eight times. */
    uint64_t _x0 = ne_now_us ? ne_now_us() : 0;
    for (int tb = 0; tb < n; tb += NE_TBLK) {
        const int nb = (n - tb < NE_TBLK) ? n - tb : NE_TBLK;
#ifdef NE_PIE
        quant_block(x + (size_t)tb * NE_DMODEL, nb, M.dec_cross[0].k->group, xqb, sxb);
#endif
        for (int L = 0; L < NE_DEC_LAYERS; L++) {
            const attn_wts_t *c = &M.dec_cross[L];
            float *ck = ctx->cross_k + (size_t)L * NE_MAX_ENC * NE_KVDIM;
            float *cv = ctx->cross_v + (size_t)L * NE_MAX_ENC * NE_KVDIM;
            project_block(&ctx->w, c->k, x + (size_t)tb * NE_DMODEL, nb,
                          ck + (size_t)tb * NE_KVDIM, NE_KVDIM, xqb, sxb);
            project_block(&ctx->w, c->v, x + (size_t)tb * NE_DMODEL, nb,
                          cv + (size_t)tb * NE_KVDIM, NE_KVDIM, xqb, sxb);
        }
    }
    for (int L = 0; L < NE_DEC_LAYERS; L++) {
        hr_arg_t hc = { ctx->cross_k + (size_t)L * NE_MAX_ENC * NE_KVDIM,
                        M.dc_kn[L], NE_KVHEADS, NE_KVDIM, 0 };
        run_for(headnorm_range, &hc, n);
    }
#ifdef NE_PIE
    /* int16 mirrors for decode, once per prompt rather than once per token.
     * quant_kT wants k_norm already applied, hence after the loop above. The
     * mirrors pack by the ACTUAL prompt length -- kq rows by S, vq rows by this
     * encode's Sp -- and decode must read them with the same strides, so Sp is
     * recorded on the context rather than re-derived. */
    ctx->cross_sp = Sp;
    for (int L = 0; L < NE_DEC_LAYERS; L++) {
        quant_kT(ctx->cross_k + (size_t)L * NE_MAX_ENC * NE_KVDIM, n,
                 ctx->cross_kq  + (size_t)L * NE_KVHEADS * NE_MAX_ENC * NE_HEADDIM,
                 ctx->cross_ksc + (size_t)L * NE_KVHEADS * NE_MAX_ENC);
        quant_vT(ctx->cross_v + (size_t)L * NE_MAX_ENC * NE_KVDIM, n, Sp,
                 ctx->cross_vq  + (size_t)L * NE_KVHEADS * NE_HEADDIM
                                  * ((((size_t)NE_MAX_ENC + 31) & ~(size_t)31)),
                 ctx->cross_vsc + (size_t)L * NE_KVDIM);
    }
#endif
    if (ne_now_us) ne_us_enc_xkv += ne_now_us() - _x0;
    return 0;
}

int ne_encode(ne_ctx_t *ctx, const int32_t *enc_ids, int n)
{
    if (n <= 0 || n > NE_MAX_ENC) return -1;   /* n==0 would softmax over nothing */
    ctx->enc_len = n;
    ctx->dec_len = 0;

    const float embed_scale = sqrtf((float)NE_DMODEL);
    float *x = ctx->enc_x;

    for (int t = 0; t < n; t++) {
        if (enc_ids[t] < 0 || enc_ids[t] >= NE_VOCAB) return -1;
        dequant_row(&ctx->w, M.embed, (uint32_t)enc_ids[t], x + (size_t)t * NE_DMODEL);
        for (int d = 0; d < NE_DMODEL; d++) x[(size_t)t * NE_DMODEL + d] *= embed_scale;
    }
#ifdef NE_TRACE
    trace_enc_pool(x, n, enc_ids, 0);          /* layer 0 = post-embedding */
    trace_enc_full(x, n, 0);
#endif
    return enc_from_layer(ctx, enc_ids, n, 0);
}

#ifdef NE_WARMSCHEMA
ne_schema_cache_t *ne_schema_capture(ne_ctx_t *ctx, const int32_t *enc_ids, int n)
{
    if (!ctx || !enc_ids || n <= 0 || n > NE_MAX_ENC) return NULL;
    int nq = ws_tools_pos(enc_ids, n);
    if (nq < 0 || nq >= n) return NULL;
    int ns = n - nq;
    ne_schema_cache_t *c = ws_cache_alloc(ns);
    if (!c) return NULL;
    memcpy(c->schema_ids, enc_ids + nq, (size_t)ns * sizeof(int32_t));
    ws_cap_nq = nq;
    ws_cap = c;
    int rc = ne_encode(ctx, enc_ids, n);
    ws_cap = NULL;
    if (rc != 0) {
        ne_schema_cache_free(c);
        return NULL;
    }
    /* Re-measure payload from the live struct (defensive; equals formula). */
    c->bytes = (size_t)ns * sizeof(int32_t)
             + (size_t)NE_WARM_F * (size_t)ns * NE_KVDIM * sizeof(float) * 2u
             + (size_t)ns * NE_DMODEL * sizeof(float);
    return c;
}

/* Warm encode with product freeze depth NE_WARM_F (= 4):
 *   layers 0..3: query rows only; schema K/V from cache
 *   splice cached schema residual post layer 3
 *   layers 4..11: full normal computation over all rows */
int ne_encode_warm(ne_ctx_t *ctx, const int32_t *enc_ids, int n,
                   const ne_schema_cache_t *cache)
{
    if (!ctx || !cache || !enc_ids) return -1;
    if (n <= 0 || n > NE_MAX_ENC) return -1;
    int nq = ws_tools_pos(enc_ids, n);
    if (nq < 0 || nq >= n) return -1;
    int ns = n - nq;
    if (ns != cache->ns) return -1;
    if (memcmp(enc_ids + nq, cache->schema_ids, (size_t)ns * sizeof(int32_t)) != 0)
        return -1;

    const int F = NE_WARM_F;

    ne_warm_nq = nq;
    ne_warm_ns = ns;
    ne_warm_ntot = n;
    /* Partial layers: nq rows projected; attention still nq queries × n keys,
     * so the honest row-layer proxy for a partial layer is nq (proj-dominated
     * and attention scales as nq/n of full). Full layers: ntot. */
    ne_warm_row_layers = F * nq + (NE_ENC_LAYERS - F) * n;
    ne_warm_row_layers_full = NE_ENC_LAYERS * n;

    ctx->enc_len = n;
    ctx->dec_len = 0;

    const float embed_scale = sqrtf((float)NE_DMODEL);
    float *x = ctx->enc_x;

    float *arena = (float *)(((uintptr_t)ctx->scratch + 63u) & ~(uintptr_t)63);
    float *q   = arena;
    float *k   = q + (size_t)NE_MAX_ENC * NE_DMODEL;
    float *v   = k + (size_t)NE_MAX_ENC * NE_KVDIM;
    float *ao  = v + (size_t)NE_MAX_ENC * NE_KVDIM;
    float *h   = ao + (size_t)NE_MAX_ENC * NE_DMODEL;
    float *kT  = h + NE_DMODEL;
    float *vT  = kT + (size_t)NE_MAX_ENC * NE_KVDIM;
    float *scb = vT + (size_t)NE_MAX_ENC * NE_KVDIM;
    float *oub = scb + (size_t)NE_QBLK * NE_MAX_ENC;
    float *hb  = oub + (size_t)NE_TBLK * NE_DMODEL;
#ifdef NE_PIE
    float  *sxb = g_sxb;
    int8_t *xqb = g_xqb;
#else
    float  *sxb = NULL;       /* the scalar path never reads them */
    int8_t *xqb = NULL;
#endif

    /* Embed query tokens; schema residual spliced after phase A. */
    for (int t = 0; t < nq; t++) {
        if (enc_ids[t] < 0 || enc_ids[t] >= NE_VOCAB) return -1;
        dequant_row(&ctx->w, M.embed, (uint32_t)enc_ids[t], x + (size_t)t * NE_DMODEL);
        for (int d = 0; d < NE_DMODEL; d++)
            x[(size_t)t * NE_DMODEL + d] *= embed_scale;
    }

    /* ---- Phase A: partial layers 0..F-1 (query rows only) ---- */
    for (int L = 0; L < F; L++) {
        const attn_wts_t *a = &M.enc[L];
        for (int tb = 0; tb < nq; tb += NE_TBLK) {
            const int nb = (nq - tb < NE_TBLK) ? nq - tb : NE_TBLK;
            nr_arg_t nr = { x + (size_t)tb * NE_DMODEL, M.enc_pre[L], hb };
            run_for(rmsnorm_range, &nr, nb);
#ifdef NE_PIE
            quant_block(hb, nb, a->q->group, xqb, sxb);   /* q, k and v share it */
#endif
            project_block(&ctx->w, a->q, hb, nb, q + (size_t)tb * NE_DMODEL, NE_DMODEL, xqb, sxb);
            project_block(&ctx->w, a->k, hb, nb, k + (size_t)tb * NE_KVDIM, NE_KVDIM, xqb, sxb);
            project_block(&ctx->w, a->v, hb, nb, v + (size_t)tb * NE_KVDIM, NE_KVDIM, xqb, sxb);
        }
        {
            hr_arg_t hq = { q, M.enc_qn[L], NE_HEADS,   NE_DMODEL, 1 };
            run_for(headnorm_range, &hq, nq);
            hr_arg_t hk = { k, M.enc_kn[L], NE_KVHEADS, NE_KVDIM,  1 };
            run_for(headnorm_range, &hk, nq);
        }
        /* Fresh query K/V | cached schema K/V. */
        ws_load_kv(cache, L, k, v, nq, ns);

        transpose_kv(k, n, kT);
        transpose_kv(v, n, vT);
        attend_seq_range(q, kT, vT, n, 0, nq, scb, ao);

        const float g = M.enc_gate[L];
        for (int tb = 0; tb < nq; tb += NE_TBLK) {
            const int nb = (nq - tb < NE_TBLK) ? nq - tb : NE_TBLK;
#ifdef NE_PIE
            quant_block(ao + (size_t)tb * NE_DMODEL, nb, a->o->group, xqb, sxb);
#endif
            project_block(&ctx->w, a->o, ao + (size_t)tb * NE_DMODEL, nb,
                          oub, NE_DMODEL, xqb, sxb);
            for (int t = 0; t < nb; t++)
                for (int d = 0; d < NE_DMODEL; d++)
                    x[(size_t)(tb + t) * NE_DMODEL + d] += g * oub[(size_t)t * NE_DMODEL + d];
        }
    }

    /* ---- Splice: schema residual = cached post layer F-1 ---- */
    ws_load_x(cache, x, nq, ns);

    /* ---- Phase B + final norm + cross tail: the identical certified
     * machinery ne_encode runs, from layer F. Under NE_PIE this is the
     * dual-core integer attention and the decode mirror computation --
     * a warm encode leaves the context exactly as ready as an exact one. */
    return enc_from_layer(ctx, enc_ids, n, F);
}

int ne_generate_warm(ne_ctx_t *ctx, const int32_t *enc_ids, int n_enc,
                     int32_t *out, int max_gen,
                     const ne_schema_cache_t *cache)
{
    if (ne_encode_warm(ctx, enc_ids, n_enc, cache) != 0) return -1;
    if (max_gen > NE_MAX_GEN - 1) max_gen = NE_MAX_GEN - 1;
    int32_t tok = NE_EOS_ID;
    int n = 0;
    for (int i = 0; i < max_gen; i++) {
        tok = ne_step(ctx, tok);
        if (tok == NE_EOS_ID) break;
        out[n++] = tok;
    }
    return n;
}
#endif /* NE_WARMSCHEMA */

/* Tool-call JSON skeleton. Decode is free until 356 (▁[{"); after that three
 * positions are fully determined by the surface grammar. 393 (arguments) is
 * NOT forced onward: empty-arg tools emit 630 (":{}}]') while others emit
 * 282 (":{"). Conservative: force only the three below.
 *
 * n_out = tokens already emitted (including prev). n_265 = how many 265s are
 * among them; the first 265 is the name/args separator, later ones are not. */
int32_t ne_forced_next(int32_t prev, int n_out, int n_265)
{
    /* 356 is forced only at its opening position: the same piece inside a
     * copied argument value must not get 'name' spliced into it. */
    if (prev == 356 && n_out == 2) return 294;   /* ▁[{"  -> name */
    /* 294 is 'name' at fixed index 2 after <tool_call> and ▁[{"; later 294
     * (if any) is not forced, and 264 also closes argument keys. */
    if (prev == 294 && n_out == 3) return 264;   /* name -> ":" */
    if (prev == 265 && n_265 == 1) return 393;   /* first "," -> arguments */
    return -1;
}

int32_t ne_step_f(ne_ctx_t *ctx, int32_t token, int32_t forced)
{
    const float embed_scale = sqrtf((float)NE_DMODEL);
    int pos = ctx->dec_len;
    if (pos >= NE_MAX_GEN) return NE_EOS_ID;   /* KV cache is full; stop */
    float x[NE_DMODEL], h[NE_DMODEL], q[NE_DMODEL], merged[NE_DMODEL], proj[NE_DMODEL];
    float *sc = ctx->scratch;   /* score row; encoder scratch is free now */

    if (token < 0 || token >= NE_VOCAB) token = NE_EOS_ID;
    dequant_row(&ctx->w, M.embed, (uint32_t)token, x);
    for (int d = 0; d < NE_DMODEL; d++) x[d] *= embed_scale;
#ifdef NE_TRACE
    ne_trace_step = pos;
    trace_emit("dembd", -1, -1, NE_DMODEL, x);
#endif

    uint64_t _tl = ne_now_us ? ne_now_us() : 0;
    for (int L = 0; L < NE_DEC_LAYERS; L++) {
        /* self-attention with KV cache (== causal mask incrementally) */
        const attn_wts_t *s = &M.dec_self[L];
        float *kc = ctx->self_k + (size_t)L * NE_MAX_GEN * NE_KVDIM;
        float *vc = ctx->self_v + (size_t)L * NE_MAX_GEN * NE_KVDIM;
        rmsnorm(x, M.ds_pre[L], NE_DMODEL, h);
#ifdef NE_PIE
        /* q/k/v share the normed row: one activation quantize, three GEMVs.
         * Same bytes as three gemv() calls, without redoing the quantize. */
        if (!ne_force_scalar) {
            quant_act(h, s->q->group, s->q->cols / s->q->group);
            gemv_xq(&ctx->w, s->q, q);
            gemv_xq(&ctx->w, s->k, kc + (size_t)pos * NE_KVDIM);
            gemv_xq(&ctx->w, s->v, vc + (size_t)pos * NE_KVDIM);
        } else
#endif
        {
            gemv(&ctx->w, s->q, h, q);
            gemv(&ctx->w, s->k, h, kc + (size_t)pos * NE_KVDIM);
            gemv(&ctx->w, s->v, h, vc + (size_t)pos * NE_KVDIM);
        }
        norm_rope_heads(q, M.ds_qn[L], NE_HEADS, pos);
        norm_rope_heads(kc + (size_t)pos * NE_KVDIM, M.ds_kn[L], NE_KVHEADS, pos);
#ifdef NE_PIE
        if (!ne_force_scalar) {
            quant_self_k_row(ctx, L, pos, kc + (size_t)pos * NE_KVDIM);
            if (ne_critical_enter) ne_critical_enter();
            attend_self_q16(ctx, L, pos + 1, q, merged);
            if (ne_critical_exit) ne_critical_exit();
        } else
#endif
        {
            attend(q, kc, vc, pos + 1, sc, merged);
        }
#ifdef NE_TRACE
        trace_ablate(merged, 1, NE_DMODEL, 2, L);
#endif
        gemv(&ctx->w, s->o, merged, proj);
        for (int d = 0; d < NE_DMODEL; d++) x[d] += M.ds_gate[L] * proj[d];
#ifdef NE_TRACE
        trace_emit("dself", L, -1, NE_DMODEL, x);
#endif

        /* cross-attention over the precomputed encoder K/V -- no RoPE */
        const attn_wts_t *c = &M.dec_cross[L];
        rmsnorm(x, M.dc_pre[L], NE_DMODEL, h);
        gemv(&ctx->w, c->q, h, q);
        norm_rope_heads(q, M.dc_qn[L], NE_HEADS, -1);
#ifdef NE_PIE
        if (!ne_force_scalar) {
            if (ne_critical_enter) ne_critical_enter();
            attend_cross_q16(ctx, L, q, merged);
            if (ne_critical_exit) ne_critical_exit();
        } else
#endif
        {   /* braces make the else body explicit; no semantic change */
#ifdef NE_TRACE
            ne_trace_xlayer = L;
#endif
            attend(q,
                   ctx->cross_k + (size_t)L * NE_MAX_ENC * NE_KVDIM,
                   ctx->cross_v + (size_t)L * NE_MAX_ENC * NE_KVDIM,
                   ctx->enc_len, sc, merged);
#ifdef NE_TRACE
            ne_trace_xlayer = -1;
#endif
        }
#ifdef NE_TRACE
        trace_ablate(merged, 1, NE_DMODEL, 3, L);
#endif
        gemv(&ctx->w, c->o, merged, proj);
        for (int d = 0; d < NE_DMODEL; d++) x[d] += M.dc_gate[L] * proj[d];
#ifdef NE_TRACE
        trace_emit("dcros", L, -1, NE_DMODEL, x);
#endif
    }

    if (ne_now_us) ne_us_proj += ne_now_us() - _tl;
    ctx->dec_len = pos + 1;

    /* Grammar-forced step: layers and KV are done; final norm only feeds the
     * logits GEMV, so both can be skipped together. */
    if (forced >= 0) {
        ne_grammar_skips++;
        return forced;
    }

    /* final norm + tied output projection over the full vocab */
    rmsnorm(x, M.dec_final_w, NE_DMODEL, h);
    uint64_t _t0 = ne_now_us ? ne_now_us() : 0;
    gemv(&ctx->w, M.embed, h, ctx->logits);
    if (ne_now_us) ne_us_logits += ne_now_us() - _t0;
#ifdef NE_TRACE
    /* the true logits, so the offline lens can be validated against layer 7 */
    trace_emit("logit", -1, -1, NE_VOCAB, ctx->logits);
#endif

    /* Start at id 1, so <pad> (id 0) can never be emitted. It is a padding
     * marker with no meaning in generated output, and its embedding row is
     * pathological: ‖row‖ = 69.8 against a median of 18.9, which makes it the
     * winner for ~15% of random directions. Trained hidden states point away
     * from it, but a degenerate state collapses there first, so the scan skips
     * it. */
    int32_t best = 1;
    float bv = ctx->logits[1];
    for (int i = 2; i < NE_VOCAB; i++)
        if (ctx->logits[i] > bv) { bv = ctx->logits[i]; best = i; }
    return best;
}

int32_t ne_step(ne_ctx_t *ctx, int32_t token)
{
    return ne_step_f(ctx, token, -1);
}

int ne_generate_cb(ne_ctx_t *ctx, const int32_t *enc_ids, int n_enc,
                   int32_t *out, int max_gen, ne_step_cb_t cb, void *user)
{
    if (ne_encode(ctx, enc_ids, n_enc) != 0) return -1;
    if (max_gen > NE_MAX_GEN - 1) max_gen = NE_MAX_GEN - 1;
    int32_t tok = NE_EOS_ID;
    int n = 0;
    int n_265 = 0;
    for (int i = 0; i < max_gen; i++) {
        int32_t forced = -1;
        if (ne_grammar_force)
            forced = ne_forced_next(tok, n, n_265);
        tok = ne_step_f(ctx, tok, forced);
        if (cb) cb(user, tok);
        if (tok == NE_EOS_ID) break;
        out[n++] = tok;
        if (tok == 265) n_265++;
    }
    return n;
}

int ne_generate(ne_ctx_t *ctx, const int32_t *enc_ids, int n_enc,
                int32_t *out, int max_gen)
{
    return ne_generate_cb(ctx, enc_ids, n_enc, out, max_gen, NULL, NULL);
}
