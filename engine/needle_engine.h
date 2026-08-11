/* Needle 26M encoder-decoder, C99, no dependencies beyond libm.
 *
 * One translation unit implements the whole forward pass; it compiles unchanged
 * for x86 (parity harness) and for the ESP32-P4 (firmware). Divergence between
 * host and board is therefore compiler/libm-level only -- the parity gate is
 * greedy-argmax sequence equality plus logit agreement, not bit equality, since
 * bit equality across libm expf/sqrtf implementations is not a promise anyone
 * can keep.
 *
 * Weights come from a .npk produced by tools/pack_npk.py: int8 or int4, quantized
 * per group of 32 or once per row (what ships),
 * symmetric along the input dim, fp16 scales, norms/gates raw fp16. The file is
 * used in place -- tensor payloads are never copied or pre-dequantized, because
 * the P4 cannot afford a float image of the model (100 MB) and the whole point
 * is to stream quantized bytes.
 */
#ifndef NEEDLE_ENGINE_H
#define NEEDLE_ENGINE_H

#include <stddef.h>
#include <stdint.h>

#define NE_DMODEL   512
#define NE_HEADS    8
#define NE_KVHEADS  4
#define NE_HEADDIM  64
#define NE_ENC_LAYERS 12
#define NE_DEC_LAYERS 8
#define NE_VOCAB    8192
#define NE_EPS      1e-6f
#define NE_ROPE_THETA 10000.0f

/* Compile-time capacity; raise if a query needs more. Encoder length covers the
 * query + <tools> + tool JSON (measured worst case in the fixture set: 271). */
#define NE_MAX_ENC  384
#define NE_MAX_GEN  64

#define NE_EOS_ID   1

/* Cap on tensors in one .npk; the fp32 scale cache is indexed by record. */
#define NE_MAX_RECS 512

/* Token-block width for the encoder projections: the weight tensor is swept
 * once per block instead of once per token. */
#define NE_TBLK     64

/* Query-block width for whole-sequence attention: K and V are re-read once per
 * block instead of once per query. Sized so one block's scores stay small
 * (NE_QBLK x NE_MAX_ENC floats) while still amortising the K/V read. */
#define NE_QBLK     16

/* One KV row: NE_KVHEADS heads of NE_HEADDIM laid out contiguously. */
#define NE_KVDIM    (NE_KVHEADS * NE_HEADDIM)

/* int4 nibble-packing chunk: 32 elements in 16 bytes, one esp.vld.128. Fixed
 * by the vector registers and independent of the quantization group -- a row
 * may carry one scale or sixteen and the bytes look identical. */
#define NE_PACK     32

enum { NE_DT_F32 = 0, NE_DT_F16 = 1, NE_DT_I8 = 2, NE_DT_I4 = 3 };

/* Quantized attention element. Default int16 (NE_AQ=4096, NE_PQ=16384),
 * byte-identical to fp32 on every fixture. NE_ATTN_I8 narrows the same scheme
 * to signed int8 (NE_AQ=NE_PQ=127) and halves the K/V mirrors and scratch;
 * measured in docs/notebook/RESULTS-INT8.md before the type existed. Everything that stores
 * or reads quantized attention values uses this type so the two builds cannot
 * disagree about layout. */
#ifdef NE_ATTN_I8
typedef int8_t  ne_aq_t;
#else
typedef int16_t ne_aq_t;
#endif

typedef struct {
    char     name[64];
    uint32_t dtype, rows, cols, group;
    uint64_t scale_off, data_off;
} ne_rec_t;

typedef struct {
    const uint8_t *blob;     /* payload region of the .npk, never modified   */
    const ne_rec_t *recs;
    uint32_t n_recs;
} ne_weights_t;

typedef struct {
    ne_weights_t w;

    /* per-query state, PSRAM-sized: filled by ne_encode(), read by ne_step() */
    int   enc_len;
    float *enc_x;                          /* [NE_MAX_ENC][512] scratch + output */
    float *cross_k, *cross_v;              /* [8][NE_MAX_ENC][256], normed, no rope */
    /* Integer mirrors of the cross-attention K/V (ne_aq_t), quantized once at
     * encode time and read by every generated token. NULL on the scalar build. */
    void    *cross_kq_raw, *cross_vq_raw;  /* malloc'd; the aligned views follow */
    ne_aq_t *cross_kq;                     /* [8][4][S][64], packed by prompt len */
    float   *cross_ksc;                    /* [8][4][NE_MAX_ENC]      */
    ne_aq_t *cross_vq;                     /* [8][4][64][cross_sp]    */
    int      cross_sp;                     /* the Sp the mirrors were packed with */
    float   *cross_vsc;                    /* [8][256]                */

    /* decoder KV cache, grows with ne_step() */
    int   dec_len;
    float *self_k, *self_v;                /* [8][NE_MAX_GEN][256] */
    /* int16 mirrors of the self-attention KV cache under NE_PIE. Packing
     * matches the encoder/cross path (quant_kT / quant_vT): K is
     * [kv][S][64] with one scale per key row; V is [kv][64][self_sp] with
     * one scale per dim. self_sp is fixed at the rounded NE_MAX_GEN so
     * pack and read share one stride (prior bugs were pack/read
     * disagreements). NULL on the scalar build. */
    void    *self_kq_raw, *self_vq_raw;    /* malloc'd; aligned views follow */
    int16_t *self_kq;                      /* [8][4][NE_MAX_GEN][64]          */
    float   *self_ksc;                     /* [8][4][NE_MAX_GEN]              */
    int16_t *self_vq;                      /* [8][4][64][self_sp]             */
    int      self_sp;                      /* the Sp the mirrors were packed with */
    float   *self_vsc;                     /* [8][256] per-dim V scales       */

    /* scratch (single decoder position / single encoder row) */
    float *scratch;                        /* >= NE_MAX_ENC*512 floats */
    float *logits;                         /* [NE_VOCAB] */
} ne_ctx_t;

/* Parse an .npk image already resident in memory (mmap, PSRAM copy, fread).
 * Returns 0 on success. The image must outlive the context. */
int ne_load(ne_ctx_t *ctx, const uint8_t *npk, size_t len);

/* Allocate the working buffers (malloc; ~10 MB). 0 on success. */
int ne_alloc(ne_ctx_t *ctx);
void ne_free(ne_ctx_t *ctx);

/* Run the encoder over enc_ids[n] and precompute cross-attention K/V.
 * Resets the decoder. Returns 0, or -1 if n exceeds NE_MAX_ENC. */
int ne_encode(ne_ctx_t *ctx, const int32_t *enc_ids, int n);

/* Feed one decoder token (the first call passes NE_EOS_ID), get back the
 * argmax of the next-token logits. ctx->logits holds the full vector. */
int32_t ne_step(ne_ctx_t *ctx, int32_t token);

/* Like ne_step, but if forced >= 0 the decoder layers still run (KV cache and
 * residual stream identical) while the final rmsnorm + vocab GEMV + argmax are
 * skipped and forced is returned. forced < 0 is identical to ne_step. */
int32_t ne_step_f(ne_ctx_t *ctx, int32_t token, int32_t forced);

/* The grammar DFA behind ne_generate_cb's forcing, exported so callers that
 * own their own decode loop (the firmware streams per token) can take the
 * same skips. prev = last emitted token, n_out = tokens emitted so far
 * (including prev), n_265 = 265s among them. Returns the forced next token
 * or -1. Call sites must mirror ne_generate_cb's counting exactly. */
int32_t ne_forced_next(int32_t prev, int n_out, int n_265);

/* Called after every ne_step inside ne_generate, before the EOS test, with the
 * token just produced; ctx->logits still holds that step's full vector. Lets a
 * caller time or record each step without forking the decode loop. On a
 * grammar-forced step logits are not recomputed and stay from the prior step. */
typedef void (*ne_step_cb_t)(void *user, int32_t token);

/* Greedy loop: encode, then decode until EOS or max_gen. Returns the number of
 * generated tokens written to out (EOS excluded), or -1 on a bad encode.
 * cb may be NULL. When ne_grammar_force is set, rigid tool-call JSON positions
 * skip the logits stage. */
int ne_generate_cb(ne_ctx_t *ctx, const int32_t *enc_ids, int n_enc,
                   int32_t *out, int max_gen, ne_step_cb_t cb, void *user);
int ne_generate(ne_ctx_t *ctx, const int32_t *enc_ids, int n_enc,
                int32_t *out, int max_gen);

/* ne_generate_cb applies the tool-call JSON DFA when non-zero (default 1).
 * Set to 0 to time the same path without logits skips. ne_grammar_skips counts
 * how many decode steps skipped the vocab GEMV; the caller may zero it. */
extern int ne_grammar_force;
extern int ne_grammar_skips;

/* Optional critical section around each vector-kernel burst.
 *
 * IDF 5.4.2's lazy PIE context save is broken (portasm.S skips q3 and misroutes
 * SAR), so a task switch inside a kernel corrupts the row being computed. The
 * kernels must therefore run with the scheduler suspended. Installing these
 * takes that guard per burst -- when the 8192-row logits GEMV runs it is still
 * the longest window (~40 ms on device); most projection GEMVs are 2-5 ms.
 * Grammar-forced steps (ne_step_f with forced >= 0) skip that GEMV entirely,
 * so not every decode token pays the 40 ms bound. Short enough for a network
 * stack between bursts. NULL means no guard, which is correct for the host
 * build. */
extern void (*ne_critical_enter)(void);
extern void (*ne_critical_exit)(void);

/* Install a microsecond clock to accumulate per-phase timings; NULL disables. */
extern uint64_t (*ne_now_us)(void);
extern uint64_t ne_us_logits, ne_us_proj;
extern uint64_t ne_us_enc_proj, ne_us_enc_attn, ne_us_enc_xkv, ne_us_enc_smax, ne_us_enc_core, ne_us_enc_kvq;

/* Set by ne_encode if a vector operand came out misaligned. Checked by the
 * firmware's boot self-test; a misaligned esp.vld.128 mis-loads silently. */
extern int ne_align_fault;

#ifdef NE_TRACE
/* Latent-path tracing, host only (main_trace.c). Every hook sits inside
 * #ifdef NE_TRACE, so the firmware build compiles the exact same object it
 * always did. When ne_trace_f is set the engine appends tagged records:
 *   char tag[8], int32 layer, int32 head, int32 step, int32 n, float v[n]
 * ne_trace_dump_model emits the dequantized embedding and the decoder final
 * norm once, so the logit lens can be computed offline.
 *
 * Tags written by the encoder path:
 *   encq  pooled mean over query tokens (before <tools> id 5)
 *   enca  pooled mean over the full sequence
 *   enct  single position: the <tools> marker (id 5)
 *   encx  full per-position state, row-major [S][DMODEL]; step holds S
 *         (emitted at every layer checkpoint, including post-embed=0 and
 *         post-final-norm=NE_ENC_LAYERS+1). Used by tools/measure_schema_drift.py.
 * Decoder tags: dembd, dself, dcros, xprob, logit; model dump: dfnw, embw. */
#include <stdio.h>
extern FILE *ne_trace_f;
void ne_trace_dump_model(ne_ctx_t *ctx);
/* Dequantized attention weights + norms + gates, one self-describing record
 * per tensor: u32 namelen, name, u32 rows, u32 cols, f32 data[rows*cols]. */
void ne_trace_dump_weights(ne_ctx_t *ctx, FILE *f);
/* Head ablation: zero one head's attention output after the fact.
 * kind 0=off, 1=encoder self, 2=decoder self, 3=decoder cross. */
extern int ne_ablate_kind, ne_ablate_layer, ne_ablate_head;
#endif

#ifdef NE_WARMSCHEMA
/* Product warm-schema F=4 (host-only, opt-in; firmware can enable later).
 *
 * Cache key = tools-JSON token sequence (ids from <tools> marker through end).
 * Contents (fp32, only what F=4 needs):
 *   schema K/V for encoder layers 0..3
 *   schema residual after layer 3 (splice source for layers 4..11)
 *
 * Capture once per schema; warm encode recomputes query rows in layers 0..3
 * (attending over cached schema K/V), splices residual, then full layers 4..11.
 * ne_encode is untouched. Key mismatch returns -1 (no silent fallback).
 *
 * Research F-sweep lives in tools/eval_warmschema.py + docs/notebook/RESULTS-HYBRID.md;
 * this product path is pinned to F=4. */
#define NE_WARM_F 4

typedef struct ne_schema_cache ne_schema_cache_t;

/* One full encode of a reference prompt that carries the target schema.
 * Returns an owned cache, or NULL on failure. Free with ne_schema_cache_free. */
ne_schema_cache_t *ne_schema_capture(ne_ctx_t *ctx, const int32_t *enc_ids, int n);
void ne_schema_cache_free(ne_schema_cache_t *cache);

/* Measured payload bytes (K + V + residual + key ids), not malloc overhead. */
size_t ne_schema_cache_bytes(const ne_schema_cache_t *cache);
int    ne_schema_cache_ns(const ne_schema_cache_t *cache);
/* Schema token key; *ns_out = length. Valid until free. */
const int32_t *ne_schema_cache_key(const ne_schema_cache_t *cache, int *ns_out);
/* 1 if enc_ids' schema portion matches the cache key. */
int ne_schema_cache_matches(const ne_schema_cache_t *cache,
                            const int32_t *enc_ids, int n);

/* Warm encode / generate. Returns -1 on key mismatch or bad input; does not
 * fall back to exact encode (caller must call ne_encode / ne_generate). */
int ne_encode_warm(ne_ctx_t *ctx, const int32_t *enc_ids, int n,
                   const ne_schema_cache_t *cache);
int ne_generate_warm(ne_ctx_t *ctx, const int32_t *enc_ids, int n_enc,
                     int32_t *out, int max_gen,
                     const ne_schema_cache_t *cache);

/* Last warm encode's row counts (FLOP reduction report). */
extern int ne_warm_nq, ne_warm_ns, ne_warm_ntot;
/* row_layers = NE_WARM_F*nq + (12-NE_WARM_F)*ntot; full = 12*ntot. */
extern int ne_warm_row_layers, ne_warm_row_layers_full;

/* Test hooks: mutate cached tensors then restore. NULL if OOB.
 * k_row: [KVDIM] at (layer, schema_row). x_row: residual [DMODEL] at schema_row. */
float *ne_schema_cache_k_row(ne_schema_cache_t *cache, int layer, int row);
float *ne_schema_cache_x_row(ne_schema_cache_t *cache, int row);
#endif

#ifdef NE_PIE
/* pie_rows.S: per-group int32 partial sums for one weight row against a
 * pre-quantized int8 activation. Both operands 16-byte aligned -- esp.vld.128
 * silently mis-loads otherwise. */
void pie_row_i8_g32(const int8_t *xq, const int8_t *w, int ngroups, int32_t *gsums);
void pie_row_i4_g32(const int8_t *xq, const uint8_t *w_packed, int ngroups, int32_t *gsums);

/* Single-drain variants for one-scale-per-row weights: the whole row lands in
 * one int32, so the caller does one multiply instead of a per-group tail. */
int32_t pie_rowsum_i8(const int8_t *xq, const int8_t *w, int nchunks);
int32_t pie_rowsum_i4(const int8_t *xq, const uint8_t *w_packed, int nchunks);

/* pie_attn.S: out[r] = dot(a, B + r*len) in int16. len must be a multiple of 32,
 * both pointers 16-byte aligned, and the first operand needs one vector of
 * readable slack past the end -- the fused load runs a chunk ahead of the
 * accumulate.
 *
 * Alignment is a choice, not a hardware limit: esp.ld.128.usar.ip plus
 * esp.src.q loads across a misaligned boundary. It costs an extra instruction
 * per load, so this engine aligns everything by construction instead. Worth
 * knowing before concluding a layout is impossible.
 *
 * This kernel accumulates into ACCX (one 40-bit accumulator): one dot product
 * per pass over B. An eight-lane QACC variant (pie_qk8_s16 in bench/gemm) was
 * measured and lost in-engine: the QACC drain emits 64 bytes per key, and the
 * score consumer's 64-byte stride costs more than the kernel saves (prefill
 * 6365 -> 6451 ms). Recorded negative; see bench/gemm/README. The landed
 * successor on the attention MAC path is pie_dots_s8 under NE_ATTN_I8
 * (docs/notebook/KERNEL-S8.md / docs/notebook/RESULTS-INT8.md) -- host-proven, board pending. */
void pie_dots_s16(const int16_t *a, const int16_t *B, int len, int nrows,
                  int32_t *out);

/* The s8 twin (NE_ATTN_I8 path): 16 lanes per vector, len must be a multiple
 * of 64 -- the engine rounds Sp to 64 under NE_ATTN_I8 for exactly this.
 * Same alignment and one-vector (16 int8) A-side slack rules. Host-verified
 * successor to the QACC experiment above; board numbers still pending. */
void pie_dots_s8(const int8_t *a, const int8_t *B, int len, int nrows,
                 int32_t *out);

/* Encoder attention, integer (ne_aq_t). Heads are independent -- each pairs
 * with kv head h/2 and writes its own 64-float slice of every output row -- so
 * a head range can run on either core. Scratch is per-core: the split gives
 * the low range to the caller and the high range to the worker. */
typedef struct {
    float   *sc;        /* [S]         score row                             */
    int32_t *iw;        /* [max(S,64)] kernel output                         */
    ne_aq_t *qq;        /* [64]        quantized query, + one vector of slack */
    ne_aq_t *pq;        /* [Sp]        quantized probabilities, + a vector    */
} ne_attn_work_t;

typedef struct {
    const float   *q;                  /* [S][NE_DMODEL], normed and roped   */
    const ne_aq_t *kq;  const float *ksc;   /* [4][S][64],  scale per row    */
    const ne_aq_t *vq;  const float *vsc;   /* [4][64][Sp], scale per dim    */
    float         *out;                /* [S][NE_DMODEL]                     */
    int S, Sp;
    ne_attn_work_t work[2];
} ne_attn_job_t;

void ne_attn_heads(const ne_attn_job_t *job, uint32_t h0, uint32_t h1);
extern void (*ne_split_attn)(const ne_attn_job_t *job, uint32_t heads);

/* Scalar parallel-for: run fn(arg, i0, i1) on both cores, covering [0, n).
 * For the quantizers and norms, which were core-0-only while core 1 idled.
 * fn must write disjoint outputs per index and use no PIE state -- it runs on
 * core 1 outside the scheduler guard. NULL means single-core. */
extern void (*ne_split_for)(void (*fn)(void *, int, int), void *arg, int n);

/* 0 = vector path (PIE kernels), 1 = scalar path. */
extern int ne_force_scalar;

/* One quantized GEMV, split into row ranges. The activation is already
 * quantized (xq/sx) and is read-only for the duration, so ranges are
 * independent and may run on different cores. */
typedef struct {
    const uint8_t *qd;       /* weight payload base                         */
    const float   *sc32;     /* fp32 weight scales, [rows][ngroups]         */
    const int8_t  *xq;       /* int8 activation, shared                     */
    const float   *sx;       /* per-group activation scales, shared         */
    size_t         stride;   /* bytes per weight row                        */
    uint32_t       ngroups;
    int            i8;       /* 1 = int8 weights, 0 = packed int4           */
    float         *y;        /* output, one float per row                   */
} ne_gemv_job_t;

void ne_gemv_rows(const ne_gemv_job_t *job, uint32_t r0, uint32_t r1);

/* Encoder projections in blocks of tokens.
 *
 * One activation vector per token means one full sweep of the weight tensor
 * per token: at 271 tokens that is 1.3 GB of PSRAM traffic for the encoder.
 * Sweeping the weights once for a block of NE_TBLK activations instead cuts it
 * by that factor, and the block (NE_TBLK * NE_DMODEL bytes) stays in cache
 * while the sweep runs. Each output is still one dot product accumulated in
 * the same order, so results are unchanged. */

typedef struct {
    const uint8_t *qd;       /* weight payload base                          */
    const float   *sc32;     /* fp32 weight scales, [rows][ngroups]          */
    const int8_t  *xq;       /* [nb][cols] int8 activations, one per token   */
    const float   *sx;       /* [nb][ngroups] activation scales              */
    size_t         stride;   /* bytes per weight row                         */
    uint32_t       ngroups;
    int            i8;
    int            nb;       /* tokens in this block                         */
    float         *y;        /* y[t * ldy + row]                             */
    size_t         ldy;
} ne_gemm_job_t;

void ne_gemm_rows(const ne_gemm_job_t *job, uint32_t r0, uint32_t r1);
extern void (*ne_split_gemm)(const ne_gemm_job_t *job, uint32_t rows);

/* Install to fan a large row loop across cores; NULL keeps everything on the
 * calling core. The callee must cover [0, rows) exactly and return only once
 * every range has been written. */
extern void (*ne_split_rows)(const ne_gemv_job_t *job, uint32_t rows);
#define NE_SPLIT_MIN_ROWS 256



#endif

#endif
