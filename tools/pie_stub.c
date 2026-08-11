/* Scalar stand-ins for the vector kernels.
 *
 * The host build compiles the engine without NE_PIE, which means the whole
 * vector code path -- the batched projections, the single-drain row sums, the
 * group-32 kernels -- is never exercised until it reaches the board. Linking
 * these against a -DNE_PIE build closes that gap: same arithmetic, no assembly,
 * so a logic error in the vector path shows up on the host.
 *
 * It cannot catch everything. Alignment is the obvious hole: esp.vld.128
 * mis-loads on a non-16-byte-aligned operand and a scalar load does not care.
 *
 * Built by verify.sh; see the "vector path logic" step. */
#include <stdint.h>
#include <stddef.h>
#define NE_PACK 32
int32_t pie_rowsum_i8(const int8_t *xq, const int8_t *w, int nchunks) {
    int32_t s = 0; for (int i = 0; i < nchunks * NE_PACK; i++) s += (int32_t)xq[i] * w[i];
    if (nchunks % 4 == 0) {          /* the fused path over-reads; so does this */
        volatile int8_t sink = 0;
        for (int i = 0; i < 16; i++) sink = xq[nchunks * NE_PACK + i];
        (void)sink;
    }
    return s;
}
int32_t pie_rowsum_i4(const int8_t *xq, const uint8_t *w, int nchunks) {
    int32_t s = 0;
    for (int c = 0; c < nchunks; c++)
        for (int k = 0; k < NE_PACK/2; k++) {
            uint8_t b = w[c*(NE_PACK/2)+k];
            int lo = (int)(b & 0xF) - ((b & 0x8) ? 16 : 0);
            int hi = (int)(b >> 4) - ((b & 0x80) ? 16 : 0);
            s += lo * xq[c*NE_PACK+k] + hi * xq[c*NE_PACK+k+NE_PACK/2];
        }
    return s;
}
/* The real kernel's fused load runs a chunk ahead, so it reads 8 lanes past the
 * end of `a`. The stub reads them too and throws them away: under a sanitizer
 * that turns a caller who forgot the slack into a diagnosed overrun instead of
 * a silent difference between host and board. */
void pie_dots_s16(const int16_t *a, const int16_t *B, int len, int nrows, int32_t *out) {
    volatile int16_t sink = 0;
    for (int r = 0; r < nrows; r++) {
        int32_t s = 0;
        for (int i = 0; i < len; i++) s += (int32_t)a[i] * (int32_t)B[(size_t)r * len + i];
        for (int i = len; i < len + 8; i++) sink = a[i];
        out[r] = s;
    }
    (void)sink;
}
/* s8 twin: one 128-bit vector is 16 int8 lanes, so the fused over-read is 16
 * elements past the end of `a`. */
void pie_dots_s8(const int8_t *a, const int8_t *B, int len, int nrows, int32_t *out) {
    volatile int8_t sink = 0;
    for (int r = 0; r < nrows; r++) {
        int32_t s = 0;
        for (int i = 0; i < len; i++) s += (int32_t)a[i] * (int32_t)B[(size_t)r * len + i];
        for (int i = len; i < len + 16; i++) sink = a[i];
        out[r] = s;
    }
    (void)sink;
}
void pie_row_i8_g32(const int8_t *xq, const int8_t *w, int ng, int32_t *g) {
    for (int i = 0; i < ng; i++) g[i] = pie_rowsum_i8(xq + i*NE_PACK, w + i*NE_PACK, 1);
}
void pie_row_i4_g32(const int8_t *xq, const uint8_t *w, int ng, int32_t *g) {
    for (int i = 0; i < ng; i++) g[i] = pie_rowsum_i4(xq + i*NE_PACK, w + i*(NE_PACK/2), 1);
}
