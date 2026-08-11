/* PIE int16 dot kernel for encoder attention.
 *
 * Attention is the one place the engine multiplies two activations rather than
 * an activation by a weight, so there is no packed weight format to lean on and
 * both operands have to be quantized at runtime. int16 keeps the fixture output
 * byte-identical to the fp32 path and takes encoder attention from 30 s to 6 s
 * at 271 tokens.
 *
 * The loop is the esp-dsp fused idiom: one vld for the B operand, one
 * multiply-accumulate that also loads the next A chunk, accumulating into xacc
 * and draining once per row. esp.vld.128 moves 8 int16 lanes.
 *
 * Four of those pairs per iteration, so the two loop-control instructions are
 * amortised over 32 lanes instead of 8: 10 instructions per 32 MACs against 16.
 * This loop runs 0.9 G MACs per prefill, so that ratio is the whole game. len
 * must be a multiple of 32 -- callers pass 64 (one head) or the key count
 * rounded up to 32.
 *
 * Overflow: the caller quantizes to +-4096 for q/k/v and 0..16384 for
 * probabilities, so the widest sum here is 64 * 4096 * 4096 = 1.07e9 (scores)
 * and about 1.7e8 (values, where the probabilities sum to one). Both fit int32,
 * which is why the drain shift is zero and the result needs no saturation.
 *
 * The .ip address register has to come from the compressed register set
 * (a0-a5); a6/a7 assemble nowhere. Both pointers must be 16-byte aligned;
 * esp.vld.128 mis-loads silently otherwise. The fused load runs one chunk
 * ahead, so it reads 16 bytes past the
 * end of the last A row -- callers leave a vector of slack there.
 */

    .text
    .align 4

/* void pie_dots_s16(const int16_t *a, const int16_t *B, int len, int nrows,
 *                   int32_t *out)
 *   out[r] = sum_i a[i] * B[r * len + i]
 */
    .global pie_dots_s16
    .type   pie_dots_s16,@function
pie_dots_s16:
    mv      t0, a4                          /* out; a4 is needed for the drain */
    srli    t1, a2, 5                       /* blocks of 32 lanes per row     */
    li      a4, 0                           /* srs shift: a-class reg required */
.pd16_row:
    esp.zero.xacc
    mv      a5, a0                          /* a5 walks A, reset every row    */
    mv      t2, t1
    esp.vld.128.ip      q0, a5, 16          /* q0 = a[0..7]                   */
.pd16_blk:
    esp.vld.128.ip      q1, a1, 16
    esp.vmulas.s16.xacc.ld.ip q0, a5, 16, q0, q1
    esp.vld.128.ip      q1, a1, 16
    esp.vmulas.s16.xacc.ld.ip q0, a5, 16, q0, q1
    esp.vld.128.ip      q1, a1, 16
    esp.vmulas.s16.xacc.ld.ip q0, a5, 16, q0, q1
    esp.vld.128.ip      q1, a1, 16
    esp.vmulas.s16.xacc.ld.ip q0, a5, 16, q0, q1
    addi    t2, t2, -1
    bgtz    t2, .pd16_blk
    esp.srs.s.xacc      t5, a4
    sw      t5, 0(t0)
    addi    t0, t0, 4
    addi    a3, a3, -1
    bgtz    a3, .pd16_row
    ret

/* void pie_dots_s8(const int8_t *a, const int8_t *B, int len, int nrows,
 *                  int32_t *out)
 *   out[r] = sum_i a[i] * B[r * len + i]
 *
 * The s8 twin of pie_dots_s16 for the NE_ATTN_I8 attention path: same fused
 * esp-dsp idiom, but esp.vld.128 moves 16 int8 lanes, so the four fused pairs
 * per iteration cover 64 MACs against 32 -- 10 instructions per 64 MACs. The
 * multiply-accumulate is esp.vmulas.s8.xacc.ld.ip, the instruction the
 * boot-verified fused path of pie_rowsum_i8 already runs.
 *
 * len must be a multiple of 64. Callers pass NE_HEADDIM (= 64) for score rows
 * and Sp for probability rows; under NE_ATTN_I8 the engine rounds Sp up to 64
 * (NE_SPALIGN in needle_engine.c) so both shapes satisfy the contract.
 *
 * Overflow: with NE_AQ = NE_PQ = 127 the widest row sum is
 * 64 * 127 * 127 = 1,032,256 (scores) and 127 * 127 = 16,129 (values, where
 * the quantized probabilities sum to ~NE_PQ). Both are far inside int32, let
 * alone the 40-bit ACCX, so the drain shift is zero and nothing saturates.
 *
 * Same operand rules as the s16 kernel: .ip address registers from a0-a5,
 * both pointers 16-byte aligned (esp.vld.128 mis-loads silently otherwise),
 * and the fused load runs one chunk ahead, reading 16 bytes -- 16 int8
 * elements -- past the end of the last A row. Callers leave a vector of slack.
 */
    .global pie_dots_s8
    .type   pie_dots_s8,@function
pie_dots_s8:
    mv      t0, a4                          /* out; a4 is needed for the drain */
    srli    t1, a2, 6                       /* blocks of 64 lanes per row     */
    li      a4, 0                           /* srs shift: a-class reg required */
.pd8_row:
    esp.zero.xacc
    mv      a5, a0                          /* a5 walks A, reset every row    */
    mv      t2, t1
    esp.vld.128.ip      q0, a5, 16          /* q0 = a[0..15]                  */
.pd8_blk:
    esp.vld.128.ip      q1, a1, 16
    esp.vmulas.s8.xacc.ld.ip q0, a5, 16, q0, q1
    esp.vld.128.ip      q1, a1, 16
    esp.vmulas.s8.xacc.ld.ip q0, a5, 16, q0, q1
    esp.vld.128.ip      q1, a1, 16
    esp.vmulas.s8.xacc.ld.ip q0, a5, 16, q0, q1
    esp.vld.128.ip      q1, a1, 16
    esp.vmulas.s8.xacc.ld.ip q0, a5, 16, q0, q1
    addi    t2, t2, -1
    bgtz    t2, .pd8_blk
    esp.srs.s.xacc      t5, a4
    sw      t5, 0(t0)
    addi    t0, t0, 4
    addi    a3, a3, -1
    bgtz    a3, .pd8_row
    ret
