/* PIE row kernels for the Needle GEMV: per-group int32 partial sums for one
 * weight row against a pre-quantized int8 activation vector. C applies the
 * combined (weight_scale x act_scale) fp32 factors.
 *
 * NPK2 int4 layout (tools/pack_npk.py): one group of 32 = 16 bytes, byte j holding
 * elem j in the low nibble and elem j+16 in the high nibble, so a single
 * vld.128 + mask/shift lands both halves in activation lane order. Nibbles are
 * two's-complement 4-bit, sign-extended in-register via (v ^ 8) - 8.
 *
 * Register-class limits: esp.srs.s.xacc rejects t1/t2/s2 as destination and
 * rejects non-a-class shift operands; a0/t3/t5 as destination with a4/a5 as the
 * shift assemble. SAR (used by esp.vsr.u32) is state, written once via
 * esp.movx.w.sar.
 *
 * All esp.vld.128 operands must be 16-byte aligned; the load mis-reads silently
 * otherwise.
 */

    .text
    .align 4

    .section .rodata
    .align 2
pie_c0f: .byte 0x0F
pie_c08: .byte 0x08
    .text

/* void pie_row_i8_g32(const int8_t *xq, const int8_t *w, int ngroups,
 *                     int32_t *gsums)
 * x and w 16-byte aligned. */
    .global pie_row_i8_g32
    .type   pie_row_i8_g32,@function
pie_row_i8_g32:
    li      a5, 0
.pi8_group:
    esp.zero.xacc
    esp.vld.128.ip      q0, a0, 16
    esp.vld.128.ip      q1, a1, 16
    esp.vmulas.s8.xacc  q0, q1
    esp.vld.128.ip      q0, a0, 16
    esp.vld.128.ip      q1, a1, 16
    esp.vmulas.s8.xacc  q0, q1
    esp.srs.s.xacc      t5, a5
    sw      t5, 0(a3)
    addi    a3, a3, 4
    addi    a2, a2, -1
    bgtz    a2, .pi8_group
    ret

/* void pie_row_i4_g32(const int8_t *xq, const uint8_t *w_packed, int ngroups,
 *                     int32_t *gsums) */
    .global pie_row_i4_g32
    .type   pie_row_i4_g32,@function
pie_row_i4_g32:
    /* esp.vldbc.8 constants must be loaded from .rodata through a plain pointer
     * with step 0, one la per constant. Sourcing them from stack scratch, or
     * stepping the .ip between the two loads, splats garbage. */
    la      a4, pie_c0f
    esp.vldbc.8.ip      q7, a4, 0           /* q7 = 0x0F in every byte lane */
    la      a4, pie_c08
    esp.vldbc.8.ip      q6, a4, 0           /* q6 = 0x08 in every byte lane */
    li      a4, 4
    esp.movx.w.sar      a4                  /* SAR = 4 for the nibble shift  */
    li      a5, 0
.pi4_group:
    esp.vld.128.ip      q0, a1, 16          /* 16 packed bytes = 32 nibbles  */
    esp.vld.128.ip      q1, a0, 16          /* xq elems 0..15                */
    esp.vld.128.ip      q2, a0, 16          /* xq elems 16..31               */
    esp.andq            q3, q0, q7          /* low nibbles, raw 0..15        */
    esp.vsr.u32         q4, q0              /* lanes >> SAR(4)               */
    esp.andq            q4, q4, q7          /* high nibbles, raw 0..15       */
    esp.xorq            q3, q3, q6
    esp.vsub.s8         q3, q3, q6          /* (v^8)-8: signed -8..7         */
    esp.xorq            q4, q4, q6
    esp.vsub.s8         q4, q4, q6
    esp.zero.xacc
    esp.vmulas.s8.xacc  q1, q3
    esp.vmulas.s8.xacc  q2, q4
    esp.srs.s.xacc      t5, a5
    sw      t5, 0(a3)
    addi    a3, a3, 4
    addi    a2, a2, -1
    bgtz    a2, .pi4_group
    ret

/* ---------------------------------------------------------------------------
 * Single-drain row kernels: the whole row accumulates in xacc and drains ONCE.
 *
 * Only usable when a row carries a single quantization scale (tools/pack_npk.py
 * --group 512): a group-32 row needs the accumulator drained and re-zeroed once
 * per group so each group's scale can be applied outside, which is what the
 * kernels above do. With one scale per row the caller applies a single multiply
 * to the returned sum.
 *
 * Overflow: xacc is wider than 32 bits and the worst case here is
 * 512 * 127 * 127 = 8.26e6, far inside int32 either way.
 *
 * pie_rowsum_i8 takes the esp-dsp fused idiom when the row is a whole number of
 * 128-element blocks, which every tensor here is: one vld for the weights and
 * one multiply-accumulate that also loads the next activation chunk, unrolled
 * eight deep. That is 18 instructions per 128 MACs against 32 for the plain
 * loop kept below for other lengths. The fused load runs a chunk ahead, so it
 * reads 16 bytes past the end of xq -- callers leave a vector of slack.
 * ------------------------------------------------------------------------- */

/* int32_t pie_rowsum_i8(const int8_t *xq, const int8_t *w, int nchunks) */
    .global pie_rowsum_i8
    .type   pie_rowsum_i8,@function
pie_rowsum_i8:
    esp.zero.xacc
    andi    a5, a2, 3
    bnez    a5, .rs8_chunk              /* not a whole 128-element block  */
    srli    a2, a2, 2
    esp.vld.128.ip      q0, a0, 16      /* preload x; the fused MAC below
                                         * loads the next chunk for free  */
.rs8_128:
    esp.vld.128.ip      q1, a1, 16
    esp.vmulas.s8.xacc.ld.ip q0, a0, 16, q0, q1
    esp.vld.128.ip      q1, a1, 16
    esp.vmulas.s8.xacc.ld.ip q0, a0, 16, q0, q1
    esp.vld.128.ip      q1, a1, 16
    esp.vmulas.s8.xacc.ld.ip q0, a0, 16, q0, q1
    esp.vld.128.ip      q1, a1, 16
    esp.vmulas.s8.xacc.ld.ip q0, a0, 16, q0, q1
    esp.vld.128.ip      q1, a1, 16
    esp.vmulas.s8.xacc.ld.ip q0, a0, 16, q0, q1
    esp.vld.128.ip      q1, a1, 16
    esp.vmulas.s8.xacc.ld.ip q0, a0, 16, q0, q1
    esp.vld.128.ip      q1, a1, 16
    esp.vmulas.s8.xacc.ld.ip q0, a0, 16, q0, q1
    esp.vld.128.ip      q1, a1, 16
    esp.vmulas.s8.xacc.ld.ip q0, a0, 16, q0, q1
    addi    a2, a2, -1
    bgtz    a2, .rs8_128
    li      a5, 0
    esp.srs.s.xacc      a0, a5
    ret
.rs8_chunk:
    esp.vld.128.ip      q0, a0, 16
    esp.vld.128.ip      q1, a1, 16
    esp.vmulas.s8.xacc  q0, q1
    esp.vld.128.ip      q0, a0, 16
    esp.vld.128.ip      q1, a1, 16
    esp.vmulas.s8.xacc  q0, q1
    addi    a2, a2, -1
    bgtz    a2, .rs8_chunk
    li      a5, 0
    esp.srs.s.xacc      a0, a5
    ret

/* int32_t pie_rowsum_i4(const int8_t *xq, const uint8_t *w_packed, int nchunks) */
    .global pie_rowsum_i4
    .type   pie_rowsum_i4,@function
pie_rowsum_i4:
    la      a4, pie_c0f
    esp.vldbc.8.ip      q7, a4, 0
    la      a4, pie_c08
    esp.vldbc.8.ip      q6, a4, 0
    li      a4, 4
    esp.movx.w.sar      a4
    esp.zero.xacc
.rs4_chunk:
    esp.vld.128.ip      q0, a1, 16          /* 32 nibbles                    */
    esp.vld.128.ip      q1, a0, 16          /* xq 0..15                      */
    esp.vld.128.ip      q2, a0, 16          /* xq 16..31                     */
    esp.andq            q3, q0, q7
    esp.vsr.u32         q4, q0
    esp.andq            q4, q4, q7
    esp.xorq            q3, q3, q6
    esp.vsub.s8         q3, q3, q6
    esp.xorq            q4, q4, q6
    esp.vsub.s8         q4, q4, q6
    esp.vmulas.s8.xacc  q1, q3
    esp.vmulas.s8.xacc  q2, q4
    addi    a2, a2, -1
    bgtz    a2, .rs4_chunk
    li      a5, 0
    esp.srs.s.xacc      a0, a5
    ret
