/* Eight query dot-products per pass over K, using QACC.
 *
 * pie_dots_s16 accumulates into ACCX -- one 40-bit accumulator -- so it yields
 * one dot product per pass and the engine streams the whole of K once per
 * query. QACC is two 256-bit accumulators, which at int16 is 8 independent
 * 64-bit lanes, and esp.vmulas.s16.qacc.ldbc.incp broadcasts one operand
 * against a vector of 8. Broadcasting the K element and holding 8 queries in
 * the vector therefore computes 8 dot products from one pass over K.
 *
 * The instruction count per MAC is the same. The point is traffic: K is
 * streamed once per eight queries instead of once per query, and bench/gemm
 * shows traffic is what binds here.
 *
 * void pie_qk8_s16(const int16_t *QT, const int16_t *K, int nkeys, void *out)
 *   QT   [64][8] int16, transposed: QT[d][q]. 1 KB, stays in L1.
 *   K    [nkeys][64] int16, streamed. Needs 2 bytes of slack past the end --
 *        the fused broadcast load runs one element ahead.
 *   out  nkeys * 64 bytes. Lane q of key j is the int32 at out[j*16 + 2*q];
 *        lanes are 64-bit, so the odd words are the high halves.
 *   All three pointers 16-byte aligned.
 *
 * Lane geometry and the 64-bit width were read off the hardware, not assumed;
 * see stage_qacc_probe in gemm_main.c.
 */
    .text
    .align 4
    .global pie_qk8_s16
    .type   pie_qk8_s16,@function
pie_qk8_s16:
.qk8_key:
    esp.zero.qacc
    mv      a4, a0                  /* QT restarts for every key            */
    esp.vldbc.16.ip     q0, a1, 0   /* broadcast K[j][0]; imm steps by 4, so */
    addi    a1, a1, 2               /* advance by one element by hand        */
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    esp.vld.128.ip      q1, a4, 16
    esp.vmulas.s16.qacc.ldbc.incp q0, a1, q0, q1
    /* The preload consumed K[j][0] without advancing and the manual bump then
     * moved to K[j][1], so 64 fused loads leave the pointer one element past
     * the start of the next key's row. Put it back. */
    addi    a1, a1, -2
    esp.st.qacc.l.l.128.ip  a3, 16
    esp.st.qacc.l.h.128.ip  a3, 16
    esp.st.qacc.h.l.128.ip  a3, 16
    esp.st.qacc.h.h.128.ip  a3, 16
    addi    a2, a2, -1
    bgtz    a2, .qk8_key
    ret
