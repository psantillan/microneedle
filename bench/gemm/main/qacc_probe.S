/* Probe for the QACC broadcast multiply-accumulate.
 *
 * The reference documentation says QACC is two 256-bit accumulators holding
 * "8 16-bit MAC results", and esp.st.qacc.* drains it in four 128-bit pieces.
 * It does not say which piece holds which lane, or how wide a lane is. This
 * runs the instruction on known inputs and dumps all 512 bits so the layout can
 * be read off the hardware instead of assumed.
 *
 * void qacc_probe(const int16_t *bc, const int16_t *vec, int n, void *out64)
 *   qacc[k] should end up as sum over i of bc[i] * vec[i*8 + k], k = 0..7.
 *   vec and out64 must be 16-byte aligned; out64 is 64 bytes.
 */
    .text
    .align 4
    .global qacc_probe
    .type   qacc_probe,@function
qacc_probe:
    esp.zero.qacc
    esp.vldbc.16.ip     q0, a0, 0           /* q0 = broadcast bc[0] */
    addi    a0, a0, 2                   /* imm steps by 4, so advance by hand */
.qp_loop:
    esp.vld.128.ip      q1, a1, 16          /* q1 = 8 vector lanes           */
    esp.vmulas.s16.qacc.ldbc.incp q0, a0, q0, q1   /* MAC, then broadcast next */
    addi    a2, a2, -1
    bgtz    a2, .qp_loop
    esp.st.qacc.l.l.128.ip  a3, 16
    esp.st.qacc.l.h.128.ip  a3, 16
    esp.st.qacc.h.l.128.ip  a3, 16
    esp.st.qacc.h.h.128.ip  a3, 16
    ret
