/* PIE s8 dot-product kernels for the utilization benchmark.
 *
 * Idiom taken from esp-dsp's dspi_dotprod_s8_arp4.S (same silicon, same
 * toolchain): preload one operand into q0, then a hardware loop whose body is
 *   esp.vld.128.ip           q1, <w ptr>, 16     ; 16 weight bytes
 *   esp.vmulas.s8.xacc.ld.ip q0, <x ptr>, 16, q0, q1  ; 16 MACs + next x load
 * i.e. 2 instructions per 16 MACs, accumulating into the wide xacc register,
 * drained once at the end with esp.srs.s.xacc (shift 0 -- our sums stay far
 * below 2^31: 512 * 127 * 127 = 8.3e6).
 *
 * pie_dot_s8_g32 is the variant the real GEMV needs: same MAC loop but the
 * accumulator is drained and re-zeroed every 32 elements so a per-group scale
 * can be applied outside. Its overhead versus the single-drain version is
 * exactly the price of group-32 quantization on this hardware -- that price is
 * what the benchmark exists to measure.
 *
 * Both functions require x and w 16-byte aligned and len a multiple of 32.
 */

    .text
    .align 4

/* int32_t pie_dot_s8(const int8_t *x, const int8_t *w, int len) */
    .global pie_dot_s8
    .type   pie_dot_s8,@function
pie_dot_s8:
    esp.zero.xacc
    esp.vld.128.ip      q0, a0, 16          /* q0 = x[0..15], a0 += 16 */
    srli    t0, a2, 4                       /* t0 = len/16 iterations   */
    esp.lp.setup 0, t0, .pd_loop_end
        esp.vld.128.ip  q1, a1, 16          /* q1 = w[0..15]            */
.pd_loop_end:
        esp.vmulas.s8.xacc.ld.ip q0, a0, 16, q0, q1
    li      a5, 0
    esp.srs.s.xacc      a0, a5              /* a0 = xacc >> 0 (low 32)  */
    ret

/* void pie_dot_s8_g32(const int8_t *x, const int8_t *w, int len,
 *                     int32_t *gsums)   -- one int32 partial per 32 elems */
    .global pie_dot_s8_g32
    .type   pie_dot_s8_g32,@function
pie_dot_s8_g32:
    srli    t0, a2, 5                       /* t0 = number of groups     */
    li      a5, 0                           /* srs shift amount (a-class reg required) */
.pg_group:
    esp.zero.xacc
    /* 32 elements = 2x (vld + fused mac/ld), straight-line, no loop setup */
    esp.vld.128.ip      q0, a0, 16
    esp.vld.128.ip      q1, a1, 16
    esp.vmulas.s8.xacc  q0, q1
    esp.vld.128.ip      q0, a0, 16
    esp.vld.128.ip      q1, a1, 16
    esp.vmulas.s8.xacc  q0, q1
    esp.srs.s.xacc      t5, a5
    sw      t5, 0(a3)
    addi    a3, a3, 4
    addi    t0, t0, -1
    bgtz    t0, .pg_group
    ret
