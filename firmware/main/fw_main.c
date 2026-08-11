/* Needle on the ESP32-P4 -- PIE-accelerated firmware. Kernel self-tests run at
 * every boot and gate the HTTP API.
 * Serial protocol: "G <n> <id>..." in, "R <n> <id>... enc_ms= tok_ms=" out.
 * Token ids in and out; tokenization runs on the host.
 */

#include <stdio.h>
#include <string.h>
#include <inttypes.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_heap_caps.h"
#include "esp_partition.h"
#include "esp_timer.h"

#include "driver/uart.h"
#include "driver/uart_vfs.h"

#include "needle_engine.h"
#include "net.h"

/* Self-test lines captured at boot, re-served over HTTP so a board whose
 * kernels failed is visible in the browser instead of quietly answering. */
static char g_selftest[512];
static int  g_selftest_failed;      /* any boot check that did not pass */
static void selftest_record(const char *line)
{
    size_t n = strlen(g_selftest);
    snprintf(g_selftest + n, sizeof g_selftest - n, "%s\"%s\"", n ? "," : "", line);
    if (strstr(line, "FAIL")) g_selftest_failed = 1;
}
const char *net_selftest_json(void) { return g_selftest; }
int net_selftest_failed(void) { return g_selftest_failed; }

static ne_ctx_t ctx;   /* ne_force_scalar + pie_row_* come from needle_engine.h */

/* esp_timer_get_time returns int64_t; calling it through a uint64_t(*)(void)
 * is an incompatible-type call, so the engine hook goes through a wrapper. */
static uint64_t now_us(void) { return (uint64_t)esp_timer_get_time(); }

/* ---------- second core ----------
 * Each core copies with plain memcpy into its own SRAM tile.
 *
 * Handshake is raw volatiles, not FreeRTOS primitives, because both cores run
 * with their schedulers suspended: IDF 5.4.2's lazy PIE context save skips q3
 * (portasm.S), so a task switch inside a kernel corrupts the row. */
static volatile const void *w_job;
static volatile int      w_kind;          /* 0 gemv rows, 1 gemm rows, 2 attention head range, 3 scalar for */
static volatile uint32_t w_r0, w_r1;
static void (*volatile w_fn)(void *, int, int);   /* kind 3: the loop body */
static void *volatile    w_arg;
static volatile int      w_go, w_done, w_ready;

static void core1_worker(void *arg)
{
    (void)arg;
    vTaskSuspendAll();                        /* this core's scheduler, only */
    w_ready = 1;
    for (;;) {
        while (!w_go) { __asm__ volatile("nop"); }
        __sync_synchronize();                 /* acquire: w_job/w_gemm/w_r0/w_r1 */
        switch (w_kind) {
        case 1:  ne_gemm_rows((const ne_gemm_job_t *)w_job, w_r0, w_r1); break;
        case 2:  ne_attn_heads((const ne_attn_job_t *)w_job, w_r0, w_r1); break;
        case 3:  w_fn(w_arg, (int)w_r0, (int)w_r1); break;
        default: ne_gemv_rows((const ne_gemv_job_t *)w_job, w_r0, w_r1); break;
        }
        __sync_synchronize();
        w_done = 1;
        while (w_go) { __asm__ volatile("nop"); }
        w_done = 0;
    }
}

/* Split by rows. The activation is already quantized and read-only, so the two
 * halves never touch the same bytes -- only the output is written, disjointly.
 * Attention splits by head index instead of by row, on the same terms -- each
 * head writes its own disjoint 64-float slice of every output row. The range is
 * an index into an INTERLEAVED head order (see ne_attn_heads): [0,4) is heads
 * 0,2,4,6 and [4,8) is 1,3,5,7, so both cores walk kv heads 0,1,2,3 together
 * and share one K/V set rather than streaming two through a 128 KB L2. All
 * three go through one handshake:
 * the close below took a while to get right and there is no reason to have two
 * copies of it. */
static void split_run(const void *job, int kind, uint32_t rows)
{
    uint32_t mid = rows / 2;
    w_job = job; w_kind = kind; w_r0 = mid; w_r1 = rows;
    __sync_synchronize();
    w_go = 1;
    switch (kind) {
    case 1:  ne_gemm_rows((const ne_gemm_job_t *)job, 0, mid); break;
    case 2:  ne_attn_heads((const ne_attn_job_t *)job, 0, mid); break;
    case 3:  w_fn(w_arg, 0, (int)mid); break;
    default: ne_gemv_rows((const ne_gemv_job_t *)job, 0, mid); break;
    }
    while (!w_done) { __asm__ volatile("nop"); }
    __sync_synchronize();                     /* acquire: the far half of y */
    w_go = 0;
    /* Wait for the worker to actually leave the round before returning.
     * Without this the handshake is only half closed: if core 1 is held in
     * `while (w_go)` by an interrupt while core 0 races ahead and raises w_go
     * for the NEXT job, core 1 never observes the 0, stays in the old round,
     * and leaves w_done set. Core 0's `while (!w_done)` then passes instantly
     * and publishes the PREVIOUS job's far half as this job's result -- silent
     * wrong logits, no hang, no crash. */
    while (w_done) { __asm__ volatile("nop"); }
    __sync_synchronize();
}

static void split_rows(const ne_gemv_job_t *job, uint32_t rows) { split_run(job, 0, rows); }
static void split_gemm(const ne_gemm_job_t *job, uint32_t rows) { split_run(job, 1, rows); }
static void split_attn(const ne_attn_job_t *job, uint32_t heads) { split_run(job, 2, heads); }

/* Scalar loops ride the same handshake as job kind 3. The bodies use no PIE
 * state, so no scheduler guard is needed around them; the close-out discipline
 * is identical. */
static void split_for(void (*fn)(void *, int, int), void *arg, int n)
{
    w_fn = fn; w_arg = arg;
    split_run(NULL, 3, (uint32_t)n);
}

/* weather_sf fixture: input ids and the expected greedy output. run_query
 * compares against SMOKE_REF, so a decoder that is wrong fails the boot
 * self-test instead of printing plausible garbage. */
static const int32_t SMOKE_REF[] = {
    4, 356, 294, 264, 358, 8062, 1331, 265, 393, 282,
    506, 264, 8074, 327, 1295, 1075, 378, 275, 8047, 503
};
static const int32_t SMOKE[] = {
    4279, 8066, 8046, 302, 1149, 362, 711, 327, 1295, 1075, 378, 275, 8047, 8105,
    5, 356, 294, 264, 358, 8062, 1331, 265, 283, 264, 618, 407, 1149, 345,
    289, 2082, 284, 318, 282, 506, 282, 298, 264, 315, 265, 283, 264, 2523,
    417, 284, 301, 262, 312, 434
};

/* Kernel unit tests on synthetic data with a known dot product. */
static void kernel_unit_tests(void)
{
    static int8_t x[64] __attribute__((aligned(16)));
    static int8_t e[64] __attribute__((aligned(16)));
    static uint8_t w4[32] __attribute__((aligned(16)));
    int32_t gs[2], ref[2];

    for (int i = 0; i < 64; i++) {
        x[i] = (int8_t)(i - 32);
        int v = (i * 7) % 15 - 7;
        e[i] = (int8_t)(v > 7 ? 7 : (v < -8 ? -8 : v));
    }
    for (int g = 0; g < 2; g++)
        for (int j = 0; j < 16; j++)
            w4[g * 16 + j] = (uint8_t)((e[g * 32 + j] & 0xF) | ((e[g * 32 + 16 + j] & 0xF) << 4));
    for (int g = 0; g < 2; g++) {
        ref[g] = 0;
        for (int k = 0; k < 32; k++) ref[g] += (int32_t)x[g * 32 + k] * e[g * 32 + k];
    }

    pie_row_i8_g32(x, e, 2, gs);
    char ln[96];
    snprintf(ln, sizeof ln, "UT i8: %s", (gs[0] == ref[0] && gs[1] == ref[1]) ? "PASS" : "FAIL");
    printf("%s\n", ln); selftest_record(ln);
    pie_row_i4_g32(x, w4, 2, gs);
    snprintf(ln, sizeof ln, "UT i4: %s", (gs[0] == ref[0] && gs[1] == ref[1]) ? "PASS" : "FAIL");
    printf("%s\n", ln); selftest_record(ln);

    /* Single-drain kernels must equal the sum of the per-group ones: same
     * MACs, one drain instead of two. nchunks=2 lands in pie_rowsum_i8's
     * fallback loop; every tensor in the engine is a whole number of 128s and
     * takes the 8-deep fused path, so test that branch too, at the engine's
     * own length. */
    int32_t whole = ref[0] + ref[1];
    int32_t r8 = pie_rowsum_i8(x, e, 2);
    int32_t r4 = pie_rowsum_i4(x, w4, 2);
    static int8_t xf[512 + 16] __attribute__((aligned(16)));
    static int8_t wf[512]      __attribute__((aligned(16)));
    int32_t reff = 0;
    for (int i = 0; i < 512; i++) {
        xf[i] = (int8_t)((i * 13 + 5) % 251 - 125);
        wf[i] = (int8_t)((i * 29 + 3) % 253 - 126);
        reff += (int32_t)xf[i] * wf[i];
    }
    int32_t rf = pie_rowsum_i8(xf, wf, 512 / 32);
    snprintf(ln, sizeof ln, "UT rowsum i8 fused: %s", rf == reff ? "PASS" : "FAIL");
    printf("%s\n", ln); selftest_record(ln);
    snprintf(ln, sizeof ln, "UT rowsum i8: %s", r8 == whole ? "PASS" : "FAIL");
    printf("%s\n", ln); selftest_record(ln);
    snprintf(ln, sizeof ln, "UT rowsum i4: %s", r4 == whole ? "PASS" : "FAIL");
    printf("%s\n", ln); selftest_record(ln);

    /* int16 attention kernel, three rows against a scalar reference. Values run
     * to the +-4096 the engine quantizes to, so the test exercises the widest
     * accumulator the kernel can be handed. */
    static int16_t a16[72] __attribute__((aligned(16)));   /* 64 + one vector of slack */
    static int16_t b16[3 * 64] __attribute__((aligned(16)));
    int32_t got[3], want[3];
    for (int i = 0; i < 64; i++) a16[i] = (int16_t)(4096 - 128 * i);
    for (int r = 0; r < 3; r++)
        for (int i = 0; i < 64; i++)
            b16[r * 64 + i] = (int16_t)(((i * 37 + r * 11) % 8193) - 4096);
    for (int r = 0; r < 3; r++) {
        want[r] = 0;
        for (int i = 0; i < 64; i++) want[r] += (int32_t)a16[i] * b16[r * 64 + i];
    }
    pie_dots_s16(a16, b16, 64, 3, got);
    int ok16 = got[0] == want[0] && got[1] == want[1] && got[2] == want[2];
    snprintf(ln, sizeof ln, "UT dots s16: %s", ok16 ? "PASS" : "FAIL");
    printf("%s\n", ln); selftest_record(ln);
}

/* ne_encode sets ne_align_fault if any vector operand it carved is not 16-byte
 * aligned. Run after the smoke query, which is the first encode. */
static void alignment_selftest(void)
{
    char ln[64];
    snprintf(ln, sizeof ln, "UT pie alignment: %s", ne_align_fault ? "FAIL" : "PASS");
    printf("%s\n", ln); selftest_record(ln);
}

/* Full-pipeline determinism check: the same probe down the scalar path twice.
 * The scalar path is bit-deterministic, so any difference in the logits or the
 * chosen token is memory corruption. */
static void determinism_selftest(void)
{
    static float ref[NE_VOCAB];
    static const int32_t probe[4] = {100, 200, 300, 400};

    ne_force_scalar = 1;
    vTaskSuspendAll();
    ne_encode(&ctx, probe, 4);
    int32_t t0 = ne_step(&ctx, NE_EOS_ID);
    xTaskResumeAll();
    memcpy(ref, ctx.logits, sizeof ref);

    vTaskSuspendAll();
    ne_encode(&ctx, probe, 4);
    int32_t t1 = ne_step(&ctx, NE_EOS_ID);
    xTaskResumeAll();
    float md = 0;
    for (int i = 0; i < NE_VOCAB; i++) {
        float d = ctx.logits[i] - ref[i];
        if (d < 0) d = -d;
        if (d > md) md = d;
    }
    ne_force_scalar = 0;

    char ln[96];
    snprintf(ln, sizeof ln, "UT determinism: %s (tok=%ld ref=%ld max|dlogit|=%.4f)",
             (md == 0.f && t1 == t0) ? "PASS" : "FAIL", (long)t1, (long)t0, md);
    printf("%s\n", ln); selftest_record(ln);
}

/* The engine calls these around each vector-kernel burst. The guard is taken
 * per burst, not per query, so a TCP stack can coexist with inference: the
 * longest window is still the 8192-row logits GEMV at ~40 ms when that GEMV
 * runs (grammar-forced steps skip it, wired below). */
static void crit_enter(void) { vTaskSuspendAll(); }
static void crit_exit(void)  { xTaskResumeAll(); }

#ifdef NE_WARMSCHEMA
/* Lazy warm-schema cache, one slot. First request whose schema (tokens from
 * the <tools> marker on) is big enough to be worth it captures during its own
 * exact encode; later requests with the SAME schema take the F=4 warm path.
 * Small (retrieval-pruned) schemas never touch the slot -- they are already
 * fast, and thrashing a 2.4 MiB capture per k=2 subset would cost more than
 * it saves. Battery on the shipping npk: 36/36 call-identical, two cases
 * re-tokenize the same argument string (docs/notebook/RESULTS-WARMF4.md + ledger). */
#define WARM_MIN_NS 128
static ne_schema_cache_t *g_warm;
char g_infer_path = 'e';               /* 'e' exact, 'c' capture, 'w' warm */
#endif

/* One query, for the HTTP front end. The engine takes the PIE guard itself. */
static int infer(const int32_t *ids, int n, int32_t *out, float *enc_ms, float *tok_ms)
{
    ne_us_logits = ne_us_proj = 0;
    ne_us_enc_proj = ne_us_enc_attn = ne_us_enc_xkv = ne_us_enc_smax = ne_us_enc_core = ne_us_enc_kvq = 0;
    int64_t t0 = esp_timer_get_time();
#ifdef NE_WARMSCHEMA
    g_infer_path = 'e';
    if (g_warm && ne_schema_cache_matches(g_warm, ids, n) &&
        ne_encode_warm(&ctx, ids, n, g_warm) == 0) {
        g_infer_path = 'w';
    } else {
        int tp = 0;
        while (tp < n && ids[tp] != 5) tp++;
        if (tp < n && n - tp >= WARM_MIN_NS) {
            ne_schema_cache_t *c = ne_schema_capture(&ctx, ids, n);
            if (c) {                    /* capture ran the full exact encode */
                if (g_warm) ne_schema_cache_free(g_warm);
                g_warm = c;
                g_infer_path = 'c';
            } else if (ne_encode(&ctx, ids, n) != 0) {
                return -1;              /* capture alloc failed; exact fallback */
            }
        } else if (ne_encode(&ctx, ids, n) != 0) {
            return -1;
        }
    }
#else
    if (ne_encode(&ctx, ids, n) != 0) return -1;
#endif
    int64_t t1 = esp_timer_get_time();
    int n_out = 0, n_265 = 0;
    int32_t tok = NE_EOS_ID;
    for (int i = 0; i < NE_MAX_GEN - 1; i++) {
        /* Same forcing ne_generate_cb applies: rigid JSON positions skip the
         * 8192-row logits GEMV. Counting mirrors that loop exactly. */
        int32_t forced = ne_grammar_force ? ne_forced_next(tok, n_out, n_265) : -1;
        tok = ne_step_f(&ctx, tok, forced);
        if (tok == NE_EOS_ID) break;
        if (tok == 265) n_265++;
        out[n_out++] = tok;
    }
    int64_t t2 = esp_timer_get_time();
    *enc_ms = (float)((double)(t1 - t0) / 1000.0);
    *tok_ms = n_out ? (float)((double)(t2 - t1) / 1000.0 / n_out) : 0.f;
    return n_out;
}

/* The IDF 5.4.2 P4 port's lazy PIE context save is incomplete: pie_save_regs /
 * pie_restore_regs in freertos/portable/riscv/portasm.S skip q3 entirely and
 * the restore writes the FFT bit-width field into SAR. Any task switch landing
 * inside a PIE kernel corrupts the row being computed. Inference therefore runs
 * with the scheduler suspended -- interrupts still fire, but no switch means
 * the PIE state is never (mis)saved. */
static void run_query_ref(const int32_t *ids, int n, const int32_t *ref, int n_ref);

static void run_query(const int32_t *ids, int n)
{
    run_query_ref(ids, n, NULL, 0);
}

static void run_query_ref(const int32_t *ids, int n,
                          const int32_t *ref, int n_ref)
{
    /* Same guard the HTTP handler takes: one query at a time against one
     * file-scope model. */
    if (!net_infer_try_lock()) { printf("E busy\n"); return; }
    ne_us_logits = ne_us_proj = 0;
    ne_us_enc_proj = ne_us_enc_attn = ne_us_enc_xkv = ne_us_enc_smax = ne_us_enc_core = ne_us_enc_kvq = 0;
    int64_t t0 = esp_timer_get_time();
    if (ne_encode(&ctx, ids, n) != 0) {
        net_infer_unlock();
        printf("E encoder length %d exceeds NE_MAX_ENC\n", n);
        return;
    }
    int64_t t1 = esp_timer_get_time();

    int32_t out[NE_MAX_GEN];
    int n_out = 0, n_265 = 0;
    int32_t tok = NE_EOS_ID;
    for (int i = 0; i < NE_MAX_GEN - 1; i++) {
        int32_t forced = ne_grammar_force ? ne_forced_next(tok, n_out, n_265) : -1;
        tok = ne_step_f(&ctx, tok, forced);
        if (tok == NE_EOS_ID) break;
        if (tok == 265) n_265++;
        out[n_out++] = tok;
    }
    int64_t t2 = esp_timer_get_time();
    double dn = n_out ? (double)n_out : 1.0;
    printf("   encoder: qkv %.0f | attn+o %.0f (kvquant %.0f, core %.0f of which softmax %.0f) | cross-kv %.0f\n",
           (double)ne_us_enc_proj / 1000.0, (double)ne_us_enc_attn / 1000.0,
           (double)ne_us_enc_kvq / 1000.0, (double)ne_us_enc_core / 1000.0,
           (double)ne_us_enc_smax / 1000.0, (double)ne_us_enc_xkv / 1000.0);
    printf("   phase/tok: logits %.1f ms  decoder-layers %.1f ms  other %.1f ms\n",
           (double)ne_us_logits / 1000.0 / dn,
           (double)ne_us_proj   / 1000.0 / dn,
           ((double)(t2 - t1) - (double)(ne_us_logits + ne_us_proj)) / 1000.0 / dn);
    ne_us_logits = ne_us_proj = 0;

    if (ref) {
        int same = (n_out == n_ref);
        for (int i = 0; same && i < n_out; i++) same = (out[i] == ref[i]);
        char sl[64];
        snprintf(sl, sizeof sl, "SMOKE %s (%d/%d tokens)", same ? "PASS" : "FAIL", n_out, n_ref);
        printf("%s\n", sl); selftest_record(sl);
    }
    printf("R %d", n_out);
    for (int i = 0; i < n_out; i++) printf(" %" PRId32, out[i]);
    net_infer_unlock();
    printf(" enc_ms=%.0f tok_ms=%.1f\n",
           (double)(t1 - t0) / 1000.0,
           n_out ? (double)(t2 - t1) / 1000.0 / n_out : 0.0);
}

void app_main(void)
{
    /* stdin arrives through the console UART, whose default VFS path has an RX
     * buffer too small for a full query line -- 271 ids is ~1400 chars, and an
     * overlong line is truncated silently. Install the driver with room to
     * spare. */
    ne_now_us = now_us;
    uart_driver_install(UART_NUM_0, 8192, 0, 0, NULL, 0);
    uart_vfs_dev_use_driver(0);

    printf("\nneedle-p4 PIE firmware\n");

    const esp_partition_t *part = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)0x40, "weights");
    if (!part) { printf("E no weights partition\n"); return; }

    uint8_t hdr[16];
    esp_partition_read(part, 0, hdr, sizeof hdr);
    uint32_t magic  = *(uint32_t *)hdr;
    uint32_t n_recs = *(uint32_t *)(hdr + 4);
    uint64_t blob   = *(uint64_t *)(hdr + 8);
    if (magic != 0x324B504Eu) {
        printf("E weights partition has no NPK2 magic (found %08" PRIx32 ")\n", magic);
        return;
    }
    size_t total = 16 + (size_t)n_recs * sizeof(ne_rec_t) + (size_t)blob;
    if (total > part->size) {
        printf("E npk header claims %u B but the partition holds %u\n",
               (unsigned)total, (unsigned)part->size);
        return;
    }
    /* esp.vld.128 requires 16-byte alignment; the packer aligns offsets within
     * the blob, so the image base itself must be 16-aligned or every weight
     * row inherits the misalignment (plain malloc gives no such promise) */
    uint8_t *img = heap_caps_aligned_alloc(16, total, MALLOC_CAP_SPIRAM);
    printf("img=%p (mod16=%u)\n", (void *)img, (unsigned)((uintptr_t)img & 0xF));
    if (!img) { printf("E PSRAM alloc failed\n"); return; }
    esp_partition_read(part, 0, img, total);
    if (ne_load(&ctx, img, total) != 0) { printf("E npk parse failed\n"); return; }
    if (ne_alloc(&ctx) != 0) { printf("E ctx alloc failed\n"); return; }
    printf("engine up; PSRAM free %u KB\n",
           (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));

    ne_critical_enter = crit_enter;
    ne_critical_exit  = crit_exit;
    kernel_unit_tests();
    determinism_selftest();
    xTaskCreatePinnedToCore(core1_worker, "ne1", 4096, NULL, 5, NULL, 1);
    while (!w_ready) vTaskDelay(1);
    ne_split_rows = split_rows;
    ne_split_gemm = split_gemm;
    ne_split_attn = split_attn;
    ne_split_for  = split_for;
    printf("second core armed\n");

    printf("smoke query (weather_sf, %d ids):\n", (int)(sizeof SMOKE / sizeof *SMOKE));
    run_query_ref(SMOKE, (int)(sizeof SMOKE / sizeof *SMOKE),
                  SMOKE_REF, (int)(sizeof SMOKE_REF / sizeof *SMOKE_REF));
    alignment_selftest();

    net_start(infer);          /* Ethernet + HTTP; DHCP prints the URL */

    static char line[4096];
    static int32_t ids[NE_MAX_ENC];
    printf("ready\n");
    for (;;) {
        if (!fgets(line, sizeof line, stdin)) { vTaskDelay(1); continue; }
        /* The auto-reset DTR/RTS toggle glitches the RX line and queues one
         * stray byte ahead of the first real command the driver ever reads.
         * Skipping to the command letter rather than to the first non-space
         * keeps that byte, or any other line noise, from eating a command. */
        char *p = line;
        while (*p && *p != 'G') p++;
        if (*p != 'G') continue;
        p++;
        int n = (int)strtol(p, &p, 10);
        if (n <= 0 || n > NE_MAX_ENC) { printf("E bad n=%d\n", n); continue; }
        int ok = 1;
        for (int i = 0; i < n; i++) {
            char *end;
            long v = strtol(p, &end, 10);
            /* end == p means no digits were consumed: the line is short, and
             * without this check strtol's 0 would pass as a valid id and the
             * rest of the list would silently be padding. */
            if (end == p || v < 0 || v >= NE_VOCAB) { ok = 0; break; }
            p = end;
            ids[i] = (int32_t)v;
        }
        if (!ok) { printf("E bad id list\n"); continue; }
        run_query(ids, n);
    }
}
