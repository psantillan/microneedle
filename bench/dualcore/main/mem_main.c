/* Does a second core buy PSRAM bandwidth, or only contend for it?
 *
 * This decides whether splitting the logits GEMV across cores is worth
 * building. The decode path is bandwidth-bound (4.19M MACs = 2.7 ms of PIE
 * compute against ~40 ms of weight fetch), so a second core only helps if two
 * cores in flight raise the ACHIEVED PSRAM read rate. If 110 MB/s is a bus
 * ceiling rather than a single-core outstanding-request limit, dual-core is
 * worthless here and tiling through SRAM is the only lever.
 *
 * Also measured: GDMA async memcpy PSRAM->SRAM, and whether PIE compute can
 * actually overlap a DMA transfer in flight (the premise of double-buffering).
 */

#include <stdio.h>
#include <string.h>
#include <inttypes.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "esp_async_memcpy.h"

#define SPAN   (2 * 1024 * 1024)   /* 2 MB per core: far past any cache */
#define TILE   (32 * 1024)

extern int32_t pie_dot_s8(const int8_t *x, const int8_t *w, int len);

static uint8_t *psram_a, *psram_b;
static volatile int      go, done_b;
static volatile uint64_t bytes_b, us_b;

/* Sum a span with plain 32-bit loads. Returned checksum keeps the reads live;
 * a memcpy would measure the copy engine, not the read path. */
static uint32_t stream_read(const uint8_t *p, size_t n)
{
    const uint32_t *w = (const uint32_t *)p;
    uint32_t acc = 0;
    for (size_t i = 0; i < n / 4; i += 8) {
        acc += w[i]   + w[i+1] + w[i+2] + w[i+3];
        acc += w[i+4] + w[i+5] + w[i+6] + w[i+7];
    }
    return acc;
}

static void core1_task(void *arg)
{
    for (;;) {
        while (!go) { __asm__ volatile("nop"); }
        int64_t t0 = esp_timer_get_time();
        volatile uint32_t s = stream_read(psram_b, SPAN);
        (void)s;
        us_b = (uint64_t)(esp_timer_get_time() - t0);
        bytes_b = SPAN;
        done_b = 1;
        while (go) { __asm__ volatile("nop"); }
        done_b = 0;
    }
}

static double mbs(size_t bytes, int64_t us) { return (double)bytes / (double)us; }

void app_main(void)
{
    printf("\n=== PSRAM bandwidth: one core vs two ===\n");
    psram_a = heap_caps_aligned_alloc(64, SPAN, MALLOC_CAP_SPIRAM);
    psram_b = heap_caps_aligned_alloc(64, SPAN, MALLOC_CAP_SPIRAM);
    printf("PSRAM free %u KB, largest block %u KB\n",
           (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM)/1024),
           (unsigned)(heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM)/1024));
    if (!psram_a || !psram_b) { printf("E alloc\n"); return; }
    memset(psram_a, 0x5A, SPAN);
    memset(psram_b, 0xA5, SPAN);
    printf("spans: a=%p b=%p  %d MB each\n", psram_a, psram_b, SPAN / (1024*1024));

    /* --- 1. single core --- */
    int64_t t0 = esp_timer_get_time();
    volatile uint32_t s1 = stream_read(psram_a, SPAN);
    int64_t t1 = esp_timer_get_time();
    (void)s1;
    double one = mbs(SPAN, t1 - t0);
    printf("\n1) core0 alone            : %7.1f MB/s  (%lld us)\n", one, (long long)(t1 - t0));

    /* --- 2. both cores, disjoint spans --- */
    xTaskCreatePinnedToCore(core1_task, "c1", 4096, NULL, 5, NULL, 1);
    vTaskDelay(pdMS_TO_TICKS(50));

    done_b = 0;
    go = 1;
    t0 = esp_timer_get_time();
    volatile uint32_t s2 = stream_read(psram_a, SPAN);
    int64_t t_a = esp_timer_get_time() - t0;
    (void)s2;
    while (!done_b) { __asm__ volatile("nop"); }
    int64_t wall = esp_timer_get_time() - t0;
    go = 0;

    double a_rate = mbs(SPAN, t_a), b_rate = mbs(SPAN, us_b);
    double agg    = mbs((size_t)SPAN * 2, wall);
    printf("2) core0 while core1 runs : %7.1f MB/s\n", a_rate);
    printf("   core1                  : %7.1f MB/s\n", b_rate);
    printf("   AGGREGATE (8 MB/wall)  : %7.1f MB/s   -> %.2fx single core\n", agg, agg / one);
    printf("   %s\n", agg > one * 1.25
           ? "second core ADDS bandwidth: splitting rows across cores is worth it"
           : "bus-limited: a second core buys nothing on a bandwidth-bound GEMV");

    /* --- 3. GDMA async memcpy PSRAM -> internal SRAM --- */
    printf("\n=== staging tiles through internal SRAM ===\n");
    uint8_t *sram = heap_caps_aligned_alloc(64, TILE * 2, MALLOC_CAP_INTERNAL | MALLOC_CAP_DMA);
    if (!sram) { printf("E no internal SRAM for tiles\n"); return; }

    /* blocking memcpy baseline */
    t0 = esp_timer_get_time();
    for (int off = 0; off + TILE <= SPAN; off += TILE) memcpy(sram, psram_a + off, TILE);
    t1 = esp_timer_get_time();
    printf("3) memcpy PSRAM->SRAM     : %7.1f MB/s\n", mbs(SPAN, t1 - t0));

    async_memcpy_config_t cfg = ASYNC_MEMCPY_DEFAULT_CONFIG();
    cfg.backlog = 4;
    async_memcpy_handle_t mcp = NULL;
    if (esp_async_memcpy_install(&cfg, &mcp) != ESP_OK) {
        printf("   (async memcpy unavailable)\n");
    } else {
        t0 = esp_timer_get_time();
        for (int off = 0; off + TILE <= SPAN; off += TILE) {
            esp_async_memcpy(mcp, sram, psram_a + off, TILE, NULL, NULL);
        }
        /* drain: a zero-length-ish final copy with a wait is not exposed, so
         * bound the transfer by issuing then measuring a final blocking copy */
        memcpy(sram + TILE, psram_a, 64);
        t1 = esp_timer_get_time();
        printf("4) GDMA async issue+run   : %7.1f MB/s\n", mbs(SPAN, t1 - t0));
    }

    /* --- 4. can PIE compute overlap a DMA in flight? --- */
    int8_t *x = heap_caps_aligned_alloc(16, 4096, MALLOC_CAP_INTERNAL);
    int8_t *w = heap_caps_aligned_alloc(16, 4096, MALLOC_CAP_INTERNAL);
    memset(x, 3, 4096); memset(w, 5, 4096);

    vTaskSuspendAll();
    t0 = esp_timer_get_time();
    volatile int32_t acc = 0;
    for (int i = 0; i < 4000; i++) acc += pie_dot_s8(x, w, 4096);
    t1 = esp_timer_get_time();
    xTaskResumeAll();
    int64_t solo = t1 - t0;
    printf("\n5) PIE compute, quiet bus : %lld us for 16.4M MAC\n", (long long)solo);

    if (mcp) {
        vTaskSuspendAll();
        t0 = esp_timer_get_time();
        for (int off = 0; off + TILE <= SPAN; off += TILE)
            esp_async_memcpy(mcp, sram, psram_a + off, TILE, NULL, NULL);
        acc = 0;
        for (int i = 0; i < 4000; i++) acc += pie_dot_s8(x, w, 4096);
        t1 = esp_timer_get_time();
        xTaskResumeAll();
        int64_t with = t1 - t0;
        printf("6) PIE compute + DMA busy : %lld us  -> %.0f%% slowdown\n",
               (long long)with, 100.0 * ((double)with / (double)solo - 1.0));
        printf("   %s\n", (double)with < (double)solo * 1.30
               ? "compute survives a busy bus: double-buffering will overlap"
               : "DMA starves the core: overlap gains are limited");
    }
    printf("\ndone. acc=%d\n", (int)acc);
}
