/* PSRAM and mmapped-flash read bandwidth on the ESP32-P4.
 *
 * This exists because Needle's decode step is bandwidth-bound, not compute-bound:
 * every generated token re-streams the whole decoder weight set. So tokens/sec is
 * set by how fast this board can read weights, and nothing else matters until that
 * number is known.
 *
 * IDF 5.4.2 offers exactly two PSRAM clocks on esp32p4 -- 20 MHz (default) and
 * 200 MHz (gated behind CONFIG_IDF_EXPERIMENTAL_FEATURES). This build reports which
 * one it was compiled with, so two flashes give the A/B.
 *
 * The reads must not be optimised away, and they must go through the cache the same
 * way a real GEMV would. Hence volatile accumulators and plain sequential loads
 * rather than DMA -- DMA would measure a path the inference kernel will not use.
 */

#include <stdio.h>
#include <string.h>
#include <inttypes.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "esp_partition.h"
#include "esp_cpu.h"
#include "esp_chip_info.h"
#include "esp_flash.h"
#include "soc/soc_caps.h"
#include "rom/ets_sys.h"

/* Needle's measured per-token weight traffic, so the benchmark reports the number
 * we actually care about instead of leaving the arithmetic to a spreadsheet. */
#define DECODE_BYTES_INT4   9700000ULL   /* 4.5 bits/param over the decode path   */
#define DECODE_BYTES_INT8  18400000ULL   /* 8.5 bits/param over the same path     */

static const char *TAG = "bench";

static double mbs(size_t bytes, int64_t us)
{
    if (us <= 0) return 0.0;
    return (double)bytes / (double)us;   /* bytes/us == MB/s */
}

/* Sequential read, 32-bit loads. This is the GEMV weight-streaming pattern. */
static double seq_read_u32(const volatile uint32_t *p, size_t words, int reps)
{
    volatile uint32_t sink = 0;
    int64_t t0 = esp_timer_get_time();
    for (int r = 0; r < reps; r++) {
        for (size_t i = 0; i < words; i += 8) {
            sink += p[i + 0]; sink += p[i + 1]; sink += p[i + 2]; sink += p[i + 3];
            sink += p[i + 4]; sink += p[i + 5]; sink += p[i + 6]; sink += p[i + 7];
        }
    }
    int64_t dt = esp_timer_get_time() - t0;
    (void)sink;
    return mbs(words * 4 * (size_t)reps, dt);
}

/* Strided read at one cache line per step: the KV-cache / gather pattern, where
 * every access pays a full line fill for a fraction of the line. */
static double strided_read(const volatile uint8_t *p, size_t bytes, size_t stride, int reps)
{
    volatile uint32_t sink = 0;
    size_t n = bytes / stride;
    int64_t t0 = esp_timer_get_time();
    for (int r = 0; r < reps; r++)
        for (size_t i = 0; i < n; i++)
            sink += *(const volatile uint32_t *)(p + i * stride);
    int64_t dt = esp_timer_get_time() - t0;
    (void)sink;
    /* Report useful bytes touched, not lines filled -- that is the honest figure
     * for a kernel that only wants 4 bytes out of every stride. */
    return mbs(n * 4 * (size_t)reps, dt);
}

/* Copy a large span in chunks, never revisiting a source block. Copying one
 * small block repeatedly measures the cache, not the bus: an earlier version of
 * this benchmark reported 325 MB/s from PSRAM whose theoretical ceiling at
 * 20 MHz hex is ~80 MB/s. If a figure here exceeds clock x 2 x 2 bytes, it is
 * still measuring cache and must not be believed. */
static double memcpy_bw_stream(void *dst, const void *src, size_t total, size_t chunk)
{
    int64_t t0 = esp_timer_get_time();
    for (size_t off = 0; off + chunk <= total; off += chunk)
        memcpy(dst, (const uint8_t *)src + off, chunk);
    int64_t dt = esp_timer_get_time() - t0;
    return mbs((total / chunk) * chunk, dt);
}

/* Cycles per byte makes the loop overhead visible: a cache-line fill from PSRAM
 * has a floor set by the bus, and anything far above it is the loop, not the RAM. */
static double cycles_per_byte(double bw_mbs, int cpu_mhz)
{
    return bw_mbs > 0.0 ? (double)cpu_mhz / bw_mbs : 0.0;
}

static void banner(void)
{
    esp_chip_info_t ci;
    esp_chip_info(&ci);
    uint32_t fsize = 0;
    esp_flash_get_size(NULL, &fsize);

    printf("\n");
    printf("================================================================\n");
    printf(" ESP32-P4 memory bandwidth  (Needle port)\n");
    printf("================================================================\n");
    printf("  chip revision      : v%d.%d\n", ci.revision / 100, ci.revision % 100);
    printf("  CPU frequency      : %" PRIu32 " MHz\n", esp_cpu_get_cycle_count() ? (uint32_t)(CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ) : 0);
    printf("  flash size         : %.2f MB\n", fsize / 1048576.0);
#ifdef CONFIG_SPIRAM_SPEED
    printf("  PSRAM clock        : %d MHz", CONFIG_SPIRAM_SPEED);
#ifdef CONFIG_IDF_EXPERIMENTAL_FEATURES
    printf("  (experimental features ON)");
#endif
    printf("\n");
#endif
#ifdef CONFIG_SPIRAM_MODE_HEX
    printf("  PSRAM mode         : HEX (16-bit)\n");
#endif
    printf("  PSRAM total        : %u KB\n",
           (unsigned)(heap_caps_get_total_size(MALLOC_CAP_SPIRAM) / 1024));
    printf("  PSRAM free         : %u KB\n",
           (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));
    printf("  internal free      : %u KB\n",
           (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024));
    printf("----------------------------------------------------------------\n");
}

void app_main(void)
{
    banner();

    const size_t BUF = 4 * 1024 * 1024;      /* big enough to defeat any cache */
    uint32_t *psram = heap_caps_malloc(BUF, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!psram) {
        printf("FAIL: could not allocate %u MB of PSRAM\n", (unsigned)(BUF / 1048576));
        return;
    }
    for (size_t i = 0; i < BUF / 4; i++) psram[i] = (uint32_t)i;   /* also proves writes work */

    const size_t ISZ = 64 * 1024;
    uint32_t *isram = heap_caps_malloc(ISZ, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);

    printf("\nPSRAM\n");
    double p_seq = seq_read_u32((volatile uint32_t *)psram, BUF / 4, 2);
    printf("  sequential read (u32)          : %8.1f MB/s\n", p_seq);
    printf("  strided read, 64 B stride      : %8.2f MB/s (useful bytes)\n",
           strided_read((volatile uint8_t *)psram, BUF, 64, 2));
    printf("  strided read, 512 B stride     : %8.2f MB/s (useful bytes)\n",
           strided_read((volatile uint8_t *)psram, BUF, 512, 2));
    if (isram) {
        double m = memcpy_bw_stream(isram, psram, BUF, ISZ);
        printf("  memcpy PSRAM -> SRAM (%u MB, %u KB chunks): %8.1f MB/s\n",
               (unsigned)(BUF / 1048576), (unsigned)(ISZ / 1024), m);
        printf("      -> %.1f cycles/byte at %d MHz\n",
               cycles_per_byte(m, CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ),
               CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ);
#ifdef CONFIG_SPIRAM_SPEED
        double ceil_mbs = (double)CONFIG_SPIRAM_SPEED * 2.0 * 2.0;
        printf("      -> bus ceiling at %d MHz hex DDR is ~%.0f MB/s; %s\n",
               CONFIG_SPIRAM_SPEED, ceil_mbs,
               m > ceil_mbs ? "ABOVE CEILING - still measuring cache, do not believe it"
                            : "below ceiling, plausible");
#endif
    }

    /* Mmapped flash: the alternative home for read-only weights. If this is close
     * to PSRAM speed, weights never need to be copied into RAM at boot at all. */
    printf("\nMMAPPED FLASH\n");
    const esp_partition_t *part = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)0x40, "weights");
    if (!part) {
        printf("  (no 'weights' partition found -- skipped)\n");
    } else {
        printf("  partition          : %s  offset 0x%06" PRIx32 "  size %.2f MB\n",
               part->label, (uint32_t)part->address, part->size / 1048576.0);
        const void *map = NULL;
        esp_partition_mmap_handle_t h;
        size_t span = part->size > BUF ? BUF : part->size;
        esp_err_t err = esp_partition_mmap(part, 0, span, ESP_PARTITION_MMAP_DATA, &map, &h);
        if (err != ESP_OK) {
            printf("  esp_partition_mmap(%u MB) FAILED: %s\n",
                   (unsigned)(span / 1048576), esp_err_to_name(err));
        } else {
            printf("  mmapped %u MB OK\n", (unsigned)(span / 1048576));
            printf("  sequential read (u32)          : %8.1f MB/s\n",
                   seq_read_u32((const volatile uint32_t *)map, span / 4, 2));
            printf("  strided read, 64 B stride      : %8.2f MB/s (useful bytes)\n",
                   strided_read((const volatile uint8_t *)map, span, 64, 1));
            esp_partition_munmap(h);
        }
    }

    printf("\n----------------------------------------------------------------\n");
    printf("WHAT THIS MEANS FOR NEEDLE\n");
    printf("  A decode step re-streams the whole decoder weight set.\n");
    printf("  int4 (4.5 bits/param): %.2f MB/token\n", DECODE_BYTES_INT4 / 1048576.0);
    printf("  int8 (8.5 bits/param): %.2f MB/token\n", DECODE_BYTES_INT8 / 1048576.0);
    if (p_seq > 1.0) {
        double t4 = DECODE_BYTES_INT4 / (p_seq * 1e6) * 1000.0;
        double t8 = DECODE_BYTES_INT8 / (p_seq * 1e6) * 1000.0;
        printf("  at the measured PSRAM sequential rate:\n");
        printf("    int4  %7.1f ms/token  ->  %5.1f s for a 40-token tool call\n", t4, t4 * 40 / 1000.0);
        printf("    int8  %7.1f ms/token  ->  %5.1f s for a 40-token tool call\n", t8, t8 * 40 / 1000.0);
        printf("  (lower bound: assumes zero compute cost and perfect streaming)\n");
    printf("\n  NOTE: a real GEMV kernel does not read weights with scalar loads --\n");
    printf("  it bulk-copies a tile into internal SRAM and computes from there.\n");
    printf("  The streamed-memcpy figure above is the rate that kernel would see.\n");
    }
    printf("================================================================\n");

    while (1) vTaskDelay(pdMS_TO_TICKS(10000));
}
