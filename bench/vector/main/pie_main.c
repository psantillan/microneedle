/* Does the PIE unit actually deliver? Measured, not asserted.
 *
 * Three inner loops over the same int8 data, results checked for exact
 * equality before any timing is believed:
 *   scalar   -- plain C int MAC, what the engine effectively does today
 *   pie_full -- esp-dsp idiom, one accumulator drain per 512-element row
 *   pie_g32  -- drain every 32 elements + fp32 scale, i.e. what the real
 *               group-32 GEMV must do; the delta vs pie_full is the price
 *               of our quantization granularity on this silicon
 *
 * Each is run from internal SRAM (the tiled-kernel case: weights staged
 * through a tile) and from PSRAM (the lazy case: PIE reading the bus
 * directly). The gap between those two answers whether tiling is mandatory.
 */

#include <stdio.h>
#include <string.h>
#include <inttypes.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_heap_caps.h"
#include "esp_cpu.h"

#define ROWS 128            /* 128x512 int8 = 64 KB: fits internal SRAM */
#define COLS 512
#define GROUP 32
#define NG   (COLS / GROUP)
#define REPS 50

extern int32_t pie_dot_s8(const int8_t *x, const int8_t *w, int len);
extern void    pie_dot_s8_g32(const int8_t *x, const int8_t *w, int len,
                              int32_t *gsums);

static int32_t scalar_dot(const int8_t *x, const int8_t *w, int len)
{
    int32_t acc = 0;
    for (int i = 0; i < len; i++) acc += (int32_t)x[i] * (int32_t)w[i];
    return acc;
}

static float scalar_dot_g32(const int8_t *x, const int8_t *w, int len,
                            const float *scales)
{
    float acc = 0.f;
    for (int g = 0; g < len / GROUP; g++) {
        int32_t d = 0;
        for (int k = 0; k < GROUP; k++)
            d += (int32_t)x[g * GROUP + k] * (int32_t)w[g * GROUP + k];
        acc += scales[g] * (float)d;
    }
    return acc;
}

static uint32_t rng = 0x1234567;
static int8_t rnd8(void) { rng = rng * 1103515245 + 12345; return (int8_t)(rng >> 16); }

typedef struct { double cyc_per_mac; double macs_per_cyc; } perf_t;

static perf_t bench(const char *label, const int8_t *x, const int8_t *W,
                    int variant, const float *scales, volatile float *sinkf)
{
    /* verify first: every variant must reproduce the scalar result exactly
     * (integer paths) before its timing means anything */
    int32_t gs[NG];
    for (int r = 0; r < ROWS; r += 37) {
        const int8_t *row = W + (size_t)r * COLS;
        int32_t ref = scalar_dot(x, row, COLS);
        if (variant == 1 && pie_dot_s8(x, row, COLS) != ref) {
            printf("  %s: VERIFY FAIL row %d\n", label, r);
            return (perf_t){0, 0};
        }
        if (variant == 2) {
            pie_dot_s8_g32(x, row, COLS, gs);
            int32_t sum = 0;
            for (int g = 0; g < NG; g++) sum += gs[g];
            if (sum != ref) {
                printf("  %s: VERIFY FAIL row %d (%" PRId32 " != %" PRId32 ")\n",
                       label, r, sum, ref);
                return (perf_t){0, 0};
            }
        }
    }

    volatile int32_t sink = 0;
    uint32_t c0 = esp_cpu_get_cycle_count();
    for (int rep = 0; rep < REPS; rep++) {
        for (int r = 0; r < ROWS; r++) {
            const int8_t *row = W + (size_t)r * COLS;
            switch (variant) {
            case 0: sink += scalar_dot(x, row, COLS); break;
            case 1: sink += pie_dot_s8(x, row, COLS); break;
            case 2: {
                pie_dot_s8_g32(x, row, COLS, gs);
                float acc = 0.f;
                for (int g = 0; g < NG; g++) acc += scales[g] * (float)gs[g];
                *sinkf += acc;
                break;
            }
            }
        }
    }
    uint32_t cyc = esp_cpu_get_cycle_count() - c0;
    (void)sink;

    double macs = (double)REPS * ROWS * COLS;
    perf_t p = { (double)cyc / macs, macs / (double)cyc };
    printf("  %-28s %10" PRIu32 " cyc  %6.3f cyc/MAC  %6.2f MAC/cyc\n",
           label, cyc, p.cyc_per_mac, p.macs_per_cyc);
    return p;
}

void app_main(void)
{
    printf("\nPIE utilization, ESP32-P4 @ %d MHz, %dx%d int8, %d reps\n",
           CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ, ROWS, COLS, REPS);

    int8_t *x  = heap_caps_aligned_alloc(16, COLS, MALLOC_CAP_INTERNAL);
    int8_t *Wi = heap_caps_aligned_alloc(16, (size_t)ROWS * COLS, MALLOC_CAP_INTERNAL);
    int8_t *Wp = heap_caps_aligned_alloc(16, (size_t)ROWS * COLS, MALLOC_CAP_SPIRAM);
    static float scales[NG];
    static volatile float sinkf;
    if (!x || !Wi || !Wp) { printf("alloc failed\n"); return; }

    for (int i = 0; i < COLS; i++) x[i] = rnd8();
    for (size_t i = 0; i < (size_t)ROWS * COLS; i++) Wi[i] = rnd8();
    memcpy(Wp, Wi, (size_t)ROWS * COLS);
    for (int g = 0; g < NG; g++) scales[g] = 0.01f * (float)(g + 1);

    printf("\nweights in INTERNAL SRAM (tiled-kernel case):\n");
    perf_t s  = bench("scalar int MAC", x, Wi, 0, scales, &sinkf);
    perf_t pf = bench("PIE, drain per row", x, Wi, 1, scales, &sinkf);
    perf_t pg = bench("PIE, drain per group of 32", x, Wi, 2, scales, &sinkf);

    printf("\nweights in PSRAM (no tiling):\n");
    bench("scalar int MAC", x, Wp, 0, scales, &sinkf);
    bench("PIE, drain per row", x, Wp, 1, scales, &sinkf);
    bench("PIE, drain per group of 32", x, Wp, 2, scales, &sinkf);

    if (s.cyc_per_mac > 0 && pf.macs_per_cyc > 0) {
        printf("\nSRAM speedups vs scalar: full-drain %.1fx, group-32 %.1fx\n",
               s.cyc_per_mac / pf.cyc_per_mac, s.cyc_per_mac / pg.cyc_per_mac);
        printf("group-32 drain tax vs single drain: %.0f%%\n",
               100.0 * (pg.cyc_per_mac / pf.cyc_per_mac - 1.0));
    }
    printf("done\n");
    while (1) vTaskDelay(pdMS_TO_TICKS(10000));
}
