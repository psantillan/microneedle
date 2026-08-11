# bench/vector

Standalone vector-unit utilization benchmark. Scalar int8 runs at 8.03
cycles/MAC; the PIE kernel with a single accumulator drain runs at 0.236, i.e.
4.24 MAC/cycle. Draining once per group of 32 instead costs 0.872 cycles/MAC,
which is the price of group-wise quantization on this silicon.

For why engine-level projections and attention cost more than this kernel rate,
see **`bench/gemm`** — the successor instrument (links `engine/pie_rows.S`,
decomposes GEMM vs attention, records the QACC negative and the s8 stage).

## The committed `sdkconfig`

The generated `sdkconfig` is tracked because two of its flags are load-bearing
and fail silently when lost:

- `CONFIG_SPI_FLASH_SUPPORT_GD_CHIP` -- without it the 32 MB flash reports as 16 MB.
- `CONFIG_IDF_EXPERIMENTAL_FEATURES` + `CONFIG_SPIRAM_SPEED_200M` -- 3.3x the PSRAM
  bandwidth, and the reason decode is not four times slower.

`sdkconfig.defaults` alone does not guarantee they survive a reconfigure.
Without the tracked `sdkconfig`, a build can come up working and much slower.
