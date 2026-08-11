# bench/bandwidth

Standalone PSRAM and flash bandwidth benchmark. Measured 33.5 MB/s at the
default 20 MHz PSRAM clock and 110.6 MB/s at 200 MHz, streamed across a span
larger than the cache. Memory-mapped flash stays at 27 MB/s either way, which
is why the weights are copied into PSRAM at boot.

## The committed `sdkconfig`

The generated `sdkconfig` is tracked because two of its flags are load-bearing
and fail silently when lost:

- `CONFIG_SPI_FLASH_SUPPORT_GD_CHIP` -- without it the 32 MB flash reports as 16 MB.
- `CONFIG_IDF_EXPERIMENTAL_FEATURES` + `CONFIG_SPIRAM_SPEED_200M` -- 3.3x the PSRAM
  bandwidth, and the reason decode is not four times slower.

`sdkconfig.defaults` alone does not guarantee they survive a reconfigure.
Without the tracked `sdkconfig`, a build can come up working and much slower.
