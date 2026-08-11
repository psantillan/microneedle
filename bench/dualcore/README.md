# bench/dualcore

Standalone dual-core bandwidth benchmark. One core streams PSRAM at 115 MB/s;
two cores aggregate 140-161 MB/s, so a second core buys about 1.3x, not 2x.
`esp_async_memcpy` measured 4.9 MB/s here and reset the board when overlapped
with vector work, which is why the copy path is a plain memcpy. Shares
`pie_dot.S` with `bench/vector` rather than copying it.

## The committed `sdkconfig`

The generated `sdkconfig` is tracked because two of its flags are load-bearing
and fail silently when lost:

- `CONFIG_SPI_FLASH_SUPPORT_GD_CHIP` -- without it the 32 MB flash reports as 16 MB.
- `CONFIG_IDF_EXPERIMENTAL_FEATURES` + `CONFIG_SPIRAM_SPEED_200M` -- 3.3x the PSRAM
  bandwidth, and the reason decode is not four times slower.

`sdkconfig.defaults` alone does not guarantee they survive a reconfigure.
Without the tracked `sdkconfig`, a build can come up working and much slower.
