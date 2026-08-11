# firmware

The board application: loads the weights, serves the demo over Ethernet, and runs kernel self-tests at boot.

## Why `sdkconfig` is committed here

The generated `sdkconfig` is tracked. Several flags in it are load-bearing and
fail silently when lost:

- `CONFIG_SPI_FLASH_SUPPORT_GD_CHIP` -- without it the 32 MB flash reports as 16 MB.
- `CONFIG_IDF_EXPERIMENTAL_FEATURES` + `CONFIG_SPIRAM_SPEED_200M` -- 3.3x the PSRAM
  bandwidth, which decode speed depends on.

`sdkconfig.defaults` alone does not guarantee they survive a reconfigure.
Without the tracked `sdkconfig`, the build still works and is quietly much
slower.
