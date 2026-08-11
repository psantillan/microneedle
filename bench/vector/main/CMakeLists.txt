# pie_dot.S lives here but is also compiled into bench/dualcore; the two benchmarks
# only compare if they measure the same instruction sequence.
idf_component_register(SRCS "pie_main.c" "pie_dot.S"
                       REQUIRES esp_psram esp_timer)
